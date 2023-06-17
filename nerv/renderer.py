import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer import NDCMultinomialRaysampler
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras, 
    look_at_rotation,
    look_at_view_transform, 
)
from diffusers import UNet2DModel
from monai.networks.nets import Unet, EfficientNetBN
from monai.networks.layers.factories import Norm

from nerfstudio.field_components import encodings

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

def inverse_look_at_view_transform(R, T, degrees=True):
    """
    This function calculates the distance (dist), elevation (elev),
    and azimuth (azim) angles from the rotation matrix (R) and
    translation vector (T) obtained from the 'look_at_view_transform' function.

    Args:
        R: Rotation matrix of shape (N, 3, 3).
        T: Translation vector of shape (N, 3).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles should be returned in degrees or radians.

    Returns:
        3-element tuple containing

        - **dist**: distance of the camera from the object(s).
        - **elev**: elevation angle between the vector from the object
            to the camera and the horizontal plane y = 0 (xz-plane).
        - **azim**: azimuth angle between the projected vector from the
            object to the camera and a reference vector at (1, 0, 0) on
            the reference plane (the horizontal plane).

    """
    R = R.view(-1, 3, 3)
    T = T.view(-1, 3)
    # Calculate the distance (dist) from the translation vector
    dist = torch.norm(T, dim=1)

    # Calculate the elevation (elev) angle
    elev = torch.asin(-R[:, 1, 1])
    # Calculate the azimuth (azim) angle
    azim = torch.atan2(R[:, 0, 2], R[:, 2, 2])
    # Normalize to -1 1
    elev = elev / 90
    azim = azim / 180
    if degrees:
        elev = elev * 180.0 / math.pi
        azim = azim * 180.0 / math.pi

    return dist, elev, azim

def make_cameras_dea(
    dist: torch.Tensor, 
    elev: torch.Tensor, 
    azim: torch.Tensor
    ):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(
        dist=dist.float(), 
        elev=elev.float() * 90, 
        azim=azim.float() * 180
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=16, znear=8.0, zfar=12.0).to(_device)

def make_cameras_RT(
    R: torch.Tensor, 
    T: torch.Tensor
    ):
    R = R.view(-1, 3, 3)
    T = T.view(-1, 3)
    assert R.device == T.device
    _device = R.device
    return FoVPerspectiveCameras(R=R, T=T, fov=16, znear=8.0, zfar=12.0).to(_device)


class NeRVFrontToBackFrustumFeaturer(nn.Module):
    def __init__(self, 
        in_channels=1, 
        out_channels=1, 
        backbone="efficientnet-b7"
    ) -> None:
        super().__init__()
        assert backbone in backbones.keys()
        self.model = EfficientNetBN(
            model_name=backbone, #(24, 32, 56, 160, 448)
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=out_channels,
            pretrained=True, 
            adv_prop=True,
        )

    def forward(self, figures):
        camfeat = self.model.forward(figures)
        return camfeat

class NeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(self, 
        in_channels=1, 
        out_channels=1, 
        img_shape=400, 
        vol_shape=256, 
        n_pts_per_ray=256, 
        sh=0, 
        pe=8, 
        backbone="efficientnet-b7"
    ) -> None:
        super().__init__()
        self.sh = sh
        self.pe = pe
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.n_pts_per_ray = n_pts_per_ray
        assert backbone in backbones.keys()
        if self.pe>0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
            num_frequencies = self.pe
            min_freq_exp = 0
            max_freq_exp = 8
            encoder = encodings.NeRFEncoding(
                in_dim=self.pe, 
                num_frequencies=num_frequencies, 
                min_freq_exp=min_freq_exp, 
                max_freq_exp=max_freq_exp
            )
            pebasis = encoder(zyx.view(-1, 3))
            pebasis = pebasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer('pebasis', pebasis)

        if self.sh > 0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
            
            encoder = encodings.SHEncoding(self.sh)
            assert out_channels == self.sh**2 if self.sh>0 else 1
            shbasis = encoder(zyx.view(-1, 3))
            shbasis = shbasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer('shbasis', shbasis)
            
        self.clarity_net = UNet2DModel(
            sample_size=self.img_shape,  
            in_channels=self.in_channels,  
            out_channels=self.n_pts_per_ray,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=backbones[backbone],  # More channels -> more parameters
            norm_num_groups=8,
            down_block_types=(
                "DownBlock2D",  
                "DownBlock2D",  
                "DownBlock2D",
                "AttnDownBlock2D",  
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",    
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",    
            ),
            class_embed_type="timestep",
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1+(2*3*self.pe),
                out_channels=1,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2+(2*3*self.pe),
                out_channels=1,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ),
        )

        self.refiner_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=3+(2*3*self.pe),
                out_channels=self.out_channels,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.5,
            ), 
        )

        self.raysampler = NDCMultinomialRaysampler(  
            image_width=self.img_shape,
            image_height=self.img_shape,
            n_pts_per_ray=self.n_pts_per_ray,  
            min_depth=8.0,
            max_depth=4.0,
        )        
        
    def forward(self, figures, elev, azim, n_views=[2, 1], resample_clarity=False, resample_volumes=False):
        _device = figures.device
        B = figures.shape[0]
        assert B==sum(n_views) # batch must be equal to number of projections
        clarity = self.clarity_net(
            figures, timestep=elev, class_labels=azim).sample.view(-1, 1, self.n_pts_per_ray, self.img_shape, self.img_shape)

        if resample_clarity:
            z = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            y = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            x = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            coords = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(B, 1, 1) # 1 DHW 3 to B DHW 3
            # Process (resample) the clarity from ray views to ndc
            dist = 10.0 * torch.ones(B, device=_device)
            cameras = make_cameras_dea(dist, elev, azim)
            
            points = cameras.transform_points_ndc(coords) # world to ndc, 1 DHW 3
            values = F.grid_sample(
                clarity,
                points.view(-1, self.vol_shape, self.vol_shape, self.vol_shape, 3),
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            
            scenes = torch.split(values, split_size_or_sections=n_views, dim=0) # 31SHW = [21SHW, 11SHW]
            interp = []
            for scene_, n_view in zip(scenes, n_views):
                value_ = scene_.mean(dim=0, keepdim=True)
                interp.append(value_)
                
            clarity = torch.cat(interp, dim=0)

        if self.pe > 0:
            density = self.density_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity], dim=1))
            mixture = self.mixture_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density], dim=1))
            shcoeff = self.refiner_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density, mixture], dim=1))
        else:
            density = self.density_net(torch.cat([clarity], dim=1)) # density = torch.add(density, clarity)
            mixture = self.mixture_net(torch.cat([clarity, density], dim=1)) # mixture = torch.add(mixture, clarity)
            shcoeff = self.refiner_net(torch.cat([clarity, density, mixture], dim=1)) # shcoeff = torch.add(shcoeff, clarity)
        if self.sh > 0:
            shcomps = torch.einsum('abcde,bcde->abcde', shcoeff, self.shbasis)
        else:
            shcomps = shcoeff 
        
        volumes = []
        for idx, n_view in enumerate(n_views):
            volume = shcomps[[idx]].repeat(n_view, 1, 1, 1, 1)
            volumes.append(volume)
        volumes = torch.cat(volumes, dim=0)
        
        if resample_volumes:
            z = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            y = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            x = torch.linspace(-1.5, 1.5, steps=self.vol_shape, device=_device)
            coords = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(B, 1, 1) # 1 DHW 3 to B DHW 3
            # Process (resample) the clarity from ray views to ndc
            dist = 10.0 * torch.ones(B, device=_device)
            cameras = make_cameras_dea(dist, elev, azim)
            
            points = cameras.transform_points_ndc(coords) # world to ndc, 1 DHW 3
            values = F.grid_sample(
                volumes,
                points.view(-1, self.vol_shape, self.vol_shape, self.vol_shape, 3),
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            
            scenes = torch.split(values, split_size_or_sections=n_views, dim=0) # 31SHW = [21SHW, 11SHW]
            interp = []
            for scene_, n_view in zip(scenes, n_views):
                value_ = scene_.mean(dim=0, keepdim=True)
                interp.append(value_)
                
            volumes = torch.cat(interp, dim=0)
        
        return volumes
