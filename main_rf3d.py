import os
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributed.fsdp.wrap import wrap
torch.set_float32_matmul_precision('medium')

from typing import Optional, NamedTuple
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel

from pytorch3d.renderer.camera_utils import join_cameras_as_batch

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
from nerv.renderer import NeRVFrontToBackInverseRenderer, NeRVFrontToBackFrustumFeaturer, make_cameras_dea 

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class RF3DLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.img = hparams.img
        self.vol = hparams.vol
        self.cam = hparams.cam
        self.sup = hparams.sup
        self.ckpt = hparams.ckpt
        self.strict = hparams.strict
        
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.ddpmbeta = hparams.ddpmbeta
        
        self.lambda_gp = hparams.lambda_gp
        self.clamp_val = hparams.clamp_val
        self.timesteps = hparams.timesteps
        
        self.ddpm_noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.timesteps, 
            beta_schedule="scaled_linear",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )
                
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.timesteps, 
            beta_schedule="scaled_linear",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
            prediction_type="sample", # or "epsilon"
            # steps_offset=20,
        )
        self.ddim_noise_scheduler.set_timesteps(50)
        
        self.logsdir = hparams.logsdir
       
        self.sh = hparams.sh
        self.pe = hparams.pe
        
        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.backbone = hparams.backbone
        self.devices = hparams.devices
        
        self.save_hyperparameters()

        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=4.0, 
            max_depth=8.0, 
            ndc_extent=4.0,
        )
        
        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh**2 if self.sh>0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray,
            sh=self.sh, 
            pe=self.pe,
            backbone=self.backbone,
        )
        
        if self.ckpt:
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)
 

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.l1loss = nn.L1Loss(reduction="mean")

    def forward_screen(self, image3d, cameras):   
        return self.fwd_renderer(image3d * 0.5 + 0.5/image3d.shape[1], cameras) * 2.0 - 1.0

    def forward_volume(self, 
                       image2d, 
                       timesteps, 
                       dist, 
                       elev, 
                       azim, 
                       path=0, 
                       n_views=[2, 1], 
                       resample_clarity=True, 
                       resample_volumes=False): 
        return self.inv_renderer(image2d, 
                                 timesteps, 
                                 dist, 
                                 elev.squeeze(1), 
                                 azim.squeeze(1), 
                                 path, 
                                 n_views, 
                                 resample_clarity=resample_clarity, 
                                 resample_volumes=resample_volumes) 

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        batchsz = image2d.shape[0]
        timezeros = torch.Tensor([0]).to(_device)
        timemones = torch.Tensor([-1]).to(_device)
            
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1 # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=30, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=30, znear=4, zfar=8)
        
        view_shape_ = [self.batch_size, 1] 
        
        path_diffuse = 1.0 * torch.ones(self.batch_size, device=_device)
        path_inverse = 0.0 * torch.ones(self.batch_size, device=_device)
        
        volume_xr_nograd = self.forward_volume(
            image2d=image2d, 
            timesteps=timezeros,
            dist=6.0,
            elev=elev_hidden.view(view_shape_), 
            azim=azim_hidden.view(view_shape_), 
            path=path_inverse, 
            n_views=[1], 
            resample_clarity=True, 
            resample_volumes=False,
        ).sum(dim=1, keepdim=True).detach()
                
        # Construct the samples in 2D
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_xr_hidden = image2d 
        
        timesteps = torch.randint(0, self.ddim_noise_scheduler.config.num_train_timesteps, (batchsz,), device=_device).long()
            
        if batch_idx%2==0:
            # Diffusion step
            volume_ct_latent = torch.randn_like(image3d)
            volume_ct_interp = self.ddim_noise_scheduler.add_noise(image3d, volume_ct_latent, timesteps=timesteps)
            volume_xr_latent = torch.randn_like(image3d)
            volume_xr_interp = self.ddim_noise_scheduler.add_noise(volume_xr_nograd, volume_xr_latent, timesteps=timesteps)
            
            figure_ct_latent = self.forward_screen(image3d=volume_ct_latent, cameras=view_random)
            figure_ct_interp = self.forward_screen(image3d=volume_ct_interp, cameras=view_random)
                    
            figure_xr_latent = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
            figure_xr_interp = self.forward_screen(image3d=volume_xr_interp, cameras=view_hidden)
                 
            # Diffuse constraint
            volume_dx_diffuse = self.forward_volume(
                image2d=torch.cat([figure_ct_interp, figure_xr_interp]),
                timesteps=timesteps,
                dist=6.0,
                elev=torch.cat([elev_random.view(view_shape_), elev_hidden.view(view_shape_)]),
                azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                path=path_diffuse,
                n_views=[1, 1],            
            )
            volume_ct_diffuse, volume_xr_diffuse = torch.split(volume_dx_diffuse, batchsz)    
            figure_ct_diffuse = self.forward_screen(image3d=volume_ct_diffuse, cameras=view_random)
            figure_xr_diffuse = self.forward_screen(image3d=volume_xr_diffuse, cameras=view_hidden)
            volume_ct_diffuse = volume_ct_diffuse.sum(dim=1, keepdim=True)
            volume_xr_diffuse = volume_xr_diffuse.sum(dim=1, keepdim=True)
                  
        elif batch_idx%2==1:    
            volume_dx_inverse = self.forward_volume(
                image2d=torch.cat([figure_ct_random, figure_xr_hidden]),
                timesteps=timezeros,
                dist=6.0,
                elev=torch.cat([elev_random.view(view_shape_), elev_hidden.view(view_shape_)]),
                azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                path=path_inverse,
                n_views=[1, 1],            
            )
            volume_ct_inverse, volume_xr_inverse = torch.split(volume_dx_inverse, batchsz)    
            figure_ct_inverse = self.forward_screen(image3d=volume_ct_inverse, cameras=view_random)
            figure_xr_inverse = self.forward_screen(image3d=volume_xr_inverse, cameras=view_hidden)
            volume_ct_inverse = volume_ct_inverse.sum(dim=1, keepdim=True)
            volume_xr_inverse = volume_xr_inverse.sum(dim=1, keepdim=True)
      
            
        if self.ddim_noise_scheduler.prediction_type=="epsilon":
            pass
        elif self.ddim_noise_scheduler.prediction_type=="sample": 
            if batch_idx%2==0:
                im3d_loss_ct = self.l1loss(volume_ct_diffuse, image3d) 
                im2d_loss_ct = self.l1loss(figure_ct_diffuse, figure_ct_random) 
                im2d_loss_xr = self.l1loss(figure_xr_diffuse, figure_xr_hidden)       
                
            elif batch_idx%2==1:      
                im3d_loss_ct = self.l1loss(volume_ct_inverse, image3d)
                im2d_loss_ct = self.l1loss(figure_ct_inverse, figure_ct_random) 
                im2d_loss_xr = self.l1loss(figure_xr_inverse, figure_xr_hidden) 
                          
            im3d_loss = im3d_loss_ct
            self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), 
                     prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            im2d_loss = im2d_loss_ct + im2d_loss_xr
            self.log(
                f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            loss = self.alpha*im3d_loss + self.gamma*im2d_loss

            alpha_t = _extract_into_tensor(
                self.ddim_noise_scheduler.alphas_cumprod.to(_device), timesteps, (image3d.shape[0], 1, 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * loss
                    
        # Visualization step 
        if batch_idx==0:
            with torch.no_grad():
                volume_output_sampled = torch.cat([volume_ct_latent.clone(), volume_xr_latent.clone()])
                figure_output_sampled = torch.cat([figure_ct_latent.clone(), figure_xr_latent.clone()])
                
                for t in tqdm(self.ddim_noise_scheduler.timesteps):
                    # 1. predict noise model_output
                    tensor_output_sampled = self.forward_volume(
                        image2d=figure_output_sampled,
                        timesteps=t,
                        dist=6.0,
                        elev=torch.cat([elev_random.view(view_shape_), elev_hidden.view(view_shape_)]),
                        azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                        path=path_diffuse,
                        n_views=[1, 1]
                    )
                    # Perform Post activation like DVGO      
                    tensor_output_sampled = tensor_output_sampled.sum(dim=1, keepdim=True)
                    
                    # 2. compute previous image: x_t -> x_t-1
                    volume_output_sampled = self.ddim_noise_scheduler.step(
                        tensor_output_sampled, 
                        t, 
                        volume_output_sampled, 
                        eta=0,
                        use_clipped_model_output=False,
                        generator=None).prev_sample
                    figure_output_sampled = self.forward_screen(
                        image3d=volume_output_sampled, 
                        cameras=join_cameras_as_batch([view_random, view_hidden]))
                    
                figure_ct_sampled, figure_xr_sampled = torch.split(figure_output_sampled, batchsz)
                volume_ct_sampled, volume_xr_sampled = torch.split(volume_output_sampled, batchsz)
                
                # Additionally estimate the volume
                volume_output_inverse = self.forward_volume(
                    image2d=torch.cat([figure_ct_random, figure_xr_hidden]), 
                    timesteps=timezeros,
                    dist=6.0,
                    elev=torch.cat([elev_random.view(view_shape_), elev_hidden.view(view_shape_)]),
                    azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                    path=path_inverse,
                    n_views=[1, 1]
                )
                figure_output_inverse = self.forward_screen(image3d=volume_output_inverse, 
                                                            cameras=join_cameras_as_batch([view_random, view_hidden]))
                
                # Perform Post activation like DVGO      
                volume_output_inverse = volume_output_inverse.sum(dim=1, keepdim=True)
                    
                figure_ct_inverse, figure_xr_inverse = torch.split(figure_output_inverse, batchsz)
                volume_ct_inverse, volume_xr_inverse = torch.split(volume_output_inverse, batchsz)

                gen2d = torch.cat([
                    torch.cat([image3d[..., self.vol_shape//2, :],
                               figure_ct_random, 
                               figure_ct_interp, 
                               volume_ct_inverse[..., self.vol_shape//2, :],
                               figure_ct_inverse,
                               volume_ct_sampled[..., self.vol_shape//2, :],
                               figure_ct_sampled,
                               ], dim=-2).transpose(2, 3),
                    torch.cat([volume_xr_nograd.sum(dim=1, keepdim=True)[..., self.vol_shape//2, :],
                               figure_xr_hidden, 
                               figure_xr_interp, 
                               volume_xr_inverse[..., self.vol_shape//2, :], 
                               figure_xr_inverse,
                               volume_xr_sampled[..., self.vol_shape//2, :],
                               figure_xr_sampled,
                               ], dim=-2).transpose(2, 3),                    
                ], dim=-2)
                        
                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(gen2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(-1., 1.) * 0.5 + 0.5
                tensorboard.add_image(f'{stage}_df_samples', grid2d, self.current_epoch*self.batch_size + batch_idx)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._common_step(batch, batch_idx, optimizer_idx, stage='train')
        self.train_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, optimizer_idx=-1, stage='validation')
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f'train_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory
        
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f'validation_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    
    # Model arguments
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    
    parser.add_argument("--img", action="store_true", help="whether to train with XR")
    parser.add_argument("--vol", action="store_true", help="whether to train with CT")
    parser.add_argument("--cam", action="store_true", help="train cam locked or hidden")
    parser.add_argument("--sup", action="store_true", help="train cam ct or not")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--delta", type=float, default=1., help="vgg loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
    parser.add_argument("--clamp_val", type=float, default=.1, help="gradient discrim clamp value")
    parser.add_argument("--ddpmbeta", type=str, default="linear")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--strategy", type=str, default='auto', help="training strategy")
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}",
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True, 
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}", 
        log_graph=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks=[
        lr_callback,
        checkpoint_callback,
    ]
    if hparams.strategy!= "fsdp":
         callbacks.append(swa_callback)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=callbacks,
        accumulate_grad_batches=4,
        strategy=hparams.strategy, #"auto", #"ddp_find_unused_parameters_true", 
        precision=16 if hparams.amp else 32,
        profiler="advanced"
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = [
    ]

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = RF3DLightningModule(
        hparams=hparams
    )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    trainer.fit(
        model,
        # compiled_model,
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=datamodule.val_dataloader(),
        # datamodule=datamodule,
        ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve