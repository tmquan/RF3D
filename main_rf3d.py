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
torch.set_float32_matmul_precision('medium')

from typing import Optional
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

class RF3DLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.gan = hparams.gan
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
            # steps_offset=20,
        )
        # set step values
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
            min_depth=8.0, 
            max_depth=12.0, 
            ndc_extent=3.0,
        )
        
        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh**2 if self.sh>0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
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

    def forward_screen(self, image3d, cameras, is_training=True):   
        return self.fwd_renderer(image3d * 0.5 + 0.5, cameras) * 2.0 - 1.0

    def forward_volume(self, image2d, elev, azim, n_views=[2, 1], is_training=True): 
        return self.inv_renderer(image2d, elev.squeeze(1), azim.squeeze(1) * 900, n_views) 
    
    def add_noise(self, x, noise, amount):
        amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
        return x*(1-amount) + noise*amount 

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        batchsz = image2d.shape[0]
        timezeros = torch.Tensor([0]).to(_device)
        timeones_ = torch.Tensor([1]).to(_device)
            
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 10.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.zeros(self.batch_size, device=_device)
        azim_random = torch.rand_like(elev_random) * 2 - 1 # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random)

        dist_hidden = 10.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden)
        
        # Construct the samples in 2D
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random, is_training=(stage=='train'))
        figure_xr_hidden = image2d 
        figure_dx_concat = torch.cat([figure_ct_random, figure_xr_hidden])
        camera_dx_concat = join_cameras_as_batch([view_random, view_hidden])
        
        view_shape_ = [self.batch_size, 1] 
            
        timesteps = torch.randint(0, self.ddim_noise_scheduler.config.num_train_timesteps, (batchsz,), device=_device).long()
        
        if batch_idx%2==0:
            volume_ct_latent = torch.randn_like(image3d)
            figure_ct_latent = self.forward_screen(image3d=volume_ct_latent, cameras=view_random, is_training=(stage=='train'))
            
            volume_ct_interp = self.ddim_noise_scheduler.add_noise(image3d, volume_ct_latent, timesteps=timesteps)
            figure_ct_interp = self.forward_screen(image3d=volume_ct_interp, cameras=view_random, is_training=(stage=='train'))
            
            volume_xr_latent = torch.randn_like(image3d)
            figure_xr_latent = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden, is_training=(stage=='train'))
            # figure_xr_latent = torch.randn_like(image2d)
            figure_xr_interp = self.ddim_noise_scheduler.add_noise(image2d, figure_xr_latent, timesteps=timesteps)
            
            volume_dx_output = self.forward_volume(
                image2d=torch.cat([figure_ct_interp, figure_xr_interp]),
                elev=torch.cat([timesteps.view(view_shape_), timesteps.view(view_shape_)]),
                azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                n_views=[1, 1]
            )
            volume_ct_output, volume_xr_output = torch.split(volume_dx_output, batchsz)
            
            figure_ct_output = self.forward_screen(image3d=volume_ct_output, cameras=view_random, is_training=(stage=='train'))
            figure_xr_output = self.forward_screen(image3d=volume_xr_output, cameras=view_hidden, is_training=(stage=='train'))
            
            diff_loss = self.l1loss(volume_ct_output, volume_ct_latent) \
                    + self.l1loss(volume_xr_output, volume_xr_latent) \
                    + self.l1loss(figure_ct_output, figure_ct_latent) \
                    + self.l1loss(figure_xr_output, figure_xr_latent) 
            
            self.log(f'{stage}_diff_loss', diff_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            loss = diff_loss
        else:
            # Additionally estimate the volume
            volume_output = self.forward_volume(
                image2d=figure_dx_concat, #torch.cat([figure_ct_latent, figure_xr_latent]),
                elev=torch.cat([timezeros.view(view_shape_), timezeros.view(view_shape_)]).to(_device),
                azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                n_views=[1, 1]
            )
            figure_output = self.forward_screen(image3d=volume_output, cameras=camera_dx_concat, is_training=(stage=='train'))
            
            # Perform Post activation like DVGO      
            volume_output = volume_output.sum(dim=1, keepdim=True)
                
            figure_ct_second, figure_xr_second = torch.split(figure_output, batchsz)
            volume_ct_second, volume_xr_second = torch.split(volume_output, batchsz)
            
            
            im3d_loss = self.l1loss(volume_ct_second, image3d) 
            self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            im2d_loss = self.l1loss(figure_xr_second, figure_xr_hidden) \
                      + self.l1loss(figure_ct_second, figure_ct_random)
            # im2d_loss = self.l1loss(figure_output, figure_dx_concat)
            self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            
            loss = im3d_loss + im2d_loss
        
        # Visualization step 
        if batch_idx==0:
            with torch.no_grad():
                # figure_output = figure_dx_latent.clone()
                volume_output = torch.cat([volume_ct_latent.clone(), volume_xr_latent.clone()])
                figure_output = torch.cat([figure_ct_latent.clone(), figure_xr_latent.clone()])
                
                for t in tqdm(self.ddim_noise_scheduler.timesteps):
                    # 1. predict noise model_output
                    tensor_output = self.forward_volume(
                        image2d=figure_output, #torch.cat([figure_ct_latent, figure_xr_latent]),
                        elev=torch.cat([t.view(view_shape_), t.view(view_shape_)]).to(_device),
                        azim=torch.cat([azim_random.view(view_shape_), azim_hidden.view(view_shape_)]),
                        n_views=[1, 1]
                    )
                    # Perform Post activation like DVGO      
                    tensor_output = tensor_output.sum(dim=1, keepdim=True)
                    
                    # 2. compute previous image: x_t -> x_t-1
                    volume_output = self.ddim_noise_scheduler.step(tensor_output, 
                                                                   t, 
                                                                   volume_output, 
                                                                   eta=0,
                                                                   use_clipped_model_output=False,
                                                                   generator=None).prev_sample
                    figure_output = self.forward_screen(image3d=volume_output, cameras=camera_dx_concat, is_training=(stage=='train'))
                    

                    
                gen_figure_ct_random, gen_figure_xr_hidden = torch.split(figure_output, batchsz)
                gen_volume_ct_random, gen_volume_xr_hidden = torch.split(volume_output, batchsz)
                
                figure_dx_zeros_ = torch.zeros_like(image2d)
               
                gen2d = torch.cat([
                    torch.cat([figure_ct_random, figure_ct_latent, figure_ct_interp, gen_figure_ct_random], dim=-2).transpose(2, 3),
                    torch.cat([figure_xr_hidden, figure_xr_latent, figure_xr_interp, gen_figure_xr_hidden], dim=-2).transpose(2, 3),
                    torch.cat([image3d[..., self.vol_shape//2, :], 
                               volume_ct_interp[..., self.vol_shape//2, :], 
                               gen_volume_ct_random[..., self.vol_shape//2, :], 
                               gen_volume_xr_hidden[..., self.vol_shape//2, :] ], dim=-2).transpose(2, 3),
                    
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
    
    parser.add_argument("--gan", action="store_true", help="whether to train with GAN")
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
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_gan{int(hparams.gan)}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}",
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
        save_dir=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_gan{int(hparams.gan)}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}", 
        log_graph=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
            swa_callback if not hparams.gan else None,
        ],
        accumulate_grad_batches=4 if not hparams.gan else 1,
        strategy="auto", #"auto", #"ddp_find_unused_parameters_true", 
        precision=16 if hparams.amp else 32,
        # gradient_clip_val=0.01, 
        # gradient_clip_algorithm="value"
        # stochastic_weight_avg=True if not hparams.gan else False,
        # deterministic=False,
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