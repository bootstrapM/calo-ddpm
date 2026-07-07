"""HuggingFace diffusers model adapter for the CelebA study.

Bridges a pretrained `UNet2DModel` (eps-prediction, VP linear schedule,
e.g. google/ddpm-ema-celebahq-256) to the repo's sampler/inpainter
convention.

TIMESTEP CONVENTION (the one subtle correctness point): this repo
conditions the network on padded indices t in 1..T, with
alphabar_ours[t] = prod_{i<=t}(1 - beta_i).  HF DDPM models are trained
on 0-indexed timesteps 0..T-1 with alphabar_HF[t] = alphabar_ours[t+1].
The wrapper therefore calls the HF UNet with (t - 1).
"""

import torch
from torch import nn

from calo_inpaint.schedule import make_subsampled_schedule

__all__ = ['EpsWrapper', 'UnitNorm', 'load_celeb_model']

DEFAULT_MODEL_ID = 'google/ddpm-ema-celebahq-256'


class EpsWrapper(nn.Module):
    """UNet2DModel -> eps-net with this repo's calling convention."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, y=None):
        return self.unet(x, (t - 1).clamp(min=0)).sample


class UnitNorm:
    """Identity 'normalization': HF DDPMs operate directly in [-1, 1]."""

    def normalize(self, x):
        return x

    def denormalize(self, y):
        return y


def load_celeb_model(model_id=DEFAULT_MODEL_ID, device='cuda'):
    """Load a pretrained HF DDPM.  Returns (net, info) with
    info = {T, beta1, betaT, shape, model_id}.  Accepts both flat model
    repos and pipeline repos (unet/ + scheduler/ subfolders), and local
    paths (offline use / tests)."""
    from diffusers import UNet2DModel, DDPMScheduler

    try:
        unet = UNet2DModel.from_pretrained(model_id)
        sch  = DDPMScheduler.from_pretrained(model_id)
    except (OSError, EnvironmentError):
        unet = UNet2DModel.from_pretrained(model_id, subfolder='unet')
        sch  = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')

    cfg = sch.config
    assert cfg.beta_schedule == 'linear', \
        f'only linear beta schedules are supported (got {cfg.beta_schedule})'
    assert getattr(cfg, 'prediction_type', 'epsilon') == 'epsilon', \
        'only eps-prediction models are supported'

    unet.to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    size = unet.config.sample_size
    info = {
        'T'       : int(cfg.num_train_timesteps),
        'beta1'   : float(cfg.beta_start),
        'betaT'   : float(cfg.beta_end),
        'shape'   : (int(unet.config.in_channels), int(size), int(size)),
        'model_id': str(model_id),
    }
    return EpsWrapper(unet), info


def celeb_schedule(info, S, device):
    return make_subsampled_schedule(
        T=info['T'], S=S, beta1=info['beta1'], betaT=info['betaT'],
        device=device
    )
