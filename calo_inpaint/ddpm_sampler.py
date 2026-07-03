"""Model loading and unconditional DDPM (ancestral) sampling.

Model directories are the unzipped Zenodo archives (record 12535659), i.e.
the jetgen training output directories:

    cent0_ddpm_seed0/
        config.json        # jetgen training config (vsched, model_args, ...)
        net_avg_gen.pth    # EMA generator weights  (used by default)
        net_gen.pth        # raw generator weights

State-dict keys carry the '_net.' prefix from jetgen's IDDPMUNet wrapper and
are stripped on load.  The eps-network is the reference improved_diffusion
UNetModel (openai/improved-diffusion @ 783b674).
"""

import json
import os

import torch

from .schedule  import make_subsampled_schedule
from .data_norm import LogNorm

__all__ = ['load_model', 'load_config', 'DDPMSampler']


def load_config(model_dir):
    with open(os.path.join(model_dir, 'config.json'), 'rt') as f:
        return json.load(f)


def load_model(model_dir, device, use_ema=True):
    """Load a pre-trained calo-ddpm model.

    Returns (net, config) where net is an improved_diffusion UNetModel in
    eval mode on `device`, and config is the parsed config.json.  The
    variance schedule parameters live in config['model_args']['vsched'].
    """
    from improved_diffusion.unet import UNetModel

    config = load_config(model_dir)

    shape      = config['data']['datasets'][0]['shape']       # [1, 24, 64]
    model_args = dict(config['generator']['model_args'])
    model_args['attention_resolutions'] = tuple(
        model_args.get('attention_resolutions') or ()
    )
    model_args['channel_mult'] = tuple(model_args['channel_mult'])

    net = UNetModel(
        in_channels  = shape[0],
        out_channels = shape[0],
        **model_args,
    )

    fname = 'net_avg_gen.pth' if use_ema else 'net_gen.pth'
    state = torch.load(
        os.path.join(model_dir, fname), map_location='cpu', weights_only=True
    )
    # strip jetgen's IDDPMUNet '_net.' prefix
    state = { (k[5:] if k.startswith('_net.') else k): v
              for (k, v) in state.items() }
    net.load_state_dict(state, strict=True)

    net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    return net, config


def schedule_from_config(config, S, device):
    """Build the (subsampled) schedule from a model's config.json."""
    vs = config['model_args']['vsched']
    assert vs['name'] == 'linear', f"unsupported vsched: {vs['name']}"
    return make_subsampled_schedule(
        T=vs['T'], S=S, beta1=vs['beta1'], betaT=vs['betaT'], device=device
    )


def lognorm_from_config(config):
    dn = config['model_args']['data_norm']
    assert dn['name'] == 'log', f"unsupported data_norm: {dn['name']}"
    return LogNorm(clip_min=dn['clip_min'])


class DDPMSampler:
    """Unconditional ancestral DDPM sampler (jetgen's `diffuse` loop).

    x_S = sqrt(vbar_S) * xi;  for s = S..1:
        eps  = net(x, t_map[s]);  x0hat = (x - sqrt(vbar_s) eps) / sbar_s
        x_{s-1} ~ q(x_{s-1} | x_s, x0hat)      (noise off at the final step)
    """

    def __init__(self, net, sched, device, seed=0, use_bf16=False):
        self.net      = net
        self.sched    = sched
        self.device   = device
        self.use_bf16 = use_bf16
        self.prg      = torch.Generator(device)
        self.prg.manual_seed(seed)

    def reseed(self, seed):
        self.prg.manual_seed(seed)

    def predict_eps(self, x, s, batch):
        t = self.sched.t_map[s].expand(batch)
        if self.use_bf16 and x.is_cuda:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                eps = self.net(x, t)
            return eps.float()
        return self.net(x, t)

    @torch.no_grad()
    def sample(self, n, shape=(1, 24, 64)):
        """Generate n events in MODEL (log) space, shape (n, 1, 24, 64)."""
        sched = self.sched
        x = sched.marginal_std() * torch.randn(
            (n, *shape), generator=self.prg, device=self.device
        )
        for s in range(sched.S, 0, -1):
            eps   = self.predict_eps(x, s, n)
            x0hat = sched.x0_from_eps(s, x, eps)
            x     = sched.ancestral_step(s, x, x0hat, generator=self.prg)
        return x
