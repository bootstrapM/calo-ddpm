"""Unconditional DDIM sampler with stochasticity parameter eta.

eta = 0   : deterministic DDIM (matches jetgen's DDIM implementation exactly)
eta = 1   : equal in law to ancestral DDPM
eta in (0,1): interpolation — the eta-sweep is a study extension not present
              in the original paper.

Note on time grids: for S < T this sampler subsamples on jetgen's
'linear-ddpm' grid (round(linspace)); jetgen's own DDIM eval used a
'linear-ddim' grid (arange with stride T//S).  At S = T both coincide.
"""

import torch

from .ddpm_sampler import DDPMSampler

__all__ = ['DDIMSampler']


class DDIMSampler(DDPMSampler):

    def __init__(self, net, sched, device, seed=0, eta=0.0, use_bf16=False):
        super().__init__(net, sched, device, seed=seed, use_bf16=use_bf16)
        self.eta = float(eta)

    @torch.no_grad()
    def sample(self, n, shape=(1, 24, 64)):
        sched = self.sched
        x = sched.marginal_std() * torch.randn(
            (n, *shape), generator=self.prg, device=self.device
        )
        for s in range(sched.S, 0, -1):
            eps   = self.predict_eps(x, s, n)
            x0hat = sched.x0_from_eps(s, x, eps)
            x     = sched.ddim_step(s, x, x0hat, eta=self.eta,
                                    generator=self.prg)
        return x
