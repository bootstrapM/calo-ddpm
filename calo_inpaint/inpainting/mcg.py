"""MCG — Manifold Constrained Gradient (Chung et al., arXiv:2206.00941),
noise-free inpainting (Algorithm 1 / Eq. 15 of the paper).

One reverse step s -> s-1:

  1. x0hat = x0hat_theta(x_s)                        (Tweedie denoiser)
  2. unconditional ancestral step:  x'_{s-1} ~ p_theta(x_{s-1} | x_s)
  3. manifold-constrained gradient: the measurement-consistency residual
     is evaluated on the KNOWN region,
         R(x_s) = || M * (y - x0hat(x_s)) ||_2 ,
     and differentiated THROUGH the network w.r.t. the full x_s,
         g = grad_{x_s} R(x_s),
     so g is nonzero everywhere -- known-region consistency couples to
     the dead pixels via the network Jacobian.  (Following the official
     implementation, the gradient of the l2 NORM — not the squared
     norm — is used, which self-normalizes the step; the overall step
     size alpha is a hyperparameter, default 1.0.)
  4. The correction is APPLIED only to the dead region — the known
     region is overwritten by the RePaint-style projection at level s-1
     (same s-1 convention/fix as repaint.py) anyway:
         x_{s-1} = M * q_sample(s-1, y) + (1 - M) * (x'_{s-1} - alpha * g)

At the final step the projection makes the known region exactly y, and the
ancestral step returns x0hat, so the output is a valid x0 sample.

Note: MCG needs gradients through the eps-network, so each step performs
one forward pass with autograd enabled (the surrounding study loop must NOT
be wrapped in torch.no_grad()).
"""

import torch

from .base_inpainter import BaseInpainter

__all__ = ['MCGInpainter']


class MCGInpainter(BaseInpainter):

    name = 'mcg'

    def __init__(self, net, sched, device, seed=0, use_bf16=False,
                 alpha=1.0):
        super().__init__(net, sched, device, seed=seed, use_bf16=use_bf16)
        self.alpha = float(alpha)

    def step(self, s, x, y, mask):
        sched = self.sched

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            x0hat, _ = self.predict_x0(x_in, s)

            residual = mask * (y - x0hat)
            # per-sample l2 norm; summed for a single backward pass
            norm = residual.flatten(1).norm(dim=1).sum()
            grad = torch.autograd.grad(norm, x_in)[0]

        x0hat = x0hat.detach()

        x_uncond = sched.ancestral_step(s, x.detach(), x0hat,
                                        generator=self.prg)
        x_dead   = x_uncond - self.alpha * grad
        x_known  = sched.q_sample(s - 1, y, noise=self.randn(x))

        return mask * x_known + (1.0 - mask) * x_dead
