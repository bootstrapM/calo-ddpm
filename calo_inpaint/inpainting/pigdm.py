"""PiGDM — Pseudoinverse-Guided Diffusion Models (Song et al., ICLR 2023),
noise-free inpainting, Algorithm 1 (VP-SDE version) of the paper.

For noiseless masking, H = H^dagger = diag(M), and the guidance simplifies
(paper Eq. 8 -> Eq. 11; the r_t^2 factor cancels exactly in the noiseless
update, Eq. 10):

    g = ( (H^dagger y - H^dagger H x0hat)^T  d x0hat / d x_s )^T
      = VJP of x0hat at x_s with cotangent  M * (y - x0hat)

computed by differentiating through the eps-network (Listing 1 of the
paper: the cotangent is detached, only x0hat is differentiated).

One step s -> s-1 (Algorithm 1):

    x_{s-1} = sbar_{s-1} x0hat + c1 xi + c2 eps_hat + sbar_s * g

    c1 = eta * sqrt(post_var[s])        (DDIM sigma; identical to the
                                         standard DDIM eta-parametrization)
    c2 = sqrt(vbar_{s-1} - c1^2)
    sbar_s * g : the "additional sqrt(alphabar_t) in front of g" required
                 for VP models (paper App. A.1).

The first three terms are exactly a DDIM(eta) step; the paper's default for
inpainting-type problems, eta = 1, makes them equal in law to the DDPM
ancestral step.  PiGDM does NOT hard-project the known region; consistency
is enforced only through the guidance term, so the known region of the
output is equal to y only approximately (the study evaluates the dead
region, where this is irrelevant).
"""

import torch

from .base_inpainter import BaseInpainter

__all__ = ['PiGDMInpainter']


class PiGDMInpainter(BaseInpainter):

    name = 'pigdm'

    def __init__(self, net, sched, device, seed=0, use_bf16=False,
                 eta=1.0):
        super().__init__(net, sched, device, seed=seed, use_bf16=use_bf16)
        self.eta = float(eta)

    def step(self, s, x, y, mask):
        sched = self.sched

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            x0hat, eps_hat = self.predict_x0(x_in, s)

            cotangent = (mask * (y - x0hat)).detach()
            vjp = torch.autograd.grad(
                (x0hat * cotangent).sum(), x_in
            )[0]

        x0hat   = x0hat.detach()
        eps_hat = eps_hat.detach()

        sb_prev = sched.sbar[s - 1]
        vb_prev = sched.vbar[s - 1]

        final = sched.t_map[s].item() <= 1

        c1_sq = (self.eta ** 2) * sched.post_var[s]
        if final:
            c1_sq = torch.zeros_like(c1_sq)
        c2 = torch.clamp(vb_prev - c1_sq, min=0.0).sqrt()

        out = sb_prev * x0hat + c2 * eps_hat + sched.sbar[s] * vjp
        if not final and c1_sq.item() > 0:
            out = out + c1_sq.sqrt() * self.randn(x)

        return out
