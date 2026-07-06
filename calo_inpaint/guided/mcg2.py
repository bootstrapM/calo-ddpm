"""MCG (Chung et al., arXiv:2206.00941), clean-room reimplementation.

Self-contained: does NOT use calo_inpaint.inpainting.  Consumes only the
schedule arrays (sbar, vbar, t_map), a frozen eps-network, and the
measurement (y, mask) in log space.  All reverse-step coefficients are
derived inline from sbar/vbar.

Differences from the first implementation, by construction:
  * SAFE residual norm  R = sqrt(sum r^2 + eps_norm^2): identical
    gradient direction and (to O(eps^2)) magnitude, but no 0/0 NaN when
    the masked residual underflows to exactly zero (torch's ||.|| has a
    NaN gradient at 0 — the identified NaN source on real data, where
    x0hat reproduces the known region to float precision late in the
    chain).
  * All network evaluations in fp32 (no autocast anywhere).
  * Finiteness tripwires every step: diverging runs RAISE with the step
    index instead of writing NaNs to disk.

Algorithm per reverse step s -> s-1 (ancestral transport):
    x0hat = (x - sqrt(vbar_s) eps) / sbar_s
    R     = sqrt(|| m (y - x0hat) ||^2 + eps_norm^2)      per sample
    g     = d R / d x                                     (through the net)
    dead : ancestral(x, x0hat) - alpha g
    known: sbar_{s-1} y + sqrt(vbar_{s-1}) xi             (level s-1)
"""

import torch

from ._runtime import (
    make_generator, expand_measurement, assert_finite
)

__all__ = ['MCG2Inpainter']


class MCG2Inpainter:

    name = 'mcg2'

    def __init__(self, net, sched, device, seed=0, alpha=1.0,
                 eps_norm=1e-8, use_bf16=False):   # use_bf16 accepted, ignored
        self.net    = net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.sched  = sched
        self.device = device
        self.alpha  = float(alpha)
        self.eps2   = float(eps_norm) ** 2
        self.prg    = make_generator(device, seed)

    def reseed(self, seed):
        self.prg.manual_seed(seed)

    def _randn(self, like):
        return torch.randn(like.shape, generator=self.prg,
                           device=like.device, dtype=like.dtype)

    def inpaint(self, y, mask, n_samples):
        sc = self.sched
        y, mask = expand_measurement(y, mask, n_samples, self.device)

        x = sc.vbar[sc.S].sqrt() * self._randn(y)          # prior marginal

        for s in range(sc.S, 0, -1):
            sb, vb           = sc.sbar[s], sc.vbar[s]
            sb_prev, vb_prev = sc.sbar[s - 1], sc.vbar[s - 1]
            t = sc.t_map[s].expand(x.shape[0])

            # ---- gradient of the smoothed residual norm (through net) ----
            with torch.enable_grad():
                x_in  = x.detach().requires_grad_(True)
                eps   = self.net(x_in, t)
                x0hat = (x_in - vb.sqrt() * eps) / sb
                r2    = (mask * (y - x0hat)).pow(2).flatten(1).sum(1)
                R     = (r2 + self.eps2).sqrt()            # never 0/0
                g     = torch.autograd.grad(R.sum(), x_in)[0]
            x0hat = x0hat.detach()
            assert_finite(x0hat, s, 'x0hat')
            assert_finite(g,     s, 'MCG gradient')

            # ---- ancestral posterior step, coefficients derived inline ----
            #   q(x_{s-1} | x_s, x0) with jump beta_k = 1 - (sb/sb_prev)^2
            if s == 1:
                x_dead = x0hat                              # exact collapse
            else:
                beta_k   = 1.0 - (sb / sb_prev) ** 2
                coef_x0  = sb_prev * beta_k / vb
                coef_xt  = (sb / sb_prev) * vb_prev / vb
                post_var = beta_k * vb_prev / vb
                x_dead = (coef_x0 * x0hat + coef_xt * x
                          + post_var.sqrt() * self._randn(x))

            x_dead = x_dead - self.alpha * g

            # ---- known region: forward-diffuse y to level s-1 -------------
            if s == 1:
                x_known = y
            else:
                x_known = sb_prev * y + vb_prev.sqrt() * self._randn(x)

            x = mask * x_known + (1.0 - mask) * x_dead
            assert_finite(x, s, 'state x')

        return x.detach()
