"""PiGDM (Song et al., ICLR 2023), clean-room reimplementation.

Self-contained: does NOT use calo_inpaint.inpainting.  Shares only the
schedule arrays, the frozen eps-network, and the measurement.

Differences from the first implementation, by construction:
  * Guidance NORM-CLIPPING instead of gradient-killing x0 clamps.  The
    v1 clamp stabilized the early chain but zeroed the guidance gradient
    wherever x0hat saturated, decoupling the trajectory from y for much
    of the reverse process (-> severe under-conditioning / shrinkage).
    Here the VJP is computed with an UNSATURATED graph; only the
    cotangent uses a range-clamped x0hat, and the finished guidance
    update is bounded per sample:
        t_g = sbar_s * vjp
        t_g <- t_g * min(1, gmax * sqrt(D) / ||t_g||_2)
    i.e. the applied per-pixel RMS never exceeds gmax (log-space units),
    which bounds the feedback loop while preserving the guidance
    direction everywhere.
  * All network evaluations in fp32; finiteness tripwires every step
    (diverging runs raise instead of writing NaNs).

Algorithm per reverse step s -> s-1 (Alg. 1, VP form, eta = 1 default):
    x0hat = (x - sqrt(vbar_s) eps) / sbar_s
    vjp   = (d x0hat / d x)^T [ m (y - clamp(x0hat)) ]
    c1    = eta * sqrt(post_var_s),   c2 = sqrt(vbar_{s-1} - c1^2)
    x_{s-1} = sbar_{s-1} x0hat + c2 eps + c1 xi + clip(sbar_s vjp)
Final step: deterministic; known region projected to y (post-processing).
"""

import torch

from ._runtime import (
    make_generator, expand_measurement, assert_finite
)

__all__ = ['PiGDM2Inpainter']


class PiGDM2Inpainter:

    name = 'pigdm2'

    def __init__(self, net, sched, device, seed=0, eta=1.0,
                 x0_clamp=(-7.0, 4.0), gmax=1.0,
                 use_bf16=False):                  # use_bf16 accepted, ignored
        assert 0.0 <= eta <= 1.0, f'eta={eta} outside [0, 1]'
        assert gmax > 0.0
        self.net    = net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.sched  = sched
        self.device = device
        self.eta    = float(eta)
        self.clamp  = None if x0_clamp is None else \
            (float(x0_clamp[0]), float(x0_clamp[1]))
        self.gmax   = float(gmax)
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
        sqrt_d = float(y[0].numel()) ** 0.5                # pixels per image

        for s in range(sc.S, 0, -1):
            sb, vb           = sc.sbar[s], sc.vbar[s]
            sb_prev, vb_prev = sc.sbar[s - 1], sc.vbar[s - 1]
            t = sc.t_map[s].expand(x.shape[0])

            # ---- pseudoinverse guidance (VJP through the net) -------------
            with torch.enable_grad():
                x_in  = x.detach().requires_grad_(True)
                eps   = self.net(x_in, t)
                x0hat = (x_in - vb.sqrt() * eps) / sb

                x0_for_res = x0hat.clamp(*self.clamp) if self.clamp else x0hat
                cot = (mask * (y - x0_for_res)).detach()   # bounded cotangent

                vjp = torch.autograd.grad((x0hat * cot).sum(), x_in)[0]
            x0hat = x0hat.detach()
            eps   = eps.detach()
            assert_finite(vjp, s, 'PiGDM vjp')

            # applied guidance, per-sample norm-clipped (bounds feedback,
            # keeps direction — no dead zones from clamp saturation)
            t_g   = sb * vjp
            norms = t_g.flatten(1).norm(dim=1).clamp_min(1e-12)
            scale = torch.clamp(self.gmax * sqrt_d / norms, max=1.0)
            t_g   = t_g * scale.view(-1, 1, 1, 1)

            # ---- DDIM(eta) transport, coefficients derived inline ---------
            if s == 1:
                out = x0hat + t_g
                out = mask * y + (1.0 - mask) * out        # final projection
            else:
                beta_k   = 1.0 - (sb / sb_prev) ** 2
                post_var = beta_k * vb_prev / vb
                c1_sq = (self.eta ** 2) * post_var
                c2    = torch.clamp(vb_prev - c1_sq, min=0.0).sqrt()
                out = (sb_prev * x0hat + c2 * eps + t_g
                       + c1_sq.sqrt() * self._randn(x))

            x = out
            assert_finite(x, s, 'state x')

        return x.detach()
