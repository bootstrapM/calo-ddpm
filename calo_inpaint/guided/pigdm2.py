"""PiGDM-style guided inpainting, adapted from Himanshu's CelebA
implementation (validated there on image inpainting with no NaN issues).

Structure per reverse step s -> s-1:
  1. x0hat from the eps-net (UNCLAMPED in the loss; clamped copy for the
     transport step).
  2. Consistency loss on the KNOWN region,
         L = mean_pixels [ m (x0hat - y) ]^2               (per sample),
     differentiated through the network w.r.t. x_s.
  3. Per-sample gradient norm clip to <= 1.
  4. Pseudoinverse-style SNR factor, clamped for stability at high noise:
         r_t = min( (1 - alphabar_s) / alphabar_s, r_max ),  r_max = 10.
  5. Annealed correction:  x_next = ddpm_step(x, x0hat_clamped)
                                    - guidance_scale (1 - alphabar_s) r_t grad.
  6. RePaint-style hard paste of the known region at level s-1; clean y at
     the final step.

NOTE (paper bookkeeping): unlike the published PiGDM (Song et al. 2023),
this variant carries a per-step known-region projection and a clamped
scalar SNR weight instead of the exact vector-Jacobian guidance
coefficient — it is "projection + annealed pseudoinverse-weighted
gradient".  Document it as such rather than as Algorithm 1 of the paper.

Adaptation notes vs. the original: mask convention flipped (1 = known),
[-1,1] -> log-space clamp, per-sample gradient clipping / loss
normalization (batch-size independence), seeded RNG, fp32 throughout,
finiteness tripwires.
"""

import torch

from ._ops import (
    make_generator, expand_measurement, assert_finite,
    predict_x0, ddpm_step, q_sample_prev, clip_grad_per_sample
)

__all__ = ['PiGDM2Inpainter']


class PiGDM2Inpainter:

    name = 'pigdm2'

    def __init__(self, net, sched, device, seed=0, guidance_scale=0.5,
                 r_max=10.0, x0_clamp=(-7.0, 4.0),
                 use_bf16=False):                       # bf16 accepted, ignored
        self.net = net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.sched  = sched
        self.device = device
        self.gs     = float(guidance_scale)
        self.r_max  = float(r_max)
        self.clamp  = (float(x0_clamp[0]), float(x0_clamp[1]))
        self.prg    = make_generator(device, seed)

    def reseed(self, seed):
        self.prg.manual_seed(seed)

    def inpaint(self, y, mask, n_samples):
        sc = self.sched
        y, mask = expand_measurement(y, mask, n_samples, self.device)

        x = sc.vbar[sc.S].sqrt() * torch.randn(
            y.shape, generator=self.prg, device=self.device)

        for s in range(sc.S, 0, -1):
            t = sc.t_map[s].expand(x.shape[0])

            with torch.enable_grad():
                x_in  = x.detach().requires_grad_(True)
                eps   = self.net(x_in, t)
                x0hat = predict_x0(sc, s, x_in, eps)
                loss  = (mask * (x0hat - y)).pow(2).flatten(1).mean(1)
                grad  = torch.autograd.grad(loss.sum(), x_in)[0]
            x0c = x0hat.detach().clamp(*self.clamp)
            assert_finite(grad, s, 'PiGDM gradient')

            with torch.no_grad():
                grad = clip_grad_per_sample(grad, 1.0)

                # clamped SNR weight (original: (1-acp)/(acp+1e-8), max 10)
                alphabar = sc.sbar[s] ** 2
                r_t = torch.clamp(sc.vbar[s] / (alphabar + 1e-8),
                                  max=self.r_max)
                scale = self.gs * sc.vbar[s] * r_t

                x_next  = ddpm_step(sc, s, x, x0c, self.prg) - scale * grad
                x_known = y if s == 1 else q_sample_prev(sc, s, y, self.prg)

                x = mask * x_known + (1.0 - mask) * x_next
            assert_finite(x, s, 'state x')

        return (mask * y + (1.0 - mask) * x).detach()
