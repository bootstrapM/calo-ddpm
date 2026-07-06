"""MCG, adapted from Himanshu's CelebA implementation (validated there on
image inpainting with no NaN issues).

Structure per reverse step s -> s-1:
  1. x0hat from the eps-net; clamped to the physical log-space range.
  2. Consistency loss on the KNOWN region of the CLAMPED x0hat,
         L = mean_pixels [ m (x0hat_clamped - y) ]^2      (per sample),
     differentiated through the network w.r.t. x_s.
  3. Per-sample gradient norm clip to <= 1.
  4. Annealed guidance scale  s_t = guidance_scale * (1 - alphabar_s):
     strong at high noise, -> 0 as the chain sharpens.
  5. Ancestral DDPM step from the clamped x0hat, minus s_t * grad.
  6. RePaint-style hard paste of the known region at level s-1
     (eliminates the boundary seam), clean y at the final step.

Adaptation notes vs. the original (see _ops.py): mask convention flipped
(here 1 = known), [-1,1] -> log-space clamp, per-sample (not global)
gradient clipping and per-sample loss normalization so behavior is
independent of the study batch size, seeded RNG.  fp32 throughout;
finiteness tripwires raise instead of writing NaNs.
"""

import torch

from ._ops import (
    make_generator, expand_measurement, assert_finite,
    predict_x0, ddpm_step, q_sample_prev, clip_grad_per_sample
)

__all__ = ['MCG2Inpainter']


class MCG2Inpainter:

    name = 'mcg2'

    def __init__(self, net, sched, device, seed=0, guidance_scale=0.5,
                 x0_clamp=(-7.0, 4.0), use_bf16=False):  # bf16 accepted, ignored
        self.net = net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.sched  = sched
        self.device = device
        self.gs     = float(guidance_scale)
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
                x0c   = predict_x0(sc, s, x_in, eps).clamp(*self.clamp)
                # per-sample mean over pixel dims (original used a global
                # .mean(); per-sample keeps the gradient scale batch-size
                # independent)
                loss = (mask * (x0c - y)).pow(2).flatten(1).mean(1)
                grad = torch.autograd.grad(loss.sum(), x_in)[0]
            x0c = x0c.detach()
            assert_finite(grad, s, 'MCG gradient')

            with torch.no_grad():
                grad  = clip_grad_per_sample(grad, 1.0)
                scale = self.gs * sc.vbar[s]          # gs * (1 - alphabar_s)

                x_next = ddpm_step(sc, s, x, x0c, self.prg) - scale * grad
                x_known = y if s == 1 else q_sample_prev(sc, s, y, self.prg)

                x = mask * x_known + (1.0 - mask) * x_next
            assert_finite(x, s, 'state x')

        # final paste of known pixels (postprocess of the original)
        return (mask * y + (1.0 - mask) * x).detach()
