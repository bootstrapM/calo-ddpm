"""RePaint (Lugmayr et al., arXiv:2201.09865), noise-free inpainting.

Algorithm 1 of the paper, one reverse step s -> s-1 (repeated n_resample
times with a one-step forward jump in between):

    x_{s-1}^known   ~ q(x_{s-1} | x0 = y)  = N(sbar_{s-1} y, vbar_{s-1} I)
    x_{s-1}^unknown ~ p_theta(x_{s-1} | x_s)          (ancestral step)
    x_{s-1} = M * x_{s-1}^known + (1 - M) * x_{s-1}^unknown

    time travel (u < n_resample):  x_s ~ q(x_s | x_{s-1})   (one-step jump,
    using the SUBSAMPLED step transition; the paper's Alg.1 writes
    beta_{t-1} here, which is a known typo — the consistent forward kernel
    is q(x_t | x_{t-1}) with beta_t).

CRITICAL FIX (this was the bug found in the earlier implementation): the
known region must be noised to level s-1, i.e. q(x_{s-1} | x0), NOT to
level s.  Both branches then live at time s-1 when they are recombined.
Noising to level s leaves the known region one step "too noisy" at every
iteration and biases the boundary of the dead region.

At the final step, sbar_0 = 1 and vbar_0 = 0, so the known region of the
output equals y exactly.
"""

from .base_inpainter import BaseInpainter

__all__ = ['RePaintInpainter']


class RePaintInpainter(BaseInpainter):

    name = 'repaint'

    def __init__(self, net, sched, device, seed=0, use_bf16=False,
                 n_resample=10):
        super().__init__(net, sched, device, seed=seed, use_bf16=use_bf16)
        assert n_resample >= 1
        self.n_resample = n_resample

    def step(self, s, x, y, mask):
        sched = self.sched

        # Single pass at s = 1: Alg. 1 never jumps back at t = 1, and its
        # remaining passes there are deterministic recomputations of the
        # identical x_0 (last-step noise gated, x_known = y exactly), so
        # one pass is exactly equivalent -- and avoids re-feeding a
        # level-0 object to the network conditioned at t = 1.
        n_inner = self.n_resample if s > 1 else 1

        for u in range(n_inner):
            # unknown region: unconditional ancestral step
            x0hat, _ = self.predict_x0(x, s)
            x_unknown = sched.ancestral_step(s, x, x0hat, generator=self.prg)

            # known region: forward-diffuse y to level s-1  (the s-1 fix)
            x_known = sched.q_sample(s - 1, y, noise=self.randn(x))

            x_prev = mask * x_known + (1.0 - mask) * x_unknown

            if u < n_inner - 1:
                # time travel: one subsampled step forward, x_{s-1} -> x_s
                x = sched.forward_step(s, x_prev, noise=self.randn(x))
            else:
                x = x_prev

        return x
