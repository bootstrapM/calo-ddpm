"""DDNM (Wang, Yu & Zhang, arXiv:2212.00490), noise-free inpainting.

Null-space decomposition: any x0 consistent with y = A x0 can be written

    x0 = A^dagger y + (I - A^dagger A) x0~

For noiseless masking, A = A^dagger = diag(M), so the range-space
rectification of the denoiser output is simply

    x0hat_proj = M * y + (1 - M) * x0hat                       (Eq. 13)

followed by the standard DDPM ancestral step using x0hat_proj (Eq. 14/15
of the paper with Sigma = 0, i.e. plain DDNM, not DDNM+ — no noisy-case
lambda/gamma corrections are needed for sigma_y = 0).

At the final step the ancestral coefficients collapse (coef_x0[1] = 1,
coef_xt[1] = 0), so the output equals x0hat_proj and the known region is
exactly y.
"""

from .base_inpainter import BaseInpainter

__all__ = ['DDNMInpainter']


class DDNMInpainter(BaseInpainter):

    name = 'ddnm'

    def step(self, s, x, y, mask):
        sched = self.sched

        x0hat, _   = self.predict_x0(x, s)
        x0hat_proj = mask * y + (1.0 - mask) * x0hat

        return sched.ancestral_step(s, x, x0hat_proj, generator=self.prg)
