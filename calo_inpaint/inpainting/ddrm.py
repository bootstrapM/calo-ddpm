"""DDRM (Kawar et al., arXiv:2201.11793), specialized to noise-free inpainting.

For pixel masking the SVD of A = diag(M) is trivial (singular values 1 on
observed pixels, 0 on dead pixels), so the spectral space coincides with
pixel space and no SVD code is needed.  With sigma_y = 0 the DDRM
variational posterior (paper Eqs. 7-8, VE variables xb_t = x_t / sbar_t,
sigma_t = sqrt(vbar_t) / sbar_t) reduces to:

  Initialization (t = S):
    observed:   xb_S ~ N(y,     sigma_S^2)
    dead:       xb_S ~ N(0,     sigma_S^2)

  Step to s-1 (given x0hat from the denoiser):
    dead pixels     ("s_i = 0" case):
        xb_{s-1} = x0hat + sqrt(1-eta^2) sigma_{s-1} (xb_s - x0hat)/sigma_s
                   + eta sigma_{s-1} xi
      Since (xb_s - x0hat)/sigma_s = eps_hat, in x-space:
        x_{s-1} = sbar_{s-1} x0hat + sqrt((1-eta^2) vbar_{s-1}) eps_hat
                  + sqrt(eta^2 vbar_{s-1}) xi
      (a DDIM-like update whose noise scale references vbar_{s-1}, not the
       ancestral posterior variance — this is DDRM-specific.)

    observed pixels (sigma_y/s_i = 0 <= sigma_t, "case 3" always):
        xb_{s-1} = (1 - eta_b) x0hat + eta_b y + sigma_{s-1} xi
      in x-space:
        x_{s-1} = sbar_{s-1} ((1-eta_b) x0hat + eta_b y) + sqrt(vbar_{s-1}) xi

  With the defaults eta = 0.85, eta_b = 1 (paper defaults), observed pixels
  are simply re-noised copies of y at each level, and at s-1 = 0
  (sigma_0 = 0, sbar_0 = 1) the output is exact: known region = y, dead
  region = x0hat.

  Note: for eta_b < 1 the published update ends with the known region at
  (1-eta_b) x0hat + eta_b y, i.e. NOT exactly consistent even though the
  measurement is noise-free.  This implementation therefore projects the
  known region to y at the final step (a no-op at eta_b = 1); the dead
  region is never affected.
"""

import torch

from .base_inpainter import BaseInpainter

__all__ = ['DDRMInpainter']


class DDRMInpainter(BaseInpainter):

    name = 'ddrm'

    def __init__(self, net, sched, device, seed=0, use_bf16=False,
                 eta=0.85, eta_b=1.0):
        super().__init__(net, sched, device, seed=seed, use_bf16=use_bf16)
        assert 0.0 <= eta <= 1.0,   f'eta={eta} outside [0, 1]'
        assert 0.0 <= eta_b <= 1.0, f'eta_b={eta_b} outside [0, 1]'
        self.eta   = float(eta)
        self.eta_b = float(eta_b)

    def init_x(self, y, n_samples=None):
        """DDRM init: observed dims centered on y, dead dims on 0.

        y arrives already expanded to the full sample batch (see base).
        """
        sched = self.sched
        noise = torch.empty(y.shape, device=self.device)
        noise.normal_(generator=self.prg)

        x_obs  = sched.sbar[sched.S] * y + sched.vbar[sched.S].sqrt() * noise
        x_dead = sched.marginal_std() * noise
        # note: same noise tensor is fine — obs/dead pixels are disjoint
        return self._mask_cache * x_obs + (1.0 - self._mask_cache) * x_dead

    def inpaint(self, y, mask, n_samples):
        # stash mask for init_x (base class calls init_x before the loop)
        self._mask_cache = self._canonical_image(
            mask.to(self.device).float(), 'mask'
        )
        try:
            return super().inpaint(y, mask, n_samples)
        finally:
            del self._mask_cache

    def step(self, s, x, y, mask):
        sched = self.sched
        x0hat, eps_hat = self.predict_x0(x, s)

        sb_prev = sched.sbar[s - 1]
        vb_prev = sched.vbar[s - 1]

        final = (s == 1)                 # last reverse step: s-1 == 0

        # dead pixels: DDIM-like update with eta
        x_dead = sb_prev * x0hat
        if not final:
            x_dead = (
                x_dead
                + ((1.0 - self.eta ** 2) * vb_prev).sqrt() * eps_hat
                + self.eta * vb_prev.sqrt() * self.randn(x)
            )

        # observed pixels: (1 - eta_b) x0hat + eta_b y, re-noised to s-1
        x_obs = sb_prev * ((1.0 - self.eta_b) * x0hat + self.eta_b * y)
        if not final:
            x_obs = x_obs + vb_prev.sqrt() * self.randn(x)

        if final:
            # Noise-free measurement: project the known region to y exactly.
            # A no-op at the default eta_b = 1 (DDRM is already exact
            # there); for eta_b < 1 the PUBLISHED update would end at
            # (1-eta_b) x0hat + eta_b y -- this projection is our
            # implementation choice on top of DDRM, and never touches the
            # dead region.
            x_obs = y

        return mask * x_obs + (1.0 - mask) * x_dead
