"""Common machinery for training-free, NOISE-FREE inpainting with a DDPM prior.

Problem setting (deliberately specialized — no noisy-measurement branches):

    y = M * x0_true          M: binary mask, 1 = known pixel, 0 = dead pixel
                             measurement noise sigma_y = 0.

All algorithms operate in MODEL (log-normalized) space and target the same
posterior p(x0 | M x0 = y) under the DDPM prior.  For this noiseless
masking case the degradation operator A = diag(M) is its own pseudoinverse
on its range (A = A^T = A^dagger = A A^T restricted to observed pixels),
which is what makes the five algorithms directly comparable.

Interface:
    inp = SomeInpainter(net, sched, device, seed=..., **algo_kwargs)
    samples = inp.inpaint(y, mask, n_samples)   # (n_samples, 1, H, W), log space

`y` must already be normalized (log space); pixels of y outside the mask are
ignored.  Posterior samples are drawn i.i.d. by batching n_samples through
the reverse process.
"""

import torch

__all__ = ['BaseInpainter']


class BaseInpainter:

    name = 'base'

    def __init__(self, net, sched, device, seed=0, use_bf16=False):
        self.net      = net
        self.sched    = sched
        self.device   = device
        self.use_bf16 = use_bf16
        self.prg      = torch.Generator(device)
        self.prg.manual_seed(seed)

        # frozen prior in eval mode: load_model() already does both, but a
        # net that bypassed it must not sample with train-mode modules
        # (dropout/BN) or build autograd graphs in the projection-type
        # algorithms; MCG/PiGDM differentiate w.r.t. x only.
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    def reseed(self, seed):
        self.prg.manual_seed(seed)

    # -- helpers -----------------------------------------------------------
    def randn(self, like):
        return torch.randn(
            like.shape, generator=self.prg, device=like.device,
            dtype=like.dtype
        )

    def predict_eps(self, x, s):
        """eps_theta(x_s, t_map[s]); x may require grad (MCG / PiGDM).

        Grad-enabled evaluations ALWAYS run in fp32: PiGDM's stability at
        early times relies on the near-cancellation
        (I - sqrt(vbar) d(eps)/dx) / sqrt(alphabar), with 1/sqrt(alphabar)
        up to ~150 for this schedule; bf16's ~0.4% relative error breaks
        the cancellation and the unprojected guidance feedback then runs
        away to inf/NaN within a few steps (verified empirically).
        """
        t = self.sched.t_map[s].expand(x.shape[0])
        if self.use_bf16 and x.is_cuda and not x.requires_grad:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                eps = self.net(x, t)
            return eps.float()
        return self.net(x, t)

    def predict_x0(self, x, s):
        eps = self.predict_eps(x, s)
        return self.sched.x0_from_eps(s, x, eps), eps

    def init_x(self, y, n_samples=None):
        """x_S ~ N(0, vbar_S I) (prior marginal).

        y arrives from inpaint() already expanded to the full sample batch
        (K * n_samples, 1, H, W); its leading dimension defines the batch.
        """
        x = torch.empty(y.shape, device=self.device)
        x.normal_(generator=self.prg)
        return self.sched.marginal_std() * x

    # -- input handling ------------------------------------------------------
    @staticmethod
    def _canonical_image(z, name='tensor'):
        """(H,W) / (C,H,W) / (B,C,H,W)  ->  (B,C,H,W)."""
        if z.dim() == 2:
            return z.unsqueeze(0).unsqueeze(0)
        if z.dim() == 3:
            return z.unsqueeze(0)
        if z.dim() == 4:
            return z
        raise ValueError(
            f'{name}: expected a 2D, 3D or 4D tensor, got shape '
            f'{tuple(z.shape)}'
        )

    # -- main loop ---------------------------------------------------------
    def inpaint(self, y, mask, n_samples):
        """Draw n_samples posterior samples per image.

        y    : (1, H, W) for a single image, or (K, 1, H, W) for K images
               batched jointly through the reverse process (GPU batch is
               then K * n_samples); mask is shared across images.
        Returns (K * n_samples, 1, H, W) in log space, ordered
        image-major: sample j of image k is row k * n_samples + j.
        """
        y    = self._canonical_image(y.to(self.device).float(), 'y')
        mask = self._canonical_image(mask.to(self.device).float(), 'mask')

        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('mask must be binary: 1 = known/live, 0 = dead')

        # expand K images to the sample batch (elementwise algorithms
        # broadcast y/mask against x throughout)
        y = y.repeat_interleave(n_samples, dim=0)   # (K * n_samples, 1, H, W)

        x = self.init_x(y, 0)                       # shape from y itself

        for s in range(self.sched.S, 0, -1):
            x = self.step(s, x, y, mask)

        return x.detach()

    def step(self, s, x, y, mask):
        raise NotImplementedError
