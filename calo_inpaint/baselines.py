"""Model-free baseline "inpainters" for the calibration study.

Both use ONLY the known pixels of each image (no network, no schedule) and
operate in the same normalized (log) space as the real algorithms, with
the same interface and file formats, so they drop into run_inpaint_study
and every downstream statistic unchanged.

  * 'noise'    : dead pixels filled i.i.d. from N(mu_i, sigma_i^2), where
                 mu_i / sigma_i are the mean / std of image i's KNOWN
                 pixels ("natural mean and variance").  A stochastic
                 no-spatial-correlations reference: for an i.i.d. prior it
                 is the exact posterior; on real data its calibration
                 deficit measures the information carried by spatial
                 structure.
  * 'meanfill' : dead pixels set to mu_i, identically for all samples.
                 A deterministic point estimator with ZERO posterior
                 width: by construction it fails every calibration test
                 (coverage -> 0, degenerate PIT/ranks, unbounded pulls) —
                 that is its purpose as a floor.  Its accuracy metrics
                 (bias, CRPS, energy sums) remain meaningful.

Expected downstream behavior is a feature, not a bug: these anchor the
comparison tables from below.
"""

import torch

__all__ = ['NoiseBaseline', 'MeanFillBaseline', 'BASELINE_INPAINTERS']


def _canonical(z, device, name):
    z = z.to(device).float()
    if z.dim() == 2:
        return z[None, None]
    if z.dim() == 3:
        return z[None]
    if z.dim() == 4:
        return z
    raise ValueError(f'{name}: expected 2D/3D/4D, got {tuple(z.shape)}')


class _StatsBaseline:
    """Shared machinery: per-image known-pixel mean/std, K-image batching."""

    def __init__(self, net=None, sched=None, device='cpu', seed=0,
                 use_bf16=False):            # net/sched/bf16 accepted, ignored
        self.device = device
        self.prg = torch.Generator(device)
        self.prg.manual_seed(seed)

    def reseed(self, seed):
        self.prg.manual_seed(seed)

    def inpaint(self, y, mask, n_samples):
        y    = _canonical(y, self.device, 'y')
        mask = _canonical(mask, self.device, 'mask')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('mask must be binary: 1 = known/live, 0 = dead')

        y = y.repeat_interleave(n_samples, dim=0)      # (K*J, C, H, W)

        # per-row stats over the KNOWN pixels (rows from the same image
        # share identical known pixels, so this is per-image by construction)
        n_known = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        mu = (y * mask).sum(dim=(1, 2, 3)) / n_known
        mu = mu.view(-1, 1, 1, 1)
        var = (((y - mu) ** 2) * mask).sum(dim=(1, 2, 3)) / n_known
        sd  = var.clamp_min(0.0).sqrt().view(-1, 1, 1, 1)

        fill = self._fill(y, mu, sd)
        return (mask * y + (1.0 - mask) * fill).detach()

    def _fill(self, y, mu, sd):
        raise NotImplementedError


class NoiseBaseline(_StatsBaseline):
    name = 'noise'

    def _fill(self, y, mu, sd):
        xi = torch.randn(y.shape, generator=self.prg, device=y.device,
                         dtype=y.dtype)
        return mu + sd * xi


class MeanFillBaseline(_StatsBaseline):
    name = 'meanfill'

    def _fill(self, y, mu, sd):
        return mu.expand_as(y)


BASELINE_INPAINTERS = {
    'noise'    : NoiseBaseline,
    'meanfill' : MeanFillBaseline,
}
