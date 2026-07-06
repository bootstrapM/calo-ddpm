"""Runtime helpers for the clean-room gradient-guided inpainters.

Deliberately independent of calo_inpaint.inpainting: this package shares
only data ingestion with the rest of the repo (schedule arrays, masks,
log-normalization).  All reverse-step math is written out locally.
"""

import torch

__all__ = ['make_generator', 'canonical_image', 'expand_measurement',
           'assert_finite']


def make_generator(device, seed):
    g = torch.Generator(device)
    g.manual_seed(seed)
    return g


def canonical_image(z, device, name='tensor'):
    """(H,W) / (C,H,W) / (K,C,H,W) -> (K,C,H,W) float32 on device."""
    z = z.to(device).float()
    if z.dim() == 2:
        return z[None, None]
    if z.dim() == 3:
        return z[None]
    if z.dim() == 4:
        return z
    raise ValueError(f'{name}: expected 2D/3D/4D, got {tuple(z.shape)}')


def expand_measurement(y, mask, n_samples, device):
    """Validate mask, expand K images to the sample batch (image-major)."""
    y    = canonical_image(y, device, 'y')
    mask = canonical_image(mask, device, 'mask')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('mask must be binary: 1 = known/live, 0 = dead')
    return y.repeat_interleave(n_samples, dim=0), mask


def assert_finite(t, step, what):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        mx  = t[torch.isfinite(t)].abs().max().item() if \
            torch.isfinite(t).any() else float('nan')
        raise FloatingPointError(
            f'non-finite {what} at reverse step s={step} '
            f'({bad} elements; max finite |value| = {mx:.3e}) — '
            f'sampler diverged, aborting instead of writing NaNs'
        )
