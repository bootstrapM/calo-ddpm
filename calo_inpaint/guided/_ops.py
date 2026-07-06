"""Shared runtime for the gradient-guided inpainters (calo_inpaint.guided).

Adapted from Himanshu's CelebA implementation.  Independent of
calo_inpaint.inpainting — shares only data ingestion (schedule arrays,
masks, log-normalization) with the rest of the repo.  All reverse-step
math is written out locally from the schedule arrays.

Convention mapping from the CelebA original:
  * mask: HERE 1 = known/live, 0 = dead   (original: 1 = inpaint region)
  * data range: log space (default clamp -7..4) instead of [-1, 1]
  * RNG: seeded torch.Generator (pipeline reproducibility/resume contract)
"""

import torch

__all__ = ['make_generator', 'expand_measurement', 'assert_finite',
           'predict_x0', 'ddpm_step', 'q_sample_prev', 'clip_grad_per_sample']


def make_generator(device, seed):
    g = torch.Generator(device)
    g.manual_seed(seed)
    return g


def _canonical(z, device, name):
    z = z.to(device).float()
    if z.dim() == 2:
        return z[None, None]
    if z.dim() == 3:
        return z[None]
    if z.dim() == 4:
        return z
    raise ValueError(f'{name}: expected 2D/3D/4D, got {tuple(z.shape)}')


def expand_measurement(y, mask, n_samples, device):
    """Validate mask (1=known), expand K images to the sample batch."""
    y    = _canonical(y, device, 'y')
    mask = _canonical(mask, device, 'mask')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('mask must be binary: 1 = known/live, 0 = dead')
    return y.repeat_interleave(n_samples, dim=0), mask


def assert_finite(t, step, what):
    if not torch.isfinite(t).all():
        raise FloatingPointError(
            f'non-finite {what} at reverse step s={step} — aborting '
            f'instead of writing NaNs')


def predict_x0(sched, s, x, eps):
    """x0hat = (x - sqrt(vbar_s) eps) / sbar_s   (Tweedie, eps-param)."""
    return (x - sched.vbar[s].sqrt() * eps) / sched.sbar[s]


def ddpm_step(sched, s, x, x0, generator):
    """Ancestral posterior step q(x_{s-1} | x_s, x0), coefficients derived
    inline from the cumulative schedule (jump-aware for subsampled grids).
    Collapses to x0 exactly at s == 1."""
    if s == 1:
        return x0
    sb, vb           = sched.sbar[s], sched.vbar[s]
    sb_prev, vb_prev = sched.sbar[s - 1], sched.vbar[s - 1]
    beta_k   = 1.0 - (sb / sb_prev) ** 2
    coef_x0  = sb_prev * beta_k / vb
    coef_xt  = (sb / sb_prev) * vb_prev / vb
    post_var = beta_k * vb_prev / vb
    noise = torch.randn(x.shape, generator=generator, device=x.device,
                        dtype=x.dtype)
    return coef_x0 * x0 + coef_xt * x + post_var.sqrt() * noise


def q_sample_prev(sched, s, y, generator):
    """Forward-diffuse the known content y to level s-1 (RePaint branch)."""
    if s == 1:
        return y
    sb_prev, vb_prev = sched.sbar[s - 1], sched.vbar[s - 1]
    noise = torch.randn(y.shape, generator=generator, device=y.device,
                        dtype=y.dtype)
    return sb_prev * y + vb_prev.sqrt() * noise


def clip_grad_per_sample(g, max_norm=1.0):
    """Per-SAMPLE norm clipping (adaptation of the original's global
    `grad / grad.norm()`: at study batch sizes of K*50 a global norm would
    shrink the guidance with batch size; per-sample preserves the original
    per-image behavior at any batch size)."""
    norms = g.flatten(1).norm(dim=1).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return g * scale.view(-1, 1, 1, 1)
