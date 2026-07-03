"""calo_inpaint: DDPM-prior inpainting study for sPHENIX calorimeter images.

Reproduction package for the calo-ddpm inpainting project.

Conventions used throughout (matching LS4GAN/calo-ddpm "jetgen" exactly):
  * Images are (1, 24, 64) tower E_T maps in the (eta, phi) plane, in GeV.
  * Model space: x = ln(clip(E_T, 1e-3)).  GeV space: E_T = exp(x).
  * The diffusion schedule is a linear-beta VP schedule with T=8000 steps,
    beta_i = linspace(beta_param/T, 1000*beta_param/T, T),
    stored in arrays of length S+1 with index 0 being the identity
    ("no noise") transition.  beta_param = 0.02 (cent0), 0.10 (cent4);
    it is read from the model's config.json, never hard-coded.
  * eps-prediction network: improved_diffusion UNetModel, conditioned on the
    ORIGINAL timestep t in [1..T] (via Schedule.t_map), also for subsampled
    schedules.
"""

from .schedule     import Schedule, make_subsampled_schedule
from .data_norm    import LogNorm
from .ddpm_sampler import load_model, DDPMSampler
from .ddim_sampler import DDIMSampler
from .masks        import square_mask

__all__ = [
    'Schedule', 'make_subsampled_schedule', 'LogNorm',
    'load_model', 'DDPMSampler', 'DDIMSampler', 'square_mask',
]
