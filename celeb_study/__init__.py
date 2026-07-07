"""Parallel CelebA-HQ inpainting study (natural-image comparison anchor).

Same algorithms, same generation machinery, same statistics as the
calorimeter study — by construction: this package IMPORTS the shared
implementations (calo_inpaint.schedule, calo_inpaint.ddpm_sampler,
calo_inpaint.inpainting, calo_inpaint.guided) rather than copying them.
Only the model adapter (HuggingFace diffusers UNet), the data range
([-1, 1] instead of log-GeV), and the mask geometry differ.
"""
