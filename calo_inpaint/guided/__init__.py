"""Gradient-guided inpainters adapted from Himanshu's CelebA implementation.

Independent of calo_inpaint.inpainting: shares only data ingestion
(schedule arrays, masks, log-normalization) with the rest of the repo.
Registered under distinct names (mcg2 / pigdm2) so runs are directly
comparable with the first implementations side by side.
"""

from .mcg2   import MCG2Inpainter
from .pigdm2 import PiGDM2Inpainter

GUIDED_INPAINTERS = {
    'mcg2'   : MCG2Inpainter,
    'pigdm2' : PiGDM2Inpainter,
}

__all__ = ['MCG2Inpainter', 'PiGDM2Inpainter', 'GUIDED_INPAINTERS']
