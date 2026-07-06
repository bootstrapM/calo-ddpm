"""Clean-room gradient-guided inpainters (independent of .inpainting).

Shares only data ingestion with the rest of the repo: schedule arrays,
masks, log-normalization, and the frozen eps-network.  Registered under
distinct names so runs are directly comparable with the first
implementations (mcg / pigdm) side by side.
"""

from .mcg2   import MCG2Inpainter
from .pigdm2 import PiGDM2Inpainter

GUIDED_INPAINTERS = {
    'mcg2'   : MCG2Inpainter,
    'pigdm2' : PiGDM2Inpainter,
}

__all__ = ['MCG2Inpainter', 'PiGDM2Inpainter', 'GUIDED_INPAINTERS']
