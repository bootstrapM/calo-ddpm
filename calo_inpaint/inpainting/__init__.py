from .base_inpainter import BaseInpainter
from .repaint        import RePaintInpainter
from .ddnm           import DDNMInpainter
from .ddrm           import DDRMInpainter
from .mcg            import MCGInpainter
from .pigdm          import PiGDMInpainter

INPAINTERS = {
    'repaint' : RePaintInpainter,
    'ddnm'    : DDNMInpainter,
    'ddrm'    : DDRMInpainter,
    'mcg'     : MCGInpainter,
    'pigdm'   : PiGDMInpainter,
}

__all__ = [
    'BaseInpainter', 'RePaintInpainter', 'DDNMInpainter', 'DDRMInpainter',
    'MCGInpainter', 'PiGDMInpainter', 'INPAINTERS',
]
