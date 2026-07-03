"""Dead-region masks.

Convention: mask == 1 on KNOWN (live) pixels, mask == 0 on the dead region
to be inpainted.  Images are (1, 24, 64) = (channel, eta, phi).
"""

import torch

__all__ = ['square_mask']


def square_mask(box, eta0=8, phi0=28, height=24, width=64, device='cpu'):
    """Square dead region of size box x box with top-left corner (eta0, phi0)."""
    assert 0 <= eta0 and eta0 + box <= height, f'box eta range out of bounds'
    assert 0 <= phi0 and phi0 + box <= width,  f'box phi range out of bounds'

    mask = torch.ones(1, height, width, device=device)
    mask[:, eta0:eta0 + box, phi0:phi0 + box] = 0.0
    return mask
