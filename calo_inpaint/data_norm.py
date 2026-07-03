"""Data normalization, matching jetgen's LogNorm(clip_min=1e-3).

GeV -> model space:  x = ln(clip(E, clip_min))
model space -> GeV:  E = exp(x)          (no clipping on the way back)
"""

import numpy as np
import torch

__all__ = ['LogNorm']


class LogNorm:

    def __init__(self, clip_min=1e-3):
        self.clip_min = clip_min

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            return np.log(np.clip(x, self.clip_min, None))
        return torch.log(torch.clamp(x, min=self.clip_min))

    def denormalize(self, y):
        if isinstance(y, np.ndarray):
            return np.exp(y)
        return torch.exp(y)
