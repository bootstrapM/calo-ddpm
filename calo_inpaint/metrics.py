"""Physics observables used for the paper-verification plots (Figs. 4 & 5).

Tower areas: a k x k tower area is the sum of E_T over a k x k patch of
towers.  Patches slide with stride 1; phi (axis -1) is periodic (wrapped),
eta (axis -2) uses valid positions only, so events (N, 24, 64) give
(N, 24-k+1, 64) area sums.
"""

import numpy as np

__all__ = ['window_sums', 'sum_et', 'sigma_et_profile', 'TOWER_AREAS']

TOWER_AREAS = (1, 4, 7, 11)


def window_sums(events, k):
    """Sliding k x k patch sums; phi periodic, eta valid.

    events: (N, H, W) in GeV  ->  (N, H-k+1, W)
    """
    ev = np.asarray(events, dtype=np.float64)
    if k == 1:
        return ev.copy()

    n, h, w = ev.shape
    ev = np.concatenate([ev, ev[:, :, :k - 1]], axis=2)   # phi wrap

    # integral image
    ii = np.zeros((n, h + 1, w + k), dtype=np.float64)
    ii[:, 1:, 1:] = ev.cumsum(axis=1).cumsum(axis=2)

    out = (
          ii[:, k:h + 1,     k:w + k]
        - ii[:, :h - k + 1,  k:w + k]
        - ii[:, k:h + 1,     :w]
        + ii[:, :h - k + 1,  :w]
    )
    return out


def sum_et(events):
    """Total E_T per event: (N, H, W) -> (N,)."""
    return np.asarray(events, dtype=np.float64).sum(axis=(1, 2))


def sigma_et_profile(events, k, bins):
    """<sigma_ET> of k x k tower areas vs Sum E_T (paper Fig. 5).

    For each event, sigma_ET = std of the k x k window sums across the
    event; the profile averages sigma_ET over events in bins of Sum E_T.

    Returns (bin_centers, mean_sigma, sem_sigma, counts).
    """
    ws  = window_sums(events, k)
    sig = ws.reshape(ws.shape[0], -1).std(axis=1)
    tot = sum_et(events)

    idx     = np.digitize(tot, bins) - 1
    nb      = len(bins) - 1
    centers = 0.5 * (bins[1:] + bins[:-1])

    mean = np.full(nb, np.nan)
    sem  = np.full(nb, np.nan)
    cnt  = np.zeros(nb, dtype=int)

    for b in range(nb):
        sel = sig[idx == b]
        cnt[b] = sel.size
        if sel.size > 0:
            mean[b] = sel.mean()
            sem[b]  = sel.std() / max(np.sqrt(sel.size), 1.0)

    return centers, mean, sem, cnt
