"""Shared analysis utilities: run loading + scalar summary metrics.

Single source of truth for sweep-style analysis (box-size sweeps, DDRM
eta sweep, baselines) used by the notebooks.  Operates purely on the
study file format (results.npy / truth.npy / metadata.json /
progress.txt) — no torch, no model.

Conventions: samples/truth stored in GeV; value-level metrics are
computed in log space by default ('log' = the raw DDPM model space; the
stored exp() is inverted losslessly).  Rank-based metrics are invariant
under this monotone map.  The finite-J calibrated pull-std reference for
J samples is sqrt((1+1/J)(J-1)/(J-3)) ~ 1.031 at J=50, reported
alongside the measured value.
"""

import glob
import json
import os

import numpy as np
from scipy import stats as sps

__all__ = ['load_run', 'summarize_run', 'collect_runs', 'pull_std_reference']

CLIP_MIN = 1e-3


def pull_std_reference(J):
    """Calibrated pull std for J posterior samples (Gaussian posterior)."""
    return float(np.sqrt((1 + 1 / J) * (J - 1) / max(J - 3, 1)))


def load_run(rundir):
    """Load one study run: progress-truncated, non-finite rows excluded.

    Returns dict with samples (n,J,b,b) GeV, truth (n,b,b) GeV, meta,
    n_excluded."""
    meta = json.load(open(os.path.join(rundir, 'metadata.json')))
    s = np.load(os.path.join(rundir, 'results.npy'))
    t = np.load(os.path.join(rundir, 'truth.npy'))

    done = s.shape[0]
    prog = os.path.join(rundir, 'progress.txt')
    if os.path.exists(prog):
        done = int(open(prog).read().split('/')[0])
    s, t = s[:done], t[:done]

    finite = np.isfinite(s).all(axis=tuple(range(1, s.ndim)))
    n_bad = int((~finite).sum())
    s, t = s[finite], t[finite]

    if s.ndim == 5:                       # celeb RGB: fold channels
        n_, j_, c_, b_, _ = s.shape
        s = s.reshape(n_, j_, c_ * b_, b_)
        t = t.reshape(t.shape[0], c_ * b_, b_)

    return {'samples': s.astype(np.float64), 'truth': t.astype(np.float64),
            'meta': meta, 'n_excluded': n_bad, 'rundir': rundir}


def summarize_run(rundir, space='log', seed=0):
    """Scalar summary metrics for one run (None if no finite images)."""
    r = load_run(rundir)
    s, t, meta = r['samples'], r['truth'], r['meta']
    if s.shape[0] == 0:
        return None
    n, J = s.shape[:2]
    rng = np.random.default_rng(seed)

    if space == 'log' and meta.get('space', 'gev') == 'gev':
        sv = np.log(np.clip(s, CLIP_MIN * 1e-3, None))
        tv = np.log(np.clip(t, CLIP_MIN * 1e-3, None))
    else:
        sv, tv = s, t

    # pulls (space-dependent); zero-width posteriors (e.g. the meanfill
    # baseline) have no valid pulls -> nan, silently
    mu = sv.mean(axis=1)
    sd = sv.std(axis=1, ddof=1)
    ok = sd > 1e-12
    z = (tv - mu)[ok] / sd[ok]
    pull_mean = float(z.mean()) if z.size else float('nan')
    pull_std  = float(z.std())  if z.size else float('nan')

    # SBC ranks (monotone-invariant), pooled chi2
    less  = (s < t[:, None]).sum(axis=1)
    equal = (s == t[:, None]).sum(axis=1)
    ranks = less + (rng.random(less.shape) * (equal + 1)).astype(int)
    counts = np.bincount(ranks.ravel(), minlength=J + 1)
    sbc_p = float(sps.chisquare(counts).pvalue) if counts.sum() else np.nan

    # central coverage (invariant)
    cov = {}
    for g in (0.5, 0.9):
        lo = np.quantile(s, (1 - g) / 2, axis=1)
        hi = np.quantile(s, 1 - (1 - g) / 2, axis=1)
        cov[g] = float(((t >= lo) & (t <= hi)).mean())

    # bias (space of `space`), with sign-fraction across pixels
    bias_map = (sv.mean(axis=1) - tv).mean(axis=0)
    bias_se  = (sv.mean(axis=1) - tv).std(axis=0) / np.sqrt(n)

    # CRPS (sorted estimator) + sharpness, in `space`
    ss = np.sort(np.moveaxis(sv, 1, -1), axis=-1)
    i = np.arange(1, J + 1)
    term2 = (2 / (J * J)) * ((2 * i - J - 1) * ss).sum(axis=-1)
    crps = float((np.abs(np.moveaxis(sv, 1, -1)
                         - tv[..., None]).mean(axis=-1) - 0.5 * term2).mean())

    # shrinkage slopes on box means (nan on degenerate inputs)
    t_m = tv.mean(axis=(1, 2))
    f_m = sv.mean(axis=(1, 2, 3))

    class _Nan:
        slope = rvalue = float('nan')
    try:
        res_f = sps.linregress(t_m, f_m)
        res_r = sps.linregress(f_m, t_m)
    except ValueError:
        res_f = res_r = _Nan()

    hp = meta.get('hyperparams', {})
    return {
        'rundir': rundir,
        'algorithm': meta['algorithm'],
        'box': int(meta['box']),
        'S': int(meta.get('S', -1)),
        'ddrm_eta': hp.get('ddrm_eta') if meta['algorithm'] == 'ddrm' else None,
        'n_images': int(n), 'n_samples': int(J),
        'n_excluded': r['n_excluded'],
        'space': space,
        'pull_mean': pull_mean, 'pull_std': pull_std,
        'pull_std_ref': pull_std_reference(J),
        'sbc_p': sbc_p,
        'cov50': cov[0.5], 'cov90': cov[0.9],
        'bias_mean': float(bias_map.mean()),
        'bias_sign_frac': float((bias_map > 0).mean()),
        'bias_maxabs_z': float(np.abs(bias_map / np.clip(bias_se, 1e-12,
                                                         None)).max()),
        'crps': crps,
        'sharpness': float(sv.std(axis=1).mean()),
        'slope_fwd': float(res_f.slope), 'slope_rev': float(res_r.slope),
        'r2': float(res_f.rvalue ** 2),
    }


def collect_runs(patterns, space='log', seed=0):
    """Summarize every run dir matching any glob pattern (str or list).

    Returns a list of summary dicts, sorted by (algorithm, box, eta)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    dirs = []
    for p in patterns:
        dirs += [d for d in glob.glob(p)
                 if os.path.exists(os.path.join(d, 'results.npy'))]
    out = []
    for d in sorted(set(dirs)):
        try:
            summ = summarize_run(d, space=space, seed=seed)
        except Exception as e:                      # noqa: BLE001
            print(f'WARNING: {d}: {e}')
            continue
        if summ is not None:
            out.append(summ)
    out.sort(key=lambda r: (r['algorithm'], r['box'],
                            r['ddrm_eta'] if r['ddrm_eta'] is not None else -1))
    return out
