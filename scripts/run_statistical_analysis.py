#!/usr/bin/env python
"""Statistical validation of inpainting posteriors (per study run directory).

The evaluation frame is Bayesian posterior validation: truth images come
from the same DDPM that serves as the prior, so any miscalibration is pure
algorithmic approximation error.  Point-estimate metrics (FID/PSNR/Pearson)
are intentionally not used; posterior-mean shrinkage (slope < 1) is expected
and reported only as a diagnostic.

For each run directory (results.npy (n_img, n_smp, b, b) + truth.npy):

  * SBC rank statistics (Talts et al., arXiv:1804.06788): per-pixel rank of
    truth among posterior samples; pooled histogram + per-pixel chi2
    p-values.  Ranks are invariant under the monotone log<->GeV map.
  * TARP expected coverage (Lemos et al., arXiv:2302.03026): joint test over
    all box pixels, random uniform reference points in standardized log
    space; ECP curve + max deviation.
  * Pull z-scores in log space: z = (truth - post_mean) / post_std, pooled
    histogram vs N(0,1), per-pixel mean/std.
  * Central credible-interval coverage at 10%..90% (per-pixel pooled), and
    for the physics observable Sum E (GeV) over the dead box.
  * Shrinkage diagnostic: posterior mean vs truth (GeV) with fitted slope.

Multiple run dirs may be given; a cross-algorithm comparison figure is then
also produced in --compare-outdir.

Example:
    python scripts/run_statistical_analysis.py \
        $ROOT/inpaint_study/repaint_box4_eta8_phi28_S1000 [more run dirs...]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLIP_MIN = 1e-3
LEVELS   = np.arange(0.1, 0.91, 0.1)


def to_log(x):
    return np.log(np.clip(x, CLIP_MIN, None))


# --------------------------------------------------------------------------
def sbc_ranks(samples, truth, rng):
    """Rank of truth among samples per (image, pixel); ties broken uniformly.

    samples: (N, J, b, b), truth: (N, b, b) -> ranks (N, b, b) in [0..J].
    """
    less  = (samples < truth[:, None]).sum(axis=1)
    equal = (samples == truth[:, None]).sum(axis=1)
    return less + (rng.random(less.shape) * (equal + 1)).astype(int)


def rank_uniformity_pvalue(ranks, n_ranks):
    """Chi-square p-value for uniformity of pooled ranks."""
    counts = np.bincount(ranks.ravel(), minlength=n_ranks)
    if counts.sum() == 0:
        return np.nan
    return float(sps.chisquare(counts).pvalue)


def tarp_ecp(samples, truth, rng):
    """TARP: returns (alpha_grid, ecp, max_dev, f_values).

    samples: (N, J, D), truth: (N, D), standardized coordinates.
    Reference points uniform over the pooled per-dim [min, max] box.
    """
    n, j, d = samples.shape
    lo = np.minimum(samples.min(axis=(0, 1)), truth.min(axis=0))
    hi = np.maximum(samples.max(axis=(0, 1)), truth.max(axis=0))

    ref = lo + (hi - lo) * rng.random((n, d))

    d_smp = np.linalg.norm(samples - ref[:, None, :], axis=2)   # (N, J)
    d_tru = np.linalg.norm(truth - ref, axis=1)                 # (N,)

    f = (d_smp < d_tru[:, None]).mean(axis=1)                   # (N,)

    alpha = np.linspace(0, 1, 101)
    ecp   = (f[None, :] <= alpha[:, None]).mean(axis=1)
    max_dev = float(np.abs(ecp - alpha).max())
    return alpha, ecp, max_dev, f


def central_coverage(samples, truth, levels):
    """Pooled central credible-interval coverage.  samples: (N,J,...)"""
    cov = []
    for g in levels:
        lo = np.quantile(samples, (1 - g) / 2, axis=1)
        hi = np.quantile(samples, (1 + g) / 2, axis=1)
        cov.append(float(((truth >= lo) & (truth <= hi)).mean()))
    return np.array(cov)


# --------------------------------------------------------------------------
def analyze_run(rundir, seed=0):
    meta = json.load(open(os.path.join(rundir, 'metadata.json')))
    samples_gev = np.load(os.path.join(rundir, 'results.npy'))  # (N,J,b,b)
    truth_gev   = np.load(os.path.join(rundir, 'truth.npy'))    # (N,b,b)

    # 'gev' (calo, log-space analysis) or 'unit' (celeb study, [-1,1]);
    # multi-channel results (N,J,C,b,b) fold channels into rows
    space = meta.get('space', 'gev')
    if samples_gev.ndim == 5:
        n_, j_, c_, b_, _ = samples_gev.shape
        samples_gev = samples_gev.reshape(n_, j_, c_ * b_, b_)
        truth_gev   = truth_gev.reshape(truth_gev.shape[0], c_ * b_, b_)

    done = int(open(os.path.join(rundir, 'progress.txt'))
               .read().split('/')[0]) \
        if os.path.exists(os.path.join(rundir, 'progress.txt')) \
        else samples_gev.shape[0]
    samples_gev = samples_gev[:done]
    truth_gev   = truth_gev[:done]

    # exclude images with non-finite samples (a NaN here means the SAMPLER
    # produced NaNs -- unfinished memmap rows are zeros, not NaNs)
    finite = np.isfinite(samples_gev).all(axis=(1, 2, 3)) \
           & np.isfinite(truth_gev).all(axis=(1, 2))
    n_bad  = int((~finite).sum())
    if n_bad:
        print(f'WARNING [{rundir}]: {n_bad}/{done} images contain '
              f'non-finite samples — excluded from analysis; '
              f'investigate the sampler run!')
        samples_gev = samples_gev[finite]
        truth_gev   = truth_gev[finite]
    if samples_gev.shape[0] == 0:
        print(f'ERROR [{rundir}]: no finite images — skipping run')
        return None

    n, j, br, bc = samples_gev.shape     # rows may be C*box (folded channels)
    b = int(meta.get('box', bc))
    rng    = np.random.default_rng(seed)
    outdir = os.path.join(rundir, 'stats')
    os.makedirs(outdir, exist_ok=True)

    if space == 'unit':                 # already a bounded model space
        s_log = samples_gev.astype(np.float64)
        t_log = truth_gev.astype(np.float64)
    else:
        s_log = to_log(samples_gev)
        t_log = to_log(truth_gev)

    summary = {'rundir': rundir, 'algorithm': meta['algorithm'],
               'box': b, 'n_images': int(n), 'n_samples': int(j),
               'space': space,
               'n_nonfinite_images_excluded': n_bad}

    # ---- SBC ranks --------------------------------------------------------
    ranks = sbc_ranks(samples_gev, truth_gev, rng)
    summary['sbc'] = {
        'pooled_chi2_pvalue': rank_uniformity_pvalue(ranks, j + 1),
        'per_pixel_chi2_pvalue': [
            [rank_uniformity_pvalue(ranks[:, r, c], j + 1) for c in range(bc)]
            for r in range(br)
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].hist(ranks.ravel(), bins=np.arange(j + 2) - 0.5,
                 density=True, histtype='stepfilled', alpha=0.7)
    axes[0].axhline(1.0 / (j + 1), color='k', ls='--', lw=1)
    axes[0].set_xlabel('SBC rank of truth')
    axes[0].set_ylabel('density')
    axes[0].set_title(
        f"pooled ranks, chi2 p = {summary['sbc']['pooled_chi2_pvalue']:.3g}")
    im = axes[1].imshow(np.array(summary['sbc']['per_pixel_chi2_pvalue']),
                        vmin=0, vmax=1, cmap='viridis')
    axes[1].set_title('per-pixel chi2 p-value')
    fig.colorbar(im, ax=axes[1])
    fig.suptitle(f"{meta['algorithm']} box={b} — SBC")
    fig.savefig(os.path.join(outdir, 'sbc_ranks.png'), dpi=150)
    plt.close(fig)

    # ---- TARP (joint over box pixels, standardized log space) -------------
    s_flat = s_log.reshape(n, j, br * bc)
    t_flat = t_log.reshape(n, br * bc)
    mu  = s_flat.mean(axis=(0, 1))
    sd  = s_flat.std(axis=(0, 1)) + 1e-12
    alpha, ecp, tarp_dev, f_vals = tarp_ecp(
        (s_flat - mu) / sd, (t_flat - mu) / sd, rng
    )
    summary['tarp'] = {
        'max_ecp_deviation': tarp_dev,
        'f_ks_pvalue': float(sps.kstest(f_vals, 'uniform').pvalue),
    }

    fig, ax = plt.subplots(figsize=(4.2, 4), constrained_layout=True)
    ax.plot(alpha, ecp, lw=2, label='TARP ECP')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='calibrated')
    ax.set_xlabel('credibility level')
    ax.set_ylabel('expected coverage')
    ax.legend()
    ax.set_title(f"{meta['algorithm']} box={b} — TARP "
                 f"(max dev {tarp_dev:.3f})")
    fig.savefig(os.path.join(outdir, 'tarp.png'), dpi=150)
    plt.close(fig)

    # ---- pull z-scores (log space) ----------------------------------------
    post_mean = s_log.mean(axis=1)
    post_std  = s_log.std(axis=1, ddof=1) + 1e-12
    z = (t_log - post_mean) / post_std
    summary['pull'] = {
        'mean': float(z.mean()), 'std': float(z.std()),
        'per_pixel_mean': z.mean(axis=0).tolist(),
        'per_pixel_std' : z.std(axis=0).tolist(),
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    grid = np.linspace(-5, 5, 200)
    ax.hist(np.clip(z.ravel(), -5, 5), bins=60, density=True, alpha=0.7)
    ax.plot(grid, sps.norm.pdf(grid), 'k--', lw=1, label='N(0,1)')
    ax.set_xlabel('pull  z = (truth − mean) / std   [log space]')
    ax.legend()
    ax.set_title(f"{meta['algorithm']} box={b} — pulls: "
                 f"mean {z.mean():.3f}, std {z.std():.3f}")
    fig.savefig(os.path.join(outdir, 'pull_z.png'), dpi=150)
    plt.close(fig)

    # ---- central credible-interval coverage -------------------------------
    cov_pix = central_coverage(s_log, t_log, LEVELS)
    sum_smp = samples_gev.sum(axis=(2, 3))            # (N, J) box Sum E
    sum_tru = truth_gev.sum(axis=(1, 2))              # (N,)
    cov_sum = central_coverage(sum_smp, sum_tru, LEVELS)
    summary['coverage'] = {
        'levels'    : LEVELS.tolist(),
        'per_pixel' : cov_pix.tolist(),
        'box_sum_e' : cov_sum.tolist(),
    }

    fig, ax = plt.subplots(figsize=(4.2, 4), constrained_layout=True)
    ax.plot(LEVELS, cov_pix, 'o-', label='per-pixel (pooled)')
    ax.plot(LEVELS, cov_sum, 's-', label=r'box $\Sigma E$')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('nominal central credibility')
    ax.set_ylabel('empirical coverage')
    ax.legend()
    ax.set_title(f"{meta['algorithm']} box={b} — coverage")
    fig.savefig(os.path.join(outdir, 'coverage.png'), dpi=150)
    plt.close(fig)

    # ---- shrinkage diagnostic (GeV) ---------------------------------------
    pm  = samples_gev.mean(axis=1).ravel()
    tr  = truth_gev.ravel()
    sel = tr > CLIP_MIN * 2 if space != 'unit' \
        else np.ones_like(tr, dtype=bool)
    slope = float(np.polyfit(tr[sel], pm[sel], 1)[0]) if sel.sum() > 10 \
        else np.nan
    summary['shrinkage_slope_gev'] = slope

    fig, ax = plt.subplots(figsize=(4.2, 4), constrained_layout=True)
    ax.plot(tr, pm, '.', ms=1, alpha=0.3)
    lim = max(tr.max(), pm.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel('truth [GeV]')
    ax.set_ylabel('posterior mean [GeV]')
    ax.set_title(f'slope = {slope:.3f}  (shrinkage < 1 expected)')
    fig.savefig(os.path.join(outdir, 'shrinkage.png'), dpi=150)
    plt.close(fig)

    with open(os.path.join(outdir, 'stats_summary.json'), 'wt') as f:
        json.dump(summary, f, indent=2)

    print(f"[{meta['algorithm']} box={b}] "
          f"SBC p={summary['sbc']['pooled_chi2_pvalue']:.3g}  "
          f"TARP dev={tarp_dev:.3f}  "
          f"pull std={z.std():.3f}  "
          f"shrinkage={slope:.3f}")
    return summary, (LEVELS, cov_pix), ranks, j


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rundirs', nargs='+')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compare-outdir', default=None)
    args = p.parse_args()

    all_summaries, compare = [], []
    for rd in args.rundirs:
        result = analyze_run(rd, seed=args.seed)
        if result is None:
            continue
        summary, cov, ranks, j = result
        all_summaries.append(summary)
        compare.append((summary['algorithm'], summary['box'],
                        cov, ranks, j, rd))

    if len(compare) > 1:
        outdir = args.compare_outdir or os.path.commonpath(args.rundirs)
        os.makedirs(outdir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4),
                                 constrained_layout=True)
        base_labs = [f'{n_} box={b_}' for n_, b_, _, _, _, _ in compare]
        for (name, box, (lv, cov), ranks, j, rd) in compare:
            lab = f'{name} box={box}'
            if base_labs.count(lab) > 1:      # sweep points: disambiguate
                lab += ' ' + os.path.basename(os.path.dirname(
                    os.path.abspath(rd)))
            axes[0].plot(lv, cov, 'o-', label=lab)
            # rank ECDF deviation from uniform
            r = np.sort(ranks.ravel()) / j
            ecdf = np.arange(1, r.size + 1) / r.size
            axes[1].plot(r, ecdf - r, lw=1, label=lab)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('nominal credibility')
        axes[0].set_ylabel('empirical per-pixel coverage')
        axes[1].axhline(0, color='k', ls='--', lw=1)
        axes[1].set_xlabel('normalized rank')
        axes[1].set_ylabel('ECDF − uniform')
        axes[0].legend(fontsize=7)
        fig.suptitle('algorithm comparison')
        fig.savefig(os.path.join(outdir, 'compare_algorithms.png'), dpi=150)
        plt.close(fig)

        with open(os.path.join(outdir, 'compare_summary.json'), 'wt') as f:
            json.dump(all_summaries, f, indent=2)
        print(f'comparison written to {outdir}')


if __name__ == '__main__':
    main()
