#!/usr/bin/env python
"""Verify generated events against the paper's figures (arXiv:2406.01602).

Reproduces, per centrality, from locally generated events:
  * Fig. 4 : tower-area E_T spectra (1x1, 4x4, 7x7, 11x11) and Sum E_T
             distribution, overlaying all seeds; ratio panels seed_k/seed_0
             (the HIJING+G4 reference is not public, so cross-seed
             consistency stands in for the paper's ML/HIJING ratio).
  * Fig. 5 : <sigma_ET> of tower areas vs Sum E_T, per seed.

Also writes summary.json with per-seed <Sum E_T>, spectrum tails and the
single-tower maximum (to monitor the known high-tower-energy anomaly), and
soft-checks <Sum E_T> against the expected value (cent4 ~ 224.5 GeV).

Example:
    python scripts/verify_paper_plots.py \
        --events $ROOT/generated_events/events_cent4_seed{0,1,2,3,4}.npy \
        --labels seed0 seed1 seed2 seed3 seed4 \
        --title "Centrality 40-50%" --outdir $ROOT/verification/cent4
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calo_inpaint.metrics import (
    window_sums, sum_et, sigma_et_profile, TOWER_AREAS
)

EXPECTED_MEAN_SUM_ET = { 'cent0': None, 'cent4': 224.5 }   # GeV, soft checks


def parse_cmdargs():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', nargs='+', required=True,
                   help='event .npy files (one per seed)')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--title',  default='')
    p.add_argument('--outdir', required=True)
    p.add_argument('--centrality', default=None, choices=['cent0', 'cent4'],
                   help='enables the <Sum E_T> soft check')
    p.add_argument('--n-max', type=int, default=None,
                   help='cap events per file')
    return p.parse_args()


def spectrum(vals, bins):
    h, _ = np.histogram(vals, bins=bins)
    widths = np.diff(bins)
    dens = h / max(vals.size, 1) / widths          # (1/N) dN/dE
    return dens, h


def main():
    args = parse_cmdargs()
    os.makedirs(args.outdir, exist_ok=True)

    labels = args.labels or [os.path.basename(p) for p in args.events]
    assert len(labels) == len(args.events)

    data = []
    for path in args.events:
        ev = np.load(path, mmap_mode='r')
        if args.n_max:
            ev = ev[:args.n_max]
        data.append(np.asarray(ev, dtype=np.float32))

    # ---- Fig. 4 style: spectra --------------------------------------------
    areas   = list(TOWER_AREAS)
    n_cols  = len(areas) + 1
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4 * n_cols, 7), sharex='col',
        gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True
    )

    summary = {'per_seed': {}, 'checks': []}
    ratios_max_dev = 0.0

    for c, k in enumerate(areas + ['sum']):
        if k == 'sum':
            vals_all = [sum_et(ev) for ev in data]
            lo, hi = (np.percentile(np.concatenate(vals_all), [0.02, 99.98]))
            bins = np.linspace(lo * 0.9, hi * 1.1, 61)
            xlabel = r'$\Sigma E_T^{tower}$ [GeV]'
        else:
            vals_all = [window_sums(ev, k).ravel() for ev in data]
            hi   = max(v.max() for v in vals_all)
            bins = np.linspace(0.0, hi * 1.05, 81)
            xlabel = rf'${k}\times{k}$ $E_T$ [GeV]'

        ref_dens = None
        for vals, lab in zip(vals_all, labels):
            dens, _ = spectrum(vals, bins)
            centers = 0.5 * (bins[1:] + bins[:-1])
            axes[0, c].step(centers, dens, where='mid', lw=1, label=lab)
            if ref_dens is None:
                ref_dens = dens
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.where(ref_dens > 0, dens / ref_dens, np.nan)
                axes[1, c].step(centers, r, where='mid', lw=1)
                good = (ref_dens * vals.size * np.diff(bins) > 100)
                if good.any():
                    ratios_max_dev = max(
                        ratios_max_dev,
                        float(np.nanmax(np.abs(r[good] - 1.0)))
                    )

        axes[0, c].set_yscale('log')
        axes[0, c].set_ylabel(r'$(1/N)\,dN/dE_T$')
        axes[1, c].axhline(1.0, color='k', lw=0.5)
        axes[1, c].set_ylim(0.5, 1.5)
        axes[1, c].set_xlabel(xlabel)
        axes[1, c].set_ylabel('seed / seed0')

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{args.title}  —  tower-area $E_T$ spectra (paper Fig. 4)')
    fig.savefig(os.path.join(args.outdir, 'fig4_spectra.png'), dpi=150)
    plt.close(fig)

    # ---- Fig. 5 style: <sigma_ET> vs Sum E_T ------------------------------
    tot_all = np.concatenate([sum_et(ev) for ev in data])
    lo, hi  = np.percentile(tot_all, [1, 99])
    pbins   = np.linspace(lo, hi, 13)

    fig, axes = plt.subplots(1, len(areas), figsize=(4 * len(areas), 3.6),
                             constrained_layout=True)
    for c, k in enumerate(areas):
        for ev, lab in zip(data, labels):
            x, m, e, _ = sigma_et_profile(ev, k, pbins)
            axes[c].errorbar(x, m, yerr=e, fmt='o-', ms=3, lw=1, label=lab)
        axes[c].set_xlabel(r'$\Sigma E_T^{tower}$ [GeV]')
        axes[c].set_ylabel(rf'$\langle\sigma_{{E_T}}\rangle$ ${k}\times{k}$ [GeV]')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{args.title}  —  $\\langle\\sigma_{{E_T}}\\rangle$ (paper Fig. 5)')
    fig.savefig(os.path.join(args.outdir, 'fig5_sigma_et.png'), dpi=150)
    plt.close(fig)

    # ---- summary + soft checks -------------------------------------------
    for ev, lab in zip(data, labels):
        tot = sum_et(ev)
        summary['per_seed'][lab] = {
            'n_events'        : int(ev.shape[0]),
            'mean_sum_et_gev' : float(tot.mean()),
            'std_sum_et_gev'  : float(tot.std()),
            'max_tower_gev'   : float(ev.max()),
            'q99999_tower_gev': float(np.quantile(ev, 0.99999)),
        }

    means = [v['mean_sum_et_gev'] for v in summary['per_seed'].values()]
    summary['cross_seed'] = {
        'mean_sum_et_spread_pct':
            float(100 * (max(means) - min(means)) / np.mean(means)),
        'spectra_max_ratio_dev_pct': float(100 * ratios_max_dev),
    }

    exp = EXPECTED_MEAN_SUM_ET.get(args.centrality or '', None)
    if exp is not None:
        dev = 100 * abs(np.mean(means) - exp) / exp
        ok  = dev < 5.0
        summary['checks'].append({
            'check'   : f'<SumET> within 5% of expected {exp} GeV',
            'value'   : float(np.mean(means)),
            'dev_pct' : float(dev),
            'passed'  : bool(ok),
        })

    # single-tower anomaly monitor (expected physical scale ~5-6 GeV;
    # pathological values ~21-22 GeV were observed in earlier studies)
    max_tower = max(v['max_tower_gev'] for v in summary['per_seed'].values())
    summary['checks'].append({
        'check'  : 'max single-tower energy below 15 GeV',
        'value'  : float(max_tower),
        'passed' : bool(max_tower < 15.0),
    })

    with open(os.path.join(args.outdir, 'summary.json'), 'wt') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    if any(not c['passed'] for c in summary['checks']):
        print('\nWARNING: some verification checks failed (see summary.json)')


if __name__ == '__main__':
    main()
