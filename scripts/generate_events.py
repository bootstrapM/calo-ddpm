#!/usr/bin/env python
"""Generate calorimeter events from a pre-trained calo-ddpm model.

Matches the paper's generation setup: ancestral DDPM with T = S = 8000
(DDIM with --dp ddim --eta ... also supported).  Output is a float32
memmap-backed .npy of shape (n, 24, 64) in GeV plus a JSON sidecar with
the full generation config.  Interrupted runs resume from progress.txt.

Example:
    python scripts/generate_events.py \
        --model-dir  $ROOT/pre-trained-model-weights/cent0_ddpm_seed0 \
        --outdir     $ROOT/generated_events \
        --tag        cent0_seed0 \
        -n 10000 --batch 1000 -S 8000 --seed 0
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calo_inpaint.ddpm_sampler import (
    load_model, schedule_from_config, lognorm_from_config
)
from calo_inpaint.ddpm_sampler import DDPMSampler
from calo_inpaint.ddim_sampler import DDIMSampler


def parse_cmdargs():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--outdir',    required=True)
    p.add_argument('--tag',       required=True,
                   help='output basename, e.g. cent0_seed0')
    p.add_argument('-n', '--n-events', type=int, default=10000)
    p.add_argument('--batch',     type=int, default=1000)
    p.add_argument('-S', '--steps', type=int, default=8000,
                   help='sampling steps (subsampled from T)')
    p.add_argument('--dp',   choices=['ddpm', 'ddim'], default='ddpm')
    p.add_argument('--eta',  type=float, default=0.0, help='DDIM eta')
    p.add_argument('--seed', type=int, default=0, help='sampler RNG seed')
    p.add_argument('--device', default='cuda')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--compile', action='store_true')
    return p.parse_args()


def main():
    args   = parse_cmdargs()
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    net, config = load_model(args.model_dir, device)
    sched   = schedule_from_config(config, S=args.steps, device=device)
    lognorm = lognorm_from_config(config)
    shape   = tuple(config['data']['datasets'][0]['shape'])   # (1, 24, 64)

    if args.compile:
        net = torch.compile(net)

    if args.dp == 'ddpm':
        sampler = DDPMSampler(net, sched, device, seed=args.seed,
                              use_bf16=args.bf16)
    else:
        sampler = DDIMSampler(net, sched, device, seed=args.seed,
                              eta=args.eta, use_bf16=args.bf16)

    out_npy  = os.path.join(args.outdir, f'events_{args.tag}.npy')
    out_json = os.path.join(args.outdir, f'events_{args.tag}.json')
    progress = os.path.join(args.outdir, f'progress_{args.tag}.txt')

    n, (_, h, w) = args.n_events, shape

    if os.path.exists(out_npy):
        events = np.lib.format.open_memmap(out_npy, mode='r+')
        assert events.shape == (n, h, w), \
            f'existing {out_npy} has shape {events.shape}, expected {(n, h, w)}'
    else:
        events = np.lib.format.open_memmap(
            out_npy, mode='w+', dtype=np.float32, shape=(n, h, w)
        )

    start = 0
    if os.path.exists(progress):
        with open(progress) as f:
            start = int(f.read().split('/')[0])
        print(f'[resume] {start} / {n} events already generated')

    # per-batch reseeding (seed*1000003 + batch offset) makes batches
    # independent and resume exact: restart from the last full batch.
    i = start - (start % args.batch)
    while i < n:
        b = min(args.batch, n - i)
        sampler.reseed(args.seed * 1000003 + i)
        x   = sampler.sample(b, shape=shape)
        gev = lognorm.denormalize(x).squeeze(1).float().cpu().numpy()

        events[i:i + b] = gev
        events.flush()
        i += b
        with open(progress, 'wt') as f:
            f.write(f'{i} / {n}\n')
        print(f'[{args.tag}] {i} / {n}   '
              f'<SumET> batch = {gev.sum(axis=(1, 2)).mean():.1f} GeV',
              flush=True)

    meta = {
        'model_dir' : os.path.abspath(args.model_dir),
        'tag'       : args.tag,
        'n_events'  : n,
        'dp'        : args.dp,
        'eta'       : args.eta,
        'T'         : config['model_args']['vsched']['T'],
        'S'         : args.steps,
        'seed'      : args.seed,
        'bf16'      : args.bf16,
        'shape'     : list(shape),
        'vsched'    : config['model_args']['vsched'],
        'data_norm' : config['model_args']['data_norm'],
        'mean_sum_et_gev' : float(np.asarray(events).sum(axis=(1, 2)).mean()),
        'created'   : datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(out_json, 'wt') as f:
        json.dump(meta, f, indent=2)

    print(f'done: {out_npy}\n      <SumET> = {meta["mean_sum_et_gev"]:.2f} GeV')


if __name__ == '__main__':
    main()
