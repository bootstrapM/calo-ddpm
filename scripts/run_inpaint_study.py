#!/usr/bin/env python
"""Posterior inpainting study: N_samples posterior samples per image over a
set of truth images, for one algorithm and one dead-region geometry.

Truth images are generated events (same DDPM as the prior — the SBC-clean
setup): each image is masked with a square dead region, the algorithm
reconstructs it, and only the dead-region pixels are stored:

    results.npy : (n_images, n_samples, box, box)  float32, GeV
    truth.npy   : (n_images, box, box)             float32, GeV
    metadata.json, progress.txt (resume; poll with `watch -n 10 cat ...`)

Example:
    python scripts/run_inpaint_study.py \
        --model-dir $ROOT/pre-trained-model-weights/cent0_ddpm_seed0 \
        --events    $ROOT/generated_events/events_cent0_seed0.npy \
        --outdir    $ROOT/inpaint_study \
        --algorithm repaint --box 4 --n-images 1000 --n-samples 50 -S 1000
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
from calo_inpaint.masks import square_mask
from calo_inpaint.inpainting import INPAINTERS


def parse_cmdargs():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--events',    required=True,
                   help='truth events .npy (N, 24, 64), GeV')
    p.add_argument('--outdir',    required=True)
    p.add_argument('--algorithm', required=True, choices=sorted(INPAINTERS))
    p.add_argument('--n-images',  type=int, default=1000)
    p.add_argument('--n-samples', type=int, default=50)
    p.add_argument('--image-offset', type=int, default=0,
                   help='skip this many events before taking truth images')
    p.add_argument('--box',   type=int, default=4)
    p.add_argument('--eta0',  type=int, default=8)
    p.add_argument('--phi0',  type=int, default=28)
    p.add_argument('-S', '--steps', type=int, default=1000)
    p.add_argument('--seed',  type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--compile', action='store_true')
    # algorithm hyperparameters (only the relevant one is used)
    p.add_argument('--repaint-resample', type=int,   default=10)
    p.add_argument('--ddrm-eta',         type=float, default=0.85)
    p.add_argument('--ddrm-eta-b',       type=float, default=1.0)
    p.add_argument('--mcg-alpha',        type=float, default=1.0)
    p.add_argument('--pigdm-eta',        type=float, default=1.0)
    return p.parse_args()


def build_inpainter(args, net, sched, device):
    kwargs = {'seed': args.seed, 'use_bf16': args.bf16}
    if args.algorithm == 'repaint':
        kwargs['n_resample'] = args.repaint_resample
    elif args.algorithm == 'ddrm':
        kwargs.update(eta=args.ddrm_eta, eta_b=args.ddrm_eta_b)
    elif args.algorithm == 'mcg':
        kwargs['alpha'] = args.mcg_alpha
    elif args.algorithm == 'pigdm':
        kwargs['eta'] = args.pigdm_eta
    return INPAINTERS[args.algorithm](net, sched, device, **kwargs)


def main():
    args   = parse_cmdargs()
    device = torch.device(args.device)

    run_name = (f'{args.algorithm}_box{args.box}'
                f'_eta{args.eta0}_phi{args.phi0}_S{args.steps}')
    rundir = os.path.join(args.outdir, run_name)
    os.makedirs(rundir, exist_ok=True)

    net, config = load_model(args.model_dir, device)
    sched   = schedule_from_config(config, S=args.steps, device=device)
    lognorm = lognorm_from_config(config)

    if args.compile:
        net = torch.compile(net)

    inp = build_inpainter(args, net, sched, device)

    events = np.load(args.events, mmap_mode='r')
    n_img  = args.n_images
    assert args.image_offset + n_img <= events.shape[0], \
        'not enough truth events'
    b = args.box

    mask = square_mask(b, args.eta0, args.phi0, device=device)
    sl_e = slice(args.eta0, args.eta0 + b)
    sl_p = slice(args.phi0, args.phi0 + b)

    res_path   = os.path.join(rundir, 'results.npy')
    truth_path = os.path.join(rundir, 'truth.npy')
    progress   = os.path.join(rundir, 'progress.txt')

    if os.path.exists(res_path):
        results = np.lib.format.open_memmap(res_path, mode='r+')
        assert results.shape == (n_img, args.n_samples, b, b)
    else:
        results = np.lib.format.open_memmap(
            res_path, mode='w+', dtype=np.float32,
            shape=(n_img, args.n_samples, b, b)
        )

    truth = np.asarray(
        events[args.image_offset:args.image_offset + n_img, sl_e, sl_p],
        dtype=np.float32
    )
    np.save(truth_path, truth)

    start = 0
    if os.path.exists(progress):
        with open(progress) as f:
            start = int(f.read().split('/')[0])
        print(f'[resume] {start} / {n_img} images done')

    meta = {
        'algorithm'   : args.algorithm,
        'model_dir'   : os.path.abspath(args.model_dir),
        'events'      : os.path.abspath(args.events),
        'image_offset': args.image_offset,
        'n_images'    : n_img,
        'n_samples'   : args.n_samples,
        'box'         : b,
        'eta0'        : args.eta0,
        'phi0'        : args.phi0,
        'T'           : config['model_args']['vsched']['T'],
        'S'           : args.steps,
        'seed'        : args.seed,
        'bf16'        : args.bf16,
        'noise_free'  : True,
        'hyperparams' : {
            'repaint_resample': args.repaint_resample,
            'ddrm_eta'        : args.ddrm_eta,
            'ddrm_eta_b'      : args.ddrm_eta_b,
            'mcg_alpha'       : args.mcg_alpha,
            'pigdm_eta'       : args.pigdm_eta,
        },
        'units'   : 'GeV (dead-region pixels only)',
        'created' : datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(rundir, 'metadata.json'), 'wt') as f:
        json.dump(meta, f, indent=2)

    t_start = datetime.datetime.now()
    for i in range(start, n_img):
        ev_gev = torch.from_numpy(
            np.asarray(events[args.image_offset + i], dtype=np.float32)
        ).to(device).unsqueeze(0)                       # (1, 24, 64)
        y = lognorm.normalize(ev_gev)                   # log space

        # per-image reseeding: reproducible and resume-exact
        inp.reseed(args.seed * 1000003 + i)
        samples = inp.inpaint(y, mask, args.n_samples)  # (n, 1, 24, 64), log

        box_gev = lognorm.denormalize(
            samples[:, 0, sl_e, sl_p]
        ).float().cpu().numpy()

        results[i] = box_gev
        results.flush()

        with open(progress, 'wt') as f:
            f.write(f'{i + 1} / {n_img}  alg={args.algorithm} box={b}\n')

        if (i + 1) % 10 == 0 or i == start:
            dt  = (datetime.datetime.now() - t_start).total_seconds()
            rate = (i + 1 - start) / max(dt, 1e-9)
            eta_s = (n_img - i - 1) / max(rate, 1e-9)
            print(f'[{run_name}] {i + 1}/{n_img} '
                  f'({rate:.2f} img/s, eta {eta_s/3600:.2f} h)', flush=True)

    print(f'done: {res_path}')


if __name__ == '__main__':
    main()
