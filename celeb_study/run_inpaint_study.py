#!/usr/bin/env python
"""CelebA inpainting study — exact analogue of scripts/run_inpaint_study.py.

Uses THE SAME algorithm implementations (calo_inpaint.inpainting +
calo_inpaint.guided, imported, not copied), the same file formats
(results.npy / truth.npy / metadata.json / progress.txt), and the same
statistics script downstream (which auto-detects space='unit' from the
metadata).  Differences: HF model adapter, data range [-1, 1] (guided
x0 clamp defaults to (-1, 1)), pixel-coordinate square masks, and
--samples-per-batch chunking (256^2 images: J=50 samples rarely fit in
one gradient batch).

Example:
    python celeb_study/run_inpaint_study.py \
        --model-id google/ddpm-ema-celebahq-256 \
        --images workdir_celeb/generated_images/images_celebahq.npy \
        --outdir workdir_celeb/inpaint_study \
        --algorithm repaint --box 64 --n-images 100 --n-samples 50 \
        -S 1000 --samples-per-batch 10
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calo_inpaint.masks import square_mask
from calo_inpaint.inpainting import INPAINTERS
from calo_inpaint.guided import GUIDED_INPAINTERS
from celeb_study.model import load_celeb_model, celeb_schedule

ALL_INPAINTERS = {**INPAINTERS, **GUIDED_INPAINTERS}


def parse_cmdargs():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', default='google/ddpm-ema-celebahq-256')
    p.add_argument('--images',   required=True,
                   help='truth images .npy (N, C, H, W) in [-1, 1]')
    p.add_argument('--outdir',   required=True)
    p.add_argument('--algorithm', required=True,
                   choices=sorted(ALL_INPAINTERS))
    p.add_argument('--n-images',  type=int, default=100)
    p.add_argument('--n-samples', type=int, default=50)
    p.add_argument('--samples-per-batch', type=int, default=10,
                   help='posterior samples per reverse chain (memory knob '
                        'at 256^2; chunks are independently seeded)')
    p.add_argument('--image-offset', type=int, default=0)
    p.add_argument('--box', type=int, default=64)
    p.add_argument('--y0',  type=int, default=96,
                   help='dead-box top row (pixel coords)')
    p.add_argument('--x0',  type=int, default=96,
                   help='dead-box left column (pixel coords)')
    p.add_argument('-S', '--steps', type=int, default=1000)
    p.add_argument('--seed',  type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--bf16', action='store_true',
                   help='bf16 autocast for the no-grad algorithms')
    # algorithm hyperparameters (defaults mirror the calo study except the
    # data-range clamp, which is [-1, 1] here)
    p.add_argument('--repaint-resample', type=int,   default=10)
    p.add_argument('--ddrm-eta',         type=float, default=0.85)
    p.add_argument('--ddrm-eta-b',       type=float, default=1.0)
    p.add_argument('--mcg-alpha',        type=float, default=1.0)
    p.add_argument('--pigdm-eta',        type=float, default=1.0)
    p.add_argument('--x0-clamp', type=float, nargs=2, default=[-1.0, 1.0],
                   metavar=('LO', 'HI'),
                   help='data-range clamp for pigdm / mcg2 / pigdm2')
    p.add_argument('--mcg2-scale',   type=float, default=0.5)
    p.add_argument('--pigdm2-scale', type=float, default=0.5)
    p.add_argument('--pigdm2-rmax',  type=float, default=10.0)
    return p.parse_args()


def build_inpainter(args, net, sched, device):
    kwargs = {'seed': args.seed, 'use_bf16': args.bf16}
    a = args.algorithm
    if a == 'repaint':
        kwargs['n_resample'] = args.repaint_resample
    elif a == 'ddrm':
        kwargs.update(eta=args.ddrm_eta, eta_b=args.ddrm_eta_b)
    elif a == 'mcg':
        kwargs['alpha'] = args.mcg_alpha
    elif a == 'pigdm':
        kwargs.update(eta=args.pigdm_eta, x0_clamp=tuple(args.x0_clamp))
    elif a == 'mcg2':
        kwargs = {'seed': args.seed, 'guidance_scale': args.mcg2_scale,
                  'x0_clamp': tuple(args.x0_clamp)}
    elif a == 'pigdm2':
        kwargs = {'seed': args.seed, 'guidance_scale': args.pigdm2_scale,
                  'r_max': args.pigdm2_rmax, 'x0_clamp': tuple(args.x0_clamp)}
    return ALL_INPAINTERS[a](net, sched, device, **kwargs)


def main():
    args   = parse_cmdargs()
    device = torch.device(args.device)

    run_name = (f'{args.algorithm}_box{args.box}'
                f'_y{args.y0}_x{args.x0}_S{args.steps}')
    rundir = os.path.join(args.outdir, run_name)
    os.makedirs(rundir, exist_ok=True)

    net, info = load_celeb_model(args.model_id, device)
    sched = celeb_schedule(info, S=args.steps, device=device)
    C, H, W = info['shape']

    inp = build_inpainter(args, net, sched, device)

    images = np.load(args.images, mmap_mode='r')
    n_img, b, J = args.n_images, args.box, args.n_samples
    assert args.image_offset + n_img <= images.shape[0]
    assert images.shape[1:] == (C, H, W), \
        f'truth images {images.shape[1:]} vs model {info["shape"]}'

    mask = square_mask(b, args.y0, args.x0, height=H, width=W,
                       device=device)
    sl_r = slice(args.y0, args.y0 + b)
    sl_c = slice(args.x0, args.x0 + b)

    res_path = os.path.join(rundir, 'results.npy')
    progress = os.path.join(rundir, 'progress.txt')

    if os.path.exists(res_path):
        results = np.lib.format.open_memmap(res_path, mode='r+')
        assert results.shape == (n_img, J, C, b, b)
    else:
        results = np.lib.format.open_memmap(
            res_path, mode='w+', dtype=np.float32,
            shape=(n_img, J, C, b, b))

    truth = np.asarray(
        images[args.image_offset:args.image_offset + n_img, :, sl_r, sl_c],
        dtype=np.float32)
    np.save(os.path.join(rundir, 'truth.npy'), truth)

    start = 0
    if os.path.exists(progress):
        start = int(open(progress).read().split('/')[0])
        print(f'[resume] {start} / {n_img} images done')

    meta = {
        'study': 'celeb', 'space': 'unit',
        'algorithm': args.algorithm, 'model_id': info['model_id'],
        'images': os.path.abspath(args.images),
        'image_offset': args.image_offset,
        'n_images': n_img, 'n_samples': J,
        'samples_per_batch': args.samples_per_batch,
        'box': b, 'y0': args.y0, 'x0': args.x0,
        'T': info['T'], 'S': args.steps, 'seed': args.seed,
        'bf16': args.bf16, 'noise_free': True,
        'hyperparams': {
            'repaint_resample': args.repaint_resample,
            'ddrm_eta': args.ddrm_eta, 'ddrm_eta_b': args.ddrm_eta_b,
            'mcg_alpha': args.mcg_alpha, 'pigdm_eta': args.pigdm_eta,
            'x0_clamp': list(args.x0_clamp),
            'mcg2_scale': args.mcg2_scale,
            'pigdm2_scale': args.pigdm2_scale,
            'pigdm2_rmax': args.pigdm2_rmax,
        },
        'units': 'model space [-1, 1] (dead-region pixels only)',
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    json.dump(meta, open(os.path.join(rundir, 'metadata.json'), 'wt'),
              indent=2)

    Jb = max(1, args.samples_per_batch)
    t_start = datetime.datetime.now()
    for i in range(start, n_img):
        y = torch.from_numpy(
            np.array(images[args.image_offset + i], dtype=np.float32)
        ).to(device)                                    # (C, H, W)

        chunks = []
        for c0 in range(0, J, Jb):
            jb = min(Jb, J - c0)
            # per-(image, chunk) seeding: reproducible for fixed Jb
            inp.reseed(args.seed * 1000003 + i * 7919 + c0)
            s = inp.inpaint(y, mask, jb)               # (jb, C, H, W)
            chunks.append(s[:, :, sl_r, sl_c].float().cpu().numpy())
        results[i] = np.concatenate(chunks, axis=0)
        results.flush()

        with open(progress, 'wt') as f:
            f.write(f'{i + 1} / {n_img}  alg={args.algorithm} box={b}\n')
        dt = (datetime.datetime.now() - t_start).total_seconds()
        rate = (i + 1 - start) / max(dt, 1e-9)
        print(f'[{run_name}] {i + 1}/{n_img} ({rate:.3f} img/s, '
              f'eta {(n_img - i - 1)/max(rate,1e-9)/3600:.2f} h)',
              flush=True)

    print(f'done: {res_path}')


if __name__ == '__main__':
    main()
