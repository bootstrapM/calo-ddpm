#!/usr/bin/env python
"""Generate truth images from the pretrained CelebA DDPM prior.

EXACTLY the calorimeter generation machinery (calo_inpaint.DDPMSampler,
same schedule construction, same per-batch reseeding and resume) applied
to the HF model — truth from the same prior used for inpainting, i.e.
the SBC-clean setup, mirroring scripts/generate_events.py.

Output: images_<tag>.npy (N, C, H, W) float32 in [-1, 1] + JSON sidecar
(+ optional preview grid PNG).

Example:
    python celeb_study/generate_images.py \
        --model-id google/ddpm-ema-celebahq-256 \
        --outdir workdir_celeb/generated_images --tag celebahq \
        -n 200 --batch 8 -S 1000 --preview
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calo_inpaint.ddpm_sampler import DDPMSampler
from celeb_study.model import load_celeb_model, celeb_schedule


def parse_cmdargs():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', default='google/ddpm-ema-celebahq-256')
    p.add_argument('--outdir',   required=True)
    p.add_argument('--tag',      default='celebahq')
    p.add_argument('-n', '--n-images', type=int, default=200)
    p.add_argument('--batch',    type=int, default=8)
    p.add_argument('-S', '--steps', type=int, default=1000)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--device',   default='cuda')
    p.add_argument('--bf16',     action='store_true')
    p.add_argument('--compile',  action='store_true',
                   help='torch.compile the UNet (A100: ~20-40%% faster)')
    p.add_argument('--preview',  action='store_true',
                   help='save a 4x4 grid PNG of the first images')
    return p.parse_args()


def main():
    args   = parse_cmdargs()
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    net, info = load_celeb_model(args.model_id, device)
    if args.compile:
        net = torch.compile(net)
    sched = celeb_schedule(info, S=args.steps, device=device)
    C, H, W = info['shape']

    sampler = DDPMSampler(net, sched, device, seed=args.seed,
                          use_bf16=args.bf16)

    out_npy  = os.path.join(args.outdir, f'images_{args.tag}.npy')
    out_json = os.path.join(args.outdir, f'images_{args.tag}.json')
    progress = os.path.join(args.outdir, f'progress_{args.tag}.txt')

    n = args.n_images
    if os.path.exists(out_npy):
        images = np.lib.format.open_memmap(out_npy, mode='r+')
        assert images.shape == (n, C, H, W)
    else:
        images = np.lib.format.open_memmap(
            out_npy, mode='w+', dtype=np.float32, shape=(n, C, H, W))

    start = 0
    if os.path.exists(progress):
        start = int(open(progress).read().split('/')[0])
        print(f'[resume] {start} / {n} images done')

    n_batches = (n + args.batch - 1) // args.batch
    print(f'[{args.tag}] {n} images, batch {args.batch} ({n_batches} '
          f'batches), S={args.steps}, T={info["T"]}, shape={info["shape"]}',
          flush=True)

    i = start - (start % args.batch)
    while i < n:
        b = min(args.batch, n - i)
        sampler.reseed(args.seed * 1000003 + i)
        with torch.no_grad():
            x = sampler.sample(b, shape=(C, H, W))
        images[i:i + b] = x.clamp(-1, 1).float().cpu().numpy()
        images.flush()
        i += b
        with open(progress, 'wt') as f:
            f.write(f'{i} / {n}\n')
        print(f'[{args.tag}] {i} / {n}', flush=True)

    meta = {
        'model_id': info['model_id'], 'tag': args.tag, 'n_images': n,
        'T': info['T'], 'S': args.steps, 'beta1': info['beta1'],
        'betaT': info['betaT'], 'seed': args.seed, 'bf16': args.bf16,
        'shape': list(info['shape']), 'space': 'unit',
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    json.dump(meta, open(out_json, 'wt'), indent=2)

    if args.preview:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        k = min(16, n)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for a, img in zip(axes.ravel(), images[:k]):
            im = (np.transpose(img, (1, 2, 0)) + 1) / 2
            a.imshow(im.squeeze().clip(0, 1)); a.axis('off')
        fig.savefig(os.path.join(args.outdir, f'preview_{args.tag}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f'done: {out_npy}')


if __name__ == '__main__':
    main()
