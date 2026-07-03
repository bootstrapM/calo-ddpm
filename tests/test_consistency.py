#!/usr/bin/env python
"""Consistency checks for schedule + samplers + all five inpainters (CPU).

Runs without pre-trained weights:

  1. Schedule identities (padding, vbar = 1 - sbar^2, step composition,
     collapse of the final ancestral step, subsampling consistency).
  2. Analytic-Gaussian posterior test: with an i.i.d. Gaussian prior
     x0 ~ N(mu0, sigma0^2) the exact eps-predictor is known in closed form;
     for noise-free masking the true posterior on dead pixels is again
     N(mu0, sigma0^2).  Every inpainter must reproduce it (mean/std), and
     the exact-consistency inpainters must return the known region == y.
  3. Unconditional DDPM/DDIM samplers reproduce the prior marginal, and
     DDIM(eta=0) is deterministic.

Usage: python tests/test_consistency.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calo_inpaint.schedule import make_subsampled_schedule
from calo_inpaint.ddpm_sampler import DDPMSampler
from calo_inpaint.ddim_sampler import DDIMSampler
from calo_inpaint.masks import square_mask
from calo_inpaint.inpainting import INPAINTERS

DEVICE = 'cpu'
T, S   = 300, 300
BETA1, BETAT = 0.02 / T, 1000 * 0.02 / T   # cent0-like, scaled to T
MU0, SIGMA0  = -3.0, 1.5
H = W = 12
PASS, FAIL = '[pass]', '[FAIL]'
n_fail = 0


def check(name, ok, detail=''):
    global n_fail
    print(f'{PASS if ok else FAIL} {name}  {detail}')
    if not ok:
        n_fail += 1


class AnalyticEpsNet(torch.nn.Module):
    """Exact eps-predictor for x0 ~ N(mu0, sigma0^2) i.i.d. pixels.

    q_t(x) = N(sbar_t mu0, sbar_t^2 sigma0^2 + vbar_t)
    eps*(x, t) = sqrt(vbar_t) (x - sbar_t mu0) / (sbar_t^2 sigma0^2 + vbar_t)

    Indexed by ORIGINAL timesteps t (validates the t_map plumbing).
    """

    def __init__(self, T, beta1, betaT, mu0, sigma0):
        super().__init__()
        beta = torch.linspace(beta1, betaT, T, dtype=torch.float64)
        beta = torch.cat([torch.zeros(1, dtype=torch.float64), beta])
        sbar = torch.exp(0.5 * torch.cumsum(torch.log1p(-beta), 0))
        vbar = 1 - sbar ** 2
        self.register_buffer('sbar', sbar.float())
        self.register_buffer('vbar', vbar.float())
        self.mu0, self.sigma0 = mu0, sigma0

    def forward(self, x, t, y=None):
        sb = self.sbar[t].view(-1, 1, 1, 1)
        vb = self.vbar[t].view(-1, 1, 1, 1)
        return vb.sqrt() * (x - sb * self.mu0) / (sb ** 2 * self.sigma0 ** 2 + vb)


def test_schedule():
    sc = make_subsampled_schedule(T, S, BETA1, BETAT, DEVICE)
    check('sched: padding', sc.sbar[0] == 1 and sc.vbar[0] == 0
          and sc.t_map[0] == 0)
    check('sched: vbar = 1 - sbar^2',
          torch.allclose(sc.vbar, 1 - sc.sbar ** 2, atol=1e-6))
    comp = sc.step_var + sc.step_scale ** 2 \
        * torch.cat([sc.vbar.new_zeros(1), sc.vbar[:-1]])
    check('sched: step composition', torch.allclose(comp[1:], sc.vbar[1:],
                                                    atol=1e-6))
    check('sched: final step collapses',
          abs(sc.coef_x0[1] - 1) < 1e-6 and abs(sc.coef_xt[1]) < 1e-6
          and sc.post_var[1] < 1e-10)

    sub = make_subsampled_schedule(T, 50, BETA1, BETAT, DEVICE)
    idx = sub.t_map
    check('sched: subsample matches full cumulative',
          torch.allclose(sub.sbar, sc.sbar[idx], atol=1e-6)
          and torch.allclose(sub.vbar, sc.vbar[idx], atol=1e-6))
    check('sched: subsample t_map endpoints',
          idx[1] == 1 and idx[-1] == T)


def test_unconditional():
    net = AnalyticEpsNet(T, BETA1, BETAT, MU0, SIGMA0).to(DEVICE)
    sc  = make_subsampled_schedule(T, S, BETA1, BETAT, DEVICE)

    x = DDPMSampler(net, sc, DEVICE, seed=1).sample(4000, shape=(1, 4, 4))
    m, s = x.mean().item(), x.std().item()
    check('ddpm: prior mean', abs(m - MU0) < 0.05 * SIGMA0, f'mean={m:.3f}')
    check('ddpm: prior std', abs(s / SIGMA0 - 1) < 0.05, f'std={s:.3f}')

    x1 = DDIMSampler(net, sc, DEVICE, seed=2, eta=0.0).sample(64, (1, 4, 4))
    x2 = DDIMSampler(net, sc, DEVICE, seed=2, eta=0.0).sample(64, (1, 4, 4))
    check('ddim(0): deterministic given seed', torch.equal(x1, x2))
    m, s = x1.mean().item(), x1.std().item()
    check('ddim(0): prior stats', abs(m - MU0) < 0.2 and abs(s / SIGMA0 - 1) < 0.1,
          f'mean={m:.3f} std={s:.3f}')


def test_inpainters():
    net = AnalyticEpsNet(T, BETA1, BETAT, MU0, SIGMA0).to(DEVICE)
    sc  = make_subsampled_schedule(T, S, BETA1, BETAT, DEVICE)

    g = torch.Generator().manual_seed(123)
    x_true = MU0 + SIGMA0 * torch.randn(1, H, W, generator=g)
    box    = 3
    mask   = square_mask(box, 4, 5, height=H, width=W, device=DEVICE)
    y      = x_true * mask
    n_smp  = 1500

    # all five now return the known region exactly (pigdm via final-step
    # post-processing projection; the others by construction)
    exact_known = {'repaint', 'ddnm', 'ddrm', 'mcg', 'pigdm'}
    # posterior on dead pixels is exactly N(mu0, sigma0^2) (iid prior).
    # DDRM at its default eta = 0.85 is intrinsically UNDER-dispersed here
    # (verified: std/sigma0 -> 1.00, 0.995, 0.96, 0.86, 0.70 for
    # eta = 0, 0.2, 0.5, 0.85, 1.0, independent of S) — a property of its
    # variational posterior, faithfully reproduced, not a bug.  The SBC
    # study is designed to quantify exactly this kind of miscalibration.
    tol_mean = {'repaint': 0.06, 'ddnm': 0.06, 'ddrm': 0.06,
                'mcg': 0.15, 'pigdm': 0.12}
    std_band = {'repaint': (0.94, 1.06), 'ddnm': (0.94, 1.06),
                'ddrm': (0.80, 1.02),    'mcg': (0.85, 1.15),
                'pigdm': (0.88, 1.12)}

    for name, cls in sorted(INPAINTERS.items()):
        kwargs = {'n_resample': 3} if name == 'repaint' else {}
        inp = cls(net, sc, DEVICE, seed=7, **kwargs)
        out = inp.inpaint(y, mask, n_smp)               # (n, 1, H, W)

        dead = out[:, 0][:, 4:4 + box, 5:5 + box]
        m  = dead.mean().item()
        s  = dead.std().item()
        lo, hi = std_band[name]
        ok = (abs(m - MU0) < tol_mean[name] * SIGMA0
              and lo < s / SIGMA0 < hi)
        check(f'{name}: dead-region posterior N(mu0, sigma0^2)', ok,
              f'mean={m:.3f} (exp {MU0}), std={s:.3f} (exp {SIGMA0})')

        kn_dev = ((out[:, 0] - x_true) * mask[0]).abs().max().item()
        if name in exact_known:
            check(f'{name}: known region exact', kn_dev < 1e-4,
                  f'max|dev|={kn_dev:.2e}')
        else:
            check(f'{name}: known region approx (no hard projection)',
                  kn_dev < 0.5 * SIGMA0, f'max|dev|={kn_dev:.2e}')

        i1 = cls(net, sc, DEVICE, seed=11, **kwargs).inpaint(y, mask, 8)
        i2 = cls(net, sc, DEVICE, seed=11, **kwargs).inpaint(y, mask, 8)
        check(f'{name}: reproducible given seed', torch.equal(i1, i2))

    # DDRM approaches exactness as eta -> 0 (regression guard on the
    # noise-free specialization):
    out = INPAINTERS['ddrm'](net, sc, DEVICE, seed=7, eta=0.2) \
        .inpaint(y, mask, n_smp)
    s = out[:, 0][:, 4:4 + box, 5:5 + box].std().item()
    check('ddrm(eta=0.2): near-exact posterior std',
          abs(s / SIGMA0 - 1) < 0.05, f'std={s:.3f}')

    # DDRM with eta_b < 1: the final projection must still return the
    # known region exactly (noise-free measurement convention):
    out = INPAINTERS['ddrm'](net, sc, DEVICE, seed=7, eta_b=0.5) \
        .inpaint(y, mask, 64)
    kn_dev = ((out[:, 0] - x_true) * mask[0]).abs().max().item()
    check('ddrm(eta_b=0.5): known region exact via final projection',
          kn_dev < 1e-6, f'max|dev|={kn_dev:.2e}')

    # hyperparameter guards
    try:
        INPAINTERS['ddrm'](net, sc, DEVICE, eta=1.5)
        check('ddrm: rejects eta > 1', False)
    except AssertionError:
        check('ddrm: rejects eta > 1', True)

    # RePaint NFE count: U evaluations per level for s > 1, exactly ONE at
    # s = 1 (Alg. 1 does no jump-back at t = 1 and its extra passes are
    # deterministic recomputations; the old buggy loop re-fed a level-0
    # object to the net conditioned at t = 1).
    class CountingNet(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.calls = inner, 0

        def forward(self, x, t, y=None):
            self.calls += 1
            return self.inner(x, t)

    cnet = CountingNet(net)
    U = 3
    INPAINTERS['repaint'](cnet, sc, DEVICE, seed=7, n_resample=U) \
        .inpaint(y, mask, 4)
    expected = U * (sc.S - 1) + 1
    check('repaint: NFE count U*(S-1)+1', cnet.calls == expected,
          f'calls={cnet.calls}, expected={expected}')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_schedule()
    test_unconditional()
    test_inpainters()
    print(f'\n{"ALL TESTS PASSED" if n_fail == 0 else f"{n_fail} FAILURES"}')
    sys.exit(1 if n_fail else 0)
