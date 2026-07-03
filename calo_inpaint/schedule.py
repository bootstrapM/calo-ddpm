"""Diffusion variance schedule, matching jetgen (LS4GAN/calo-ddpm) exactly.

jetgen parametrizes each forward transition as

    q(x_i | x_{i-1}) = N(x_i ; scale_i * x_{i-1}, var_i),   i = 1..T
    scale_i = sqrt(1 - beta_i),   var_i = beta_i
    beta_i  = linspace(beta1, betaT, T)   with   beta1 = beta_param / T,
                                                 betaT = 1000 * beta_param / T

and stores arrays of length T+1 with an identity transition padded at
index 0 (scale=1, var=0).  Cumulative ("jump") transitions are

    q(x_i | x_0) = N(x_i ; sbar_i * x_0, vbar_i)
    sbar_i = prod_{j<=i} scale_j          (sbar_0 = 1)
    vbar_i = 1 - sbar_i^2                 (vbar_0 = 0)

Subsampling ("linear-ddpm"): the S+1 sub-times are
    tau = [0] + round(linspace(1, T, S))
and the subsampled process has step transitions
    q(x_{tau_s} | x_{tau_{s-1}}) = N(.; (sbar_ts/sbar_ts-1) x, vbar_ts - (sbar_ts/sbar_ts-1)^2 vbar_ts-1)

All schedule tensors are created directly on `device` (a previous bug had
them on CPU); computation is done in float64 and cast to float32.
"""

import torch

__all__ = ['Schedule', 'make_subsampled_schedule']


class Schedule:
    """Subsampled VP diffusion schedule.

    All arrays have length S+1 and are indexed by the subsampled step
    s in [0..S], where s=0 is the clean data ("identity" padding).

    Attributes
    ----------
    t_map      : (S+1,) long   original timesteps (0..T); feed t_map[s] to the net
    sbar, vbar : (S+1,)        cumulative signal scale / noise variance
    step_scale : (S+1,)        per-step transition scale  q(x_s | x_{s-1});  [0] = 1
    step_var   : (S+1,)        per-step transition var                        [0] = 0
    coef_x0    : (S+1,)        DDPM posterior q(x_{s-1} | x_s, x0) mean coef on x0
    coef_xt    : (S+1,)        ... mean coef on x_s
    post_var   : (S+1,)        ... variance (beta-tilde)

    Useful identities (hold exactly by construction):
      vbar[s] = 1 - sbar[s]^2
      step_var[s] + step_scale[s]^2 * vbar[s-1] = vbar[s]
      coef_x0[1] = 1, coef_xt[1] = 0, post_var[1] = 0 when t_map[0] = 0
        => the final ancestral step returns x0hat exactly.
    """

    def __init__(self, t_map, sbar, vbar, step_scale, step_var, T, device):
        self.T      = T
        self.S      = len(t_map) - 1
        self.device = device

        self.t_map      = t_map.to(device=device, dtype=torch.long)
        self.sbar       = sbar.to(device=device, dtype=torch.float32)
        self.vbar       = vbar.to(device=device, dtype=torch.float32)
        self.step_scale = step_scale.to(device=device, dtype=torch.float32)
        self.step_var   = step_var.to(device=device, dtype=torch.float32)

        # DDPM posterior q(x_{s-1} | x_s, x0) coefficients, s = 1..S:
        #   mean = coef_x0[s] * x0 + coef_xt[s] * x_s ,  var = post_var[s]
        # from Bayes inversion of the (subsampled) step transition:
        #   coef_xt = step_scale * vbar_{s-1} / vbar_s
        #   coef_x0 = sbar_{s-1} * step_var / vbar_s
        #   post_var = step_var * vbar_{s-1} / vbar_s
        vb_prev = torch.cat([self.vbar.new_zeros(1), self.vbar[:-1]])
        sb_prev = torch.cat([self.sbar.new_ones(1),  self.sbar[:-1]])
        vb      = self.vbar.clone()
        vb[0]   = 1.0  # avoid 0/0 at unused index 0

        self.coef_xt  = self.step_scale * vb_prev / vb
        self.coef_x0  = sb_prev * self.step_var / vb
        self.post_var = self.step_var * vb_prev / vb

        self.coef_xt[0]  = 0.0
        self.coef_x0[0]  = 1.0
        self.post_var[0] = 0.0

    # -- shape helper ------------------------------------------------------
    @staticmethod
    def _bc(a):
        """Broadcast a scalar schedule entry over an image batch."""
        return a  # entries indexed with python ints are 0-dim tensors

    # -- forward -----------------------------------------------------------
    def q_sample(self, s, x0, noise=None, generator=None):
        """Sample q(x_s | x_0) = N(sbar_s x0, vbar_s).  s: python int."""
        if s == 0:
            return x0.clone() if noise is None else x0.clone()
        if noise is None:
            noise = _randn(x0, generator)
        return self.sbar[s] * x0 + self.vbar[s].sqrt() * noise

    def forward_step(self, s, x_prev, noise=None, generator=None):
        """Sample q(x_s | x_{s-1}) — one subsampled step forward."""
        if noise is None:
            noise = _randn(x_prev, generator)
        return self.step_scale[s] * x_prev + self.step_var[s].sqrt() * noise

    # -- eps <-> x0 --------------------------------------------------------
    def x0_from_eps(self, s, x, eps):
        return (x - self.vbar[s].sqrt() * eps) / self.sbar[s]

    def eps_from_x0(self, s, x, x0):
        return (x - self.sbar[s] * x0) / self.vbar[s].sqrt()

    # -- reverse -----------------------------------------------------------
    def ancestral_step(self, s, x, x0hat, generator=None, noise=None):
        """Sample the DDPM posterior q(x_{s-1} | x_s, x0hat).

        Noise is suppressed on the final step s == 1 (where post_var = 0 by
        the vbar_0 = 0 padding and the posterior collapses to x0hat exactly;
        equivalent to jetgen's `map_time(t) > 1` gate since t_map[1] = 1).
        """
        mean = self.coef_x0[s] * x0hat + self.coef_xt[s] * x
        if s <= 1:
            return mean
        if noise is None:
            noise = _randn(x, generator)
        return mean + self.post_var[s].sqrt() * noise

    def ddim_step(self, s, x, x0hat, eta=0.0, generator=None, noise=None):
        """DDIM update x_s -> x_{s-1} with stochasticity eta in [0, 1].

            sigma_s = eta * sqrt(post_var[s])
            x_{s-1} = sbar_{s-1} x0hat + sqrt(vbar_{s-1} - sigma_s^2) eps_hat
                      + sigma_s xi

        eta=0 reproduces jetgen's deterministic DDIM; eta=1 is equivalent in
        law to the DDPM ancestral step.
        """
        eps_hat = self.eps_from_x0(s, x, x0hat) if s > 0 else torch.zeros_like(x)
        sb_prev = self.sbar[s - 1]
        vb_prev = self.vbar[s - 1]

        sigma2 = (eta ** 2) * self.post_var[s]
        if s <= 1:                       # final step: deterministic, exact
            sigma2 = torch.zeros_like(sigma2)

        dir_coef = torch.clamp(vb_prev - sigma2, min=0.0).sqrt()
        out = sb_prev * x0hat + dir_coef * eps_hat

        if sigma2.item() > 0:
            if noise is None:
                noise = _randn(x, generator)
            out = out + sigma2.sqrt() * noise
        return out

    def marginal_std(self):
        """sqrt(vbar_S): std of the t=T marginal (matches jetgen init)."""
        return self.vbar[self.S].sqrt()


def _randn(like, generator=None):
    return torch.randn(
        like.shape, generator=generator, device=like.device, dtype=like.dtype
    )


def make_subsampled_schedule(T, S, beta1, betaT, device):
    """Build the (optionally subsampled) linear schedule on `device`.

    Parameters mirror the jetgen vsched config: {'name': 'linear', 'T', 'beta1',
    'betaT'}.  S = T gives the full schedule.  Subsampling follows jetgen's
    'linear-ddpm' rule: tau = [0] + round(linspace(1, T, S)).
    """
    assert 1 <= S <= T, f'S={S} must be in [1, {T}]'

    beta = torch.linspace(float(beta1), float(betaT), T, dtype=torch.float64)
    beta = torch.cat([torch.zeros(1, dtype=torch.float64), beta])   # pad idx 0

    log_alpha = torch.log1p(-beta)
    sbar = torch.exp(0.5 * torch.cumsum(log_alpha, dim=0))          # sbar_i
    vbar = 1.0 - sbar ** 2

    if S == T:
        tau = torch.arange(0, T + 1, dtype=torch.long)
    else:
        tau = torch.round(torch.linspace(1, T, S, dtype=torch.float64)).long()
        tau = torch.cat([torch.zeros(1, dtype=torch.long), tau])
        assert torch.all(tau[1:] > tau[:-1]), 'subsampled times must increase'

    sb = sbar[tau]
    vb = vbar[tau]

    step_scale = torch.ones_like(sb)
    step_var   = torch.zeros_like(vb)
    step_scale[1:] = sb[1:] / sb[:-1]
    step_var[1:]   = vb[1:] - (step_scale[1:] ** 2) * vb[:-1]

    return Schedule(
        t_map=tau, sbar=sb, vbar=vb,
        step_scale=step_scale, step_var=step_var,
        T=T, device=device,
    )
