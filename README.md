# calo-ddpm inpainting study

Training-free, noise-free inpainting of dead sPHENIX calorimeter regions
using a pre-trained DDPM (arXiv:2406.01602, weights: Zenodo 12535659) as the
prior, plus Bayesian posterior validation (SBC / TARP / pulls / coverage).

## Setup (fully self-contained — no pre-existing env, code, or files needed)

```bash
bash scripts/setup_env.sh                # fresh conda env "calo-ddpm" (or .venv),
                                         # torch from PyPI, improved-diffusion
                                         # cloned fresh from OpenAI @ 783b674
                                         # into external/ (its setup.py is broken
                                         # for pip git-installs, so it is
                                         # registered via a .pth file instead),
                                         # then runs the consistency tests
conda activate calo-ddpm                 # (or: source .venv/bin/activate)
```

The only external inputs, both downloaded fresh by the pipeline itself:
improved-diffusion (OpenAI GitHub, pinned commit) and the pre-trained
weights (Zenodo 12535659, md5-verified by `scripts/download_weights.sh`).

## Reproduce everything

```bash
./run_all.sh                             # all stages, outputs in ./workdir
./run_all.sh weights generate            # or selected stages
```

Stages: `weights` (download models) → `generate` (events, cent0/cent4 ×
seeds 0–4, T=S=8000 DDPM) → `verify` (paper Fig. 4/5 observables) →
`inpaint` (RePaint / DDNM / DDRM / MCG / ΠGDM, box sizes 2/4/8 at
(η,φ)=(8,28), 50 posterior samples × 1000 images) → `stats`
(SBC ranks, TARP, pull z-scores, credible-interval coverage, comparison).

All knobs are environment variables (see header of `run_all.sh`); every
long stage checkpoints via `progress.txt` and resumes on rerun
(`watch -n 10 cat <dir>/progress.txt` to monitor).

## Layout

```
calo_inpaint/            library (schedule, samplers, data norm, masks, metrics)
calo_inpaint/inpainting/ base_inpainter, repaint, ddnm, ddrm, mcg, pigdm
scripts/                 generate_events, verify_paper_plots,
                         run_inpaint_study, run_statistical_analysis,
                         download_weights.sh
notebooks/               experimentation.ipynb — step-by-step interactive pipeline
tests/                   consistency checks (tiny CPU model, no weights needed)
run_all.sh               master pipeline
```

For interactive exploration: `jupyter lab notebooks/experimentation.ipynb`
(knobs like `ROOT`, `CENT`, `S_FAST`, `N_POST` are env-overridable; defaults
target quick GPU turnaround, use `S_FAST=8000` for paper-quality generation).

Conventions: images are (1, 24, 64) tower E_T maps in GeV; model space is
`ln(clip(E, 1e-3))`; schedules replicate jetgen's padded linear-beta
parametrization exactly (index 0 = identity); the network is conditioned on
original timesteps 1..8000 also when subsampled. All inpainters are
specialized to noiseless masking (σ_y = 0); the known region is noised to
level **s−1** (not s) in RePaint/MCG projections — this was the earlier bug.

## Tests

```bash
python tests/test_consistency.py
```
