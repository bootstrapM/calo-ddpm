#!/usr/bin/env bash
# Bootstrap a COMPLETELY FRESH environment — no reliance on any pre-existing
# env, package, or file from earlier iterations of this project:
#   * conda env "calo-ddpm" (or a local .venv if conda is unavailable)
#   * torch + scientific stack from PyPI
#   * improved-diffusion freshly cloned from OpenAI at the exact commit used
#     by LS4GAN/calo-ddpm (783b6740edb79fdb7d063250db2c51cc9545dcd1)
#
# NOTE on improved-diffusion: its setup.py at that commit is broken for
# modern pip (declares py_modules instead of the package, producing an empty
# wheel on `pip install git+...`).  We therefore clone it into
# external/improved-diffusion and register it with a .pth file — equivalent
# to the `setup.py develop` install the upstream README uses, but
# deterministic across pip versions.
#
# Usage:  bash scripts/setup_env.sh        (idempotent; safe to re-run)
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="calo-ddpm"
PY_VER="3.11"
IDIFF_COMMIT="783b6740edb79fdb7d063250db2c51cc9545dcd1"
IDIFF_DIR="external/improved-diffusion"

# ---- 1. python environment -------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        echo "[setup] conda env '${ENV_NAME}' already exists — reusing it"
        echo "        (for a truly fresh env: conda env remove -n ${ENV_NAME}, then re-run)"
    else
        echo "[setup] creating fresh conda env '${ENV_NAME}' (python ${PY_VER})"
        conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
    fi
    PYTHON="conda run -n ${ENV_NAME} python"
    ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
    echo "[setup] conda not found — using local .venv"
    [[ -d .venv ]] || python3 -m venv .venv
    PYTHON=".venv/bin/python"
    ACTIVATE_HINT="source .venv/bin/activate"
fi

# ---- 2. python packages ------------------------------------------------------
echo "[setup] installing requirements (torch, numpy, scipy, matplotlib, tqdm, blobfile)"
${PYTHON} -m pip install --upgrade pip
${PYTHON} -m pip install -r requirements.txt

# remove any broken metadata-only install from an earlier pip git+ attempt
${PYTHON} -m pip uninstall -y improved-diffusion >/dev/null 2>&1 || true

# ---- 3. improved-diffusion @ pinned OpenAI commit ----------------------------
if [[ -d "${IDIFF_DIR}/.git" ]]; then
    echo "[setup] ${IDIFF_DIR} exists — pinning to ${IDIFF_COMMIT:0:9}"
    git -C "${IDIFF_DIR}" fetch -q origin
else
    echo "[setup] cloning openai/improved-diffusion -> ${IDIFF_DIR}"
    mkdir -p external
    git clone -q https://github.com/openai/improved-diffusion "${IDIFF_DIR}"
fi
git -C "${IDIFF_DIR}" checkout -q "${IDIFF_COMMIT}"

# register via .pth (equivalent to `setup.py develop`, but version-proof)
SITE_DIR="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$(pwd)/${IDIFF_DIR}" > "${SITE_DIR}/improved_diffusion_repo.pth"
echo "[setup] registered ${IDIFF_DIR} in ${SITE_DIR}/improved_diffusion_repo.pth"

# ---- 4. verify ---------------------------------------------------------------
echo "[setup] verifying install"
${PYTHON} - <<'EOF'
import torch, numpy, scipy, matplotlib
import improved_diffusion.unet as u
import inspect, os, subprocess
repo = os.path.dirname(os.path.dirname(inspect.getfile(u)))
sha  = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'],
                      capture_output=True, text=True).stdout.strip()
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('improved_diffusion from', repo, '@', sha[:9])
assert sha.startswith('783b6740'), f'wrong improved-diffusion commit: {sha}'
EOF

echo "[setup] running consistency tests (CPU, ~30 s)"
${PYTHON} tests/test_consistency.py

echo
echo "[setup] done.  Activate with:  ${ACTIVATE_HINT}"
