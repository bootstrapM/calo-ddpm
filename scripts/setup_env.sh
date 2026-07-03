#!/usr/bin/env bash
# Bootstrap a COMPLETELY FRESH environment — no reliance on any pre-existing
# env, package, or file from earlier iterations of this project:
#   * new conda env "calo-ddpm" (or a local .venv if conda is unavailable)
#   * torch + scientific stack from PyPI
#   * improved-diffusion freshly cloned from OpenAI at the exact commit used
#     by LS4GAN/calo-ddpm (783b6740edb79fdb7d063250db2c51cc9545dcd1)
#   * consistency tests run at the end as an install check
#
# Usage:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="calo-ddpm"
PY_VER="3.11"

if command -v conda >/dev/null 2>&1; then
    echo "[setup] creating fresh conda env '${ENV_NAME}' (python ${PY_VER})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
    PYTHON="conda run -n ${ENV_NAME} python"
    ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
    echo "[setup] conda not found — creating local .venv"
    python3 -m venv .venv
    PYTHON=".venv/bin/python"
    ACTIVATE_HINT="source .venv/bin/activate"
fi

echo "[setup] installing requirements (torch, scipy, matplotlib, tqdm,"
echo "        improved-diffusion @ openai commit 783b674...)"
${PYTHON} -m pip install --upgrade pip
${PYTHON} -m pip install -r requirements.txt

echo "[setup] verifying install"
${PYTHON} -c "
import torch, improved_diffusion.unet, numpy, scipy, matplotlib
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
"

echo "[setup] running consistency tests (CPU, ~30 s)"
${PYTHON} tests/test_consistency.py

echo
echo "[setup] done.  Activate with:  ${ACTIVATE_HINT}"
