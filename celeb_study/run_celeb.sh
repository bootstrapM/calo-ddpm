#!/usr/bin/env bash
# ============================================================================
# CelebA-HQ inpainting study — parallel pipeline to run_all.sh
#   generate : sample truth images from the HF prior (SBC-clean setup)
#   inpaint  : all algorithms on the shared truth set
#   stats    : the SAME statistics script as the calo study (auto-detects
#              space='unit' from the run metadata)
#
# Usage:  ./celeb_study/run_celeb.sh [generate inpaint stats]
#         MODEL_ID=google/ddpm-ema-celebahq-256 N_IMAGES=100 ...
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-./workdir_celeb}"
MODEL_ID="${MODEL_ID:-google/ddpm-ema-celebahq-256}"
IMAGES_DIR="${IMAGES_DIR:-${ROOT}/generated_images}"
STUDY_DIR="${STUDY_DIR:-${ROOT}/inpaint_study}"

N_GEN="${N_GEN:-200}"                 # truth images to generate
GEN_BATCH="${GEN_BATCH:-8}"
N_IMAGES="${N_IMAGES:-100}"           # truth images per study run
N_SAMPLES="${N_SAMPLES:-50}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-10}"
BOX="${BOX:-64}"                      # dead box (pixels), corner (Y0, X0)
Y0="${Y0:-96}"
X0="${X0:-96}"
STEPS="${STEPS:-1000}"
ALGORITHMS="${ALGORITHMS:-repaint ddnm ddrm mcg2 pigdm2}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"

STAGES="${*:-generate inpaint stats}"
echo "stages: ${STAGES} | model: ${MODEL_ID} | root: ${ROOT}"

if [[ " ${STAGES} " == *" generate "* ]]; then
    ${PYTHON} celeb_study/generate_images.py \
        --model-id "${MODEL_ID}" --outdir "${IMAGES_DIR}" --tag celebahq \
        -n "${N_GEN}" --batch "${GEN_BATCH}" -S "${STEPS}" \
        --device "${DEVICE}" --bf16 --preview
fi

if [[ " ${STAGES} " == *" inpaint "* ]]; then
    for alg in ${ALGORITHMS}; do
        ${PYTHON} celeb_study/run_inpaint_study.py \
            --model-id "${MODEL_ID}" \
            --images "${IMAGES_DIR}/images_celebahq.npy" \
            --outdir "${STUDY_DIR}" \
            --algorithm "${alg}" --box "${BOX}" --y0 "${Y0}" --x0 "${X0}" \
            --n-images "${N_IMAGES}" --n-samples "${N_SAMPLES}" \
            --samples-per-batch "${SAMPLES_PER_BATCH}" \
            -S "${STEPS}" --device "${DEVICE}" --bf16
    done
fi

if [[ " ${STAGES} " == *" stats "* ]]; then
    rundirs=()
    for alg in ${ALGORITHMS}; do
        rd="${STUDY_DIR}/${alg}_box${BOX}_y${Y0}_x${X0}_S${STEPS}"
        [[ -f "${rd}/results.npy" ]] && rundirs+=("${rd}")
    done
    [[ ${#rundirs[@]} -gt 0 ]] && ${PYTHON} scripts/run_statistical_analysis.py \
        "${rundirs[@]}" --compare-outdir "${STUDY_DIR}/comparison"
fi

echo "run_celeb.sh: done."
