#!/usr/bin/env bash
# ============================================================================
# calo-ddpm inpainting study — master pipeline
#
# Stages:
#   weights   : download pre-trained models (Zenodo 12535659)
#   generate  : event generation, cent0 + cent4, model seeds 0..4
#   verify    : reproduce paper Fig. 4 / Fig. 5 observables per centrality
#   inpaint   : noise-free inpainting study, 5 algorithms x box sizes
#   stats     : SBC / TARP / pull / coverage analysis + comparison
#
# Usage:
#   ./run_all.sh                 # everything, fully self-contained in ./workdir
#   ./run_all.sh generate verify # selected stages
#   ROOT=/data/mystudy N_EVENTS=100000 ./run_all.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- configuration (override via environment) ------------------------------
ROOT="${ROOT:-./workdir}"                       # all outputs live under here
WEIGHTS_DIR="${WEIGHTS_DIR:-${ROOT}/pre-trained-model-weights}"
EVENTS_DIR="${EVENTS_DIR:-${ROOT}/generated_events}"
VERIFY_DIR="${VERIFY_DIR:-${ROOT}/verification}"
STUDY_DIR="${STUDY_DIR:-${ROOT}/inpaint_study}"

CENTS="${CENTS:-0 4}"                           # centrality classes
SEEDS="${SEEDS:-0 1 2 3 4}"                     # model training seeds
N_EVENTS="${N_EVENTS:-10000}"                   # events per (cent, seed)
GEN_BATCH="${GEN_BATCH:-1000}"
GEN_STEPS="${GEN_STEPS:-8000}"                  # T = S = 8000 (paper setup)

ALGORITHMS="${ALGORITHMS:-repaint ddnm ddrm mcg pigdm}"
BOX_SIZES="${BOX_SIZES:-2 4 8}"                 # dead-region sizes
MASK_ETA0="${MASK_ETA0:-8}"                     # dead-region corner
MASK_PHI0="${MASK_PHI0:-28}"
INPAINT_STEPS="${INPAINT_STEPS:-1000}"          # subsampled S for inpainting
N_IMAGES="${N_IMAGES:-1000}"                    # truth images per study
N_SAMPLES="${N_SAMPLES:-50}"                    # posterior samples per image
STUDY_CENT="${STUDY_CENT:-0}"                   # centrality used in the study
STUDY_SEED="${STUDY_SEED:-0}"                   # model seed used in the study

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
EXTRA_FLAGS="${EXTRA_FLAGS:---bf16}"            # e.g. "--bf16 --compile"

STAGES="${*:-weights generate verify inpaint stats}"
echo "stages: ${STAGES}"
echo "root:   ${ROOT}"

# ---- stage: weights ---------------------------------------------------------
if [[ " ${STAGES} " == *" weights "* ]]; then
    bash scripts/download_weights.sh "${WEIGHTS_DIR}"
fi

# ---- stage: generate --------------------------------------------------------
if [[ " ${STAGES} " == *" generate "* ]]; then
    for cent in ${CENTS}; do
        for seed in ${SEEDS}; do
            ${PYTHON} scripts/generate_events.py \
                --model-dir "${WEIGHTS_DIR}/cent${cent}_ddpm_seed${seed}" \
                --outdir    "${EVENTS_DIR}" \
                --tag       "cent${cent}_seed${seed}" \
                -n "${N_EVENTS}" --batch "${GEN_BATCH}" \
                -S "${GEN_STEPS}" --dp ddpm --seed 0 \
                --device "${DEVICE}" ${EXTRA_FLAGS}
        done
    done
fi

# ---- stage: verify ----------------------------------------------------------
if [[ " ${STAGES} " == *" verify "* ]]; then
    declare -A TITLE=( [0]="Centrality 0-10%" [4]="Centrality 40-50%" )
    for cent in ${CENTS}; do
        evts=()
        labs=()
        for seed in ${SEEDS}; do
            evts+=("${EVENTS_DIR}/events_cent${cent}_seed${seed}.npy")
            labs+=("seed${seed}")
        done
        ${PYTHON} scripts/verify_paper_plots.py \
            --events "${evts[@]}" --labels "${labs[@]}" \
            --title "${TITLE[$cent]:-cent${cent}}" \
            --centrality "cent${cent}" \
            --outdir "${VERIFY_DIR}/cent${cent}"
    done
fi

# ---- stage: inpaint ---------------------------------------------------------
if [[ " ${STAGES} " == *" inpaint "* ]]; then
    for alg in ${ALGORITHMS}; do
        for box in ${BOX_SIZES}; do
            ${PYTHON} scripts/run_inpaint_study.py \
                --model-dir "${WEIGHTS_DIR}/cent${STUDY_CENT}_ddpm_seed${STUDY_SEED}" \
                --events    "${EVENTS_DIR}/events_cent${STUDY_CENT}_seed${STUDY_SEED}.npy" \
                --outdir    "${STUDY_DIR}" \
                --algorithm "${alg}" --box "${box}" \
                --eta0 "${MASK_ETA0}" --phi0 "${MASK_PHI0}" \
                --n-images "${N_IMAGES}" --n-samples "${N_SAMPLES}" \
                -S "${INPAINT_STEPS}" --seed 0 \
                --device "${DEVICE}" ${EXTRA_FLAGS}
        done
    done
fi

# ---- stage: stats -----------------------------------------------------------
if [[ " ${STAGES} " == *" stats "* ]]; then
    rundirs=()
    for alg in ${ALGORITHMS}; do
        for box in ${BOX_SIZES}; do
            rd="${STUDY_DIR}/${alg}_box${box}_eta${MASK_ETA0}_phi${MASK_PHI0}_S${INPAINT_STEPS}"
            [[ -f "${rd}/results.npy" ]] && rundirs+=("${rd}")
        done
    done
    if [[ ${#rundirs[@]} -gt 0 ]]; then
        ${PYTHON} scripts/run_statistical_analysis.py \
            "${rundirs[@]}" --compare-outdir "${STUDY_DIR}/comparison"
    else
        echo "no completed study runs found in ${STUDY_DIR}"
    fi
fi

echo "run_all.sh: done."
