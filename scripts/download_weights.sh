#!/usr/bin/env bash
# Download pre-trained DDPM weights (Zenodo record 12535659) for both
# centralities, seeds 0..4, and unpack into
#   $1/cent{0,4}_ddpm_seed{0..4}/{config.json,net_avg_gen.pth,...}
set -euo pipefail

WEIGHTS_DIR="${1:?usage: download_weights.sh WEIGHTS_DIR}"
ZENODO="https://zenodo.org/records/12535659/files"

mkdir -p "${WEIGHTS_DIR}"

for cent in 0 4; do
    for seed in 0 1 2 3 4; do
        name="cent${cent}_ddpm_seed${seed}"
        dest="${WEIGHTS_DIR}/${name}"

        if [[ -f "${dest}/config.json" ]]; then
            echo "[skip] ${name} already present"
            continue
        fi

        echo "[download] ${name}"
        curl -L --fail -o "${WEIGHTS_DIR}/${name}.zip" \
            "${ZENODO}/${name}.zip?download=1"

        unzip -q -o "${WEIGHTS_DIR}/${name}.zip" -d "${WEIGHTS_DIR}/${name}_tmp"

        # flatten: config.json may sit at top level or inside a subdirectory
        cfg_path="$(find "${WEIGHTS_DIR}/${name}_tmp" -name config.json | head -1)"
        [[ -n "${cfg_path}" ]] || { echo "no config.json in ${name}.zip"; exit 1; }
        mkdir -p "${dest}"
        mv "$(dirname "${cfg_path}")"/* "${dest}/"

        rm -rf "${WEIGHTS_DIR}/${name}_tmp" "${WEIGHTS_DIR}/${name}.zip"
        echo "[ok] ${dest}"
    done
done

echo "all weights in ${WEIGHTS_DIR}"
