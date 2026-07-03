#!/usr/bin/env bash
# Fresh download of the pre-trained DDPM weights from Zenodo record 12535659
# (both centralities, seeds 0..4), with md5 verification against the
# checksums published on the Zenodo record, unpacked into
#   $1/cent{0,4}_ddpm_seed{0..4}/{config.json,net_avg_gen.pth,...}
set -euo pipefail

WEIGHTS_DIR="${1:?usage: download_weights.sh WEIGHTS_DIR}"
ZENODO="https://zenodo.org/records/12535659/files"

declare -A MD5=(
    [cent0_ddpm_seed0]=4dcbc7f0b3c857735ddef1acfc478fd1
    [cent0_ddpm_seed1]=2dd3782202316d4c6c7f55627ae12c60
    [cent0_ddpm_seed2]=4afad38ae83a743f414ede83468200e1
    [cent0_ddpm_seed3]=cc16985d64a62a511459802bfd48c41d
    [cent0_ddpm_seed4]=1a9694ee5f76fd7c68f9d3a324e46dee
    [cent4_ddpm_seed0]=a914d675d31762e10886b86637fe6732
    [cent4_ddpm_seed1]=29268db3bc6966cc038d3fcce5c6bfc0
    [cent4_ddpm_seed2]=6dfe5caac1b51447bc645925b7dc93c0
    [cent4_ddpm_seed3]=f18249ae92c0e91751d5bcf14b0d6ba1
    [cent4_ddpm_seed4]=5dbc20e64655dc91d8643e85e3315b91
)

mkdir -p "${WEIGHTS_DIR}"

for cent in 0 4; do
    for seed in 0 1 2 3 4; do
        name="cent${cent}_ddpm_seed${seed}"
        dest="${WEIGHTS_DIR}/${name}"

        if [[ -f "${dest}/config.json" && -f "${dest}/net_avg_gen.pth" ]]; then
            echo "[skip] ${name} already present"
            continue
        fi

        echo "[download] ${name}"
        curl -L --fail -o "${WEIGHTS_DIR}/${name}.zip" \
            "${ZENODO}/${name}.zip?download=1"

        echo "${MD5[$name]}  ${WEIGHTS_DIR}/${name}.zip" | md5sum -c - \
            || { echo "md5 MISMATCH for ${name}.zip — corrupted download"; exit 1; }

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

echo "all weights verified & unpacked in ${WEIGHTS_DIR}"
