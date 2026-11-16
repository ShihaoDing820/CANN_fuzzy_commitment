#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build"

if [[ -n "${CANN_ACTIVATE_SCRIPT:-}" ]]; then
    source "${CANN_ACTIVATE_SCRIPT}"
elif [[ -f /home/dsh/cann-dev/activate.sh ]]; then
    source /home/dsh/cann-dev/activate.sh
elif [[ -f "${HOME}/Ascend/cann/set_env.sh" ]]; then
    source "${HOME}/Ascend/cann/set_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "CANN environment is not active." >&2
    exit 1
fi

cmake --fresh -S "${project_dir}" -B "${build_dir}" \
    -DCMAKE_ASC_RUN_MODE=cpu \
    -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "${build_dir}" --parallel 2
"${build_dir}/fuzzy_commitment_standard_kernel_cpu_test"
