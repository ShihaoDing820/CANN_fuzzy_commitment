#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_full=false
if [[ "${1:-}" == "--full" ]]; then
    run_full=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--full]" >&2
    exit 2
fi

if [[ -f /home/dsh/cann-dev/activate.sh ]]; then
    source /home/dsh/cann-dev/activate.sh
elif [[ -f "${HOME}/Ascend/cann/set_env.sh" ]]; then
    source "${HOME}/Ascend/cann/set_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "CANN environment is not active." >&2
    exit 1
fi

echo "[1/3] Build and package standard custom OPP"
(cd "${project_dir}/custom_opp" && ./build.sh)

echo "[2/3] Run exact standard-kernel CPU Debug test"
"${project_dir}/tests/run-standard-kernel-cpu-test.sh"

echo "[3/3] Run 10-pair real LFW regression"
"${project_dir}/prototype/run-tests.sh"

if [[ "${run_full}" == true ]]; then
    echo "[FULL] Run all 6000 LFW pairs"
    "${project_dir}/prototype/run-all.sh"
fi

echo "[PASS] All requested fuzzy commitment checks completed"
