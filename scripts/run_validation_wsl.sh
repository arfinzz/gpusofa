#!/usr/bin/env bash
# Run one validation scene (testscenes/validationtests/) on the CPU (SOFA's
# components), on the GPU (this plugin), or both, for a number of steps.
# Usage (from WSL):
#   bash scripts/run_validation_wsl.sh <scene.py> <steps> [cpu|gpu|both|compare]
#     compare = the GPU side with SOFA_VALIDATION_COMPARE=1 (stage by stage against SOFA's CPU components)
# The scene reads SOFA_VALIDATION_MATERIAL and its own switches from the environment;
# logs go to SOFA_BENCHMARK_LOG_DIR (default output/benchmark_logs/validation_<date>).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
SCENE="${REPO_DIR}/testscenes/validationtests/${1:?scene file}"
STEPS="${2:?steps}"
WHICH="${3:-both}"
OUT="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/validation_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_ROOT
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export PYTHONPATH="${SOFA_ROOT}/plugins/SofaPython3/lib/python3/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export SOFA_BENCHMARK_LOG_DIR="${OUT}"
mkdir -p "${OUT}"

run_side() {
    local side="$1"; shift
    local name
    # SOFA_VALIDATION_RUN_TAG (optional) keeps runs of one scene with other settings apart.
    name="$(basename "${SCENE}" .py)${SOFA_VALIDATION_RUN_TAG:+_${SOFA_VALIDATION_RUN_TAG}}_${SOFA_VALIDATION_MATERIAL:-default}_${side}"
    local extra=()
    [ "${side}" != cpu ] && extra=(-l SofaCUDA -l "${LIB}")
    env SOFA_VALIDATION_SIDE="${side%%_*}" "$@" "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
        -l SofaPython3 "${extra[@]}" "${SCENE}" > "${OUT}/${name}.log" 2>&1 < /dev/null
    local code=$?
    echo "${name}: exit=${code}"
    grep -E '\[ERROR\]|Traceback|Error:' "${OUT}/${name}.log" | head -5
}

case "${WHICH}" in
    cpu)     run_side cpu ;;
    gpu)     run_side gpu ;;
    compare) run_side gpu_compare SOFA_VALIDATION_COMPARE=1 ;;
    both)    run_side cpu; run_side gpu ;;
    *)       echo "usage: $0 <scene.py> <steps> [cpu|gpu|both|compare]"; exit 2 ;;
esac
echo "OUT=${OUT}"
echo VALIDATION_RUN_DONE
