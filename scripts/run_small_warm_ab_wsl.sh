#!/usr/bin/env bash
# Warm re-run of the small one-tissue/one-blade scene in both fast-path and
# validation modes. Used to get a representative fast-path number after the GPU
# is already warm (the first leg of a batch is cold-clock-contaminated).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
LIB="${SOFA_GPU_COLLISION_LIB:-${REPO_DIR}/SofaGpuCollision/build-profile/libSofaGpuCollision.so}"
RUNDIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/small_warm_$(date +%Y%m%d_%H%M%S)}"
STEPS="${SOFA_BENCHMARK_STEPS:-160}"
SCENE="${REPO_DIR}/testscenes/one_tissue_one_blade.py"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${SOFA_PLUGIN_LIB_PATHS%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"

run_leg() {
    local leg="$1"; local counter="$2"
    local leg_dir="${RUNDIR}/${leg}"
    mkdir -p "${leg_dir}"
    echo "--- LEG ${leg} (counter=${counter}) ---"
    env SOFA_USE_FEATURE_BASED_PROXIMITY=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER="${counter}" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${leg}" SOFA_BENCHMARK_LOG_DIR="${leg_dir}" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}"
}

mkdir -p "${RUNDIR}"
echo "small warm A/B -> ${RUNDIR}"
# Warm-up leg discarded for thermal ramp, then the two measured legs.
run_leg warmup_discard 0
run_leg small_fast_warm 0
run_leg small_valid_warm 1
echo "DONE ${RUNDIR}"
