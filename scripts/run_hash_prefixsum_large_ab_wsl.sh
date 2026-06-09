#!/usr/bin/env bash
# A/B run for the EXPERIMENTAL spatial-hash + prefix-sum broad cull
# (experiment/hash-prefixsum-broadphase branch).
#
# Runs the SAME large-tissue + large-tool scene twice:
#   A) baseline  : dense grid + tool-active-cell generation (current default)
#   B) experiment: spatial hash + prefix-sum work expansion
# Both feed the identical FBP narrow kernel, so the per-frame contact count
# MUST match between A and B. The wall-clock / FPS difference is the payoff.
#
# This script ONLY runs the experiment scene; it never touches the dense path's
# own benchmarks. Usage:
#   scripts/run_hash_prefixsum_large_ab_wsl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${REPO_DIR}/SofaGpuCollision/build-profile/libSofaGpuCollision.so}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/hash_prefixsum_large_${RUN_STAMP}}"
STEPS="${SOFA_BENCHMARK_STEPS:-120}"
SCENE="${REPO_DIR}/testscenes/hash_prefixsum_large.py"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${SOFA_PLUGIN_LIB_PATHS%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"

# Shared FBP knobs (identical for both legs so only the broad cull differs).
export SOFA_USE_TOOL_ACTIVE_CELL_GENERATION="${SOFA_USE_TOOL_ACTIVE_CELL_GENERATION:-1}"
# Read the contact counter every frame so the A/B correctness check is exact.
export SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
export SOFA_GPU_COLLISION_LIB

run_leg() {
    local label="$1"        # human label
    local use_hash="$2"     # 0 or 1
    local leg_dir="${BASE_LOG_DIR}/${label}"
    mkdir -p "${leg_dir}"
    echo "============================================================"
    echo "  LEG: ${label}  (useHashPrefixSumGeneration=${use_hash})"
    echo "  Logs -> ${leg_dir}"
    echo "============================================================"
    SOFA_USE_HASH_PREFIXSUM_GENERATION="${use_hash}" \
    SOFA_BENCHMARK_LABEL_SUFFIX="_${label}" \
    SOFA_BENCHMARK_LOG_DIR="${leg_dir}" \
    "${SOFA_ROOT}/bin/runSofa" \
        -g batch \
        -n "${STEPS}" \
        -l SofaPython3 \
        -l SofaCUDA \
        -l "${SOFA_GPU_COLLISION_LIB}" \
        "${SCENE}"
}

mkdir -p "${BASE_LOG_DIR}"
echo "Base log dir -> ${BASE_LOG_DIR}"

# Leg A: dense-grid baseline (the current default path; hash flag OFF).
run_leg "dense_baseline" 0
# Leg B: spatial-hash + prefix-sum experiment.
run_leg "hash_prefixsum" 1

echo
echo "============================================================"
echo "  DONE. Compare per-frame contact counts between:"
echo "    ${BASE_LOG_DIR}/dense_baseline"
echo "    ${BASE_LOG_DIR}/hash_prefixsum"
echo "  Contact counts MUST be identical; FPS/wall-clock is the payoff."
echo "============================================================"
