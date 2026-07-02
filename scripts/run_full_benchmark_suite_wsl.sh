#!/usr/bin/env bash
# Full GPU-collision benchmark suite — one pass over every current scene, in
# both the production fast path (detection-only, no counter readback) and the
# validation path (counter readback on, so contact counts are visible).
#
# Scenes:
#   small      one-tissue / one-blade        (tri-tri FBP)
#   large      181x181 tissue + subdiv blade (tri-tri FBP, 79,520 elements)
#   vt_self    2-layer slab self-collision   (vertex-triangle)
#   vt_cross   tissue grid + tool point cloud(vertex-triangle cross-model)
#   hash       large tissue + large tool     (dense vs hash+prefix-sum A/B)
#
# Each leg writes its own summary under $RUNDIR/<leg>/. Compare against the
# documented historical numbers in guide/plan.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${REPO_DIR}/SofaGpuCollision/build-profile/libSofaGpuCollision.so}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUNDIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/full_suite_${RUN_STAMP}}"
STEPS="${SOFA_BENCHMARK_STEPS:-160}"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${SOFA_PLUGIN_LIB_PATHS%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"
export SOFA_GPU_COLLISION_LIB

mkdir -p "${RUNDIR}"
echo "Full suite run dir -> ${RUNDIR}"

# run_leg <leg-name> <scene-file> <KEY=VAL ...extra env>
run_leg() {
    local leg="$1"; shift
    local scene="$1"; shift
    local leg_dir="${RUNDIR}/${leg}"
    mkdir -p "${leg_dir}"
    echo "------------------------------------------------------------"
    echo "  LEG: ${leg}   scene=$(basename "${scene}")   extra: $*"
    echo "------------------------------------------------------------"
    env "$@" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${leg}" \
        SOFA_BENCHMARK_LOG_DIR="${leg_dir}" \
        "${SOFA_ROOT}/bin/runSofa" \
            -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${SOFA_GPU_COLLISION_LIB}" \
            "${scene}"
}

SMALL="${REPO_DIR}/testscenes/one_tissue_one_blade.py"
LARGE="${REPO_DIR}/testscenes/large_tissue_blade.py"
VT_SELF="${REPO_DIR}/testscenes/self_collision_vertex_triangle.py"
VT_CROSS="${REPO_DIR}/testscenes/cross_model_vertex_triangle.py"
HASH="${REPO_DIR}/testscenes/hash_prefixsum_large.py"

# --- tri-tri FBP: small ---
run_leg small_fastpath   "${SMALL}" SOFA_USE_FEATURE_BASED_PROXIMITY=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=0
run_leg small_validation "${SMALL}" SOFA_USE_FEATURE_BASED_PROXIMITY=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

# --- tri-tri FBP: large (79,520 elements) ---
run_leg large_fastpath   "${LARGE}" SOFA_USE_FEATURE_BASED_PROXIMITY=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=0
run_leg large_validation "${LARGE}" SOFA_USE_FEATURE_BASED_PROXIMITY=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

# --- vertex-triangle (scene defaults already enable FBP + v-t + counter) ---
run_leg vt_self_validation  "${VT_SELF}"  SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
run_leg vt_self_fastpath    "${VT_SELF}"  SOFA_PROXIMITY_READ_CONTACT_COUNTER=0
run_leg vt_cross_validation "${VT_CROSS}" SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
run_leg vt_cross_fastpath   "${VT_CROSS}" SOFA_PROXIMITY_READ_CONTACT_COUNTER=0

# --- broad-cull 5-way (large tissue + large tool): dense, optimised hash, simple hash, sorted grid ---
run_leg hash_dense  "${HASH}" SOFA_USE_HASH_PREFIXSUM_GENERATION=0 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
run_leg hash_on     "${HASH}" SOFA_USE_HASH_PREFIXSUM_GENERATION=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
run_leg hash_simple "${HASH}" SOFA_USE_SIMPLE_HASH_GENERATION=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
run_leg hash_sorted "${HASH}" SOFA_USE_SORTED_GRID_GENERATION=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

echo
echo "============================================================"
echo "  DONE. Summaries under ${RUNDIR}/<leg>/"
echo "============================================================"
