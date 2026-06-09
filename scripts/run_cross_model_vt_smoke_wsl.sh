#!/usr/bin/env bash
# Cross-model vertex-triangle smoke (Phase 12 cross-model wiring).
# Tissue (CudaTriangleCollisionModel) vs tool (CudaPointCollisionModel).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${REPO_DIR}/SofaGpuCollision/build-profile/libSofaGpuCollision.so}"
LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/xm_vt_smoke_$(date +%Y%m%d_%H%M%S)}"
STEPS="${SOFA_BENCHMARK_STEPS:-20}"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${SOFA_PLUGIN_LIB_PATHS%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"

export SOFA_USE_FEATURE_BASED_PROXIMITY=1
export SOFA_USE_VERTEX_TRIANGLE_PROXIMITY=1
export SOFA_PROXIMITY_READ_CONTACT_COUNTER="${SOFA_PROXIMITY_READ_CONTACT_COUNTER:-1}"
export SOFA_PROXIMITY_KEEP_CONTACTS_ON_DEVICE="${SOFA_PROXIMITY_KEEP_CONTACTS_ON_DEVICE:-1}"
export SOFA_PROXIMITY_COMPUTE_BARYCENTRICS=1
export SOFA_USE_INDEXED_DENSE_GRID_INPUT=1
export SOFA_USE_DIRECT_DEVICE_POSITIONS=1
export SOFA_COPY_CONTACTS_TO_HOST="${SOFA_COPY_CONTACTS_TO_HOST:-0}"
export SOFA_DEDUPLICATE_PAIRS=1
export SOFA_USE_GPU_HASH_DEDUPE=1
export SOFA_BENCHMARK_LOG_DIR="${LOG_DIR}"
export SOFA_GPU_COLLISION_LIB

mkdir -p "${LOG_DIR}"
echo "Logs -> ${LOG_DIR}"

"${SOFA_ROOT}/bin/runSofa" \
    -g batch \
    -n "${STEPS}" \
    -l SofaPython3 \
    -l SofaCUDA \
    -l "${SOFA_GPU_COLLISION_LIB}" \
    "${REPO_DIR}/testscenes/cross_model_vertex_triangle.py"
