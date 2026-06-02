#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${SOFA_ROOT}/plugins/SofaGpuCollision/lib/libSofaGpuCollision.so}"
LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/large_tissue_blade_$(date +%Y%m%d_%H%M%S)}"
STEPS="${SOFA_BENCHMARK_STEPS:-60}"

export SOFA_BENCHMARK_LOG_DIR="${LOG_DIR}"
export SOFA_BENCHMARK_LABEL_SUFFIX="${SOFA_BENCHMARK_LABEL_SUFFIX:-_large_$(date +%Y%m%d_%H%M%S)}"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
SOFA_PLUGIN_REPOSITORIES="${SOFA_PLUGIN_LIB_PATHS%:}"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins${SOFA_PLUGIN_REPOSITORIES:+:${SOFA_PLUGIN_REPOSITORIES}}${SOFA_PLUGIN_PATH:+:${SOFA_PLUGIN_PATH}}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"

mkdir -p "${LOG_DIR}"

run_sofa() {
    "${SOFA_ROOT}/bin/runSofa" \
        -l SofaPython3 \
        -l SofaCUDA \
        -l "${SOFA_GPU_COLLISION_LIB}" \
        "$@"
}

echo "Writing benchmark logs to ${LOG_DIR}"
echo "Using benchmark label suffix ${SOFA_BENCHMARK_LABEL_SUFFIX}"
echo "Running CPU large tissue/blade benchmark..."
run_sofa -g batch -n "${STEPS}" "${REPO_DIR}/test_cpu_large_tissue_blade_benchmark.py"

echo "Running GPU dense-grid large tissue/blade benchmark..."
run_sofa -g batch -n "${STEPS}" "${REPO_DIR}/test_gpu_large_tissue_blade_dense_grid_benchmark.py"

echo "CPU summary label: cpu_large_tissue_blade_benchmark${SOFA_BENCHMARK_LABEL_SUFFIX}"
echo "GPU summary label: gpu_large_tissue_blade_dense_grid_benchmark${SOFA_BENCHMARK_LABEL_SUFFIX}"
