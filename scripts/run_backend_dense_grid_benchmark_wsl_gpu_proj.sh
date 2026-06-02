#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD_DIR="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${SOFA_BACKEND_PROFILE_ROOT:-${REPO_DIR}/output/benchmark_logs/backend_dense_grid_${TIMESTAMP}}"
BENCH="${BUILD_DIR}/SofaGpuCollisionDenseGridBackendBench"
NSIGHT_SYSTEMS_HOST_DIR="${NSIGHT_SYSTEMS_HOST_DIR:-/usr/lib/nsight-systems/host-linux-x64}"

mkdir -p "${LOG_ROOT}/nsight"

import_nsys_report() {
    local stem="$1"
    local importer="${NSIGHT_SYSTEMS_HOST_DIR}/QdstrmImporter"
    if [[ -f "${stem}.qdstrm" && -x "${importer}" ]]; then
        LD_LIBRARY_PATH="${NSIGHT_SYSTEMS_HOST_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${importer}" \
            -i "${stem}.qdstrm" \
            -o "${stem}.nsys-rep" \
            -f \
            > "${stem}_import_stdout.txt" 2> "${stem}_import_stderr.txt" || true
    fi
    if [[ -f "${stem}.nsys-rep" ]]; then
        nsys stats "${stem}.nsys-rep" \
            > "${stem}_stats_stdout.txt" 2> "${stem}_stats_stderr.txt" || true
    fi
}

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export LD_LIBRARY_PATH="${BUILD_DIR}:${SOFA_ROOT}/lib:${NSIGHT_SYSTEMS_HOST_DIR}:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SOFA_ROOT

cmake -S "${REPO_DIR}/SofaGpuCollision" \
    -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${SOFA_ROOT}" \
    -DSOFAGPUCOLLISION_ENABLE_CUDA=ON
cmake --build "${BUILD_DIR}" --target SofaGpuCollisionDenseGridBackendBench -j"$(nproc)"

if [[ ! -x "${BENCH}" ]]; then
    echo "Backend benchmark executable not found: ${BENCH}" >&2
    exit 1
fi

echo "Running standalone dense-grid backend benchmark..."
SOFA_BACKEND_BENCH_CSV="${LOG_ROOT}/backend_dense_grid_timings.csv" \
SOFA_GPU_DETAILED_PROFILING="${SOFA_GPU_DETAILED_PROFILING:-1}" \
SOFA_COPY_CONTACTS_TO_HOST="${SOFA_COPY_CONTACTS_TO_HOST:-0}" \
SOFA_DEDUPLICATE_PAIRS="${SOFA_DEDUPLICATE_PAIRS:-1}" \
SOFA_USE_GPU_HASH_DEDUPE="${SOFA_USE_GPU_HASH_DEDUPE:-1}" \
SOFA_CANONICAL_PAIR_EMISSION="${SOFA_CANONICAL_PAIR_EMISSION:-0}" \
"${BENCH}" > "${LOG_ROOT}/backend_dense_grid_summary.txt"

set +e
echo "Running Nsight Systems on standalone backend benchmark..."
SOFA_BACKEND_BENCH_CSV="${LOG_ROOT}/nsight/backend_dense_grid_nsys_timings.csv" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS="${SOFA_DEDUPLICATE_PAIRS:-1}" \
SOFA_USE_GPU_HASH_DEDUPE="${SOFA_USE_GPU_HASH_DEDUPE:-1}" \
SOFA_CANONICAL_PAIR_EMISSION="${SOFA_CANONICAL_PAIR_EMISSION:-0}" \
nsys profile \
    --trace=cuda,nvtx \
    --capture-range=nvtx \
    --nvtx-capture='SofaGpuCollision dense-grid narrow phase' \
    --capture-range-end=repeat:4 \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${LOG_ROOT}/nsight/backend_dense_grid_nsys" \
    "${BENCH}" \
    > "${LOG_ROOT}/nsight/nsys_stdout.txt" 2> "${LOG_ROOT}/nsight/nsys_stderr.txt"
NSYS_STATUS=$?
import_nsys_report "${LOG_ROOT}/nsight/backend_dense_grid_nsys"

echo "Running Nsight Compute on standalone backend benchmark..."
SOFA_BACKEND_BENCH_CSV="${LOG_ROOT}/nsight/backend_dense_grid_ncu_timings.csv" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS="${SOFA_DEDUPLICATE_PAIRS:-1}" \
SOFA_USE_GPU_HASH_DEDUPE="${SOFA_USE_GPU_HASH_DEDUPE:-1}" \
SOFA_CANONICAL_PAIR_EMISSION="${SOFA_CANONICAL_PAIR_EMISSION:-0}" \
ncu \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name 'regex:.*(resetDenseGridKernel|insertPackedTrianglesKernel|insertIndexedTrianglesKernel|generateDenseGridCandidatePairsKernel|generateDenseGridUniqueCandidatePairsKernel|exactDenseGridContactKernel|exactDenseGridIndexedContactKernel).*' \
    --launch-count "${SOFA_NCU_LAUNCH_COUNT:-40}" \
    --force-overwrite \
    --export "${LOG_ROOT}/nsight/backend_dense_grid_ncu" \
    "${BENCH}" \
    > "${LOG_ROOT}/nsight/ncu_stdout.txt" 2> "${LOG_ROOT}/nsight/ncu_stderr.txt"
NCU_STATUS=$?
set -e

{
    echo "log_root=${LOG_ROOT}"
    echo "sofa_root=${SOFA_ROOT}"
    echo "build_dir=${BUILD_DIR}"
    echo "benchmark=${BENCH}"
    echo "nsys_status=${NSYS_STATUS}"
    echo "ncu_status=${NCU_STATUS}"
    echo "nsys_qdstrm=${LOG_ROOT}/nsight/backend_dense_grid_nsys.qdstrm"
    echo "nsys_report=${LOG_ROOT}/nsight/backend_dense_grid_nsys.nsys-rep"
    echo "nsys_sqlite=${LOG_ROOT}/nsight/backend_dense_grid_nsys.sqlite"
} > "${LOG_ROOT}/manifest.txt"

echo "Backend dense-grid benchmark complete."
echo "Log root: ${LOG_ROOT}"
echo "Nsight Systems status: ${NSYS_STATUS}"
echo "Nsight Compute status: ${NCU_STATUS}"
