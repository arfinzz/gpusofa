#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export SOFA_BENCHMARK_STEPS="${SOFA_BENCHMARK_STEPS:-80}"
export SOFA_LARGE_WARMUP_STEPS="${SOFA_LARGE_WARMUP_STEPS:-10}"
export SOFA_BENCHMARK_LABEL_SUFFIX="${SOFA_BENCHMARK_LABEL_SUFFIX:-_kernel_profile_${TIMESTAMP}}"
export SOFA_BENCHMARK_LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/gpu_kernel_profile_${TIMESTAMP}}"

bash "${SCRIPT_DIR}/run_large_tissue_blade_benchmarks_wsl.sh"

python3 "${SCRIPT_DIR}/analyze_benchmark_compute_only.py" \
    "${SOFA_BENCHMARK_LOG_DIR}" \
    --output "${SOFA_BENCHMARK_LOG_DIR}/gpu_kernel_profile_summary.txt"

echo
echo "GPU kernel profile written to ${SOFA_BENCHMARK_LOG_DIR}/gpu_kernel_profile_summary.txt"
