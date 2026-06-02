#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SOFA_BENCHMARK_STEPS="${SOFA_BENCHMARK_STEPS:-120}"
export SOFA_LARGE_WARMUP_STEPS="${SOFA_LARGE_WARMUP_STEPS:-10}"
export SOFA_BENCHMARK_LABEL_SUFFIX="${SOFA_BENCHMARK_LABEL_SUFFIX:-_large_compute_$(date +%Y%m%d_%H%M%S)}"
export SOFA_BENCHMARK_LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/large_tissue_blade_compute_$(date +%Y%m%d_%H%M%S)}"

bash "${SCRIPT_DIR}/run_large_tissue_blade_benchmarks_wsl.sh"

python3 "${SCRIPT_DIR}/analyze_benchmark_compute_only.py" \
    "${SOFA_BENCHMARK_LOG_DIR}" \
    --output "${SOFA_BENCHMARK_LOG_DIR}/compute_only_summary.txt" \
    --markdown "${SOFA_BENCHMARK_LOG_DIR}/gpu_collision_profile_report.md" \
    --svg "${SOFA_BENCHMARK_LOG_DIR}/gpu_collision_stage_times.svg"
