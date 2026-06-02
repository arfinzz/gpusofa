#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG_DIR="${SOFA_SCALE_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/scene_size_scaling_${TIMESTAMP}}"
REPORT_DIR="${SOFA_REPORT_DIR:-${REPO_DIR}/reports}"
STEPS="${SOFA_BENCHMARK_STEPS:-70}"
WARMUP="${SOFA_LARGE_WARMUP_STEPS:-10}"

CASES=(
  "medium 121 96 18 4 96 8 96 192 192 8000000"
  "large 181 128 24 4 96 8 96 192 192 8000000"
  "xlarge 241 192 36 4 128 8 128 256 256 8000000"
  "xxlarge 301 256 48 4 160 8 160 384 384 8000000"
)

mkdir -p "${BASE_LOG_DIR}"
mkdir -p "${REPORT_DIR}"

echo "Writing scaling benchmark logs to ${BASE_LOG_DIR}"
echo "Steps=${STEPS}, warmup=${WARMUP}"

for case_spec in "${CASES[@]}"; do
    read -r name tissue_n blade_x blade_y blade_z grid_x grid_y grid_z max_tissue max_tool max_candidates <<< "${case_spec}"

    echo
    echo "=== Running scale case: ${name} ==="
    echo "tissue=${tissue_n}x${tissue_n}, blade=${blade_x}x${blade_y}x${blade_z}, grid=${grid_x}x${grid_y}x${grid_z}"

    export SOFA_BENCHMARK_STEPS="${STEPS}"
    export SOFA_LARGE_WARMUP_STEPS="${WARMUP}"
    export SOFA_BENCHMARK_LABEL_SUFFIX="_scale_${name}_${TIMESTAMP}"
    export SOFA_BENCHMARK_LOG_DIR="${BASE_LOG_DIR}/${name}"
    export SOFA_LARGE_TISSUE_NX="${tissue_n}"
    export SOFA_LARGE_TISSUE_NZ="${tissue_n}"
    export SOFA_LARGE_BLADE_SEGMENTS_X="${blade_x}"
    export SOFA_LARGE_BLADE_SEGMENTS_Y="${blade_y}"
    export SOFA_LARGE_BLADE_SEGMENTS_Z="${blade_z}"
    export SOFA_GRID_RESOLUTION_X="${grid_x}"
    export SOFA_GRID_RESOLUTION_Y="${grid_y}"
    export SOFA_GRID_RESOLUTION_Z="${grid_z}"
    export SOFA_MAX_TISSUE_TRIANGLES_PER_CELL="${max_tissue}"
    export SOFA_MAX_TOOL_TRIANGLES_PER_CELL="${max_tool}"
    export SOFA_MAX_CANDIDATE_PAIRS="${max_candidates}"

    bash "${SCRIPT_DIR}/run_large_tissue_blade_compute_breakdown_wsl.sh"
done

python3 "${SCRIPT_DIR}/generate_scene_size_comparison.py" "${BASE_LOG_DIR}" \
    --csv "${REPORT_DIR}/scene_size_scaling_summary.csv" \
    --svg "${REPORT_DIR}/scene_size_scaling_comparison.svg" \
    --md "${REPORT_DIR}/scene_size_scaling_benchmark_report.md"

echo
echo "Scaling benchmark complete."
echo "Base log dir: ${BASE_LOG_DIR}"
echo "Report: ${REPORT_DIR}/scene_size_scaling_benchmark_report.md"
echo "Graph: ${REPORT_DIR}/scene_size_scaling_comparison.svg"
