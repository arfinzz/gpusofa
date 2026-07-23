#!/usr/bin/env bash
# Nsight Compute per-kernel metrics for the way-6 (big-cell fused) pipeline,
# at the historical 14k/80k/200k triangle-count scales using the winning factor-2
# configuration. Same metric set as run_profile_5way_wsl.sh.
# Usage (from WSL): bash scripts/run_ncu_bigcell_wsl.sh
set -uo pipefail

REPO=/home/arfin/gpu-sofa
BUILD=$REPO/SofaGpuCollision/build-profile
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
OUT=$REPO/output/benchmark_logs/profiling_bigcell_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="$BUILD:${SOFA_ROOT}/lib:${PLP}"
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread"

profile_leg() {
    local label="$1" nx="$2" factor="$3"
    echo "=== NCU: $label (nx=$nx factor=$factor) ==="
    local -a geometry_env
    case "$label" in
        bigcell_14k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=30 SOFA_LARGE_BLADE_SEGMENTS_Y=8 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=8.0 SOFA_BACKEND_BLADE_HEIGHT=0.4 SOFA_GRID_MIN_X=-4.5 SOFA_GRID_MIN_Y=-0.5 SOFA_GRID_MIN_Z=-4.5 SOFA_GRID_MAX_X=4.5 SOFA_GRID_MAX_Y=0.5 SOFA_GRID_MAX_Z=4.5 SOFA_GRID_RESOLUTION_X=64 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=64 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=128 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=128) ;;
        bigcell_80k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=128 SOFA_LARGE_BLADE_SEGMENTS_Y=24 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=12.0 SOFA_BACKEND_BLADE_HEIGHT=0.8 SOFA_GRID_RESOLUTION_X=96 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=96 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=192 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=192) ;;
        bigcell_200k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=30 SOFA_LARGE_BLADE_SEGMENTS_Y=8 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=8.0 SOFA_BACKEND_BLADE_HEIGHT=0.4 SOFA_GRID_MIN_X=-4.5 SOFA_GRID_MIN_Y=-0.5 SOFA_GRID_MIN_Z=-4.5 SOFA_GRID_MAX_X=4.5 SOFA_GRID_MAX_Y=0.5 SOFA_GRID_MAX_Z=4.5 SOFA_GRID_RESOLUTION_X=128 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=128 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=256 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=256) ;;
        *) echo "unknown ncu label: $label" >&2; return 2 ;;
    esac
    env "${geometry_env[@]}" SOFA_LARGE_TISSUE_NX="$nx" \
        SOFA_BACKEND_BENCH_RUN_FBP=0 SOFA_BACKEND_BENCH_RUN_VT=0 \
        SOFA_BACKEND_BENCH_RUN_HASH=0 SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH=0 \
        SOFA_BACKEND_BENCH_RUN_SORTED_GRID=0 \
        SOFA_BACKEND_BENCH_BIGCELL_FACTOR="$factor" \
        SOFA_BACKEND_BENCH_CSV="$OUT/${label}_timings.csv" \
        SOFA_BACKEND_BENCH_STEPS=4 SOFA_BACKEND_BENCH_WARMUP=1 \
        SOFA_GPU_DETAILED_PROFILING=0 \
        SOFA_BIGCELL_CUDA_GRAPH=0 \
        ncu --target-processes all --kernel-name-base function \
        --launch-count 40 --launch-skip 10 \
        --force-overwrite --export "$OUT/${label}_profile" \
        --metrics "$METRICS" \
        "$BENCH" > "$OUT/${label}_ncu_capture.log" 2>"$OUT/${label}_ncu_err.txt" || true
    if [[ -f "$OUT/${label}_profile.ncu-rep" ]]; then
        ncu --import "$OUT/${label}_profile.ncu-rep" --csv --page raw --metrics "$METRICS" \
            > "$OUT/${label}_ncu.csv" 2>>"$OUT/${label}_ncu_err.txt" || true
    fi
    echo "--- $label ncu rows: $(grep -c '","' "$OUT/${label}_ncu.csv" 2>/dev/null || echo 0) ---"
}

profile_leg bigcell_14k_f2  81  2
profile_leg bigcell_80k_f2  181 2
profile_leg bigcell_200k_f2 316 2

echo "OUT_DIR=$OUT"
echo NCU_BIGCELL_DONE
