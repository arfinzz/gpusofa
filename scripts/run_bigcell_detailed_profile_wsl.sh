#!/usr/bin/env bash
# Detailed profiling campaign for the winning big-cell configuration.
# Produces three views at the historical 14k/80k/200k triangle-count scales.
#   production: graph-enabled aggregate timing, no diagnostic perturbation
#   events:     graph-disabled CUDA-event timing for every pipeline stage
#   internals:  separate clock64/counter instrumented fused kernel
set -euo pipefail

REPO=${REPO:-/home/arfin/gpu-sofa}
BUILD=${SOFA_GPU_COLLISION_BUILD_DIR:-$REPO/SofaGpuCollision/build-profile}
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
OUT=${SOFA_BIGCELL_PROFILE_OUT:-$REPO/output/bigcell_detailed_profile_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD:/opt/sofa/install/v25.12/lib:${LD_LIBRARY_PATH:-}"

run_leg() {
    local label=$1 nx=$2 steps=$3 warmup=$4 detailed=$5 internals=$6 graph=$7
    local base="$OUT/$label.csv" log="$OUT/$label.log"
    local -a geometry_env
    case "$label" in
        14k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=30 SOFA_LARGE_BLADE_SEGMENTS_Y=8 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=8.0 SOFA_BACKEND_BLADE_HEIGHT=0.4 SOFA_GRID_MIN_X=-4.5 SOFA_GRID_MIN_Y=-0.5 SOFA_GRID_MIN_Z=-4.5 SOFA_GRID_MAX_X=4.5 SOFA_GRID_MAX_Y=0.5 SOFA_GRID_MAX_Z=4.5 SOFA_GRID_RESOLUTION_X=64 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=64 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=128 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=128) ;;
        80k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=128 SOFA_LARGE_BLADE_SEGMENTS_Y=24 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=12.0 SOFA_BACKEND_BLADE_HEIGHT=0.8 SOFA_GRID_RESOLUTION_X=96 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=96 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=192 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=192) ;;
        200k_*) geometry_env=(SOFA_LARGE_BLADE_SEGMENTS_X=30 SOFA_LARGE_BLADE_SEGMENTS_Y=8 SOFA_LARGE_BLADE_SEGMENTS_Z=4 SOFA_BACKEND_TISSUE_SIZE=8.0 SOFA_BACKEND_BLADE_HEIGHT=0.4 SOFA_GRID_MIN_X=-4.5 SOFA_GRID_MIN_Y=-0.5 SOFA_GRID_MIN_Z=-4.5 SOFA_GRID_MAX_X=4.5 SOFA_GRID_MAX_Y=0.5 SOFA_GRID_MAX_Z=4.5 SOFA_GRID_RESOLUTION_X=128 SOFA_GRID_RESOLUTION_Y=8 SOFA_GRID_RESOLUTION_Z=128 SOFA_MAX_TISSUE_TRIANGLES_PER_CELL=256 SOFA_MAX_TOOL_TRIANGLES_PER_CELL=256) ;;
        *) echo "unknown profile label: $label" >&2; return 2 ;;
    esac
    env "${geometry_env[@]}" SOFA_LARGE_TISSUE_NX="$nx" \
        SOFA_BACKEND_BENCH_STEPS="$steps" SOFA_BACKEND_BENCH_WARMUP="$warmup" \
        SOFA_BACKEND_BENCH_RUN_FBP=0 SOFA_BACKEND_BENCH_RUN_VT=0 \
        SOFA_BACKEND_BENCH_RUN_HASH=0 SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH=0 \
        SOFA_BACKEND_BENCH_RUN_SORTED_GRID=0 SOFA_BACKEND_BENCH_RUN_BIGCELL=1 \
        SOFA_BACKEND_BENCH_BIGCELL_FACTOR=2 SOFA_BACKEND_BENCH_BIGCELL_TILE=256 \
        SOFA_BACKEND_BENCH_BIGCELL_HASH_BUILD=0 SOFA_BACKEND_BENCH_BIGCELL_SHARED_BUILD=1 \
        SOFA_BACKEND_BENCH_BIGCELL_PROFILE_INTERNALS="$internals" \
        SOFA_GPU_DETAILED_PROFILING="$detailed" SOFA_BIGCELL_CUDA_GRAPH="$graph" \
        SOFA_BACKEND_BENCH_CSV="$base" "$BENCH" >"$log" 2>&1
    grep '^bigcell_' "$log"
}

for spec in 14k:81 80k:181 200k:316; do
    label=${spec%%:*}
    nx=${spec##*:}
    run_leg "${label}_production" "$nx" 80 12 0 0 1
    run_leg "${label}_events" "$nx" 30 6 1 0 0
    run_leg "${label}_internals" "$nx" 12 3 1 1 0
done

cuobjdump --dump-resource-usage "$BUILD/libSofaGpuCollision.so" >"$OUT/resource_usage.txt"
printf 'PROFILE_OUT=%s\n' "$OUT"