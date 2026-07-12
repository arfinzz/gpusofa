#!/usr/bin/env bash
# Full profiling capture for ALL FIVE broad-cull execution modes, via the
# standalone backend bench (no SOFA): per-mode Nsight Systems per-kernel time
# summary + per-mode Nsight Compute per-kernel metrics (SOL, DRAM, occupancy,
# registers). CUDA graphs are disabled during capture so every kernel is an
# individual launch with clean attribution (kernel times are unaffected; only
# launch overhead differs). Modes: dense(FBP) / opt hash / simple hash /
# sorted grid. Outputs land in output/benchmark_logs/profiling_5way_<ts>/.
# Usage (from WSL): bash scripts/run_profile_5way_wsl.sh
set -uo pipefail

REPO=/home/arfin/gpu-sofa
BUILD=$REPO/SofaGpuCollision/build-profile
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
OUT=$REPO/output/benchmark_logs/profiling_5way_$(date +%Y%m%d_%H%M%S)
NSIGHT_HOST=/usr/lib/nsight-systems/host-linux-x64
mkdir -p "$OUT"
PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="$BUILD:${SOFA_ROOT}/lib:${PLP}"

# mode name -> leg gate env values: FBP VT HASH SIMPLE SORTED BIGCELL
run_env() {
    local fbp="$1" vt="$2" hash="$3" simple="$4" sorted="$5" bigcell="${6:-0}"
    echo "SOFA_BACKEND_BENCH_RUN_FBP=$fbp SOFA_BACKEND_BENCH_RUN_VT=$vt SOFA_BACKEND_BENCH_RUN_HASH=$hash SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH=$simple SOFA_BACKEND_BENCH_RUN_SORTED_GRID=$sorted SOFA_BACKEND_BENCH_RUN_BIGCELL=$bigcell"
}

profile_mode() {
    local mode="$1"; shift
    local gates="$*"
    echo "=== MODE: $mode ==="

    # --- Nsight Systems: per-kernel time summary over 12 steady frames ---
    env $gates \
        SOFA_BACKEND_BENCH_CSV="$OUT/${mode}_timings.csv" \
        SOFA_BACKEND_BENCH_STEPS=12 SOFA_BACKEND_BENCH_WARMUP=3 \
        SOFA_GPU_DETAILED_PROFILING=0 \
        SOFA_HASH_CUDA_GRAPH=0 SOFA_SIMPLE_HASH_CUDA_GRAPH=0 SOFA_SORTED_GRID_CUDA_GRAPH=0 SOFA_BIGCELL_CUDA_GRAPH=0 \
        nsys profile -t cuda --force-overwrite=true -o "$OUT/${mode}" \
        "$BENCH" > "$OUT/${mode}_bench_stdout.txt" 2>&1
    if [[ -f "$OUT/${mode}.qdstrm" && -x "$NSIGHT_HOST/QdstrmImporter" ]]; then
        LD_LIBRARY_PATH="$NSIGHT_HOST:$LD_LIBRARY_PATH" "$NSIGHT_HOST/QdstrmImporter" \
            -i "$OUT/${mode}.qdstrm" -o "$OUT/${mode}.nsys-rep" -f >/dev/null 2>&1 || true
    fi
    if [[ -f "$OUT/${mode}.nsys-rep" ]]; then
        nsys stats --report gpukernsum --format csv "$OUT/${mode}.nsys-rep" \
            > "$OUT/${mode}_kernsum.csv" 2>"$OUT/${mode}_kernsum_err.txt" || true
        echo "--- $mode nsys kernel summary (top 14 by total time) ---"
        grep -v '^$' "$OUT/${mode}_kernsum.csv" | tail -n +2 | head -15
    else
        echo "($mode: no nsys-rep produced — see ${mode}_bench_stdout.txt)"
    fi

    # --- Nsight Compute: per-kernel metrics, 45 launches after skipping 12 ---
    env $gates \
        SOFA_BACKEND_BENCH_CSV="$OUT/${mode}_ncu_timings.csv" \
        SOFA_BACKEND_BENCH_STEPS=4 SOFA_BACKEND_BENCH_WARMUP=1 \
        SOFA_GPU_DETAILED_PROFILING=0 \
        SOFA_HASH_CUDA_GRAPH=0 SOFA_SIMPLE_HASH_CUDA_GRAPH=0 SOFA_SORTED_GRID_CUDA_GRAPH=0 SOFA_BIGCELL_CUDA_GRAPH=0 \
        ncu --target-processes all --kernel-name-base function \
        --launch-count 45 --launch-skip 12 \
        --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread \
        --csv "$BENCH" > "$OUT/${mode}_ncu.csv" 2>"$OUT/${mode}_ncu_err.txt" || true
    echo "--- $mode ncu rows: $(grep -c '","' "$OUT/${mode}_ncu.csv" 2>/dev/null || echo 0) ---"
    echo
}

# Optional filter: pass mode names as args to profile a subset (batching under
# a caller wall-clock cap); no args = all five modes.
MODES=("$@")
should_run() {
    [ ${#MODES[@]} -eq 0 ] && return 0
    local m
    for m in "${MODES[@]}"; do [ "$m" = "$1" ] && return 0; done
    return 1
}

should_run dense   && profile_mode dense   "$(run_env 1 0 0 0 0 0)"
should_run hash    && profile_mode hash    "$(run_env 0 0 1 0 0 0)"
should_run simple  && profile_mode simple  "$(run_env 0 0 0 1 0 0)"
should_run sorted  && profile_mode sorted  "$(run_env 0 0 0 0 1 0)"
should_run bigcell && profile_mode bigcell "$(run_env 0 0 0 0 0 1)"

echo "OUT_DIR=$OUT"
echo PROFILE5WAY_DONE
