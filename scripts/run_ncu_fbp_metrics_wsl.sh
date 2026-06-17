#!/usr/bin/env bash
# Nsight Compute profile of featureBasedProximityKernel (the FBP narrow kernel),
# via the standalone backend bench (large geometry, no SOFA). Reports the limiter:
# compute SOL %, DRAM %, achieved occupancy %, registers/thread.
set -uo pipefail
REPO=/home/arfin/gpu-sofa
BUILD=$REPO/SofaGpuCollision/build-profile
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
OUT=$REPO/output/benchmark_logs/ncu_fbp_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD:/opt/sofa/install/v25.12/lib"

echo "=== ncu on featureBasedProximityKernel (8 launches) ==="
ncu --target-processes all --kernel-name-base function \
    --kernel-name 'regex:featureBasedProximityKernel' \
    --launch-count 8 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__waves_per_multiprocessor \
    --csv "$BENCH" 2>"$OUT/ncu_stderr.txt" | tee "$OUT/ncu_fbp.csv"
echo
echo "ncu output -> $OUT/ncu_fbp.csv"
echo NCU_FBP_DONE
