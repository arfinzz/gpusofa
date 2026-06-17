#!/usr/bin/env bash
# Deep profiling to find the next optimization target:
#  (A) nsys per-kernel time summary  -> which kernel dominates the frame
#  (B) ncu section deep-dive on featureBasedProximityKernel -> stall reasons +
#      occupancy limiter (why it's slow, what to change)
# Uses the standalone backend bench (large geometry: FBP runs 322,560 candidate
# pairs -> 8018 contacts, the same workload as the large scene).
set -uo pipefail
SOFA_ROOT=/opt/sofa/install/v25.12
REPO=/home/arfin/gpu-sofa
BUILD=$REPO/SofaGpuCollision/build-profile
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
WIN=$(ls -d /mnt/c/Users/arfin/Desktop/GPU*SOFA)
OUT=$REPO/output/benchmark_logs/profile_deep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD:$SOFA_ROOT/lib"

echo "=== sync + incremental build ==="
rsync -a "$WIN/SofaGpuCollision/src/" "$REPO/SofaGpuCollision/src/"
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -2
[ -x "$BENCH" ] || { echo "BENCH MISSING"; exit 1; }

echo
echo "=== (A) nsys per-kernel time summary ==="
nsys profile --force-overwrite=true --stats=true --trace=cuda \
    -o "$OUT/nsys_bench" "$BENCH" >"$OUT/nsys_full.txt" 2>&1 || true
# print the CUDA kernel summary table
awk '/cuda_gpu_kern_sum|CUDA GPU Kernel Summary|Kernel Name/{f=1} f{print} /gpu_mem|Memory Operation/{if(f) exit}' "$OUT/nsys_full.txt" | head -45
echo "(full nsys stats: $OUT/nsys_full.txt)"

echo
echo "=== (B) ncu deep-dive on featureBasedProximityKernel ==="
ncu --target-processes all --kernel-name-base function \
    --kernel-name 'regex:featureBasedProximityKernel' \
    --launch-count 2 \
    --section SpeedOfLight --section Occupancy \
    --section WarpStateStats --section SchedulerStats --section LaunchStats \
    "$BENCH" >"$OUT/fbp_deep.txt" 2>"$OUT/fbp_deep_err.txt" || true
# print the most actionable parts
grep -iE "Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|Registers Per Thread|Block Limit|Theoretical Occupancy|Stall|Warp Cycles|Issued Warp|Eligible|Issue Slots|warps per scheduler" "$OUT/fbp_deep.txt" | head -60
echo "(full ncu report: $OUT/fbp_deep.txt)"
echo PROFILE_DEEP_DONE
