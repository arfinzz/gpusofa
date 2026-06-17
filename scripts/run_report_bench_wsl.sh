#!/usr/bin/env bash
# Fresh clean benchmark run on the current (reverted-baseline) build for the new
# report: warms the GPU, then full suite + mode comparison + ncu FBP profile +
# backend parity. Captures everything under one timestamped dir.
set -uo pipefail
SOFA_ROOT=/opt/sofa/install/v25.12
REPO=/home/arfin/gpu-sofa
BUILD=$REPO/SofaGpuCollision/build-profile
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
WIN=$(ls -d /mnt/c/Users/arfin/Desktop/GPU*SOFA)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/output/benchmark_logs/fresh_report_$STAMP
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD:$SOFA_ROOT/lib"

echo "=== sync + build ==="
rsync -a "$WIN/SofaGpuCollision/src/" "$REPO/SofaGpuCollision/src/"
rsync -a --delete "$WIN/testscenes/" "$REPO/testscenes/"
rsync -a "$WIN/scripts/" "$REPO/scripts/"
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -2 | grep -iE "error|Built target" || true
[ -x "$BENCH" ] || { echo BUILD_FAILED; exit 1; }
nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv,noheader || true

echo "=== warmup (prime GPU clocks) ==="
SOFA_BACKEND_BENCH_RUN_HASH=1 "$BENCH" >/dev/null 2>&1 || true

echo "=== backend bench parity ==="
SOFA_BACKEND_BENCH_RUN_HASH=1 "$BENCH" 2>&1 | grep -iE "fbp_contacts|hash_contacts|unique|overflow|occupied|vt_contacts" | tee "$OUT/bench.txt"

echo
echo "=== full suite ==="
bash "$REPO/scripts/run_full_benchmark_suite_wsl.sh" >"$OUT/suite.log" 2>&1 || true
SD=$(ls -dt $REPO/output/benchmark_logs/full_suite_* | head -1)
echo "suite dir: $SD" | tee "$OUT/suite_summary.txt"
for f in $(ls "$SD"/*/*summary*.txt 2>/dev/null | sort); do
  leg=$(basename "$(dirname "$f")")
  fps=$(grep -E '^avg_fps=' "$f"|cut -d= -f2); nk=$(grep -E '^avg_narrow_kernel_ms=' "$f"|cut -d= -f2)
  nw=$(grep -E '^avg_narrow_wall_ms=' "$f"|cut -d= -f2); cc=$(grep -E '^avg_narrow_output_contact_count=' "$f"|cut -d= -f2)
  vf=$(grep -E '^avg_narrow_vf_contact_count=' "$f"|cut -d= -f2); fv=$(grep -E '^avg_narrow_fv_contact_count=' "$f"|cut -d= -f2)
  ee=$(grep -E '^avg_narrow_ee_contact_count=' "$f"|cut -d= -f2); ov=$(grep -E '^avg_narrow_overflow_count=' "$f"|cut -d= -f2)
  kl=$(grep -E '^avg_kernel_launch_count=' "$f"|cut -d= -f2)
  printf '%-20s fps=%-9s nkern=%-9s nwall=%-9s contacts=%-6s vffvee=%s/%s/%s ovf=%s launches=%s\n' "$leg" "${fps%.*}" "$nk" "$nw" "$cc" "$vf" "$fv" "$ee" "$ov" "${kl%.*}" | tee -a "$OUT/suite_summary.txt"
done

echo
echo "=== mode comparison (GPU warm) ==="
bash "$REPO/scripts/run_mode_comparison_ab_wsl.sh" 2>&1 | tee "$OUT/mode_cmp.txt" | tail -6

echo
echo "=== ncu FBP profile (registers / occupancy / stalls / duration) ==="
ncu --target-processes all --kernel-name-base function --kernel-name 'regex:featureBasedProximityKernel' --launch-count 4 \
  --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio \
  --csv "$BENCH" 2>/dev/null | grep -iE "registers_per_thread|warps_active|sm__throughput|dram_throughput|time_duration|stalled" | sed 's/.*"\([a-z_]*[a-z]\)","[^"]*","\([0-9.]*\)"$/\1 = \2/' | tee "$OUT/ncu_fbp.txt"
echo
echo "OUT=$OUT"
echo FRESH_BENCH_DONE
