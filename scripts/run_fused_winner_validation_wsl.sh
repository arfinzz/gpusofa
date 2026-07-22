#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/arfin/gpu-sofa}
BUILD=${SOFA_GPU_COLLISION_BUILD_DIR:-$REPO/SofaGpuCollision/build-profile}
BENCH=$BUILD/SofaGpuCollisionDenseGridBackendBench
OUT=${SOFA_FUSED_WINNER_OUT:-$REPO/output/fused_winner_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD:/opt/sofa/install/v25.12/lib:${LD_LIBRARY_PATH:-}"

# Factors 1/2/4, tile chunking, CSR/hash, and both shared builders.
SOFA_BACKEND_BENCH_STEPS=2 SOFA_BACKEND_BENCH_WARMUP=1 \
    bash "$REPO/scripts/run_bigcell_parity_wsl.sh" >"$OUT/parity.log" 2>&1

printf 'round,tissue_n,kernel_ms,wall_ms,pairs,contacts,vf,fv,ee,total_overflow\n' >"$OUT/backend_replicated.csv"
for round in 0 1 2 3; do
    for n in 181 316; do
        log="$OUT/r${round}_n${n}.log"
        env SOFA_BACKEND_BENCH_STEPS=100 SOFA_BACKEND_BENCH_WARMUP=15 \
            SOFA_LARGE_TISSUE_NX="$n" SOFA_BACKEND_BENCH_RUN_FBP=0 \
            SOFA_BACKEND_BENCH_RUN_VT=0 SOFA_BACKEND_BENCH_RUN_HASH=0 \
            SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH=0 SOFA_BACKEND_BENCH_RUN_SORTED_GRID=0 \
            SOFA_BACKEND_BENCH_RUN_BIGCELL=1 SOFA_BACKEND_BENCH_BIGCELL_FACTOR=2 \
            SOFA_BACKEND_BENCH_BIGCELL_TILE=256 SOFA_BACKEND_BENCH_BIGCELL_SHARED_BUILD=1 \
            SOFA_BACKEND_BENCH_CSV="$OUT/r${round}_n${n}.csv" \
            "$BENCH" >"$log" 2>&1
        kernel=$(sed -n 's/^bigcell_gpu_kernel_avg_ms=//p' "$log")
        wall=$(sed -n 's/^bigcell_wall_avg_ms=//p' "$log")
        pairs=$(sed -n 's/^bigcell_pairs_tested=//p' "$log")
        read -r contacts vf fv ee < <(sed -n 's/^bigcell_contacts=\([0-9]*\) (vf=\([0-9]*\) fv=\([0-9]*\) ee=\([0-9]*\)).*/\1 \2 \3 \4/p' "$log")
        overflow=$(sed -n 's/^bigcell_total_overflow=//p' "$log")
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$round" "$n" "$kernel" "$wall" "$pairs" "$contacts" "$vf" "$fv" "$ee" "$overflow" \
            | tee -a "$OUT/backend_replicated.csv"
    done
done

# The two big-cell scenes used in the performance reports.
SOFA_BENCHMARK_STEPS=160 \
SOFA_BENCHMARK_LOG_DIR="$OUT/scenes" \
SOFA_SUITE_ONLY='^(hash_bigcell|xlarge_bigcell)$' \
    bash "$REPO/scripts/run_full_benchmark_suite_wsl.sh" >"$OUT/scenes.log" 2>&1

cuobjdump --dump-resource-usage "$BUILD/libSofaGpuCollision.so" >"$OUT/resource_usage.txt"
echo "WINNER_OUT=$OUT"
