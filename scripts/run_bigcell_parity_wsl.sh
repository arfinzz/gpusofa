#!/usr/bin/env bash
# Big-cell fused ("6th way") parity: runs the standalone backend bench across
# the way-6 toggle matrix and prints the contact-parity lines. Every combo must
# report the same contacts as the dense FBP leg, and bigcell_pairs_tested must
# equal sortedgrid_unique_pairs (both apply the home-cell rule at small-cell
# granularity). The tile=32 leg forces the oversized-big-cell chunk loop.
# Usage (from WSL): bash scripts/run_bigcell_parity_wsl.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
B="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="${B}:${SOFA_ROOT}/lib:${PLP}"

run_combo() {
    local label="$1" factor="$2" tile="$3" hashbuild="$4" runothers="$5"
    echo "=== BIGCELL PARITY: ${label} (factor=${factor} tile=${tile} hash_build=${hashbuild}) ==="
    SOFA_BACKEND_BENCH_CSV="/tmp/bigcell_parity_${label}.csv" \
    SOFA_GPU_DETAILED_PROFILING=0 \
    SOFA_BACKEND_BENCH_RUN_HASH="${runothers}" \
    SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH="${runothers}" \
    SOFA_BACKEND_BENCH_BIGCELL_FACTOR="${factor}" \
    SOFA_BACKEND_BENCH_BIGCELL_TILE="${tile}" \
    SOFA_BACKEND_BENCH_BIGCELL_HASH_BUILD="${hashbuild}" \
    "${B}/SofaGpuCollisionDenseGridBackendBench" 2>&1 \
      | grep -iE 'fbp_contacts|sortedgrid_(unique|contacts)|bigcell_(build|factor|tool_tile|gpu_kernel|pairs_tested|contacts|entry_overflow|build_overflow)' \
      | grep -v csv
    echo
}

run_combo csr_f4_t256   4 256 0 1
run_combo csr_f4_t32    4 32  0 0
run_combo csr_f2_t256   2 256 0 0
run_combo csr_f1_t256   1 256 0 0
run_combo hash_f4_t256  4 256 1 0
run_combo hash_f4_t32   4 32  1 0
echo BIGCELL_PARITY_DONE
