#!/usr/bin/env bash
# Sorted-grid ("5th way") toggle parity: runs the standalone backend bench for
# all four sort-engine x dedup combos and prints the contact-parity lines. Every
# combo must report the same contacts as the dense FBP leg (8018 on the default
# bench geometry). The cub combos may print the frame-0 fallback notice on
# systems where CUB DeviceRadixSort is unreliable (see
# reports/five_way_broadcull_comparison_20260703.md) — that is the safety net
# working, not a failure.
# Usage (from WSL): bash scripts/run_sorted_grid_parity_wsl.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
B="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null)"
export LD_LIBRARY_PATH="${B}:${SOFA_ROOT}/lib:${PLP}"

run_combo() {
    local label="$1" cubflag="$2" pairflag="$3" runothers="$4"
    echo "=== SORTED PARITY: ${label} (cub=${cubflag} pairhash=${pairflag}) ==="
    SOFA_BACKEND_BENCH_CSV="/tmp/sorted_parity_${label}.csv" \
    SOFA_GPU_DETAILED_PROFILING=0 \
    SOFA_BACKEND_BENCH_RUN_HASH="${runothers}" \
    SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH="${runothers}" \
    SOFA_BACKEND_BENCH_SORTED_CUB="${cubflag}" \
    SOFA_BACKEND_BENCH_SORTED_PAIRHASH="${pairflag}" \
    SOFA_SORTED_GRID_VERIFY=1 \
    "${B}/SofaGpuCollisionDenseGridBackendBench" 2>&1 \
      | grep -iE 'fbp_contacts|hash_contacts=|simplehash_contacts|sortedgrid_(engine|dedup|gpu_kernel|unique|contacts|incidence_overflow|verify)|falling back' \
      | grep -v csv
    echo
}

run_combo counting_homecell 0 0 1
run_combo cub_homecell      1 0 0
run_combo counting_pairhash 0 1 0
run_combo cub_pairhash      1 1 0
echo SORTED_PARITY_DONE
