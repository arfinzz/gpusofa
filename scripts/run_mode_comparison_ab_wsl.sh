#!/usr/bin/env bash
# Same-session comparison of all 12 execution modes (the 6 broad-cull ways and
# their toggle combinations) on the 14,368-triangle scene hash_prefixsum_large.py.
# Legs run back-to-back (thermally fair) with counter readback on, so every leg's
# contact count can be checked identical; the summary ranks them by kernel time.
# Leg names and what each one means: README.md, "The 12 execution modes".
#
# Usage (from WSL):  bash scripts/run_mode_comparison_ab_wsl.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
SCENE="${REPO_DIR}/testscenes/collisiondetectiontests/hash_prefixsum_large.py"
STEPS="${SOFA_BENCHMARK_STEPS:-160}"
BASE="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/mode_comparison_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
mkdir -p "${BASE}"

run_mode() {
    label="$1"; hashflag="$2"; activecells="$3"; simplehash="${4:-0}"; sorted="${5:-0}"; cub="${6:-0}"; pairdedup="${7:-0}"; bigcell="${8:-0}"; sharedbuild="${9:-0}"; hashbuild="${10:-0}"
    d="${BASE}/${label}"; mkdir -p "${d}"
    env SOFA_USE_HASH_PREFIXSUM_GENERATION="${hashflag}" \
        SOFA_USE_SIMPLE_HASH_GENERATION="${simplehash}" \
        SOFA_USE_SORTED_GRID_GENERATION="${sorted}" \
        SOFA_SORTED_GRID_CUB_SORT="${cub}" \
        SOFA_SORTED_GRID_PAIRHASH_DEDUP="${pairdedup}" \
        SOFA_USE_BIGCELL_FUSED_GENERATION="${bigcell}" \
        SOFA_BIGCELL_SHARED_BUILD="${sharedbuild}" \
        SOFA_BIGCELL_HASH_BUILD="${hashbuild}" \
        SOFA_BIGCELL_HASH_SLOTS=2048 \
        SOFA_USE_TOOL_ACTIVE_CELL_GENERATION="${activecells}" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${label}" \
        SOFA_BENCHMARK_LOG_DIR="${d}" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}" >"${d}/run.log" 2>&1
    echo "  ${label} done -> ${d}"
}

nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv,noheader || true
run_mode dense_plain          0 0 0 0 0 0 0 0 0
run_mode dense_active         0 1 0 0 0 0 0 0 0
run_mode hash_opt             1 1 0 0 0 0 0 0 0
run_mode simple_hash          0 1 1 0 0 0 0 0 0
run_mode sorted_grid          0 1 0 1 0 0 0 0 0
run_mode sorted_cub           0 1 0 1 1 0 0 0 0
run_mode sorted_pairhash      0 1 0 1 0 1 0 0 0
run_mode sorted_cub_pairhash  0 1 0 1 1 1 0 0 0
run_mode bigcell_direct       0 1 0 0 0 0 1 0 0
run_mode bigcell_sharedhash   0 1 0 0 0 0 1 1 0
run_mode bigcell_sharedsort   0 1 0 0 0 0 1 2 0
run_mode bigcell_globalhash   0 1 0 0 0 0 1 0 1

echo
echo "=== SUMMARY (kernel time is the robust metric) ==="
for leg in dense_plain dense_active hash_opt simple_hash sorted_grid sorted_cub sorted_pairhash sorted_cub_pairhash bigcell_direct bigcell_sharedhash bigcell_sharedsort bigcell_globalhash; do
    f="$(ls "${BASE}/${leg}"/*summary*.txt 2>/dev/null | head -1)"
    [ -z "${f}" ] && { echo "${leg}: NO SUMMARY (see ${BASE}/${leg}/run.log)"; continue; }
    fps=$(grep -E '^avg_fps=' "${f}"|cut -d= -f2)
    nk=$(grep -E '^avg_narrow_kernel_ms=' "${f}"|cut -d= -f2)
    cc=$(grep -E '^avg_narrow_output_contact_count=' "${f}"|cut -d= -f2)
    vf=$(grep -E '^avg_narrow_vf_contact_count=' "${f}"|cut -d= -f2)
    fv=$(grep -E '^avg_narrow_fv_contact_count=' "${f}"|cut -d= -f2)
    ee=$(grep -E '^avg_narrow_ee_contact_count=' "${f}"|cut -d= -f2)
    ov=$(grep -E '^avg_narrow_overflow_count=' "${f}"|cut -d= -f2)
    kl=$(grep -E '^avg_kernel_launch_count=' "${f}"|cut -d= -f2)
    printf '%-14s fps=%-9s nkern=%-9s contacts=%-6s vffvee=%s/%s/%s ovf=%s launches=%s\n' \
        "${leg}" "${fps}" "${nk}" "${cc}" "${vf}" "${fv}" "${ee}" "${ov}" "${kl%.*}"
done
echo MODE_COMPARISON_DONE
