#!/usr/bin/env bash
# Same-session 4-way* broad-cull comparison on the large-tissue + large-tool scene.
# Runs three modes back-to-back (thermally fair), counter readback on so contacts
# can be checked identical:
#   1. dense_plain   : dense grid, Phase-15 tool-active-cell generation OFF
#   2. dense_phase15 : dense grid, Phase-15 ON   (the "optimised dense" path)
#   3. hash_opt      : optimised spatial-hash + prefix-sum broad cull
# (*The "earlier hash" 4th mode is a prior build; compare via its documented
#  kernel time in reports/hash_optimized_broadphase_20260617.md.)
#
# Usage (from WSL):  bash scripts/run_mode_comparison_ab_wsl.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
SCENE="${REPO_DIR}/testscenes/hash_prefixsum_large.py"
STEPS="${SOFA_BENCHMARK_STEPS:-160}"
BASE="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/mode_comparison_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
mkdir -p "${BASE}"

run_mode() {
    label="$1"; hashflag="$2"; phase15="$3"
    d="${BASE}/${label}"; mkdir -p "${d}"
    env SOFA_USE_HASH_PREFIXSUM_GENERATION="${hashflag}" \
        SOFA_USE_TOOL_ACTIVE_CELL_GENERATION="${phase15}" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${label}" \
        SOFA_BENCHMARK_LOG_DIR="${d}" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}" >"${d}/run.log" 2>&1
    echo "  ${label} done -> ${d}"
}

nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv,noheader || true
run_mode dense_plain   0 0
run_mode dense_phase15 0 1
run_mode hash_opt      1 1

echo
echo "=== SUMMARY (kernel time is the robust metric) ==="
for leg in dense_plain dense_phase15 hash_opt; do
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
