#!/usr/bin/env bash
# Branch comparison A/B: dense (feature-branch behavior) vs hash+prefix-sum
# (experiment-branch addition) on BOTH a small-tool scene and a large-tool scene.
#
# Same binary, same FBP narrow kernel; only the broad cull differs, toggled by
# SOFA_USE_HASH_PREFIXSUM_GENERATION. Contact counts MUST match dense vs hash
# within each scene. The point of running BOTH scenes is to show the honest
# regime split:
#   * small tool / large tissue  -> Phase 15 dense path is favored
#   * large tool / large tissue  -> hash + prefix-sum is favored
#
# Usage (from WSL):  bash scripts/run_branch_comparison_ab_wsl.sh
set -uo pipefail

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
REPO="${REPO:-/home/arfin/gpu-sofa}"
LIB="${SOFA_GPU_COLLISION_LIB:-${REPO}/SofaGpuCollision/build-profile/libSofaGpuCollision.so}"
SCENE="${REPO}/test_gpu_hash_prefixsum_large.py"
STEPS="${SOFA_BENCHMARK_STEPS:-160}"
BASE="${REPO}/output/benchmark_logs/branch_cmp_20260609"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${PLP}"
export SOFA_PROXIMITY_READ_CONTACT_COUNTER=1
export SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1

mkdir -p "${BASE}"

run_leg() {
    label="$1"; hashflag="$2"; bx="$3"; by="$4"; bz="$5"
    d="${BASE}/${label}"
    mkdir -p "${d}"
    echo "=== leg ${label}: hash=${hashflag} blade=${bx}x${by}x${bz} ==="
    env SOFA_USE_HASH_PREFIXSUM_GENERATION="${hashflag}" \
        SOFA_HASH_BLADE_SEGMENTS_X="${bx}" \
        SOFA_HASH_BLADE_SEGMENTS_Y="${by}" \
        SOFA_HASH_BLADE_SEGMENTS_Z="${bz}" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${label}" \
        SOFA_BENCHMARK_LOG_DIR="${d}" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}" \
            >"${d}/run.log" 2>&1
    echo "    done -> ${d}"
}

nvidia-smi --query-gpu=temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader || true

# Small tool (tiny blade ~24 tris) vs large tissue: Phase-15 favorable regime.
run_leg small_dense 0 2 2 1
run_leg small_hash  1 2 2 1
# Large tool (full subdivided blade ~1568 tris) vs large tissue: hash favorable.
run_leg large_dense 0 30 8 4
run_leg large_hash  1 30 8 4

echo
echo "=== SUMMARIES ==="
for leg in small_dense small_hash large_dense large_hash; do
    f="$(ls "${BASE}/${leg}"/*summary*.txt 2>/dev/null | head -1)"
    [ -z "${f}" ] && { echo "${leg}: NO SUMMARY"; continue; }
    fps=$(grep -E '^avg_fps=' "${f}" | cut -d= -f2)
    nw=$(grep -E '^avg_narrow_wall_ms=' "${f}" | cut -d= -f2)
    nk=$(grep -E '^avg_narrow_kernel_ms=' "${f}" | cut -d= -f2)
    cc=$(grep -E '^avg_narrow_output_contact_count=' "${f}" | cut -d= -f2)
    vf=$(grep -E '^avg_narrow_vf_contact_count=' "${f}" | cut -d= -f2)
    fv=$(grep -E '^avg_narrow_fv_contact_count=' "${f}" | cut -d= -f2)
    ee=$(grep -E '^avg_narrow_ee_contact_count=' "${f}" | cut -d= -f2)
    rc=$(grep -E '^avg_raw_candidate_count=' "${f}" | cut -d= -f2)
    uc=$(grep -E '^avg_unique_candidate_count=' "${f}" | cut -d= -f2)
    ov=$(grep -E '^avg_narrow_overflow_count=' "${f}" | cut -d= -f2)
    kl=$(grep -E '^avg_kernel_launch_count=' "${f}" | cut -d= -f2)
    printf '%-12s fps=%-9s nwall=%-9s nkern=%-9s contacts=%-6s vf/fv/ee=%s/%s/%s raw=%s uniq=%s ovf=%s launches=%s\n' \
        "${leg}" "${fps}" "${nw}" "${nk}" "${cc}" "${vf}" "${fv}" "${ee}" "${rc}" "${uc}" "${ov}" "${kl}"
done
echo BRANCH_CMP_DONE
