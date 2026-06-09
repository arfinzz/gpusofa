#!/usr/bin/env bash
# Tiny-scene dense-vs-hash A/B: small tissue (~800 tris) + tiny tool (~24 tris).
# This is the regime where the spatial-hash + prefix-sum broad cull is LEAST
# likely to pay off (few occupied cells, so the prefix-sum/scan overhead is a
# bigger fraction of the work). Completes the small<->large picture.
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
export SOFA_HASH_TISSUE_NX=21
export SOFA_HASH_TISSUE_NZ=21
export SOFA_HASH_BLADE_SEGMENTS_X=2
export SOFA_HASH_BLADE_SEGMENTS_Y=2
export SOFA_HASH_BLADE_SEGMENTS_Z=1

mkdir -p "${BASE}"

run_leg() {
    label="$1"; hashflag="$2"
    d="${BASE}/${label}"
    mkdir -p "${d}"
    echo "=== leg ${label}: hash=${hashflag} (tiny tissue 21x21, tiny tool 2x2x1) ==="
    env SOFA_USE_HASH_PREFIXSUM_GENERATION="${hashflag}" \
        SOFA_BENCHMARK_LABEL_SUFFIX="_${label}" \
        SOFA_BENCHMARK_LOG_DIR="${d}" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
            -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}" \
            >"${d}/run.log" 2>&1
    echo "    done -> ${d}"
}

nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv,noheader || true
run_leg tiny_dense 0
run_leg tiny_hash  1

echo
echo "=== TINY SUMMARIES ==="
for leg in tiny_dense tiny_hash; do
    f="$(ls "${BASE}/${leg}"/*summary*.txt 2>/dev/null | head -1)"
    if [ -z "${f}" ]; then echo "${leg}: NO SUMMARY -- see ${BASE}/${leg}/run.log"; continue; fi
    fps=$(grep -E '^avg_fps=' "${f}" | cut -d= -f2)
    nw=$(grep -E '^avg_narrow_wall_ms=' "${f}" | cut -d= -f2)
    nk=$(grep -E '^avg_narrow_kernel_ms=' "${f}" | cut -d= -f2)
    cc=$(grep -E '^avg_narrow_output_contact_count=' "${f}" | cut -d= -f2)
    vf=$(grep -E '^avg_narrow_vf_contact_count=' "${f}" | cut -d= -f2)
    fv=$(grep -E '^avg_narrow_fv_contact_count=' "${f}" | cut -d= -f2)
    ee=$(grep -E '^avg_narrow_ee_contact_count=' "${f}" | cut -d= -f2)
    ov=$(grep -E '^avg_narrow_overflow_count=' "${f}" | cut -d= -f2)
    kl=$(grep -E '^avg_kernel_launch_count=' "${f}" | cut -d= -f2)
    printf '%-11s fps=%-9s nwall=%-9s nkern=%-9s contacts=%-5s vffvee=%s/%s/%s ovf=%s launches=%s\n' \
        "${leg}" "${fps}" "${nw}" "${nk}" "${cc}" "${vf}" "${fv}" "${ee}" "${ov}" "${kl}"
done
echo TINY_DONE
