#!/usr/bin/env bash
# Nsight Compute capture targeting the Phase 11 (FBP tri-tri) and Phase 12
# (v-t) kernels specifically. Runs three scenes back to back:
#   1. tri-tri FBP one-tissue/one-blade
#   2. v-t self-collision two-layer slab
#   3. v-t cross-model tissue + tool cloud
# Each gets its own .ncu-rep and a CSV exported with the metric set that
# answers "is this kernel compute-, memory-, or atomic-bound on the
# GTX 1650 Ti".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD_DIR="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD_DIR}/libSofaGpuCollision.so}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${SOFA_PROFILE_ROOT:-${REPO_DIR}/output/benchmark_logs/fbp_nsight_${TIMESTAMP}}"
NCU_STEPS="${SOFA_NCU_STEPS:-4}"
NCU_LAUNCH_COUNT="${SOFA_NCU_LAUNCH_COUNT:-40}"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${SOFA_PLUGIN_LIB_PATHS%:}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"

if [[ ! -f "${SOFA_GPU_COLLISION_LIB}" ]]; then
    echo "Profiling plugin not found: ${SOFA_GPU_COLLISION_LIB}" >&2
    exit 1
fi

mkdir -p "${PROFILE_ROOT}"
echo "Nsight FBP profile root: ${PROFILE_ROOT}"

# Metric set focused on "what bounds this kernel?" for the GTX 1650 Ti.
METRICS='gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__registers_per_thread,launch__block_size,sm__inst_executed.avg.pct_of_peak_sustained_elapsed'

run_scene() {
    local label="$1"
    local scene_py="$2"
    shift 2
    local outdir="${PROFILE_ROOT}/${label}"
    mkdir -p "${outdir}"
    local ncu_rep="${outdir}/profile.ncu-rep"
    local stdout_log="${outdir}/ncu_stdout.txt"
    local stderr_log="${outdir}/ncu_stderr.txt"
    local csv="${outdir}/profile.csv"

    echo
    echo "=== Profiling ${label} ==="
    echo "Scene: ${scene_py}"
    echo "Report: ${ncu_rep}"

    # Run NCU. Combine base env + caller-supplied scene env vars via `env`.
    env \
        SOFA_GPU_DETAILED_PROFILING=0 \
        SOFA_COPY_CONTACTS_TO_HOST=0 \
        SOFA_DEDUPLICATE_PAIRS=1 \
        SOFA_USE_GPU_HASH_DEDUPE=1 \
        SOFA_USE_INDEXED_DENSE_GRID_INPUT=1 \
        SOFA_USE_DIRECT_DEVICE_POSITIONS=1 \
        "$@" \
        ncu --launch-skip 0 --launch-count "${NCU_LAUNCH_COUNT}" \
            --metrics "${METRICS}" \
            --target-processes all \
            --force-overwrite \
            --export "${ncu_rep%.*}" \
            "${SOFA_ROOT}/bin/runSofa" \
                -l SofaPython3 \
                -l SofaCUDA \
                -l "${SOFA_GPU_COLLISION_LIB}" \
                -g batch -n "${NCU_STEPS}" \
                "${scene_py}" \
        > "${stdout_log}" 2> "${stderr_log}" || echo "NCU returned non-zero for ${label}"

    if [[ -f "${ncu_rep}" ]]; then
        ncu --import "${ncu_rep}" --csv --page raw > "${csv}" 2>> "${stderr_log}" || true
        echo "CSV: ${csv}"
    else
        echo "WARNING: ${ncu_rep} was not produced. See ${stderr_log}."
    fi
}

# --- Run 1: tri-tri FBP ---
run_scene "tri_tri_fbp" "${REPO_DIR}/test_gpu_one_tissue_one_blade_dense_grid_benchmark.py" \
    SOFA_USE_FEATURE_BASED_PROXIMITY=1 \
    SOFA_USE_VERTEX_TRIANGLE_PROXIMITY=0 \
    SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

# --- Run 2: v-t self-collision ---
run_scene "vt_self_collision" "${REPO_DIR}/test_gpu_self_collision_vertex_triangle_smoke.py" \
    SOFA_USE_FEATURE_BASED_PROXIMITY=1 \
    SOFA_USE_VERTEX_TRIANGLE_PROXIMITY=1 \
    SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

# --- Run 3: v-t cross-model ---
run_scene "vt_cross_model" "${REPO_DIR}/test_gpu_cross_model_vertex_triangle_smoke.py" \
    SOFA_USE_FEATURE_BASED_PROXIMITY=1 \
    SOFA_USE_VERTEX_TRIANGLE_PROXIMITY=1 \
    SOFA_PROXIMITY_READ_CONTACT_COUNTER=1

echo
echo "All scenes profiled. Reports under ${PROFILE_ROOT}/"
