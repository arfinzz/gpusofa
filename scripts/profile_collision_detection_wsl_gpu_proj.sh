#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD_DIR="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${SOFA_PROFILE_ROOT:-${REPO_DIR}/output/benchmark_logs/collision_profile_${TIMESTAMP}}"
STEPS="${SOFA_BENCHMARK_STEPS:-40}"
WARMUP="${SOFA_LARGE_WARMUP_STEPS:-5}"
REPORT_DIR="${SOFA_REPORT_DIR:-${REPO_DIR}/reports}"

mkdir -p "${PROFILE_ROOT}"
mkdir -p "${REPORT_DIR}"

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
SOFA_PLUGIN_REPOSITORIES="${SOFA_PLUGIN_LIB_PATHS%:}"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins${SOFA_PLUGIN_REPOSITORIES:+:${SOFA_PLUGIN_REPOSITORIES}}${SOFA_PLUGIN_PATH:+:${SOFA_PLUGIN_PATH}}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"
export SOFA_ROOT

echo "Profile root: ${PROFILE_ROOT}"
echo "SOFA root: ${SOFA_ROOT}"
echo "Repo: ${REPO_DIR}"

cmake -S "${REPO_DIR}/SofaGpuCollision" \
    -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${SOFA_ROOT}" \
    -DSOFAGPUCOLLISION_ENABLE_CUDA=ON
cmake --build "${BUILD_DIR}" -j"$(nproc)"

SOFA_GPU_COLLISION_LIB="${BUILD_DIR}/libSofaGpuCollision.so"
export SOFA_GPU_COLLISION_LIB

echo "Profiling plugin: ${SOFA_GPU_COLLISION_LIB}"
ldd "${SOFA_GPU_COLLISION_LIB}" > "${PROFILE_ROOT}/ldd_SofaGpuCollision.txt"
ldd "${SOFA_ROOT}/plugins/SofaCUDA/lib/libSofaCUDA.so" > "${PROFILE_ROOT}/ldd_SofaCUDA.txt"
if grep -q "not found" "${PROFILE_ROOT}/ldd_SofaGpuCollision.txt" "${PROFILE_ROOT}/ldd_SofaCUDA.txt"; then
    echo "Missing shared libraries detected. See ldd logs in ${PROFILE_ROOT}." >&2
    exit 1
fi

run_sofa() {
    "${SOFA_ROOT}/bin/runSofa" \
        -l SofaPython3 \
        -l SofaCUDA \
        -l "${SOFA_GPU_COLLISION_LIB}" \
        "$@"
}

run_sofa --help > "${PROFILE_ROOT}/runSofa_plugin_help.txt" 2>&1 || true

echo "Running correctness smoke with contacts copied to host..."
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/smoke" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=1 \
SOFA_DEDUPLICATE_PAIRS=1 \
run_sofa -g batch -n 5 "${REPO_DIR}/test_gpu_phase5_overlap_validation.py"

echo "Running one tissue / one blade detection-only profile..."
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/one_detection_only" \
SOFA_BENCHMARK_LABEL_SUFFIX="_one_detection_${TIMESTAMP}" \
SOFA_BENCHMARK_STEPS="${STEPS}" \
SOFA_LARGE_WARMUP_STEPS="${WARMUP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=1 \
bash "${SCRIPT_DIR}/run_one_tissue_one_blade_benchmarks_wsl.sh"
python3 "${SCRIPT_DIR}/analyze_benchmark_compute_only.py" "${PROFILE_ROOT}/one_detection_only" \
    --output "${PROFILE_ROOT}/one_detection_only/compute_summary.txt" \
    --markdown "${PROFILE_ROOT}/one_detection_only/profile_report.md" \
    --svg "${PROFILE_ROOT}/one_detection_only/stage_times.svg"

echo "Running one tissue / one blade contact-copy profile..."
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/one_contact_copy" \
SOFA_BENCHMARK_LABEL_SUFFIX="_one_contact_copy_${TIMESTAMP}" \
SOFA_BENCHMARK_STEPS="${STEPS}" \
SOFA_LARGE_WARMUP_STEPS="${WARMUP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=1 \
SOFA_DEDUPLICATE_PAIRS=1 \
bash "${SCRIPT_DIR}/run_one_tissue_one_blade_benchmarks_wsl.sh"
python3 "${SCRIPT_DIR}/analyze_benchmark_compute_only.py" "${PROFILE_ROOT}/one_contact_copy" \
    --output "${PROFILE_ROOT}/one_contact_copy/compute_summary.txt" \
    --markdown "${PROFILE_ROOT}/one_contact_copy/profile_report.md" \
    --svg "${PROFILE_ROOT}/one_contact_copy/stage_times.svg"

echo "Running large detection-only profile..."
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/large_detection_only" \
SOFA_BENCHMARK_LABEL_SUFFIX="_large_detection_${TIMESTAMP}" \
SOFA_BENCHMARK_STEPS="${STEPS}" \
SOFA_LARGE_WARMUP_STEPS="${WARMUP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=1 \
bash "${SCRIPT_DIR}/run_large_tissue_blade_compute_breakdown_wsl.sh"

echo "Running large dedupe-off profile..."
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/large_dedupe_off" \
SOFA_BENCHMARK_LABEL_SUFFIX="_large_dedupe_off_${TIMESTAMP}" \
SOFA_BENCHMARK_STEPS="${STEPS}" \
SOFA_LARGE_WARMUP_STEPS="${WARMUP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=0 \
bash "${SCRIPT_DIR}/run_large_tissue_blade_compute_breakdown_wsl.sh"

echo "Running grid-size sweep..."
SOFA_SCALE_LOG_DIR="${PROFILE_ROOT}/scene_size_sweep" \
SOFA_REPORT_DIR="${REPORT_DIR}" \
SOFA_BENCHMARK_STEPS="${STEPS}" \
SOFA_LARGE_WARMUP_STEPS="${WARMUP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=1 \
bash "${SCRIPT_DIR}/run_scene_size_scaling_benchmarks_wsl.sh"

echo "Capturing Nsight Systems and Nsight Compute profiles..."
SOFA_PROFILE_ROOT="${PROFILE_ROOT}" \
SOFA_GPU_COLLISION_BUILD_DIR="${BUILD_DIR}" \
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB}" \
bash "${SCRIPT_DIR}/run_nsight_collision_profile_wsl_gpu_proj.sh"
NSYS_STATUS="$(sed -n 's/^nsys_status=//p' "${PROFILE_ROOT}/nsight/nsight_manifest.txt" | tail -n 1)"
NCU_STATUS="$(sed -n 's/^ncu_status=//p' "${PROFILE_ROOT}/nsight/nsight_manifest.txt" | tail -n 1)"

{
    echo "profile_root=${PROFILE_ROOT}"
    echo "sofa_root=${SOFA_ROOT}"
    echo "profiling_plugin=${SOFA_GPU_COLLISION_LIB}"
    echo "nsys_status=${NSYS_STATUS}"
    echo "ncu_status=${NCU_STATUS}"
} > "${PROFILE_ROOT}/profile_run_manifest.txt"

echo "Collision profiling complete."
echo "Profile root: ${PROFILE_ROOT}"
