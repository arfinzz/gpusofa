#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD_DIR="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
SOFA_GPU_COLLISION_LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD_DIR}/libSofaGpuCollision.so}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${SOFA_PROFILE_ROOT:-${REPO_DIR}/output/benchmark_logs/collision_nsight_${TIMESTAMP}}"
NSYS_STEPS="${SOFA_NSYS_STEPS:-8}"
NSYS_FULL_STEPS="${SOFA_NSYS_FULL_STEPS:-4}"
NCU_STEPS="${SOFA_NCU_STEPS:-6}"
NCU_LAUNCH_COUNT="${SOFA_NCU_LAUNCH_COUNT:-40}"
NSIGHT_SYSTEMS_HOST_DIR="${NSIGHT_SYSTEMS_HOST_DIR:-/usr/lib/nsight-systems/host-linux-x64}"
NSIGHT_SYSTEMS_INSTALL_DIR="${NSIGHT_SYSTEMS_INSTALL_DIR:-/usr/lib/nsight-systems}"

mkdir -p "${PROFILE_ROOT}/nsight"

import_nsys_report() {
    local stem="$1"
    local importer="${NSIGHT_SYSTEMS_HOST_DIR}/QdstrmImporter"
    if [[ -f "${stem}.qdstrm" && -x "${importer}" ]]; then
        LD_LIBRARY_PATH="${NSIGHT_SYSTEMS_HOST_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${importer}" \
            -i "${stem}.qdstrm" \
            -o "${stem}.nsys-rep" \
            -f \
            > "${stem}_import_stdout.txt" 2> "${stem}_import_stderr.txt" || true
    fi
    if [[ -f "${stem}.nsys-rep" ]]; then
        nsys stats "${stem}.nsys-rep" \
            > "${stem}_stats_stdout.txt" 2> "${stem}_stats_stderr.txt" || true
    fi
}

SOFA_PLUGIN_LIB_PATHS="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf "%p:" 2>/dev/null || true)"
SOFA_PLUGIN_REPOSITORIES="${SOFA_PLUGIN_LIB_PATHS%:}"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins${SOFA_PLUGIN_REPOSITORIES:+:${SOFA_PLUGIN_REPOSITORIES}}${SOFA_PLUGIN_PATH:+:${SOFA_PLUGIN_PATH}}"
export LD_LIBRARY_PATH="${SOFA_ROOT}/lib:${NSIGHT_SYSTEMS_HOST_DIR}:${SOFA_PLUGIN_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_DATA_PATH="${SOFA_ROOT}/plugins/Sofa.Qt/share/sofa/gui/qt:${SOFA_ROOT}/share/sofa/qt${QT_DATA_PATH:+:${QT_DATA_PATH}}"
export QUADD_INSTALL_DIR="${NSIGHT_SYSTEMS_INSTALL_DIR}"
export SOFA_ROOT
export SOFA_GPU_COLLISION_LIB

if [[ ! -f "${SOFA_GPU_COLLISION_LIB}" ]]; then
    echo "Profiling plugin not found: ${SOFA_GPU_COLLISION_LIB}" >&2
    exit 1
fi

echo "Nsight profile root: ${PROFILE_ROOT}"
echo "Profiling plugin: ${SOFA_GPU_COLLISION_LIB}"

set +e
SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/nsight/nsys_logs" \
SOFA_BENCHMARK_LABEL_SUFFIX="_nsys_${TIMESTAMP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=1 \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=nvtx \
    --nvtx-capture='SOFA narrow phase' \
    --capture-range-end=repeat:4 \
    --force-overwrite=true \
    --output="${PROFILE_ROOT}/nsight/gpu_collision_nsys" \
    "${SOFA_ROOT}/bin/runSofa" \
        -l SofaPython3 \
        -l SofaCUDA \
        -l "${SOFA_GPU_COLLISION_LIB}" \
        -g batch -n "${NSYS_STEPS}" \
        "${REPO_DIR}/testscenes/one_tissue_one_blade.py" \
    > "${PROFILE_ROOT}/nsight/nsys_stdout.txt" 2> "${PROFILE_ROOT}/nsight/nsys_stderr.txt"
NSYS_STATUS=$?
import_nsys_report "${PROFILE_ROOT}/nsight/gpu_collision_nsys"

NSYS_FULL_STATUS=0
if [[ ! -f "${PROFILE_ROOT}/nsight/gpu_collision_nsys.nsys-rep" ]]; then
    SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/nsight/nsys_full_logs" \
    SOFA_BENCHMARK_LABEL_SUFFIX="_nsys_full_${TIMESTAMP}" \
    SOFA_GPU_DETAILED_PROFILING=1 \
    SOFA_COPY_CONTACTS_TO_HOST=0 \
    SOFA_DEDUPLICATE_PAIRS=1 \
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${PROFILE_ROOT}/nsight/gpu_collision_nsys_full" \
        "${SOFA_ROOT}/bin/runSofa" \
            -l SofaPython3 \
            -l SofaCUDA \
            -l "${SOFA_GPU_COLLISION_LIB}" \
            -g batch -n "${NSYS_FULL_STEPS}" \
            "${REPO_DIR}/testscenes/one_tissue_one_blade.py" \
        > "${PROFILE_ROOT}/nsight/nsys_full_stdout.txt" 2> "${PROFILE_ROOT}/nsight/nsys_full_stderr.txt"
    NSYS_FULL_STATUS=$?
fi

NSYS_LIGHT_STATUS=0
if [[ "${NSYS_STATUS}" != "0" ]]; then
    SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/nsight/nsys_light_logs" \
    SOFA_BENCHMARK_LABEL_SUFFIX="_nsys_light_${TIMESTAMP}" \
    SOFA_GPU_DETAILED_PROFILING=1 \
    SOFA_COPY_CONTACTS_TO_HOST=0 \
    SOFA_DEDUPLICATE_PAIRS=1 \
    nsys profile \
        --trace=cuda,nvtx \
        --capture-range=nvtx \
        --nvtx-capture='SOFA narrow phase' \
        --capture-range-end=repeat:4 \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${PROFILE_ROOT}/nsight/gpu_collision_nsys_light" \
        "${SOFA_ROOT}/bin/runSofa" \
            -l SofaPython3 \
            -l SofaCUDA \
            -l "${SOFA_GPU_COLLISION_LIB}" \
            -g batch -n "${NSYS_STEPS}" \
            "${REPO_DIR}/testscenes/one_tissue_one_blade.py" \
        > "${PROFILE_ROOT}/nsight/nsys_light_stdout.txt" 2> "${PROFILE_ROOT}/nsight/nsys_light_stderr.txt"
    NSYS_LIGHT_STATUS=$?
    import_nsys_report "${PROFILE_ROOT}/nsight/gpu_collision_nsys_light"
fi

SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/nsight/ncu_logs" \
SOFA_BENCHMARK_LABEL_SUFFIX="_ncu_${TIMESTAMP}" \
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_COPY_CONTACTS_TO_HOST=0 \
SOFA_DEDUPLICATE_PAIRS=1 \
ncu \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name 'regex:.*(resetDenseGridKernel|insertPackedTrianglesKernel|insertIndexedTrianglesKernel|insertPackedTrianglePairKernel|insertIndexedTrianglePairKernel|compactActiveDenseGridCellsKernel|generateDenseGridCandidatePairsKernel|generateDenseGridUniqueCandidatePairsKernel|generateActiveDenseGridCandidatePairsKernel|generateActiveDenseGridUniqueCandidatePairsKernel|exactDenseGridContactKernel|exactDenseGridIndexedContactKernel).*' \
    --launch-count "${NCU_LAUNCH_COUNT}" \
    --force-overwrite \
    --export "${PROFILE_ROOT}/nsight/gpu_collision_ncu" \
    "${SOFA_ROOT}/bin/runSofa" \
        -l SofaPython3 \
        -l SofaCUDA \
        -l "${SOFA_GPU_COLLISION_LIB}" \
        -g batch -n "${NCU_STEPS}" \
        "${REPO_DIR}/testscenes/one_tissue_one_blade.py" \
    > "${PROFILE_ROOT}/nsight/ncu_stdout.txt" 2> "${PROFILE_ROOT}/nsight/ncu_stderr.txt"
NCU_STATUS=$?

NCU_LAUNCH_STATUS=0
if [[ "${NCU_STATUS}" != "0" ]]; then
    SOFA_BENCHMARK_LOG_DIR="${PROFILE_ROOT}/nsight/ncu_launch_logs" \
    SOFA_BENCHMARK_LABEL_SUFFIX="_ncu_launch_${TIMESTAMP}" \
    SOFA_GPU_DETAILED_PROFILING=1 \
    SOFA_COPY_CONTACTS_TO_HOST=0 \
    SOFA_DEDUPLICATE_PAIRS=1 \
    ncu \
        --target-processes all \
        --kernel-name-base function \
        --kernel-name 'regex:.*(resetDenseGridKernel|insertPackedTrianglesKernel|insertIndexedTrianglesKernel|insertPackedTrianglePairKernel|insertIndexedTrianglePairKernel|compactActiveDenseGridCellsKernel|generateDenseGridCandidatePairsKernel|generateDenseGridUniqueCandidatePairsKernel|generateActiveDenseGridCandidatePairsKernel|generateActiveDenseGridUniqueCandidatePairsKernel|exactDenseGridContactKernel|exactDenseGridIndexedContactKernel).*' \
        --section LaunchStats \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --force-overwrite \
        --export "${PROFILE_ROOT}/nsight/gpu_collision_ncu_launchstats" \
        "${SOFA_ROOT}/bin/runSofa" \
            -l SofaPython3 \
            -l SofaCUDA \
            -l "${SOFA_GPU_COLLISION_LIB}" \
            -g batch -n "${NCU_STEPS}" \
            "${REPO_DIR}/testscenes/one_tissue_one_blade.py" \
        > "${PROFILE_ROOT}/nsight/ncu_launch_stdout.txt" 2> "${PROFILE_ROOT}/nsight/ncu_launch_stderr.txt"
    NCU_LAUNCH_STATUS=$?
fi
set -e

{
    echo "profile_root=${PROFILE_ROOT}"
    echo "sofa_root=${SOFA_ROOT}"
    echo "profiling_plugin=${SOFA_GPU_COLLISION_LIB}"
    echo "nsight_systems_install_dir=${NSIGHT_SYSTEMS_INSTALL_DIR}"
    echo "nsys_status=${NSYS_STATUS}"
    echo "nsys_full_status=${NSYS_FULL_STATUS}"
    echo "nsys_light_status=${NSYS_LIGHT_STATUS}"
    echo "ncu_status=${NCU_STATUS}"
    echo "ncu_launch_status=${NCU_LAUNCH_STATUS}"
    echo "nsys_importer=${NSIGHT_SYSTEMS_HOST_DIR}/QdstrmImporter"
    echo "nsys_importer_available=$([[ -x "${NSIGHT_SYSTEMS_HOST_DIR}/QdstrmImporter" ]] && echo 1 || echo 0)"
    echo "nsys_report=${PROFILE_ROOT}/nsight/gpu_collision_nsys.nsys-rep"
    echo "nsys_report_exists=$([[ -f "${PROFILE_ROOT}/nsight/gpu_collision_nsys.nsys-rep" ]] && echo 1 || echo 0)"
    echo "nsys_full_report=${PROFILE_ROOT}/nsight/gpu_collision_nsys_full.nsys-rep"
    echo "nsys_full_report_exists=$([[ -f "${PROFILE_ROOT}/nsight/gpu_collision_nsys_full.nsys-rep" ]] && echo 1 || echo 0)"
    echo "nsys_light_report=${PROFILE_ROOT}/nsight/gpu_collision_nsys_light.nsys-rep"
    echo "nsys_light_report_exists=$([[ -f "${PROFILE_ROOT}/nsight/gpu_collision_nsys_light.nsys-rep" ]] && echo 1 || echo 0)"
    echo "ncu_report=${PROFILE_ROOT}/nsight/gpu_collision_ncu.ncu-rep"
    echo "ncu_report_exists=$([[ -f "${PROFILE_ROOT}/nsight/gpu_collision_ncu.ncu-rep" ]] && echo 1 || echo 0)"
    echo "ncu_launch_report=${PROFILE_ROOT}/nsight/gpu_collision_ncu_launchstats.ncu-rep"
    echo "ncu_launch_report_exists=$([[ -f "${PROFILE_ROOT}/nsight/gpu_collision_ncu_launchstats.ncu-rep" ]] && echo 1 || echo 0)"
} > "${PROFILE_ROOT}/nsight/nsight_manifest.txt"

echo "nsys_status=${NSYS_STATUS}"
echo "nsys_full_status=${NSYS_FULL_STATUS}"
echo "nsys_light_status=${NSYS_LIGHT_STATUS}"
echo "ncu_status=${NCU_STATUS}"
echo "ncu_launch_status=${NCU_LAUNCH_STATUS}"
echo "Manifest: ${PROFILE_ROOT}/nsight/nsight_manifest.txt"
