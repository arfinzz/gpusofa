#!/usr/bin/env bash
# Sync the Windows working tree into the WSL repo copy, then build the plugin
# and the standalone backend bench. Resolves the spaced Windows mount path via a
# glob so no literal space is typed. Run from WSL:
#   cp /mnt/c/Users/arfin/Desktop/GPU*SOFA/scripts/sync_and_build_wsl.sh /home/arfin/_sb.sh && bash /home/arfin/_sb.sh
# The copy never deletes: a file removed or moved on Windows keeps its old copy in WSL.
set -uo pipefail

SRC="$(ls -d /mnt/c/Users/arfin/Desktop/GPU*SOFA)"
DST=/home/arfin/gpu-sofa
echo "SRC=${SRC}"
echo "DST=${DST}"

rsync -a "${SRC}/SofaGpuCollision/src/" "${DST}/SofaGpuCollision/src/"
# CMakeLists must sync too: adding a source file on Windows and forgetting this
# produced a GREEN build that silently omitted the new components (2026-07-15).
rsync -a "${SRC}/SofaGpuCollision/CMakeLists.txt" "${DST}/SofaGpuCollision/CMakeLists.txt"
rsync -a "${SRC}/testscenes/"           "${DST}/testscenes/"
rsync -a "${SRC}/scripts/"              "${DST}/scripts/"

echo "--- sync markers (expect non-zero) ---"
grep -c computeSimpleHashProximityContacts "${DST}/SofaGpuCollision/src/SofaGpuCollision/cuda/detail/SimpleHash.cuh"
grep -c useSimpleHashGeneration            "${DST}/SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp"
grep -c computeSimpleHashProximityContacts "${DST}/SofaGpuCollision/src/tools/DenseGridBackendBench.cpp"
grep -c CudaContactPenaltyForceField       "${DST}/SofaGpuCollision/CMakeLists.txt"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${DST}/SofaGpuCollision/build-profile"
# Optimised CPU code, and GPU code compiled for the GTX 1650 Ti itself (compute
# capability 7.5) instead of 5.2 code that the driver translates at load time.
BUILD_TYPE="${SOFA_GPU_BUILD_TYPE:-Release}"
CUDA_ARCH="${SOFA_GPU_CUDA_ARCH:-75}"

echo "=== CMAKE CONFIGURE (build type ${BUILD_TYPE}, CUDA arch ${CUDA_ARCH}) ==="
cmake -S "${DST}/SofaGpuCollision" -B "${BUILD}" \
    -DCMAKE_PREFIX_PATH="${SOFA_ROOT}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DSOFAGPUCOLLISION_ENABLE_CUDA=ON > /home/arfin/_cfg.log 2>&1
echo "configure_exit=$?"

echo "=== CMAKE BUILD (this compiles the big .cu; be patient) ==="
cmake --build "${BUILD}" -j"$(nproc)" > /home/arfin/_build.log 2>&1
echo "build_exit=$?"
echo "--- build log tail ---"
tail -25 /home/arfin/_build.log
echo "BUILD_SCRIPT_DONE"
