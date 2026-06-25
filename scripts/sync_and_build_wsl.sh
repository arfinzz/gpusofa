#!/usr/bin/env bash
# Sync the Windows working tree into the WSL repo copy, then build the plugin
# and the standalone backend bench. Resolves the spaced Windows mount path via a
# glob so no literal space is typed. Run from WSL:
#   cp /mnt/c/Users/arfin/Desktop/GPU*SOFA/scripts/_wsl_sync_build.sh /home/arfin/_sb.sh && bash /home/arfin/_sb.sh
set -uo pipefail

SRC="$(ls -d /mnt/c/Users/arfin/Desktop/GPU*SOFA)"
DST=/home/arfin/gpu-sofa
echo "SRC=${SRC}"
echo "DST=${DST}"

rsync -a "${SRC}/SofaGpuCollision/src/" "${DST}/SofaGpuCollision/src/"
rsync -a "${SRC}/testscenes/"           "${DST}/testscenes/"
rsync -a "${SRC}/scripts/"              "${DST}/scripts/"

echo "--- sync markers (expect non-zero) ---"
grep -c computeSimpleHashProximityContacts "${DST}/SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu"
grep -c useSimpleHashGeneration            "${DST}/SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp"
grep -c computeSimpleHashProximityContacts "${DST}/SofaGpuCollision/src/tools/DenseGridBackendBench.cpp"

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${DST}/SofaGpuCollision/build-profile"

echo "=== CMAKE CONFIGURE ==="
cmake -S "${DST}/SofaGpuCollision" -B "${BUILD}" \
    -DCMAKE_PREFIX_PATH="${SOFA_ROOT}" \
    -DSOFAGPUCOLLISION_ENABLE_CUDA=ON > /home/arfin/_cfg.log 2>&1
echo "configure_exit=$?"

echo "=== CMAKE BUILD (this compiles the big .cu; be patient) ==="
cmake --build "${BUILD}" -j"$(nproc)" > /home/arfin/_build.log 2>&1
echo "build_exit=$?"
echo "--- build log tail ---"
tail -25 /home/arfin/_build.log
echo "BUILD_SCRIPT_DONE"
