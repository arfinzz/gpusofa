#!/usr/bin/env bash
# Run the fully GPU-resident FEM + contact scene (Tier 1).
# Prints the residency verdict (Gate 5), contact/force diagnostics and timing.
# Usage (from WSL): bash scripts/run_gpu_resident_scene_wsl.sh [steps]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${BUILD}/libSofaGpuCollision.so"
SCENE="${REPO_DIR}/testscenes/gpu_resident_fem_contact.py"
STEPS="${1:-${SOFA_BENCHMARK_STEPS:-60}}"
OUT="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/gpu_resident_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="${BUILD}:${SOFA_ROOT}/lib:${PLP}"
mkdir -p "${OUT}"

echo "scene=${SCENE}"
echo "steps=${STEPS}"
env SOFA_BENCHMARK_LOG_DIR="${OUT}" \
    "${SOFA_ROOT}/bin/runSofa" -g batch -n "${STEPS}" \
        -l SofaPython3 -l SofaCUDA -l "${LIB}" "${SCENE}" > "${OUT}/run.log" 2>&1
echo "exit=$?"

echo
echo "=== GATE 5: residency verdict ==="
if grep -qi 'DEVICE->HOST TRANSFER DETECTED' "${OUT}/run.log"; then
    grep -i -A2 'DEVICE->HOST TRANSFER DETECTED' "${OUT}/run.log" | head -12
else
    echo "PASS — no device-to-host transfer of x/v/f detected."
fi
grep -i 'GPU residency @frame' "${OUT}/run.log" | tail -4

echo
echo "=== errors / warnings ==="
grep -iE '\[ERROR\]|\[WARNING\]' "${OUT}/run.log" | grep -viE 'deprecat|RegisterObject' | head -12

echo
echo "=== simulation ran? ==="
grep -iE 'iterations done|FPS' "${OUT}/run.log" | tail -3

echo
echo "=== summary ==="
S="$(ls "${OUT}"/*summary*.txt 2>/dev/null | head -1)"
if [ -n "${S}" ]; then
    grep -E '^(avg_fps|avg_narrow_kernel_ms|avg_narrow_wall_ms|avg_narrow_output_contact_count|collision_element_count)=' "${S}"
else
    echo "(no summary written — see ${OUT}/run.log)"
fi
echo "OUT=${OUT}"
echo GPU_RESIDENT_RUN_DONE
