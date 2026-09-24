#!/usr/bin/env bash
# Run the tissue-poke scenes (testscenes/surgicalsimulationtests/) and print
# their summaries: peak force, indentation, relaxation during the hold, recovery.
# The CPU scene loads no GPU plugin; the GPU scene loads SofaCUDA and this
# plugin and falls back to the CPU for anything missing (it prints where each
# piece ran).
# Usage (from WSL): bash scripts/run_tissue_poke_wsl.sh [cpu|gpu|both] [frames]
#   frames defaults to one full poke (settle, press, hold, retract, rest).
# To watch a scene in SOFA's window instead (press Animate to start):
#   bash scripts/run_tissue_poke_wsl.sh view-cpu     (or view-gpu; extra arguments go to runSofa)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
SCENES="${REPO_DIR}/testscenes/surgicalsimulationtests"
WHICH="${1:-both}"
OUT="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/tissue_poke_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_ROOT
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export PYTHONPATH="${SOFA_ROOT}/plugins/SofaPython3/lib/python3/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}"

# Watch mode: SOFA's window. WSLg draws on the GPU only with GALLIUM_DRIVER=d3d12
# (otherwise it falls back to software rendering).
case "${WHICH}" in
    view-cpu|view-gpu)
        export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
        scene="${WHICH#view-}"
        extra=()
        [ "${scene}" = gpu ] && extra=(-l SofaCUDA -l "${LIB}")
        echo "logs=${OUT}"
        # Any further arguments go to runSofa (for example: -g imgui).
        exec env SOFA_BENCHMARK_LOG_DIR="${OUT}" "${SOFA_ROOT}/bin/runSofa" \
            -l SofaPython3 "${extra[@]}" "${@:2}" "${SCENES}/tissue_poke_${scene}.py"
        ;;
esac

FRAMES="${2:-$(cd "${SCENES}" && python3 -c 'import poke_common as pc; print(pc.total_steps())' 2>/dev/null < /dev/null | tail -1)}"
echo "frames=${FRAMES}  logs=${OUT}"

run_scene() {
    local which="$1"; shift
    local log="${OUT}/tissue_poke_${which}.log"
    echo
    echo "=== ${which^^} scene ==="
    env SOFA_BENCHMARK_LOG_DIR="${OUT}" "${SOFA_ROOT}/bin/runSofa" -g batch -n "${FRAMES}" \
        -l SofaPython3 "$@" "${SCENES}/tissue_poke_${which}.py" > "${log}" 2>&1 < /dev/null
    echo "exit=$?"
    grep -E 'TissuePoke|placement' "${log}" | head -3
    grep -E '\[ERROR\]|Traceback|Error:' "${log}" | head -8
    grep -E '\[WARNING\]' "${log}" | grep -viE 'deprecat|RegisterObject' | sort | uniq -c | sort -rn | head -6
    if [ -f "${OUT}/tissue_poke_${which}_summary.txt" ]; then
        cat "${OUT}/tissue_poke_${which}_summary.txt"
    else
        echo "(no summary: the run did not reach the end; see ${log})"
    fi
}

case "${WHICH}" in
    cpu)  run_scene cpu ;;
    gpu)  run_scene gpu -l SofaCUDA -l "${LIB}" ;;
    both) run_scene cpu; run_scene gpu -l SofaCUDA -l "${LIB}" ;;
    *)    echo "usage: $0 [cpu|gpu|both] [frames]   or   $0 view-cpu|view-gpu"; exit 2 ;;
esac
echo "OUT=${OUT}"
echo TISSUE_POKE_DONE
