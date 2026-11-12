#!/usr/bin/env bash
# Hard-poke stress tests (testscenes/contactstresstests/): each scene run end to end
# once per kind of clock, while nvidia-smi logs the GPU's clocks, temperature and memory:
#   plain             no timers: the real time per frame (frame_ms in <label>.csv)
#   gpuevents         CUDA events around every GPU stage (tissue and contact) and around
#                     the collision; the events make the CPU wait at each stage's end
#   sofatimer         SOFA's own timer, every stage of every step; SofaCUDA makes it wait
#                     for the GPU at each stage boundary, so the GPU's time lands in its stage
#   sofatimer_nosync  the same with CUDA_TIMER_SYNC=0: where the CPU really waits
# Then scripts/hard_poke_report.py writes report.md (every number, per phase and at the
# step with the most contacts) and steps_merged.csv (one row per step, all clocks).
# Usage (from WSL): bash scripts/run_hard_poke_wsl.sh [diaphragm|cuboid|both] [gpu|cpu]
#   cpu runs SOFA's own CPU components (plain and sofatimer only; minutes per run).
# Scene settings can be changed with SOFA_HARD_POKE_* (testscenes/contactstresstests/hard_poke_common.py).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
SCENES="${REPO_DIR}/testscenes/contactstresstests"
WHICH="${1:-both}"
SIDE="${2:-gpu}"
OUT="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/hard_poke_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_ROOT
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export PYTHONPATH="${SOFA_ROOT}/plugins/SofaPython3/lib/python3/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}"
echo "logs=${OUT}"

# ---- the machine, once ----------------------------------------------------------
{
    echo "date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "kernel=$(uname -r)"
    echo "os=$(. /etc/os-release && echo "${PRETTY_NAME}")"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "cpu_threads_visible=$(nproc)"
    echo "memory_total_gb=$(awk '/MemTotal/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader)"
    echo "cuda_toolkit=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
    echo "sofa=${SOFA_ROOT}"
    echo "plugin=${LIB} ($(stat -c '%y' "${LIB}" 2>/dev/null | cut -d. -f1))"
    echo "side=${SIDE}"
} > "${OUT}/machine.txt"
cat "${OUT}/machine.txt"

# ---- GPU clocks, temperature and memory, every 200 ms, for the whole session -----
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -lms 200 > "${OUT}/gpu_monitor.csv" 2>/dev/null &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null' EXIT
echo "run,start,end,exit,wall_s" > "${OUT}/run_marks.csv"

run_kind() {
    local scene="$1" kind="$2"; shift 2
    local dir="${OUT}/${scene}/${kind}"
    mkdir -p "${dir}"
    local frames
    frames="$(cd "${SCENES}" && python3 -c "import hard_poke_common as hp; print(hp.total_steps('${scene}'))" 2>/dev/null < /dev/null | tail -1)"
    local extra=() timer=()
    [ "${SIDE}" = gpu ] && extra=(-l SofaCUDA -l "${LIB}")
    case "${kind}" in sofatimer*) timer=(--computationTimeSampling 1 -o stdout) ;; esac
    local start t0 t1 code
    start="$(date '+%Y/%m/%d %H:%M:%S.%3N')"
    t0="$(date +%s.%N)"
    env SOFA_BENCHMARK_LOG_DIR="${dir}" SOFA_VALIDATION_SIDE="${SIDE}" SOFA_HARD_POKE_VISUAL=0 "$@" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "${frames}" "${timer[@]}" -l SofaPython3 "${extra[@]}" \
        "${SCENES}/${scene}_poke.py" > "${dir}/run.log" 2>&1 < /dev/null
    code=$?
    t1="$(date +%s.%N)"
    echo "${scene}/${kind},${start},$(date '+%Y/%m/%d %H:%M:%S.%3N'),${code},$(echo "${t1} - ${t0}" | bc)" >> "${OUT}/run_marks.csv"
    echo "${scene}/${kind}: frames=${frames} exit=${code} wall=$(echo "${t1} - ${t0}" | bc) s"
    grep -E 'HardPoke |HARD_POKE_SUMMARY' "${dir}/run.log" | head -2
    grep -E '\[ERROR\]|Traceback|Error:' "${dir}/run.log" | head -5
}

run_scene() {
    local scene="$1"
    echo
    echo "=== ${scene} (${SIDE}) ==="
    run_kind "${scene}" plain
    if [ "${SIDE}" = gpu ]; then
        run_kind "${scene}" gpuevents SOFA_VALIDATION_MEASURE_TIMES=1 SOFA_HARD_POKE_COLLISION_TIMES=1
        run_kind "${scene}" sofatimer
        run_kind "${scene}" sofatimer_nosync CUDA_TIMER_SYNC=0
    else
        run_kind "${scene}" sofatimer
    fi
}

case "${WHICH}" in
    diaphragm|cuboid) run_scene "${WHICH}" ;;
    both) run_scene diaphragm; run_scene cuboid ;;
    *) echo "usage: $0 [diaphragm|cuboid|both] [gpu|cpu]"; exit 2 ;;
esac

kill "${MONITOR_PID}" 2>/dev/null
python3 "${SCRIPT_DIR}/hard_poke_report.py" "${OUT}"
echo "OUT=${OUT}"
echo HARD_POKE_RUN_DONE
