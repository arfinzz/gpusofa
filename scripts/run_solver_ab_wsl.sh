#!/usr/bin/env bash
# A/B of the GPU contact solver (GpuContactConstraintSolver): the old method against
# its three new options and their combinations.
#   A    the old method: one GPU block, every solve from zero, contacts on both sides
#   W    warmStart=true            each solve starts from the previous step's multipliers
#   T    solverKernel=tiled        the same Gauss-Seidel on the whole GPU
#   D    contactSides=deformable   contacts on the tissue's vertices only
#   WT, WTD                        combinations (WT is the default since 2026-11-12)
# In order:
#   1. the kernel checks (SofaGpuCollisionConstraintChecks: C1-C4, against SOFA's own solver)
#   2. the hard pokes (diaphragm, cuboid) and the tissue poke, each variant:
#        plain    no timers, twice (the second round in reverse order, against heat drift)
#        events   CUDA events per stage
#        verify   every 10th solve checked on the CPU in double (A, W, T, D, WTD)
#   3. the contact validation tests (incline, plate, grasp, cutting): SOFA's CPU side once,
#      the GPU side per variant (A, W, T, D, WTD), each against SOFA and its known answer
#   4. scripts/solver_ab_report.py: report.md
# nvidia-smi logs the GPU every 200 ms throughout.
# Usage (from WSL): bash scripts/run_solver_ab_wsl.sh [log folder]   (about an hour)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/install/v25.12}"
BUILD="${SOFA_GPU_COLLISION_BUILD_DIR:-${REPO_DIR}/SofaGpuCollision/build-profile}"
LIB="${SOFA_GPU_COLLISION_LIB:-${BUILD}/libSofaGpuCollision.so}"
OUT="${1:-${REPO_DIR}/output/benchmark_logs/solver_ab_$(date +%Y%m%d_%H%M%S)}"

PLP="$(find "${SOFA_ROOT}/plugins" -type d -name lib -printf '%p:' 2>/dev/null || true)"
export SOFA_ROOT
export SOFA_PLUGIN_PATH="${SOFA_ROOT}/lib:${SOFA_ROOT}/plugins:${PLP%:}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${BUILD}:${SOFA_ROOT}/lib:${PLP}"
export PYTHONPATH="${SOFA_ROOT}/plugins/SofaPython3/lib/python3/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}"
echo "logs=${OUT}"

VARIANTS=(A W T D WT WTD)
VERIFIED=(A W T D WTD)
variant_env() {
    case "$1" in
        A)   echo "SOFA_CONTACT_WARM_START=0 SOFA_CONTACT_SOLVER_KERNEL=oneBlock" ;;
        W)   echo "SOFA_CONTACT_WARM_START=1 SOFA_CONTACT_SOLVER_KERNEL=oneBlock" ;;
        T)   echo "SOFA_CONTACT_WARM_START=0 SOFA_CONTACT_SOLVER_KERNEL=tiled" ;;
        D)   echo "SOFA_CONTACT_WARM_START=0 SOFA_CONTACT_SOLVER_KERNEL=oneBlock SOFA_CONTACT_SIDES=deformable" ;;
        WT)  echo "SOFA_CONTACT_WARM_START=1 SOFA_CONTACT_SOLVER_KERNEL=tiled" ;;
        WTD) echo "SOFA_CONTACT_WARM_START=1 SOFA_CONTACT_SOLVER_KERNEL=tiled SOFA_CONTACT_SIDES=deformable" ;;
    esac
}

{
    echo "date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    echo "plugin=${LIB} ($(stat -c '%y' "${LIB}" 2>/dev/null | cut -d. -f1))"
} > "${OUT}/machine.txt"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -lms 200 > "${OUT}/gpu_monitor.csv" 2>/dev/null &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null' EXIT
echo "run,start,end,exit,wall_s" > "${OUT}/run_marks.csv"

# ---- 1. kernel checks ------------------------------------------------------------
"${BUILD}/SofaGpuCollisionConstraintChecks" > "${OUT}/constraint_checks.txt" 2>&1
echo "constraint checks: $(grep -E '^CONSTRAINT_CHECKS' "${OUT}/constraint_checks.txt")"

# ---- 2. scenes ---------------------------------------------------------------------
HARD="${REPO_DIR}/testscenes/contactstresstests"
POKE="${REPO_DIR}/testscenes/surgicalsimulationtests"
frames_of() {
    case "$1" in
        poke) (cd "${POKE}" && python3 -c 'import poke_common as pc; print(pc.total_steps())' 2>/dev/null < /dev/null | tail -1) ;;
        *)    (cd "${HARD}" && python3 -c "import hard_poke_common as hp; print(hp.total_steps('$1'))" 2>/dev/null < /dev/null | tail -1) ;;
    esac
}
scene_file() { case "$1" in poke) echo "${POKE}/tissue_poke_gpu.py" ;; *) echo "${HARD}/$1_poke.py" ;; esac; }

run_scene() {   # scene variant kind [extra env...]
    local scene="$1" variant="$2" kind="$3"; shift 3
    local dir="${OUT}/${scene}/${variant}/${kind}"
    mkdir -p "${dir}"
    local start t0 t1 code
    start="$(date '+%Y/%m/%d %H:%M:%S.%3N')"
    t0="$(date +%s.%N)"
    # shellcheck disable=SC2046
    env SOFA_BENCHMARK_LOG_DIR="${dir}" SOFA_HARD_POKE_VISUAL=0 SOFA_POKE_VISUAL=0 $(variant_env "${variant}") "$@" \
        "${SOFA_ROOT}/bin/runSofa" -g batch -n "$(frames_of "${scene}")" -l SofaPython3 -l SofaCUDA -l "${LIB}" \
        "$(scene_file "${scene}")" > "${dir}/run.log" 2>&1 < /dev/null
    code=$?
    t1="$(date +%s.%N)"
    echo "${scene}/${variant}/${kind},${start},$(date '+%Y/%m/%d %H:%M:%S.%3N'),${code},$(echo "${t1} - ${t0}" | bc)" >> "${OUT}/run_marks.csv"
    echo "${scene}/${variant}/${kind}: exit=${code} wall=$(echo "${t1} - ${t0}" | bc) s"
    grep -E '\[ERROR\]|Traceback' "${dir}/run.log" | head -3
}

for scene in diaphragm cuboid poke; do
    for v in "${VARIANTS[@]}"; do run_scene "${scene}" "${v}" plain1; done
    for (( i=${#VARIANTS[@]}-1; i>=0; i-- )); do run_scene "${scene}" "${VARIANTS[$i]}" plain2; done
    for v in "${VARIANTS[@]}"; do
        run_scene "${scene}" "${v}" events SOFA_VALIDATION_MEASURE_TIMES=1 SOFA_POKE_MEASURE_TIMES=1
    done
    for v in "${VERIFIED[@]}"; do run_scene "${scene}" "${v}" verify SOFA_CONTACT_VERIFY_EVERY=10; done
done

# ---- 3. contact validation tests -----------------------------------------------------
validation() {   # dir side [env...]
    local dir="$1" side="$2"; shift 2
    (
        export SOFA_BENCHMARK_LOG_DIR="${dir}"
        run() { env "$@" bash "${SCRIPT_DIR}/run_validation_wsl.sh" "${TEST}" "${STEPS}" "${side}" | grep -v "^OUT=\|VALIDATION_RUN_DONE"; }
        for a in 10 25; do TEST=incline_friction.py STEPS=50 run "$@" SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_ANGLE=$a; done
        TEST=plate_compression.py STEPS=200 run "$@" SOFA_VALIDATION_MATERIAL=neohookean
        for mu in 0.25 0.5; do
            TEST=grasp_lift.py STEPS=360 run "$@" SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_FRICTION=$mu SOFA_VALIDATION_RUN_TAG=mu$mu
        done
        TEST=cutting.py STEPS=300 run "$@" SOFA_VALIDATION_MATERIAL=neohookean
    )
}
validation "${OUT}/validation/cpu" cpu
for v in "${VERIFIED[@]}"; do
    # shellcheck disable=SC2046
    validation "${OUT}/validation/${v}" gpu $(variant_env "${v}") SOFA_CONTACT_VERIFY_EVERY=1
    cp "${OUT}/validation/cpu/"*_cpu* "${OUT}/validation/${v}/" 2>/dev/null
    python3 "${SCRIPT_DIR}/compare_validation.py" "${OUT}/validation/${v}" > "${OUT}/validation/${v}/compare.txt" 2>&1
done

kill "${MONITOR_PID}" 2>/dev/null
python3 "${SCRIPT_DIR}/solver_ab_report.py" "${OUT}"
echo "OUT=${OUT}"
echo SOLVER_AB_DONE
