#!/usr/bin/env bash
# CPU (SOFA) against GPU time per step for growing scene sizes.
#
# Part 1 - the tissue alone: testscenes/validationtests/material_check.py (a cube
# of tissue under a tilted gravity) with SOFA_VALIDATION_DIVISIONS cells per edge,
# for each material, on the CPU (SOFA's components: EulerImplicitSolver +
# SparseLDLSolver + MeshMatrixMass + the material's force fields) and on the GPU
# (GpuTissueSolver). Positions are not logged (no GPU read-back).
# Part 2 - the whole tissue poke (tissue + collision + constraint contact) at
# several tissue mesh sizes (SOFA_POKE_FINE_STEP: element size under the probe).
#
# Usage (from WSL):
#   bash scripts/scaling_study_wsl.sh [tissue|poke|both]
#   SCALING_DIVISIONS="4 6 8 10 12 14 16"  SCALING_MATERIALS="neohookean ogden_maxwell"
#   SCALING_FINE_STEPS="0.003 0.002 0.0015"  SCALING_POKE_FRAMES=300  SCALING_POKE_CPU=1
# Then: python3 scripts/summarize_scaling.py <log dir>
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHICH="${1:-both}"
OUT="${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/scaling_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT}"
DIVISIONS="${SCALING_DIVISIONS:-4 6 8 10 12 14 16}"
MATERIALS="${SCALING_MATERIALS:-neohookean ogden_maxwell}"
FINE_STEPS="${SCALING_FINE_STEPS:-0.003 0.002 0.0015}"
POKE_FRAMES="${SCALING_POKE_FRAMES:-300}"
echo "logs=${OUT}"

if [ "${WHICH}" = tissue ] || [ "${WHICH}" = both ]; then
    for mat in ${MATERIALS}; do
        for d in ${DIVISIONS}; do
            # The CPU is slow on big meshes: fewer steps there (the first two are warm-up).
            cpu_steps=12; [ "${d}" -ge 12 ] && cpu_steps=6
            for side in gpu cpu; do
                steps=30; [ "${side}" = cpu ] && steps=${cpu_steps}
                dir="${OUT}/tissue_${mat}_d${d}"
                mkdir -p "${dir}"
                env SOFA_BENCHMARK_LOG_DIR="${dir}" SOFA_VALIDATION_MATERIAL="${mat}" SOFA_VALIDATION_DIVISIONS="${d}" \
                    SOFA_VALIDATION_LOG_POSITIONS=0 SOFA_VALIDATION_MEASURE_TIMES=1 \
                    timeout 3600 bash "${SCRIPT_DIR}/run_validation_wsl.sh" material_check.py "${steps}" "${side}" \
                    | grep -v "^OUT=\|VALIDATION_RUN_DONE"
            done
        done
    done
fi

if [ "${WHICH}" = poke ] || [ "${WHICH}" = both ]; then
    for fs in ${FINE_STEPS}; do
        for side in gpu cpu; do
            [ "${side}" = cpu ] && [ "${SCALING_POKE_CPU:-1}" != 1 ] && continue
            dir="${OUT}/poke_fine${fs}_${side}"
            env SOFA_BENCHMARK_LOG_DIR="${dir}" SOFA_POKE_FINE_STEP="${fs}" SOFA_POKE_MEASURE_TIMES=1 SOFA_POKE_VISUAL=0 \
                timeout 14400 bash "${SCRIPT_DIR}/run_tissue_poke_wsl.sh" "${side}" "${POKE_FRAMES}" > "${dir}.out" 2>&1
            echo "poke fine step ${fs} ${side}: $(grep -h 'TissuePoke.*nodes' "${dir}"/*.log | head -1 | sed 's/.*: //')"
        done
    done
fi
echo "OUT=${OUT}"
echo SCALING_DONE
