#!/usr/bin/env bash
# Every validation test (testscenes/validationtests/), CPU (SOFA) and GPU, then the
# summary: bash scripts/run_validation_suite_wsl.sh [log dir]
#   material_check        all 7 materials, both sides, and the GPU's stage-by-stage
#                         comparison with SOFA's components (compare)
#   confined_compression  10% compression (StVK: 5%), 5 core materials + Ogden-Maxwell,
#                         both sides; the Maxwell creep curve; stage-by-stage runs
#                         (Ogden and NeoHookean)
#   beam_bending          small load at 4 mesh sizes (convergence to beam theory),
#                         large load, both sides
#   incline_friction      0, 10 and 25 degrees, both sides
#   plate_compression     3 elastic materials, both sides, and a stage-by-stage
#                         contact comparison
#   grasp_lift            two jaws and a pedestal (three rigid bodies), friction
#                         0.05, 0.25, 0.3 and 0.5 across Coulomb's threshold, both sides
#   cutting               a blade cuts a slot into a loaded beam (tetrahedra removed),
#                         two materials, both sides
# Summary: python3 scripts/compare_validation.py <log dir>
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${SOFA_BENCHMARK_LOG_DIR:-${REPO_DIR}/output/benchmark_logs/validation_suite_$(date +%Y%m%d_%H%M%S)}}"
export SOFA_BENCHMARK_LOG_DIR="${OUT}"
mkdir -p "${OUT}"
run() { bash "${SCRIPT_DIR}/run_validation_wsl.sh" "$@" | grep -v "^OUT=\|VALIDATION_RUN_DONE"; }

for m in neohookean stable_neohookean stvk mooney_rivlin ogden ogden_maxwell sls_ogden_sofa; do
    SOFA_VALIDATION_MATERIAL=$m run material_check.py 60 both
    SOFA_VALIDATION_MATERIAL=$m run material_check.py 60 compare
done

for m in neohookean stable_neohookean stvk mooney_rivlin ogden; do
    SOFA_VALIDATION_MATERIAL=$m run confined_compression.py 300 both
done
SOFA_VALIDATION_MATERIAL=ogden_maxwell run confined_compression.py 1500 both
SOFA_VALIDATION_MATERIAL=ogden_maxwell SOFA_VALIDATION_LOAD=creep run confined_compression.py 1500 both
SOFA_VALIDATION_MATERIAL=ogden run confined_compression.py 150 compare
SOFA_VALIDATION_MATERIAL=neohookean run confined_compression.py 150 compare

for d in 2 3 4 6; do
    SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_DIVISIONS=$d run beam_bending.py 500 both
done
SOFA_VALIDATION_MATERIAL=ogden SOFA_VALIDATION_DIVISIONS=4 run beam_bending.py 500 both
SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_DIVISIONS=3 SOFA_VALIDATION_LOAD=large run beam_bending.py 500 both

for a in 0 10 25; do
    SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_ANGLE=$a run incline_friction.py 50 both
done

for m in neohookean mooney_rivlin ogden; do
    SOFA_VALIDATION_MATERIAL=$m run plate_compression.py 200 both
done
SOFA_VALIDATION_MATERIAL=neohookean run plate_compression.py 200 compare

for mu in 0.05 0.25 0.3 0.5; do
    SOFA_VALIDATION_MATERIAL=neohookean SOFA_VALIDATION_FRICTION=$mu SOFA_VALIDATION_RUN_TAG=mu$mu run grasp_lift.py 360 both
done

for m in neohookean ogden_maxwell; do
    SOFA_VALIDATION_MATERIAL=$m run cutting.py 300 both
done

python3 "${SCRIPT_DIR}/compare_validation.py" "${OUT}"
echo "OUT=${OUT}"
echo VALIDATION_SUITE_DONE
