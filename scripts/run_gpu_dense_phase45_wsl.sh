#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-60}"
ROOT_DIR="/mnt/c/Users/arfin/Desktop/GPU SOFA"

source "$HOME/sofa-venv/bin/activate"
export SOFA_BENCHMARK_LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-$HOME/gpu-sofa-benchmark-logs}"
export QT_ROOT="${QT_ROOT:-$HOME/Qt/6.11.0/gcc_64}"
export LD_LIBRARY_PATH="$QT_ROOT/lib:$ROOT_DIR/SofaGpuCollision/build-wsl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$QT_ROOT/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
export QML2_IMPORT_PATH="$QT_ROOT/qml${QML2_IMPORT_PATH:+:$QML2_IMPORT_PATH}"
export SOFA_PIPELINE_LABEL="gpu_dense_collision_phase45_compact_candidates"
export SOFA_PIPELINE_PHASE="phase45-experimental"
export SOFA_PIPELINE_NOTES="Phase 4/5 experimental compact candidate pipeline: broad phase uses compact GPU pair generation instead of dense flag readback, narrow phase emits explicit GPU leaf-box contact candidates, and the dual-mesh CUDA collision surface remains GPU-resident. Final exact contact resolution still falls back to the CPU response path."
export SOFA_PIPELINE_WARMUP_STEPS="${SOFA_PIPELINE_WARMUP_STEPS:-0}"
export SOFA_GPU_NARROW_MIN_PAIR_COUNT="${SOFA_GPU_NARROW_MIN_PAIR_COUNT:-1}"
export SOFA_BLADE_STATIC_OVERLAP="${SOFA_BLADE_STATIC_OVERLAP:-1}"
export SOFA_BLADE_START_Y="${SOFA_BLADE_START_Y:-0.0}"
export SOFA_BLADE_SETTLE_STEPS="${SOFA_BLADE_SETTLE_STEPS:-0}"
export SOFA_BLADE_DESCEND_STEPS="${SOFA_BLADE_DESCEND_STEPS:-1}"
export SOFA_BLADE_TOTAL_DOWN="${SOFA_BLADE_TOTAL_DOWN:-0.0}"

"$HOME/sofa/install/v25.12/bin/runSofa" \
  -g batch \
  -l "$ROOT_DIR/SofaGpuCollision/build-wsl/libSofaGpuCollision.so" \
  -l SofaPython3 \
  -l SofaCUDA \
  -n "$ITERATIONS" \
  "$ROOT_DIR/test_gpu_dense_phase45_validation.py"
