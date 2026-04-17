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

"$HOME/sofa/install/v25.12/bin/runSofa" \
  -g batch \
  -l "$ROOT_DIR/SofaGpuCollision/build-wsl/libSofaGpuCollision.so" \
  -l SofaPython3 \
  -l SofaCUDA \
  -n "$ITERATIONS" \
  "$ROOT_DIR/test_gpu_tissue_phase3_dual_mesh_experimental.py"
