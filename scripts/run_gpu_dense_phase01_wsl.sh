#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-60}"
ROOT="/mnt/c/Users/arfin/Desktop/GPU SOFA"
PLUGIN_SO="${ROOT}/SofaGpuCollision/build-wsl/libSofaGpuCollision.so"

source /home/arfin/sofa-venv/bin/activate

export SOFA_BENCHMARK_LOG_DIR="${SOFA_BENCHMARK_LOG_DIR:-/home/arfin/gpu-sofa-benchmark-logs}"
export QT_ROOT="${QT_ROOT:-/home/arfin/Qt/6.11.0/gcc_64}"
export LD_LIBRARY_PATH="${QT_ROOT}/lib:${ROOT}/SofaGpuCollision/build-wsl${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_PLUGIN_PATH="${QT_ROOT}/plugins${QT_PLUGIN_PATH:+:${QT_PLUGIN_PATH}}"
export QML2_IMPORT_PATH="${QT_ROOT}/qml${QML2_IMPORT_PATH:+:${QML2_IMPORT_PATH}}"

mkdir -p "${SOFA_BENCHMARK_LOG_DIR}"

"/home/arfin/sofa/install/v25.12/bin/runSofa" \
  -g batch \
  -l "${PLUGIN_SO}" \
  -l SofaPython3 \
  -l SofaCUDA \
  -n "${ITERATIONS}" \
  "${ROOT}/test_gpu_dense_collision_plugin_benchmark.py"
