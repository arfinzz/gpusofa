#!/usr/bin/env bash
set -euo pipefail
VERIFY_DIR="${1:-/home/arfin/gpu-sofa/output/benchmark_logs/phase12_full_verification_20260525}"

for run in tri_tri_fbp vt_self_collision vt_cross_model; do
    echo "=== ${run} ==="
    f="$(ls ${VERIFY_DIR}/${run}/*summary*.txt 2>/dev/null | head -1)"
    if [[ -z "${f}" ]]; then
        echo "(no summary file found)"
        echo
        continue
    fi
    grep -E "collision_vertex_count|collision_element_count|avg_fps|avg_narrow_wall_ms|avg_narrow_kernel_ms|avg_narrow_feature_based_proximity_kernel_ms|avg_narrow_host_synchronization_ms|avg_narrow_output_contact_count|avg_narrow_vf_contact_count|avg_narrow_fv_contact_count|avg_narrow_ee_contact_count|avg_narrow_overflow_count|avg_kernel_launch_count|avg_cuda_memset_count|avg_host_to_device_bytes|avg_device_to_host_bytes" "${f}"
    echo
done
