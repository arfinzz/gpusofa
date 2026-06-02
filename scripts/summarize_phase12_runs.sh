#!/usr/bin/env bash
set -u
BASE="${1:-/home/arfin/gpu-sofa/output/benchmark_logs}"
for d in phase12_verify_tri_tri phase12_verify_vt_self phase12_verify_vt_cross; do
    echo "=== ${d} ==="
    summary_file="$(ls ${BASE}/${d}/*summary*.txt 2>/dev/null | head -1)"
    if [[ -z "${summary_file}" ]]; then
        echo "(no summary file)"
        echo
        continue
    fi
    grep -E '^(collision_vertex_count|collision_element_count|avg_fps|avg_narrow_wall_ms|avg_narrow_kernel_ms|avg_narrow_feature_based_proximity_kernel_ms|avg_narrow_host_synchronization_ms|avg_narrow_output_contact_count|avg_narrow_vf_contact_count|avg_narrow_fv_contact_count|avg_narrow_ee_contact_count|avg_narrow_overflow_count|avg_kernel_launch_count|avg_cuda_memset_count|avg_host_to_device_bytes|avg_device_to_host_bytes)=' "${summary_file}"
    echo
done
