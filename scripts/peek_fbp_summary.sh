#!/usr/bin/env bash
set -u
DIR="${1}"
shopt -s nullglob
for f in ${DIR}/*summary*.txt; do
    echo "=== ${f} ==="
    grep -E '^(collision_vertex_count|collision_element_count|avg_fps|avg_narrow_wall_ms|avg_narrow_kernel_ms|avg_narrow_feature_based_proximity_kernel_ms|avg_narrow_host_synchronization_ms|avg_narrow_output_contact_count|avg_narrow_vf_contact_count|avg_narrow_fv_contact_count|avg_narrow_ee_contact_count|avg_narrow_overflow_count|avg_kernel_launch_count|avg_cuda_memset_count|avg_host_to_device_bytes|avg_device_to_host_bytes)=' "${f}"
done
