#!/usr/bin/env bash
# Summarize the seven runs of full_e2e_verification_<date>.
set -u
ROOT="${1}"

peek() {
    local label="$1"
    local dir="$2"
    echo "=== ${label} ==="
    local f="$(ls ${dir}/*summary*.txt 2>/dev/null | head -1)"
    if [[ -z "${f}" ]]; then
        echo "(no summary)"; echo; return
    fi
    grep -E '^(collision_vertex_count|collision_element_count|avg_fps|avg_narrow_wall_ms|avg_narrow_kernel_ms|avg_narrow_feature_based_proximity_kernel_ms|avg_narrow_host_synchronization_ms|avg_narrow_output_contact_count|avg_narrow_vf_contact_count|avg_narrow_fv_contact_count|avg_narrow_ee_contact_count|avg_narrow_overflow_count|avg_narrow_unique_candidate_count|avg_narrow_raw_candidate_count|avg_kernel_launch_count|avg_cuda_memset_count|avg_host_to_device_bytes|avg_device_to_host_bytes|avg_workspace_resize_count)=' "${f}"
    echo
}

for run in \
    "01: tri-tri FBP fast path:${ROOT}/01_tri_tri_fbp_fast" \
    "02: tri-tri FBP validation:${ROOT}/02_tri_tri_fbp_validation" \
    "03: v-t self-collision:${ROOT}/03_vt_self_collision" \
    "04: v-t cross-model detection:${ROOT}/04_vt_cross_model_detection" \
    "05: v-t cross-model publication:${ROOT}/05_vt_cross_model_publication"
do
    label="${run%%:*}"
    rest="${run#*:}"
    name="${rest%%:*}"
    dir="${rest#*:}"
    peek "${label} ${name}" "${dir}"
done

echo "=== 06: standalone backend bench ==="
if [[ -f "${ROOT}/06_backend_bench/output.txt" ]]; then
    tail -40 "${ROOT}/06_backend_bench/output.txt"
fi
echo
