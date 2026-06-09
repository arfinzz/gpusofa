#!/usr/bin/env bash
set -u
RUNDIR="${1}"
shopt -s nullglob
for leg in small_fastpath small_validation large_fastpath large_validation \
           vt_self_fastpath vt_self_validation vt_cross_fastpath vt_cross_validation \
           hash_dense hash_on; do
    d="${RUNDIR}/${leg}"
    [[ -d "${d}" ]] || continue
    f="$(ls ${d}/*summary*.txt 2>/dev/null | head -1)"
    [[ -z "${f}" ]] && continue
    fps=$(grep -E '^avg_fps=' "${f}" | cut -d= -f2)
    nw=$(grep -E '^avg_narrow_wall_ms=' "${f}" | cut -d= -f2)
    nk=$(grep -E '^avg_narrow_kernel_ms=' "${f}" | cut -d= -f2)
    cc=$(grep -E '^avg_narrow_output_contact_count=' "${f}" | cut -d= -f2)
    vf=$(grep -E '^avg_narrow_vf_contact_count=' "${f}" | cut -d= -f2)
    fv=$(grep -E '^avg_narrow_fv_contact_count=' "${f}" | cut -d= -f2)
    ee=$(grep -E '^avg_narrow_ee_contact_count=' "${f}" | cut -d= -f2)
    ov=$(grep -E '^avg_narrow_overflow_count=' "${f}" | cut -d= -f2)
    kl=$(grep -E '^avg_kernel_launch_count=' "${f}" | cut -d= -f2)
    printf "%-20s fps=%-10s nwall=%-10s nkern=%-10s contacts=%-6s vf/fv/ee=%s/%s/%s ovf=%s launches=%s\n" \
        "${leg}" "${fps%.*}" "${nw}" "${nk}" "${cc}" "${vf}" "${fv}" "${ee}" "${ov}" "${kl%.*}"
done
