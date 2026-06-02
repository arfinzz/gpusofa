#!/usr/bin/env bash
# Pulls the FBP / v-t kernel rows out of the NCU CSV exports and prints a
# compact table.
set -u
PROFILE_ROOT="${1}"

show() {
    local label="$1"
    local csv="$2"
    local kernel="$3"
    echo "=== ${label} :: ${kernel} ==="
    if [[ ! -f "${csv}" ]]; then
        echo "(no csv)"
        echo
        return
    fi
    # CSV header is the second line of `ncu --import --csv --page raw` output.
    # Print the columns we care about for any row whose Kernel Name contains $kernel.
    python3 - <<PY
import csv, sys
csv_path = "${csv}"
needle = "${kernel}"
cols_keep = [
    "Kernel Name",
    "Block Size",
    "launch__grid_size",
    "launch__registers_per_thread",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
]
with open(csv_path, newline="") as fh:
    rows = list(csv.reader(fh))
if len(rows) < 2:
    print("(empty)")
    sys.exit(0)
header = rows[0]
units  = rows[1]
data   = rows[2:]
col_idx = {c: i for i, c in enumerate(header) if c in cols_keep}
matches = [r for r in data if needle in r[col_idx["Kernel Name"]]]
if not matches:
    print(f"(no rows matching kernel substring '{needle}')")
    sys.exit(0)
for r in matches:
    parts = []
    for col in cols_keep:
        if col == "Kernel Name":
            continue
        i = col_idx.get(col)
        val = r[i] if i is not None else ""
        # trim long values
        if len(val) > 12:
            val = val[:12]
        unit = units[i] if i is not None and i < len(units) else ""
        parts.append(f"{col.split('.')[0]}={val}{unit if unit and unit != '' else ''}")
    print("  ", " ".join(parts))
PY
    echo
}

for run in tri_tri_fbp vt_self_collision vt_cross_model; do
    csv="${PROFILE_ROOT}/${run}/profile.csv"
    case "${run}" in
        tri_tri_fbp)
            show "${run}" "${csv}" "featureBasedProximityKernel"
            show "${run}" "${csv}" "insertIndexedTrianglesKernel"
            show "${run}" "${csv}" "generateDenseGridUniqueCandidatePairsKernel"
            ;;
        vt_self_collision|vt_cross_model)
            show "${run}" "${csv}" "featureBasedVertexTriangleProximityKernel"
            show "${run}" "${csv}" "insertIndexedPointsKernel"
            show "${run}" "${csv}" "insertIndexedTrianglesKernel"
            show "${run}" "${csv}" "generateDenseGridUniqueCandidatePairsKernel"
            ;;
    esac
done
