"""Summarise validation runs (testscenes/validationtests/): CPU (SOFA) against GPU.

Usage: python3 scripts/compare_validation.py <log dir> [test name]

For every <test>_<material>_cpu.csv with a matching _gpu.csv it reports the
largest difference of the logged quantities over the run (positions in metres,
forces in newtons; relative to the largest value on the CPU side), and for every
*_gpu_tissue_compare.csv the largest stage-by-stage differences of the GPU tissue
step against SOFA's CPU components (forces and matrix relative to their largest
values, dv relative to the largest dv of the run). Writes <log dir>/validation_summary.csv too.
"""

import csv
import glob
import math
import os
import sys


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# Solver statistics (sweeps, errors, row counts) and GPU-only outputs are not physics to compare.
SKIP = ("time", "wall_ms", "phase", "constraints", "iterations", "error", "right_jaw_contact_fy")


def numeric_columns(rows, skip=SKIP):
    def numeric(k):
        try:
            float(rows[0][k])
            return True
        except (TypeError, ValueError):
            return False
    return [k for k in rows[0].keys() if k not in skip and numeric(k)]


def run_pairs(base, test=None):
    pairs = []
    for cpu in sorted(glob.glob(os.path.join(base, "*_cpu.csv"))):
        stem = os.path.basename(cpu)[:-len("_cpu.csv")]
        if test and not stem.startswith(test):
            continue
        gpu = os.path.join(base, stem + "_gpu.csv")
        if os.path.isfile(gpu):
            pairs.append((stem, cpu, gpu))
    return pairs


def compare_runs(cpu_rows, gpu_rows):
    n = min(len(cpu_rows), len(gpu_rows))
    cols = numeric_columns(cpu_rows)
    worst_abs, worst_rel, worst_col, worst_t = 0.0, 0.0, "", 0.0
    per_col = {}
    for c in cols:
        scale = max(abs(float(r[c])) for r in cpu_rows[:n]) or 1.0
        d, t = max((abs(float(a[c]) - float(b[c])), float(a["time"])) for a, b in zip(cpu_rows[:n], gpu_rows[:n]))
        per_col[c] = (d, d / scale)
        if d / scale > worst_rel:
            worst_abs, worst_rel, worst_col, worst_t = d, d / scale, c, t
    wall = lambda rows: sum(float(r["wall_ms"]) for r in rows[1:n]) / max(1, n - 1)
    return {"steps": n, "worst_abs": worst_abs, "worst_rel": worst_rel, "worst_col": worst_col, "worst_t": worst_t,
            "cpu_ms": wall(cpu_rows), "gpu_ms": wall(gpu_rows), "per_col": per_col}


def main():
    base = sys.argv[1]
    test = sys.argv[2] if len(sys.argv) > 2 else None
    out_rows = []
    print(f"{'run':44s} {'steps':>5s} {'max diff':>10s} {'rel':>9s}  where                      "
          f"{'CPU ms/step':>11s} {'GPU ms/step':>11s}")
    for stem, cpu, gpu in run_pairs(base, test):
        r = compare_runs(load(cpu), load(gpu))
        print(f"{stem:44s} {r['steps']:5d} {r['worst_abs']:10.3g} {r['worst_rel']:9.2e}  {r['worst_col']:18s} "
              f"t={r['worst_t']:<5.2f} {r['cpu_ms']:11.1f} {r['gpu_ms']:11.1f}")
        out_rows.append({"run": stem, "kind": "cpu_vs_gpu_run", "steps": r["steps"], "max_abs_diff": r["worst_abs"],
                         "max_rel_diff": r["worst_rel"], "where": r["worst_col"], "cpu_ms_per_step": r["cpu_ms"],
                         "gpu_ms_per_step": r["gpu_ms"]})
    compares = sorted(glob.glob(os.path.join(base, "*_gpu_tissue_compare.csv")))
    if compares:
        print(f"\n{'stage by stage (GPU tissue step vs SOFA CPU components)':58s} {'force':>9s} {'matrix':>9s} "
              f"{'dv':>9s} {'xfree m':>9s} {'vfree':>9s} {'GPU ms':>7s} {'CPU ms':>7s}")
    for path in compares:
        rows = load(path)
        if not rows:
            continue
        mx = lambda k: max(float(r[k]) for r in rows if not math.isnan(float(r[k])))
        mean = lambda k: sum(float(r[k]) for r in rows[1:]) / max(1, len(rows) - 1)
        stem = os.path.basename(path)[:-len("_gpu_tissue_compare.csv")]
        # dv relative to the run's largest dv (a step-wise ratio means nothing once the body is at rest).
        dv = mx("dv_max_abs_diff") / max(mx("dv_max"), 1e-300) if "dv_max" in rows[0] else mx("dv_rel_diff")
        print(f"{stem:58s} {mx('force_rel_diff'):9.2e} {mx('matrix_rel_diff'):9.2e} {dv:9.2e} "
              f"{mx('xfree_max_abs_diff_m'):9.2e} {mx('vfree_max_abs_diff'):9.2e} {mean('gpu_step_ms'):7.2f} "
              f"{mean('cpu_free_motion_ms'):7.1f}")
        out_rows.append({"run": stem, "kind": "stage_by_stage", "steps": len(rows),
                         "force_rel_diff": mx("force_rel_diff"), "matrix_rel_diff": mx("matrix_rel_diff"),
                         "dv_rel_diff": dv, "xfree_max_abs_diff_m": mx("xfree_max_abs_diff_m"),
                         "vfree_max_abs_diff": mx("vfree_max_abs_diff"), "gpu_ms_per_step": mean("gpu_step_ms"),
                         "cpu_ms_per_step": mean("cpu_free_motion_ms")})
    contact = sorted(glob.glob(os.path.join(base, "*_contact_compare.csv")))
    if contact:
        print(f"\n{'stage by stage (GPU contact response vs SOFA CPU pipeline, same contacts)':58s} {'rows':>9s} "
              f"{'W rel':>9s} {'lambda':>9s} {'dx m':>9s} {'steps':>6s}")
    for path in contact:
        rows = [r for r in load(path) if int(float(r.get("rows", 0) or 0)) > 0]
        if not rows:
            continue
        mx = lambda k: max((float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")), default=float("nan"))
        stem = os.path.basename(path)[:-len("_contact_compare.csv")]
        print(f"{stem:58s} {mx('rows_max_diff'):9.2e} {mx('compliance_rel_diff'):9.2e} "
              f"{mx('lambda_rel_diff_same_input'):9.2e} {mx('correction_diff_same_lambda_m'):9.2e} {len(rows):6d}")
        out_rows.append({"run": stem, "kind": "contact_stage_by_stage", "steps": len(rows),
                         "rows_max_diff": mx("rows_max_diff"), "compliance_rel_diff": mx("compliance_rel_diff"),
                         "lambda_rel_diff_same_input": mx("lambda_rel_diff_same_input"),
                         "correction_diff_same_lambda_m": mx("correction_diff_same_lambda_m")})

    # The known answers: each test's summary lines (key=value), grouped by test.
    summaries = []
    for path in sorted(glob.glob(os.path.join(base, "*_summary.txt"))):
        for line in open(path):
            fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
            if "test" in fields:
                fields["run"] = os.path.basename(path)[:-len("_summary.txt")]
                summaries.append(fields)
    if summaries:
        print("\nKnown answers:")
        columns = {
            "confined_compression": ["load", "stretch", "expected", "error"],
            "beam_bending": ["load", "divisions", "nodes", "tip_deflection_m", "timoshenko_m", "ratio"],
            "incline_friction": ["angle_deg", "mu", "com_dx_m", "expected_dx_m", "error_m"],
            "plate_compression": ["force_n", "expected_force_n", "force_error_percent", "lateral_stretch", "expected_lateral"],
            "grasp_lift": ["mu", "squeeze_lift_n", "friction_capacity_ratio", "expected", "jaw_lift_m", "block_lift_m",
                           "slip_m", "result"],
        }
        for test in sorted({s["test"] for s in summaries}):
            cols = columns.get(test, [k for k in summaries[0] if k not in ("test", "run")])
            print(f"  {test}")
            print(f"    {'material':16s} {'side':4s} " + " ".join(f"{c:>18s}" for c in cols))
            for s in [s for s in summaries if s["test"] == test]:
                print(f"    {s.get('material', ''):16s} {s.get('side', ''):4s} " + " ".join(f"{s.get(c, ''):>18s}" for c in cols))
            for s in [s for s in summaries if s["test"] == test]:
                out_rows.append(dict(kind="known_answer", **s))

    keys = sorted({k for r in out_rows for k in r.keys()}, key=lambda k: (k not in ("run", "kind", "steps"), k))
    with open(os.path.join(base, "validation_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
