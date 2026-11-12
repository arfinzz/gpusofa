#!/usr/bin/env python3
"""Report of an A/B session of the GPU contact solver (scripts/run_solver_ab_wsl.sh).

Usage: python3 scripts/solver_ab_report.py <session dir>
Writes <session dir>/report.md: the kernel checks, then per scene the speed (plain runs),
the stage times (CUDA events), how far each variant's motion and force are from the old
method's (against the old method's own run-to-run spread), the CPU check of the
solutions (overlap, friction cone, distance to a reference solution), and the contact
validation tests against SOFA and their known answers.
"""

import csv
import glob
import json
import math
import os
import re
import statistics
import sys

VARIANTS = ["A", "W", "T", "D", "WT", "WTD"]
NAMES = {"A": "old method", "W": "warm start", "T": "tiled kernel", "D": "tissue-side contacts",
         "WT": "warm + tiled", "WTD": "warm + tiled + tissue-side"}


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def fmt(v, digits=2):
    if isinstance(v, str):
        return v
    if v is None or v != v:
        return "-"
    if v != 0 and (abs(v) < 10 ** -digits or abs(v) >= 1e6):
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return out


# ---- per scene: which files and columns ---------------------------------------------
def scene_files(scene):
    if scene == "poke":
        return {"main": "tissue_poke_gpu.csv", "stats": "tissue_poke_gpu_constraints.csv",
                "verify": "tissue_poke_gpu_contact_verify.csv"}
    return {"main": f"{scene}_poke_gpu.csv", "stats": f"{scene}_poke_gpu.csv", "verify": f"{scene}_poke_gpu_contact_verify.csv"}


def load_run(out, scene, variant, kind):
    d = os.path.join(out, scene, variant, kind)
    f = scene_files(scene)
    main = read_csv(os.path.join(d, f["main"]))
    stats = read_csv(os.path.join(d, f["stats"])) if f["stats"] != f["main"] else main
    verify = read_csv(os.path.join(d, f["verify"]))
    return main, stats, verify


def frame_ms(scene, row):
    return num(row["wall_ms"] if scene == "poke" else row["frame_ms"])


def rows_of(scene, row):
    return num(row["rows"])


def physics(scene, rows):
    """Per step: (time, tool y, surface y, force y)."""
    if scene == "poke":
        return [(num(r["time"]), num(r["probe_tip_y"]), num(r["surface_center_y"]), num(r["force_y"])) for r in rows]
    return [(num(r["time"]), num(r["tool_tip_y"]), num(r["monitor_y"]), num(r["spring_force_y"])) for r in rows]


def physics_diff(scene, a_rows, b_rows):
    a, b = physics(scene, a_rows), physics(scene, b_rows)
    n = min(len(a), len(b))
    if n == 0:
        return None
    peak = max(abs(x[3]) for x in a[:n]) or 1e-30
    return {"steps": n,
            "tool_mm": 1000 * max(abs(a[i][1] - b[i][1]) for i in range(n)),
            "surface_mm": 1000 * max(abs(a[i][2] - b[i][2]) for i in range(n)),
            "force_rel": max(abs(a[i][3] - b[i][3]) for i in range(n)) / peak,
            "peak_a": max(x[3] for x in a[:n]), "peak_b": max(x[3] for x in b[:n]),
            "bitwise": all(a[i] == b[i] for i in range(n)) and len(a) == len(b)}


def scene_section(out, scene):
    lines = []
    title = {"diaphragm": "Hard poke: diaphragm", "cuboid": "Hard poke: cuboid", "poke": "Tissue poke (1,800 nodes, 95 contacts)"}[scene]
    lines += [f"## {title}", ""]
    present = [v for v in VARIANTS if os.path.isdir(os.path.join(out, scene, v))]
    if not present:
        return lines + ["(no runs)", ""]

    # ---- speed: plain runs ----------------------------------------------------------
    speed = {}
    for v in present:
        reps = []
        for kind in ("plain1", "plain2"):
            main, stats, _ = load_run(out, scene, v, kind)
            if not main:
                continue
            by_time = {round(num(r["time"]), 6): r for r in stats}
            contact = [frame_ms(scene, r) for i, r in enumerate(main)
                       if i > 0 and rows_of(scene, by_time.get(round(num(r["time"]), 6), {"rows": 0})) > 0]
            free = [frame_ms(scene, r) for i, r in enumerate(main)
                    if i > 0 and rows_of(scene, by_time.get(round(num(r["time"]), 6), {"rows": 0})) == 0]
            heavy = sorted(contact)[len(contact) // 2:] if contact else []
            reps.append({"contact": mean(contact), "free": mean(free), "all": mean([frame_ms(scene, r) for r in main[1:]]),
                         "steps_contact": len(contact), "heaviest_half": mean([x for x in contact if contact and x >= statistics.median(contact)])})
        if reps:
            speed[v] = {k: mean([r[k] for r in reps]) for k in reps[0]}
            speed[v]["spread"] = (abs(reps[0]["contact"] - reps[1]["contact"]) / mean([r["contact"] for r in reps]) * 100
                                  if len(reps) > 1 else float("nan"))
    if speed:
        base = speed.get("A", {}).get("contact", float("nan"))
        base_all = speed.get("A", {}).get("all", float("nan"))
        lines += ["### Speed (plain runs, no timers; mean of two runs, the second in reverse order)", ""]
        lines += table(["Variant", "Steps in contact", "Frame in contact ms", "Speed-up", "Heaviest half ms",
                        "Frame without contact ms", "Whole run, mean frame ms", "Speed-up (whole run)", "Run-to-run spread %"],
                       [[f"{v} ({NAMES[v]})", fmt(s["steps_contact"], 0), s["contact"], f"x{base / s['contact']:.2f}",
                         s["heaviest_half"], s["free"], s["all"], f"x{base_all / s['all']:.2f}", s["spread"]]
                        for v, s in speed.items()])
        lines.append("")

    # ---- stages: CUDA events ---------------------------------------------------------
    stage_rows = []
    for v in present:
        main, stats, _ = load_run(out, scene, v, "events")
        sel = [r for r in stats if rows_of(scene, r) > 0]
        if not sel:
            continue
        if scene == "poke":
            g = lambda r, k: num(r[{"gs": "solve_ms", "w": "compliance_ms", "rows": "rows_ms", "corr": "correction_ms",
                                     "contact": "gpu_ms", "tissue": "tissue_ms"}[k]])
            warm = lambda r: num(r.get("warm_started")) / max(num(r["contacts"]), 1)
            sides = lambda r: (num(r.get("contacts_on_tissue_vertices")), num(r.get("contacts_on_tool_vertices")))
            separation = lambda r: (num(r.get("min_separation_tissue_vertices_m")), num(r.get("min_separation_tool_vertices_m")))
        else:
            g = lambda r, k: num(r[{"gs": "gauss_seidel_ms", "w": "compliance_ms", "rows": "rows_ms", "corr": "correction_ms",
                                     "contact": "contact_gpu_ms", "tissue": "tissue_gpu_ms"}[k]])
            warm = lambda r: num(r["warm_started"]) / max(num(r["contacts"]), 1)
            sides = lambda r: (num(r["contacts_on_tissue_vertices"]), num(r["contacts_on_tool_vertices"]))
            separation = lambda r: (num(r["min_separation_tissue_vertices_m"]), num(r["min_separation_tool_vertices_m"]))
        peak = max(sel, key=lambda r: rows_of(scene, r))
        stage_rows.append([f"{v} ({NAMES[v]})", mean([num(r["contacts"]) for r in sel]), mean([rows_of(scene, r) for r in sel]),
                           mean([sides(r)[0] for r in sel]), mean([sides(r)[1] for r in sel]),
                           mean([num(r["sweeps"]) for r in sel]), 100 * mean([warm(r) for r in sel]),
                           mean([g(r, "rows") for r in sel]), mean([g(r, "w") for r in sel]), mean([g(r, "gs") for r in sel]),
                           mean([g(r, "corr") for r in sel]), mean([g(r, "contact") for r in sel]),
                           mean([g(r, "tissue") for r in sel]),
                           1000 * min((separation(r)[0] for r in sel if separation(r)[0] == separation(r)[0]), default=float("nan")),
                           1000 * min((separation(r)[1] for r in sel if separation(r)[1] == separation(r)[1]), default=float("nan")),
                           f"{fmt(rows_of(scene, peak), 0)} rows, {fmt(num(peak['sweeps']), 0)} sweeps, GS {fmt(g(peak, 'gs'))} ms"])
    if stage_rows:
        lines += ["### Stages (CUDA events; mean over the steps in contact)", ""]
        lines += table(["Variant", "Contacts", "Rows", "On tissue vertices", "On tool vertices", "Sweeps", "Warm-started %",
                        "A rows ms", "B W ms", "C Gauss-Seidel ms", "D correction ms", "Contact total ms", "Tissue ms",
                        "Closest tissue vertex mm", "Closest tool vertex mm", "Step with most rows"], stage_rows)
        lines += ["", "Closest vertex: over the run, the smallest side-aware distance of a tissue (tool) vertex from its own closest "
                      "face of the other surface, whether or not its contact is kept (the constraint keeps 0.5 mm; below 0 = the "
                      "vertex is inside the other body)."]
        lines.append("")

    # ---- physics: each variant against the old method ----------------------------------
    a1, _, _ = load_run(out, scene, "A", "plain1")
    a2, _, _ = load_run(out, scene, "A", "plain2")
    phys = []
    noise = physics_diff(scene, a1, a2) if a1 and a2 else None
    if noise:
        phys.append(["A run 1 vs A run 2 (the old method's own spread)", noise["tool_mm"], noise["surface_mm"],
                     100 * noise["force_rel"], noise["peak_a"], noise["peak_b"], "yes" if noise["bitwise"] else "no"])
    for v in present:
        if v == "A":
            continue
        b1, _, _ = load_run(out, scene, v, "plain1")
        b2, _, _ = load_run(out, scene, v, "plain2")
        d = physics_diff(scene, a1, b1) if a1 and b1 else None
        same = physics_diff(scene, b1, b2) if b1 and b2 else None
        if d:
            phys.append([f"A vs {v} ({NAMES[v]})", d["tool_mm"], d["surface_mm"], 100 * d["force_rel"], d["peak_a"], d["peak_b"],
                         ("yes" if same and same["bitwise"] else "no") + " (its own 2 runs)"])
    if phys:
        lines += ["### Motion and force against the old method (plain run 1, every step)", "",
                  "Largest difference over the whole run: the tool's height, the tissue surface under it, and the force "
                  "on the tool (relative to the peak force). The first row is the old method against itself.", ""]
        lines += table(["Comparison", "Tool mm", "Surface mm", "Force % of peak", "Peak force A N", "Peak force B N",
                        "Bit-identical"], phys)
        lines.append("")

    # ---- CPU check of the solutions -------------------------------------------------
    ver = []
    for v in present:
        _, _, verify = load_run(out, scene, v, "verify")
        if not verify:
            continue
        mx = lambda k: max(num(r[k]) for r in verify)
        av = lambda k: mean([num(r[k]) for r in verify])
        ver.append([f"{v} ({NAMES[v]})", str(len(verify)), av("rows"), av("sweeps"), mx("check_error_over_tolerance"),
                    1000 * mx("max_overlap_m"), 1000 * mx("max_open_gap_with_force_m"), mx("max_cone_excess_rel"),
                    1000 * mx("max_stick_slip_m"), 1000 * mx("distance_to_reference_m"), 1000 * av("distance_to_reference_m"),
                    100 * mx("normal_impulse_error_rel"), f"{int(sum(num(r['reference_converged']) for r in verify))}/{len(verify)}"])
    if ver:
        lines += ["### The solutions checked on the CPU (every 10th solve, double precision)", "",
                  "From W, dfree and the GPU's lambda: overlap left (a contact closer than the contact distance), a "
                  "contact pushing while apart, friction beyond the cone, slip while sticking; one more exact "
                  "Gauss-Seidel sweep (its error over the solver's tolerance: below 1 means converged by SOFA's rule); "
                  "and the distance to a reference solved 10^5 times tighter from the same point.", ""]
        lines += table(["Variant", "Steps checked", "Rows", "Sweeps", "Next-sweep error / tolerance (max)", "Overlap max mm",
                        "Pushing while apart max mm", "Cone excess max", "Slip while sticking max mm",
                        "Distance to reference max mm", "mean mm", "Normal impulse error max %", "Reference converged"], ver)
        lines.append("")
    return lines


def checks_section(out):
    path = os.path.join(out, "constraint_checks.txt")
    if not os.path.isfile(path):
        return []
    text = open(path).read()
    lines = ["## Kernel checks (SofaGpuCollisionConstraintChecks, against SOFA's own Gauss-Seidel)", ""]
    for key in ("C1_FRICTION_SOLVER", "C2_COMPLIANCE", "C3_TILED_KERNEL", "C4_WARM_START", "CONSTRAINT_CHECKS"):
        m = re.search(rf"^{key}=(\w+)", text, re.M)
        lines.append(f"- {key}: **{m.group(1) if m else 'missing'}**")
    lines += ["", "```"] + [l for l in text.splitlines() if l.startswith(("C3 ", "C4 "))] + ["```", ""]
    return lines


def validation_section(out):
    base = os.path.join(out, "validation")
    if not os.path.isdir(base):
        return []
    lines = ["## Contact validation tests (each GPU variant against SOFA's CPU run and the known answer)", ""]
    known_keys = {"incline_friction": ("angle_deg", ["com_dx_m", "expected_dx_m", "error_m"]),
                  "plate_compression": ("material", ["force_n", "expected_force_n", "force_error_percent"]),
                  "grasp_lift": ("mu", ["squeeze_lift_n", "block_lift_m", "slip_m", "result"])}
    rows = []
    for v in VARIANTS:
        summary = read_csv(os.path.join(base, v, "validation_summary.csv"))
        if not summary:
            continue
        for r in summary:
            if r.get("kind") == "cpu_vs_gpu_run":
                rows.append([f"{v} ({NAMES[v]})", r["run"], num(r["max_rel_diff"]), r["where"], num(r["cpu_ms_per_step"]),
                             num(r["gpu_ms_per_step"])])
    if rows:
        lines += ["### Each GPU run against SOFA's CPU run (largest difference of the logged quantities)", ""]
        lines += table(["Variant", "Run", "Max relative difference", "Where", "CPU ms/step", "GPU ms/step"], rows)
        lines.append("")
    known = []
    for v in VARIANTS:
        summary = read_csv(os.path.join(base, v, "validation_summary.csv"))
        for r in summary:
            if r.get("kind") != "known_answer" or r.get("side") != "gpu":
                continue
            test = r.get("test", "")
            keys = known_keys.get(test, (None, [k for k in r if k not in ("kind", "run", "test", "side", "steps")][:4]))[1]
            known.append([f"{v} ({NAMES[v]})", test, r.get("run", ""), "; ".join(f"{k}={r.get(k, '')}" for k in keys)])
        if v == "A":
            for r in summary:
                if r.get("kind") == "known_answer" and r.get("side") == "cpu":
                    test = r.get("test", "")
                    keys = known_keys.get(test, (None, [k for k in r if k not in ("kind", "run", "test", "side", "steps")][:4]))[1]
                    known.append(["SOFA CPU", test, r.get("run", ""), "; ".join(f"{k}={r.get(k, '')}" for k in keys)])
    if known:
        lines += ["### Known answers", ""]
        lines += table(["Variant", "Test", "Run", "Result"], known)
        lines.append("")
    ver = []
    for v in VARIANTS:
        for path in sorted(glob.glob(os.path.join(base, v, "*_contact_verify.csv"))):
            rs = read_csv(path)
            if not rs:
                continue
            mx = lambda k: max(num(r[k]) for r in rs)
            ver.append([f"{v} ({NAMES[v]})", os.path.basename(path)[:-len("_contact_verify.csv")], str(len(rs)),
                        mx("check_error_over_tolerance"), 1000 * mx("max_overlap_m"), mx("max_cone_excess_rel"),
                        1000 * mx("distance_to_reference_m")])
    if ver:
        lines += ["### Every solve of the validation runs checked on the CPU", ""]
        lines += table(["Variant", "Run", "Solves", "Next-sweep error / tolerance max", "Overlap max mm", "Cone excess max",
                        "Distance to reference max mm"], ver)
        lines.append("")
    return lines


def main():
    out = sys.argv[1]
    lines = ["# GPU contact solver: old method against the new options (A/B)", ""]
    machine = os.path.join(out, "machine.txt")
    if os.path.isfile(machine):
        lines += [f"- {l.strip()}" for l in open(machine) if l.strip()] + [""]
    lines += ["Variants: " + ", ".join(f"**{v}** = {NAMES[v]}" for v in VARIANTS) + ".", ""]
    lines += checks_section(out)
    for scene in ("diaphragm", "cuboid", "poke"):
        lines += scene_section(out, scene)
    lines += validation_section(out)
    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"report={path}")


if __name__ == "__main__":
    main()
