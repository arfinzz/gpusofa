#!/usr/bin/env python3
"""Report of a hard-poke session (scripts/run_hard_poke_wsl.sh): every clock of every run,
per phase of the poke, against the number of contacts, and at the step with the most.

Usage: python3 scripts/hard_poke_report.py <session dir>
Writes <session dir>/report.md and, per scene, <session dir>/<scene>/steps_merged.csv
(one row per step: the plain run's frame time, the GPU stage times, the collision's
times and counts, and SOFA's timer stages with and without its GPU waits).
"""

import csv
import datetime
import json
import os
import statistics
import sys

PHASES = ["settle", "press", "hold", "retract", "rest"]
CONTACT_STAGES = ["rows_ms", "contact_factorize_ms", "compliance_ms", "gauss_seidel_ms", "correction_ms"]
TISSUE_STAGES = ["tissue_material_ms", "tissue_assembly_ms", "tissue_factorize_ms", "tissue_solve_ms"]
COLLISION_COLUMNS = ["narrow_kernel_ms", "narrow_wall_ms", "narrow_host_synchronization_ms", "broad_wall_ms",
                     "narrow_output_contact_count", "narrow_vf_contact_count", "narrow_fv_contact_count",
                     "narrow_ee_contact_count", "narrow_unique_candidate_count", "narrow_overflow_count",
                     "kernel_launch_count", "host_to_device_bytes", "device_to_host_bytes"]
# SOFA timer stages put in steps_merged.csv (their path in SOFA's timer tree).
TIMER_STAGES = {
    "animate": "Simulation::animate",
    "begin_event": "Simulation::animate/AnimateBeginEvent",
    "free_motion_and_collision": "Simulation::animate/FreeMotion+CollisionDetection",
    "free_motion": "Simulation::animate/FreeMotion+CollisionDetection/FreeMotion",
    "collision": "Simulation::animate/FreeMotion+CollisionDetection/CollisionDetection",
    "narrow_phase": "Simulation::animate/FreeMotion+CollisionDetection/CollisionDetection/CollisionVisitor/"
                    "doCollisionDetection/NarrowPhase",
    "constraint_solver": "Simulation::animate/ConstraintSolver",
    "prepare_states": "Simulation::animate/ConstraintSolver/SolveConstraint/PrepareState",
    "build_system": "Simulation::animate/ConstraintSolver/SolveConstraint/BuildSystem",
    "solve_system": "Simulation::animate/ConstraintSolver/SolveConstraint/SolveSystem",
    "apply_correction": "Simulation::animate/ConstraintSolver/SolveConstraint/ApplyCorrection",
    "end_event": "Simulation::animate/AnimateEndEvent",
}


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def pct(values, q):
    values = sorted(v for v in values if v == v)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def fmt(v, digits=2):
    if isinstance(v, str):
        return v
    if v != v:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return out


def parse_sofa_timer(path):
    """SOFA's per-step timer tables ('Steps Duration Statistics'): one dict per step,
    label path -> total ms in that step (a label met several times in a step is summed)."""
    steps = []
    if not os.path.isfile(path):
        return steps
    current, stack = None, []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("Steps Duration Statistics"):
                current, stack = {}, []
                steps.append(current)
                continue
            if current is None:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            try:
                level = int(parts[0])
                total = float(parts[7])
            except ValueError:
                continue
            label = parts[-1].strip().lstrip(".")
            del stack[level:]
            stack.append(label)
            if level == 0:
                continue                     # TOTAL
            key = "/".join(stack[1:])
            current[key] = current.get(key, 0.0) + total
    return steps


def gpu_monitor(out_dir):
    """nvidia-smi samples per run (run_marks.csv gives each run's start and end)."""
    fmt_ts = "%Y/%m/%d %H:%M:%S.%f"
    samples = []
    for row in csv.reader(open(os.path.join(out_dir, "gpu_monitor.csv"))) if os.path.isfile(
            os.path.join(out_dir, "gpu_monitor.csv")) else []:
        if len(row) < 9:
            continue
        try:
            ts = datetime.datetime.strptime(row[0].strip(), fmt_ts)
        except ValueError:
            continue
        samples.append((ts, [c.strip() for c in row[1:]]))
    runs = {}
    for mark in read_csv(os.path.join(out_dir, "run_marks.csv")):
        start = datetime.datetime.strptime(mark["start"], fmt_ts)
        end = datetime.datetime.strptime(mark["end"], fmt_ts)
        inside = [s for ts, s in samples if start <= ts <= end]
        if not inside:
            continue
        temp = [num(s[0]) for s in inside]
        sm = [num(s[1]) for s in inside]
        mem_clock = [num(s[2]) for s in inside]
        power = [num(s[3]) for s in inside]
        util = [num(s[4]) for s in inside]
        mem_used = [num(s[6]) for s in inside]
        # throttle bitmask without "GPU idle" (0x1) and "applications clocks setting" (0x2)
        throttled = [s for s in inside if s[7].startswith("0x") and int(s[7], 16) & ~0x3]
        thermal = [s for s in inside if s[7].startswith("0x") and int(s[7], 16) & (0x20 | 0x40)]
        runs[mark["run"]] = {
            "samples": len(inside), "wall_s": num(mark["wall_s"]), "exit": mark["exit"],
            "temp_mean": mean(temp), "temp_max": max(temp), "sm_mean": mean(sm), "sm_min": min(sm), "sm_max": max(sm),
            "mem_clock_mean": mean(mem_clock), "power_mean": mean(power), "util_mean": mean(util),
            "mem_used_max": max(mem_used), "throttled_pct": 100.0 * len(throttled) / len(inside),
            "thermal_pct": 100.0 * len(thermal) / len(inside)}
    return runs


def phase_rows(rows, phase, skip_first=True):
    return [r for i, r in enumerate(rows) if r["phase"] == phase and not (skip_first and i == 0)]


def scene_report(out_dir, scene, monitor):
    lines = []
    base = os.path.join(out_dir, scene)
    side = "gpu" if os.path.isdir(os.path.join(base, "gpuevents")) else "cpu"
    label = f"{scene}_poke_{side}"
    params_path = os.path.join(base, "plain", label + "_parameters.json")
    if not os.path.isfile(params_path):
        return [f"## {scene}: no run found", ""]
    params = json.load(open(params_path))
    plain = read_csv(os.path.join(base, "plain", label + ".csv"))
    events = read_csv(os.path.join(base, "gpuevents", label + ".csv"))
    collision = read_csv(os.path.join(base, "gpuevents", label + "_collision_timings.csv"))
    timer = parse_sofa_timer(os.path.join(base, "sofatimer", "run.log"))
    nosync = parse_sofa_timer(os.path.join(base, "sofatimer_nosync", "run.log"))
    summary = {}
    for line in open(os.path.join(base, "plain", label + "_summary.txt")) if os.path.isfile(
            os.path.join(base, "plain", label + "_summary.txt")) else []:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            summary[k] = v

    m, t, mo, c, co = params["mesh"], params["tool"], params["motion"], params["contact"], params["collision"]
    lines += [f"## {scene.capitalize()} ({side.upper()})", "", params["description"] + ".", "",
              "### Every setting", ""]
    lines += table(["Setting", "Value"], [
        ["Tissue mesh", f"{m['nodes']:,} nodes, {m['tetrahedra']:,} tetrahedra, {m['surface_triangles']:,} surface "
                        f"triangles ({m['surface_nodes']:,} surface nodes)"],
        ["Mesh grid (x, y, z lines)", f"{m['grid']['x']} x {m['grid']['y']} x {m['grid']['z']}; finest spacing "
                                      f"{1000 * m['fine_step_m']:.2f} mm within {1000 * m['fine_half_width_m']:.0f} mm "
                                      f"of the middle, growing x{m['growth']} outside"],
        ["Fixed nodes", f"{m['fixed_nodes']:,} ({'the rim' if scene == 'diaphragm' else 'the base'})"],
        ["Material", params["material"]["note"] + f", density {params['material']['density_kg_m3']:.0f} kg/m3"],
        ["Tool", f"rigid rod, {1000 * t['radius_m']:.0f} mm radius hemispherical tip, {1000 * t['length_m']:.0f} mm long: "
                 f"{t['vertices']:,} vertices, {t['triangles']:,} triangles ({t['segments']} around, {t['rings']} rings)"],
        ["Tool drive", f"{t['mass_kg']} kg, pulled by a spring of {t['coupling_stiffness_N_m']:,.0f} N/m "
                       f"(angular {t['coupling_angular_stiffness']} N m/rad)"],
        ["Motion", f"starts {1000 * mo['start_height_m']:.0f} mm above, down to {1000 * mo['depth_m']:.0f} mm below the "
                   f"undeformed top at {1000 * mo['speed_m_s']:.0f} mm/s; settle {mo['settle_s']} s, press "
                   f"{mo['press_s']:.2f} s, hold {mo['hold_s']} s, retract {mo['press_s']:.2f} s, rest {mo['rest_s']} s"],
        ["Time step", f"{mo['dt_s']} s, {mo['steps']} steps ({mo['total_time_s']:.2f} s simulated), gravity {mo['gravity'][1]} m/s2"],
        ["Contact", f"constraints with friction mu {c['friction']}, contact distance {1000 * c['contact_distance_m']:.1f} mm, "
                    f"alarm {1000 * c['alarm_distance_m']:.1f} mm, filter {c['filter']}"],
        ["Gauss-Seidel", f"tolerance {c['tolerance']:g}, at most {c['max_iterations']} sweeps"],
        ["Collision", f"way {co['way']}; grid {co['resolution'][0]} x {co['resolution'][1]} x {co['resolution'][2]} cells of "
                      f"{1000 * co['cell_m']:.0f} mm; room for {co['max_raw_contacts']:,} raw contacts"],
        ["Tissue factorisation", params["factorization"]],
    ])
    lines.append("")

    # ---- end to end ----------------------------------------------------------------
    lines += ["### One frame end to end (plain run, no timers)", "",
              "frame = wall time from the end of one step to the end of the next (Python clock); "
              "inside step = from SOFA's AnimateBeginEvent to its AnimateEndEvent. The first frame "
              "(start-up) is left out.", ""]
    rows = []
    for phase in PHASES + ["all"]:
        sel = plain[1:] if phase == "all" else phase_rows(plain, phase)
        if not sel:
            continue
        f = [num(r["frame_ms"]) for r in sel]
        rows.append([phase, str(len(sel)), mean([num(r["contacts"]) for r in sel]), mean([num(r["rows"]) for r in sel]),
                     mean(f), statistics.median(f), pct(f, 0.95), max(f), mean([num(r["step_ms"]) for r in sel]),
                     1000.0 / mean(f)])
    lines += table(["Phase", "Steps", "Contacts", "Rows", "Frame mean ms", "Median", "95%", "Max",
                    "Inside step ms", "Frames/s"], rows)
    lines += ["", f"First frame (start-up): {fmt(num(plain[0]['frame_ms']))} ms. Whole run: "
                  f"{fmt(monitor.get(scene + '/plain', {}).get('wall_s', float('nan')))} s wall "
                  f"(runSofa start to exit), {summary.get('run_wall_s', '-')} s from the first step to the last. "
                  f"Peak force {summary.get('peak_force_N', '-')} N at {summary.get('peak_force_time_s', '-')} s; "
                  f"most contacts {summary.get('max_contacts', '-')} ({summary.get('max_rows', '-')} rows) at "
                  f"{summary.get('max_rows_time_s', '-')} s; most Gauss-Seidel sweeps {summary.get('max_sweeps', '-')}; "
                  f"smallest element volume ratio {summary.get('min_volume_ratio', '-')}.", ""]

    if side == "gpu" and events:
        by_step = {int(r["step"]): r for r in collision}

        def stage_values(sel):
            out = {}
            for key in ["frame_ms", "step_ms", "tissue_gpu_ms"] + TISSUE_STAGES + ["contact_gpu_ms"] + CONTACT_STAGES:
                out[key] = mean([num(r[key]) for r in sel])
            for key in COLLISION_COLUMNS:
                out[key] = mean([num(by_step[int(r["step"])][key]) for r in sel if int(r["step"]) in by_step])
            for key in ["contacts", "rows", "sweeps", "gs_error"]:
                out[key] = mean([num(r[key]) for r in sel])
            gpu = out["tissue_gpu_ms"] + out["narrow_kernel_ms"] + out["contact_gpu_ms"]
            out["gpu_sum"] = gpu
            out["rest"] = out["frame_ms"] - gpu
            return out

        groups = [(p, phase_rows(events, p)) for p in PHASES]
        groups = [(p, sel) for p, sel in groups if sel]
        peak_row = max(events, key=lambda r: num(r["rows"]))
        groups.append((f"peak (step {peak_row['step']})", [peak_row]))
        values = [(p, stage_values(sel)) for p, sel in groups]
        lines += ["### Every stage of the step (CUDA events)", "",
                  "Each GPU stage timed with CUDA events (the CPU waits at the end of each, so this run's "
                  "frames are slower than the plain run's). 'Rest' = frame minus the GPU stages: SOFA's "
                  "own work, Python, the tool's CPU solve, launches and waiting.", ""]
        stage_rows = [
            ["**Tissue: free motion (GPU)**", "tissue_gpu_ms"],
            ["  material (stress + stiffness per tetrahedron)", "tissue_material_ms"],
            ["  matrix assembly", "tissue_assembly_ms"],
            ["  band factorisation", "tissue_factorize_ms"],
            ["  solve + refinement", "tissue_solve_ms"],
            ["**Collision (GPU kernels)**", "narrow_kernel_ms"],
            ["  collision as the CPU sees it", "narrow_wall_ms"],
            ["  of which waiting for the GPU", "narrow_host_synchronization_ms"],
            ["**Contact response (GPU)**", "contact_gpu_ms"],
            ["  A. choose contacts + build rows", "rows_ms"],
            ["  (factorisation: reused from the tissue)", "contact_factorize_ms"],
            ["  B. compliance W", "compliance_ms"],
            ["  C. Gauss-Seidel", "gauss_seidel_ms"],
            ["  D. correction", "correction_ms"],
            ["**GPU stages, total**", "gpu_sum"],
            ["**Rest (CPU, launches, waits)**", "rest"],
            ["**Frame (this run)**", "frame_ms"],
            ["Raw contacts from collision", "narrow_output_contact_count"],
            ["  vertex-face / face-vertex / edge-edge", None],
            ["  overflow (must be 0)", "narrow_overflow_count"],
            ["Contacts kept", "contacts"],
            ["Constraint rows", "rows"],
            ["Gauss-Seidel sweeps", "sweeps"],
            ["Collision kernel launches", "kernel_launch_count"],
        ]
        out_rows = []
        for name, key in stage_rows:
            if key is None:
                out_rows.append([name] + [f"{fmt(v['narrow_vf_contact_count'], 0)} / {fmt(v['narrow_fv_contact_count'], 0)}"
                                          f" / {fmt(v['narrow_ee_contact_count'], 0)}" for _, v in values])
            else:
                digits = 0 if "count" in key or key in ("contacts", "rows") else (1 if key == "sweeps" else 2)
                out_rows.append([name] + [fmt(v[key], digits) for _, v in values])
        lines += table(["Stage (ms per step)"] + [p for p, _ in values], out_rows)
        lines.append("")

        # ---- against the number of rows -------------------------------------------
        bins = [(0, 0), (1, 500), (501, 1000), (1001, 1500), (1501, 2000), (2001, 2500), (2501, 10 ** 9)]
        plain_by_step = {int(r["step"]): r for r in plain}
        bin_rows = []
        for lo, hi in bins:
            sel = [r for r in events[1:] if lo <= num(r["rows"]) <= hi]
            if not sel:
                continue
            v = stage_values(sel)
            pf = mean([num(plain_by_step[int(r["step"])]["frame_ms"]) for r in sel if int(r["step"]) in plain_by_step])
            bin_rows.append([f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+", str(len(sel)), v["narrow_output_contact_count"],
                             v["contacts"], v["sweeps"], v["tissue_gpu_ms"], v["narrow_kernel_ms"], v["rows_ms"],
                             v["compliance_ms"], v["gauss_seidel_ms"], v["correction_ms"], v["rest"], pf])
        lines += ["### Time against the number of contacts", "",
                  "Steps grouped by their constraint rows (3 per kept contact). Stage times from the CUDA-event run; "
                  "the last column is the plain run's frame for the same steps.", ""]
        lines += table(["Rows", "Steps", "Raw contacts", "Kept", "Sweeps", "Tissue", "Collision", "A rows", "B W",
                        "C Gauss-Seidel", "D correction", "Rest", "Frame (plain)"], bin_rows)
        lines.append("")

    # ---- SOFA's timer ------------------------------------------------------------------
    def timer_means(steps, sel_steps):
        sums, counts = {}, {}
        for i in sel_steps:
            if i >= len(steps):
                continue
            for k, v in steps[i].items():
                sums[k] = sums.get(k, 0.0) + v
            for k in steps[i]:
                counts[k] = counts.get(k, 0) + 1
        n = len([i for i in sel_steps if i < len(steps)])
        return {k: s / n for k, s in sums.items()} if n else {}

    contact_steps = [i for i, r in enumerate(plain) if num(r["rows"]) > 0 and i > 0]
    free_steps = [i for i, r in enumerate(plain) if num(r["rows"]) == 0 and i > 0]
    if timer:
        a = timer_means(timer, contact_steps)
        b = timer_means(nosync, contact_steps) if nosync else {}
        f = timer_means(timer, free_steps)
        keys = sorted(set(a) | set(b), key=lambda k: list(timer[contact_steps[0]].keys()).index(k)
                      if contact_steps and k in timer[contact_steps[0]] else 10 ** 6)
        tree = []
        for k in keys:
            va, vb, vf = a.get(k, float("nan")), b.get(k, float("nan")), f.get(k, float("nan"))
            if max(x for x in (va, vb, vf, 0.0) if x == x) < 0.05:
                continue
            depth = k.count("/")
            tree.append([("&nbsp;&nbsp;" * depth) + k.split("/")[-1], va, vb, vf])
        lines += ["### SOFA's own timer, every stage (mean ms per step)", "",
                  f"Steps in contact ({len(contact_steps)}) and steps without contact ({len(free_steps)}). "
                  "'Synced': SofaCUDA makes the timer wait for the GPU at every stage boundary, so each "
                  "stage holds its own GPU time. 'No sync' (CUDA_TIMER_SYNC=0): the GPU runs ahead and "
                  "the time shows up where the CPU waits for it. Stages under 0.05 ms are left out.", ""]
        lines += table(["Stage", "In contact, synced", "In contact, no sync", "No contact, synced"], tree)
        lines.append("")

    # ---- GPU clocks ------------------------------------------------------------------
    gm = [(k.split("/", 1)[1], v) for k, v in monitor.items() if k.startswith(scene + "/")]
    if gm:
        lines += ["### The GPU during each run (nvidia-smi every 200 ms)", ""]
        lines += table(["Run", "Wall s", "Samples", "Temp mean / max C", "SM clock mean (min-max) MHz",
                        "Memory clock MHz", "Power W", "GPU busy %", "Memory used max MiB",
                        "Slowed down % (thermal %)"],
                       [[k, v["wall_s"], str(v["samples"]), f"{fmt(v['temp_mean'], 0)} / {fmt(v['temp_max'], 0)}",
                         f"{fmt(v['sm_mean'], 0)} ({fmt(v['sm_min'], 0)}-{fmt(v['sm_max'], 0)})", v["mem_clock_mean"],
                         v["power_mean"], v["util_mean"], v["mem_used_max"],
                         f"{fmt(v['throttled_pct'], 0)} ({fmt(v['thermal_pct'], 0)})"] for k, v in gm])
        lines.append("")

    # ---- merged per-step CSV ----------------------------------------------------------
    merged_path = os.path.join(base, "steps_merged.csv")
    ev_by_step = {int(r["step"]): r for r in events}
    col_by_step = {int(r["step"]): r for r in collision}
    header = (["step", "time", "phase", "contacts", "rows", "sweeps", "frame_ms_plain", "step_ms_plain"]
              + [f"ev_{k}" for k in ["frame_ms", "tissue_gpu_ms"] + TISSUE_STAGES + ["contact_gpu_ms"] + CONTACT_STAGES]
              + [f"col_{k}" for k in COLLISION_COLUMNS]
              + [f"timer_{k}" for k in TIMER_STAGES] + [f"nosync_{k}" for k in TIMER_STAGES])
    with open(merged_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, r in enumerate(plain):
            s = int(r["step"])
            ev, co_ = ev_by_step.get(s, {}), col_by_step.get(s, {})
            row = [s, r["time"], r["phase"], r["contacts"], r["rows"], r["sweeps"], r["frame_ms"], r["step_ms"]]
            row += [ev.get(k, "") for k in ["frame_ms", "tissue_gpu_ms"] + TISSUE_STAGES + ["contact_gpu_ms"] + CONTACT_STAGES]
            row += [co_.get(k, "") for k in COLLISION_COLUMNS]
            for steps in (timer, nosync):
                row += [f"{steps[i].get(p, 0.0):.4f}" if i < len(steps) else "" for p in TIMER_STAGES.values()]
            w.writerow(row)
    lines += [f"Per-step data, all clocks: `{os.path.relpath(merged_path, out_dir)}`", ""]
    return lines


def main():
    out_dir = sys.argv[1]
    monitor = gpu_monitor(out_dir)
    lines = ["# Hard-poke stress test: every clock", ""]
    machine = os.path.join(out_dir, "machine.txt")
    if os.path.isfile(machine):
        lines += ["## Machine", ""] + table(["", ""], [line.strip().split("=", 1) for line in open(machine) if "=" in line])
        lines.append("")
    for scene in ("diaphragm", "cuboid"):
        if os.path.isdir(os.path.join(out_dir, scene)):
            lines += scene_report(out_dir, scene, monitor)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"report={path}")


if __name__ == "__main__":
    main()
