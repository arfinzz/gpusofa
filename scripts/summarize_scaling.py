"""Summarise scripts/scaling_study_wsl.sh: CPU (SOFA) against GPU time per step for
growing scene sizes.

Usage: python3 scripts/summarize_scaling.py <log dir>

Writes <log dir>/scaling_tissue.csv, scaling_poke.csv and, when matplotlib is
available, scaling.png. Times are wall-clock milliseconds per step (the time a
user waits for a step, everything included), averaged after two warm-up steps;
for the GPU also the GPU time of the tissue step by stage.
"""

import csv
import glob
import os
import re
import statistics
import sys


def bandwidth_in(paths):
    """The GPU tissue's half-bandwidth from a run's log (0: dense), or None."""
    for path in paths:
        for line in open(path, errors="replace"):
            m = re.search(r"factorisation: (band, half-bandwidth (\d+)|dense)", line)
            if m:
                return int(m.group(2)) if m.group(2) else 0
    return None


def mean_after_warmup(values, skip=2):
    values = values[skip:] if len(values) > skip + 1 else values
    return statistics.fmean(values) if values else float("nan")


def tissue_table(base):
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "tissue_*_d*"))):
        m = re.match(r"tissue_(.+)_d(\d+)$", os.path.basename(d))
        if not m:
            continue
        material, div = m.group(1), int(m.group(2))
        entry = {"material": material, "divisions": div, "nodes": (div + 1) ** 3, "tetrahedra": 6 * div ** 3,
                 "dofs": 3 * (div + 1) ** 3}
        for side in ("cpu", "gpu"):
            path = os.path.join(d, f"material_check_{material}_{side}.csv")
            if not os.path.isfile(path):
                continue
            data = list(csv.DictReader(open(path)))
            entry[f"{side}_ms"] = mean_after_warmup([float(r["wall_ms"]) for r in data])
            entry[f"{side}_steps"] = len(data)
            if side == "gpu" and data and "gpu_ms" in data[0]:
                for k in ("gpu_ms", "gpu_material_ms", "gpu_assembly_ms", "gpu_factorize_ms", "gpu_solve_ms"):
                    entry[k] = mean_after_warmup([float(r[k]) for r in data])
            if side == "gpu":
                entry["gpu_half_bandwidth"] = bandwidth_in(glob.glob(os.path.join(d, "*gpu*.log")))
        if "cpu_ms" in entry and "gpu_ms" in entry:
            entry["speedup"] = entry["cpu_ms"] / entry["gpu_ms"] if entry.get("gpu_ms") else float("nan")
        rows.append(entry)
    rows.sort(key=lambda r: (r["material"], r["divisions"]))
    return rows


def poke_table(base):
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "poke_fine*_*"))):
        m = re.match(r"poke_fine([0-9.]+)_(cpu|gpu)$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        fine, side = float(m.group(1)), m.group(2)
        path = os.path.join(d, f"tissue_poke_{side}.csv")
        if not os.path.isfile(path):
            continue
        data = list(csv.DictReader(open(path)))
        nodes = tets = None
        for log in glob.glob(os.path.join(d, "*.log")):
            for line in open(log, errors="replace"):
                mm = re.search(r"(\d+) nodes, (\d+) tetrahedra", line)
                if mm:
                    nodes, tets = int(mm.group(1)), int(mm.group(2))
                    break
        contact = [float(r["wall_ms"]) for r in data if r["phase"] in ("press", "hold") and float(r["force_y"]) > 1e-4]
        band = bandwidth_in(glob.glob(os.path.join(d, "*.log"))) if side == "gpu" else None
        rows.append({"fine_step_m": fine, "side": side, "nodes": nodes, "tetrahedra": tets, "steps": len(data),
                     "gpu_half_bandwidth": band,
                     "ms_all": mean_after_warmup([float(r["wall_ms"]) for r in data]),
                     "ms_contact": statistics.fmean(contact) if contact else float("nan"),
                     "contact_steps": len(contact)})
    # The same mesh on both sides: fill a side's size from the other if only one logged it.
    for r in rows:
        if r["nodes"] is None:
            other = next((o for o in rows if o["fine_step_m"] == r["fine_step_m"] and o["nodes"]), None)
            if other:
                r["nodes"], r["tetrahedra"] = other["nodes"], other["tetrahedra"]
    rows.sort(key=lambda r: (r["fine_step_m"], r["side"]))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    base = sys.argv[1]
    tissue = tissue_table(base)
    poke = poke_table(base)
    write_csv(os.path.join(base, "scaling_tissue.csv"), tissue)
    write_csv(os.path.join(base, "scaling_poke.csv"), poke)

    if tissue:
        print("Tissue alone (material_check cube), ms per step:")
        print(f"{'material':15s} {'cells':>5s} {'nodes':>6s} {'tets':>7s} {'CPU ms':>9s} {'GPU ms':>8s} {'x':>6s}"
              f"   GPU: {'material':>8s} {'assembly':>8s} {'factor':>8s} {'solve':>7s} {'band':>6s}")
        for r in tissue:
            print(f"{r['material']:15s} {r['divisions']:5d} {r['nodes']:6d} {r['tetrahedra']:7d} "
                  f"{r.get('cpu_ms', float('nan')):9.1f} {r.get('gpu_ms', float('nan')):8.2f} "
                  f"{r.get('speedup', float('nan')):6.1f}   "
                  f"     {r.get('gpu_material_ms', float('nan')):8.2f} {r.get('gpu_assembly_ms', float('nan')):8.2f} "
                  f"{r.get('gpu_factorize_ms', float('nan')):8.2f} {r.get('gpu_solve_ms', float('nan')):7.2f} "
                  f"{r.get('gpu_half_bandwidth') if r.get('gpu_half_bandwidth') is not None else '':>6}")
    if poke:
        print("\nWhole poke (tissue + collision + constraint contact), ms per step:")
        print(f"{'fine step':>9s} {'side':>4s} {'nodes':>6s} {'tets':>7s} {'all steps':>10s} {'contact steps':>14s}")
        for r in poke:
            print(f"{r['fine_step_m'] * 1000:8.1f}mm {r['side']:>4s} {r['nodes'] or 0:6d} {r['tetrahedra'] or 0:7d} "
                  f"{r['ms_all']:10.1f} {r['ms_contact']:14.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2 if poke else 1, figsize=(12 if poke else 6.5, 4.8))
    axes = axes if poke else [axes]
    ax = axes[0]
    for material in sorted({r["material"] for r in tissue}):
        sel = [r for r in tissue if r["material"] == material]
        for side, style in (("cpu", "o-"), ("gpu", "s--")):
            pts = [(r["nodes"], r[f"{side}_ms"]) for r in sel if f"{side}_ms" in r]
            if pts:
                ax.plot(*zip(*pts), style, label=f"{material} {side.upper()}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tissue nodes")
    ax.set_ylabel("ms per step (wall clock)")
    ax.set_title("Tissue alone: SOFA CPU vs GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if poke:
        ax = axes[1]
        for side, style in (("cpu", "o-"), ("gpu", "s--")):
            pts = [(r["nodes"], r["ms_contact"]) for r in poke if r["side"] == side and r["nodes"]]
            if pts:
                ax.plot(*zip(*sorted(pts)), style, label=f"{side.upper()} (steps in contact)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("tissue nodes")
        ax.set_ylabel("ms per step (wall clock)")
        ax.set_title("Whole poke: SOFA CPU vs GPU")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(base, "scaling.png"), dpi=130)


if __name__ == "__main__":
    main()
