"""Plots of validation runs (testscenes/validationtests/): SOFA's CPU components
(solid) against the GPU (dashed), with the known answer where there is one.

Usage: python3 scripts/plot_validation.py <log dir> [output png]

One figure with a panel per test found in the log dir: the grasp (block rise
against the jaws' for each friction), cutting (tip deflection through the cut),
the incline (slide against Coulomb's law) and confined compression (stretch against
the exact answer).
"""

import csv
import glob
import math
import os
import re
import sys


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def column(rows, key, scale=1.0):
    return [float(r[key]) * scale for r in rows if r.get(key) not in (None, "", "nan")]


def pairs(base, pattern):
    """(label, cpu rows, gpu rows) for every <stem>_cpu.csv / _gpu.csv matching pattern."""
    out = []
    for cpu in sorted(glob.glob(os.path.join(base, pattern + "_cpu.csv"))):
        gpu = cpu[:-len("_cpu.csv")] + "_gpu.csv"
        if os.path.isfile(gpu):
            out.append((os.path.basename(cpu)[:-len("_cpu.csv")], load(cpu), load(gpu)))
    return out


def main():
    base = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "validation.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    grasp = pairs(base, "grasp_lift_mu*")
    if grasp:
        panels.append("grasp")
    cutting = pairs(base, "cutting_*")
    if cutting:
        panels.append("cutting")
    incline = pairs(base, "incline_friction_a*")
    if incline:
        panels.append("incline")
    confined = pairs(base, "confined_compression_large_*")
    if confined:
        panels.append("confined")
    if not panels:
        print("no validation runs found")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    axes = axes if len(panels) > 1 else [axes]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, panel in zip(axes, panels):
        if panel == "grasp":
            for k, (label, c, g) in enumerate(grasp):
                mu = re.search(r"mu([0-9.]+)", label).group(1)
                col = colors[k % len(colors)]
                ax.plot(column(c, "time"), column(c, "block_rise", 1000), "-", color=col, label=f"mu={mu} CPU")
                ax.plot(column(g, "time"), column(g, "block_rise", 1000), "--", color=col, label=f"mu={mu} GPU")
            c = grasp[0][1]
            ax.plot(column(c, "time"), column(c, "jaw_rise", 1000), ":", color="black", label="jaws")
            ax.set_title("Grasp and lift: block rise")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("rise (mm)")
        elif panel == "cutting":
            for k, (label, c, g) in enumerate(cutting):
                material = label[len("cutting_"):]
                col = colors[k % len(colors)]
                ax.plot(column(c, "time"), [-v for v in column(c, "tip_dy", 1000)], "-", color=col, label=f"{material} CPU")
                ax.plot(column(g, "time"), [-v for v in column(g, "tip_dy", 1000)], "--", color=col, label=f"{material} GPU")
            ax.axvspan(2.0, 2.5, color="grey", alpha=0.15, label="cutting")
            ax.set_title("Cutting a slot into a loaded beam")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("tip deflection (mm)")
        elif panel == "incline":
            for k, (label, c, g) in enumerate(incline):
                angle = re.search(r"_a([0-9.]+)_", label).group(1)
                col = colors[k % len(colors)]
                ax.plot(column(c, "time"), column(c, "com_dx", 1000), "-", color=col, label=f"{angle} deg CPU")
                ax.plot(column(g, "time"), column(g, "com_dx", 1000), "--", color=col, label=f"{angle} deg GPU")
                if "expected_dx" in c[0]:
                    ax.plot(column(c, "time"), column(c, "expected_dx", 1000), ":", color=col, label=f"{angle} deg Coulomb")
            ax.set_title("Block on an incline (mu = 0.3)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("slide (mm)")
        elif panel == "confined":
            for k, (label, c, g) in enumerate(confined):
                material = label[len("confined_compression_large_"):]
                col = colors[k % len(colors)]
                ax.plot(column(c, "time"), column(c, "stretch"), "-", color=col, label=f"{material} CPU")
                ax.plot(column(g, "time"), column(g, "stretch"), "--", color=col, label=f"{material} GPU")
                ax.plot(column(c, "time"), column(c, "expected"), ":", color=col)
            ax.set_title("Confined compression (dotted: exact)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("stretch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    print(target)


if __name__ == "__main__":
    main()
