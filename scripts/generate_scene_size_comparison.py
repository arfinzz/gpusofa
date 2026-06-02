#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def read_key_value_file(path):
    values = {}
    with open(path, encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value
    return values


def as_float(values, key, default=0.0):
    try:
        return float(values.get(key, default))
    except ValueError:
        return default


def as_int(values, key, default=0):
    try:
        return int(float(values.get(key, default)))
    except ValueError:
        return default


def case_sort_key(name):
    order = {"medium": 0, "large": 1, "xlarge": 2, "xxlarge": 3}
    return order.get(name, 100), name


def load_cases(base_dir):
    base = Path(base_dir)
    cases = []
    for case_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: case_sort_key(p.name)):
        cpu_summaries = sorted(case_dir.glob("cpu_large_tissue_blade_benchmark*_summary.txt"))
        gpu_summaries = sorted(case_dir.glob("gpu_large_tissue_blade_dense_grid_benchmark*_summary.txt"))
        compute_summary = case_dir / "compute_only_summary.txt"
        if not cpu_summaries or not gpu_summaries or not compute_summary.exists():
            continue

        cpu = read_key_value_file(cpu_summaries[-1])
        gpu = read_key_value_file(gpu_summaries[-1])
        compute = read_compute_summary(compute_summary)
        gpu_compute = compute.get("gpu", {})

        total_tris = as_int(gpu, "collision_element_count")
        row = {
            "case": case_dir.name,
            "collision_triangles": total_tris,
            "cpu_wall_ms": as_float(cpu, "avg_step_seconds") * 1000.0,
            "gpu_wall_ms": as_float(gpu, "avg_step_seconds") * 1000.0,
            "gpu_compute_ms": as_float(gpu_compute, "gpu_compute_avg_seconds") * 1000.0,
            "gpu_pipeline_wall_ms": as_float(gpu_compute, "pipeline_wall_avg_seconds") * 1000.0,
            "gpu_overhead_ms": as_float(gpu_compute, "wall_minus_gpu_compute_avg_seconds") * 1000.0,
            "h2d_bytes": as_int(gpu, "avg_host_to_device_bytes"),
            "d2h_bytes": as_int(gpu, "avg_device_to_host_bytes"),
            "raw_candidates": as_int(gpu, "avg_narrow_raw_candidate_count"),
            "unique_candidates": as_int(gpu, "avg_narrow_unique_candidate_count"),
            "contacts": as_int(gpu, "avg_narrow_output_contact_count"),
            "grid_cells": as_int(gpu, "avg_narrow_grid_cell_count"),
            "overflow": as_int(gpu, "avg_narrow_overflow_count"),
            "measured_steps": as_int(gpu, "measured_steps"),
        }
        cases.append(row)
    return cases


def read_compute_summary(path):
    sections = {}
    current = None
    with open(path, encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1]
                current = "gpu" if name.startswith("gpu_") else "cpu"
                sections[current] = {}
                continue
            if current is None or "=" not in line:
                continue
            key, value = line.split("=", 1)
            sections[current][key] = value
    return sections


def write_csv(cases, path):
    fieldnames = [
        "case",
        "collision_triangles",
        "measured_steps",
        "cpu_wall_ms",
        "gpu_wall_ms",
        "gpu_compute_ms",
        "gpu_pipeline_wall_ms",
        "gpu_overhead_ms",
        "h2d_bytes",
        "d2h_bytes",
        "raw_candidates",
        "unique_candidates",
        "contacts",
        "grid_cells",
        "overflow",
    ]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in cases:
            writer.writerow(row)


def svg_line(points, color):
    if not points:
        return ""
    d = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />'
        for x, y in points
    )
    return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3" />\n{circles}'


def write_svg(cases, path):
    width, height = 1120, 720
    left, right, top, bottom = 95, 40, 60, 115
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_values = [row["collision_triangles"] for row in cases]
    y_values = []
    for row in cases:
        y_values.extend([row["cpu_wall_ms"], row["gpu_wall_ms"], row["gpu_compute_ms"]])

    min_x, max_x = min(x_values), max(x_values)
    max_y = max(y_values) * 1.15
    max_y = max(1.0, max_y)

    def x_map(value):
        if max_x == min_x:
            return left + plot_w * 0.5
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y_map(value):
        return top + plot_h - (value / max_y) * plot_h

    cpu_points = [(x_map(row["collision_triangles"]), y_map(row["cpu_wall_ms"])) for row in cases]
    gpu_wall_points = [(x_map(row["collision_triangles"]), y_map(row["gpu_wall_ms"])) for row in cases]
    gpu_compute_points = [(x_map(row["collision_triangles"]), y_map(row["gpu_compute_ms"])) for row in cases]

    y_ticks = 6
    grid = []
    for i in range(y_ticks + 1):
        value = max_y * i / y_ticks
        y = y_map(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9dee7" />'
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )

    x_labels = []
    for row in cases:
        x = x_map(row["collision_triangles"])
        x_labels.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#edf0f5" />'
            f'<text x="{x:.2f}" y="{height-bottom+28}" text-anchor="middle" font-size="12">{row["case"]}</text>'
            f'<text x="{x:.2f}" y="{height-bottom+47}" text-anchor="middle" font-size="11">{row["collision_triangles"]:,} tris</text>'
        )

    labels = []
    for row in cases:
        x = x_map(row["collision_triangles"])
        labels.append(
            f'<text x="{x:.2f}" y="{y_map(row["gpu_wall_ms"])-10:.2f}" text-anchor="middle" font-size="11" fill="#b42318">{row["gpu_wall_ms"]:.2f}</text>'
        )
        labels.append(
            f'<text x="{x:.2f}" y="{y_map(row["gpu_compute_ms"])+18:.2f}" text-anchor="middle" font-size="11" fill="#1c6b3c">{row["gpu_compute_ms"]:.2f}</text>'
        )
        labels.append(
            f'<text x="{x:.2f}" y="{y_map(row["cpu_wall_ms"])-10:.2f}" text-anchor="middle" font-size="11" fill="#1f5f99">{row["cpu_wall_ms"]:.2f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-weight="700">Scene Size Scaling: CPU vs GPU Dense Grid</text>
  <text x="{width/2}" y="52" text-anchor="middle" font-size="13" fill="#58606c">Wall time includes SOFA/frame overhead. GPU compute-only is CUDA event kernel timing.</text>
  {"".join(grid)}
  {"".join(x_labels)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#2b2f36" stroke-width="2" />
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#2b2f36" stroke-width="2" />
  <text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-size="14">Average time per step (ms)</text>
  <text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14">Collision triangle count</text>
  {svg_line(cpu_points, "#1f5f99")}
  {svg_line(gpu_wall_points, "#b42318")}
  {svg_line(gpu_compute_points, "#1c6b3c")}
  {"".join(labels)}
  <rect x="790" y="86" width="285" height="96" rx="8" fill="#fff" stroke="#ccd3dd" />
  <line x1="810" y1="112" x2="850" y2="112" stroke="#1f5f99" stroke-width="3" />
  <text x="860" y="116" font-size="13">CPU wall time</text>
  <line x1="810" y1="138" x2="850" y2="138" stroke="#b42318" stroke-width="3" />
  <text x="860" y="142" font-size="13">GPU wall time</text>
  <line x1="810" y1="164" x2="850" y2="164" stroke="#1c6b3c" stroke-width="3" />
  <text x="860" y="168" font-size="13">GPU compute-only</text>
</svg>
'''
    Path(path).write_text(svg, encoding="utf-8")


def write_markdown(cases, csv_path, svg_path, path):
    lines = [
        "# Scene Size Scaling Benchmark",
        "",
        "Generated from scaling benchmark logs.",
        "",
        f"- CSV: `{csv_path}`",
        f"- SVG graph: `{svg_path}`",
        "",
        "| Case | Collision triangles | CPU wall ms | GPU wall ms | GPU compute-only ms | H2D bytes/step | Raw candidates | Unique candidates | Contacts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            f"| {row['case']} | {row['collision_triangles']} | "
            f"{row['cpu_wall_ms']:.6f} | {row['gpu_wall_ms']:.6f} | {row['gpu_compute_ms']:.6f} | "
            f"{row['h2d_bytes']} | {row['raw_candidates']} | {row['unique_candidates']} | {row['contacts']} |"
        )

    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `GPU wall ms` is the full measured step time from the GPU benchmark scene.",
        "- `GPU compute-only ms` is `broad_kernel_ms + narrow_kernel_ms` from CUDA event timing.",
        "- H2D transfer bytes scale directly with uploaded triangle records because the current path uploads packed `DeviceTriangle` arrays every step.",
        "- If GPU compute-only approaches CPU wall time while GPU wall time remains much higher, the missing speedup is mostly transfer/orchestration rather than raw device kernel throughput.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate scene-size scaling CSV, SVG, and Markdown report.")
    parser.add_argument("base_log_dir")
    parser.add_argument("--csv", default="reports/scene_size_scaling_summary.csv")
    parser.add_argument("--svg", default="reports/scene_size_scaling_comparison.svg")
    parser.add_argument("--md", default="reports/scene_size_scaling_benchmark_report.md")
    args = parser.parse_args()

    cases = load_cases(args.base_log_dir)
    if not cases:
        raise SystemExit(f"No complete benchmark cases found under {args.base_log_dir}")

    write_csv(cases, args.csv)
    write_svg(cases, args.svg)
    write_markdown(cases, args.csv, args.svg, args.md)

    print(f"cases={len(cases)}")
    print(f"csv={args.csv}")
    print(f"svg={args.svg}")
    print(f"md={args.md}")


if __name__ == "__main__":
    main()
