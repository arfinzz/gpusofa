#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


GPU_PROFILE_COLUMNS = [
    ("host_preparation", "narrow_host_preparation_ms"),
    ("sofa_triangle_extraction", "narrow_sofa_triangle_extraction_ms"),
    ("backend_triangle_pack", "narrow_backend_triangle_pack_ms"),
    ("h2d_upload", "narrow_h2d_ms"),
    ("device_allocation", "narrow_device_allocation_ms"),
    ("clear_grid", "narrow_clear_grid_ms"),
    ("clear_counters", "narrow_counter_clear_ms"),
    ("tissue_aabb", "narrow_tissue_aabb_ms"),
    ("tool_aabb", "narrow_tool_aabb_ms"),
    ("insert_tissue", "narrow_insert_tissue_ms"),
    ("insert_tool", "narrow_insert_tool_ms"),
    ("generate_pairs", "narrow_generate_pairs_ms"),
    ("candidate_readback", "narrow_candidate_readback_ms"),
    ("sort_unique", "narrow_sort_unique_ms"),
    ("sort_unique_host", "narrow_sort_unique_host_ms"),
    ("exact_contact", "narrow_exact_contact_ms"),
    ("contact_count_readback", "narrow_contact_count_readback_ms"),
    ("contact_download", "narrow_contact_download_ms"),
    ("sofa_output_publish", "narrow_sofa_output_publish_ms"),
]

GPU_COUNT_COLUMNS = [
    ("raw_candidates", "narrow_raw_candidate_count"),
    ("unique_candidates", "narrow_unique_candidate_count"),
    ("duplicate_reduction_ratio", "narrow_duplicate_reduction_ratio"),
    ("contacts", "narrow_output_contact_count"),
    ("grid_cells", "narrow_grid_cell_count"),
    ("active_mixed_cells", "narrow_active_mixed_cell_count"),
    ("tissue_inserts", "narrow_tissue_insert_count"),
    ("tool_inserts", "narrow_tool_insert_count"),
    ("max_tissue_cell_occupancy", "narrow_max_tissue_cell_occupancy"),
    ("max_tool_cell_occupancy", "narrow_max_tool_cell_occupancy"),
    ("overflow", "narrow_overflow_count"),
    ("hash_dedupe_probe_overflow", "narrow_hash_dedupe_probe_overflow_count"),
    ("kernel_launch_count", "kernel_launch_count"),
    ("cuda_memset_count", "cuda_memset_count"),
    ("workspace_resize_count", "workspace_resize_count"),
]


def parse_float(row, key, default=0.0):
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def parse_int(row, key, default=0):
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def read_measured_rows(csv_path):
    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if parse_int(row, "included_in_stats") == 1]
    return rows


def summarize_csv(csv_path):
    rows = read_measured_rows(csv_path)
    if not rows:
        return None

    headers = set(rows[0].keys())
    measured_steps = len(rows)
    wall_seconds = sum(parse_float(row, "duration_seconds") for row in rows)

    has_gpu_columns = {
        "broad_kernel_ms",
        "narrow_kernel_ms",
        "host_to_device_bytes",
        "device_to_host_bytes",
    }.issubset(headers)

    summary = {
        "csv_path": str(csv_path),
        "measured_steps": measured_steps,
        "wall_total_seconds": wall_seconds,
        "wall_avg_seconds": wall_seconds / measured_steps,
        "has_gpu_columns": has_gpu_columns,
    }

    if has_gpu_columns:
        pure_compute_ms = sum(
            parse_float(row, "broad_kernel_ms") + parse_float(row, "narrow_kernel_ms")
            for row in rows
        )
        narrow_wall_ms = sum(parse_float(row, "narrow_wall_ms") for row in rows)
        broad_wall_ms = sum(parse_float(row, "broad_wall_ms") for row in rows)
        h2d_bytes = sum(parse_int(row, "host_to_device_bytes") for row in rows)
        d2h_bytes = sum(parse_int(row, "device_to_host_bytes") for row in rows)

        wall_avg_ms = wall_seconds * 1000.0 / measured_steps
        summary.update({
            "gpu_compute_total_seconds": pure_compute_ms / 1000.0,
            "gpu_compute_avg_seconds": pure_compute_ms / (1000.0 * measured_steps),
            "pipeline_wall_total_seconds": (broad_wall_ms + narrow_wall_ms) / 1000.0,
            "pipeline_wall_avg_seconds": (broad_wall_ms + narrow_wall_ms) / (1000.0 * measured_steps),
            "wall_avg_ms": wall_avg_ms,
            "host_to_device_total_bytes": h2d_bytes,
            "host_to_device_avg_bytes": h2d_bytes // measured_steps,
            "device_to_host_total_bytes": d2h_bytes,
            "device_to_host_avg_bytes": d2h_bytes // measured_steps,
            "wall_minus_gpu_compute_total_seconds": wall_seconds - pure_compute_ms / 1000.0,
            "wall_minus_gpu_compute_avg_seconds": wall_seconds / measured_steps - pure_compute_ms / (1000.0 * measured_steps),
            "avg_raw_candidates": sum(parse_int(row, "narrow_raw_candidate_count") for row in rows) // measured_steps,
            "avg_unique_candidates": sum(parse_int(row, "narrow_unique_candidate_count") for row in rows) // measured_steps,
            "avg_contacts": sum(parse_int(row, "narrow_output_contact_count") for row in rows) // measured_steps,
            "avg_grid_cells": sum(parse_int(row, "narrow_grid_cell_count") for row in rows) // measured_steps,
            "avg_overflow": sum(parse_int(row, "narrow_overflow_count") for row in rows) // measured_steps,
        })

        profile_columns = [
            (label, column)
            for label, column in GPU_PROFILE_COLUMNS
            if column in headers
        ]
        if profile_columns:
            summary["gpu_profile"] = [
                {
                    "label": label,
                    "column": column,
                    "total_ms": sum(parse_float(row, column) for row in rows),
                    "avg_ms": sum(parse_float(row, column) for row in rows) / measured_steps,
                    "min_ms": min(parse_float(row, column) for row in rows),
                    "max_ms": max(parse_float(row, column) for row in rows),
                    "percent_of_wall": (
                        (sum(parse_float(row, column) for row in rows) / measured_steps) / wall_avg_ms * 100.0
                        if wall_avg_ms > 0.0 else 0.0
                    ),
                }
                for label, column in profile_columns
            ]
            summary["top_gpu_profile"] = sorted(
                summary["gpu_profile"],
                key=lambda item: item["avg_ms"],
                reverse=True,
            )[:3]

        count_columns = [
            (label, column)
            for label, column in GPU_COUNT_COLUMNS
            if column in headers
        ]
        if count_columns:
            summary["gpu_counts"] = [
                {
                    "label": label,
                    "column": column,
                    "avg": sum(parse_float(row, column) for row in rows) / measured_steps,
                    "min": min(parse_float(row, column) for row in rows),
                    "max": max(parse_float(row, column) for row in rows),
                }
                for label, column in count_columns
            ]

    return summary


def find_timing_csvs(log_dir):
    paths = sorted(Path(log_dir).glob("*_timings.csv"))
    return paths


def format_summary(summary):
    lines = []
    name = Path(summary["csv_path"]).name
    lines.append(f"[{name}]")
    lines.append(f"measured_steps={summary['measured_steps']}")
    lines.append(f"wall_total_seconds={summary['wall_total_seconds']:.9f}")
    lines.append(f"wall_avg_seconds={summary['wall_avg_seconds']:.9f}")

    if not summary["has_gpu_columns"]:
        lines.append("gpu_compute_total_seconds=not_available")
        lines.append("gpu_compute_avg_seconds=not_available")
        lines.append("note=CSV has no GPU stage columns; for the CPU scene, there is no H2D/D2H transfer to exclude.")
        return lines

    lines.append(f"gpu_compute_total_seconds={summary['gpu_compute_total_seconds']:.9f}")
    lines.append(f"gpu_compute_avg_seconds={summary['gpu_compute_avg_seconds']:.9f}")
    lines.append(f"pipeline_wall_total_seconds={summary['pipeline_wall_total_seconds']:.9f}")
    lines.append(f"pipeline_wall_avg_seconds={summary['pipeline_wall_avg_seconds']:.9f}")
    lines.append(f"wall_minus_gpu_compute_total_seconds={summary['wall_minus_gpu_compute_total_seconds']:.9f}")
    lines.append(f"wall_minus_gpu_compute_avg_seconds={summary['wall_minus_gpu_compute_avg_seconds']:.9f}")
    lines.append(f"host_to_device_total_bytes={summary['host_to_device_total_bytes']}")
    lines.append(f"host_to_device_avg_bytes={summary['host_to_device_avg_bytes']}")
    lines.append(f"device_to_host_total_bytes={summary['device_to_host_total_bytes']}")
    lines.append(f"device_to_host_avg_bytes={summary['device_to_host_avg_bytes']}")
    lines.append(f"avg_raw_candidates={summary['avg_raw_candidates']}")
    lines.append(f"avg_unique_candidates={summary['avg_unique_candidates']}")
    lines.append(f"avg_contacts={summary['avg_contacts']}")
    lines.append(f"avg_grid_cells={summary['avg_grid_cells']}")
    lines.append(f"avg_overflow={summary['avg_overflow']}")
    if summary.get("gpu_profile"):
        lines.append("gpu_stage_profile_avg_ms:")
        for item in summary["gpu_profile"]:
            lines.append(
                f"  {item['label']}={item['avg_ms']:.9f},min={item['min_ms']:.9f},"
                f"max={item['max_ms']:.9f},total={item['total_ms']:.9f},pct_wall={item['percent_of_wall']:.3f}"
            )
        lines.append("gpu_stage_profile_total_ms:")
        for item in summary["gpu_profile"]:
            lines.append(f"  {item['label']}={item['total_ms']:.9f}")
        lines.append("top_gpu_stage_bottlenecks:")
        for index, item in enumerate(summary.get("top_gpu_profile", []), start=1):
            lines.append(f"  {index}. {item['label']}={item['avg_ms']:.9f} ms ({item['percent_of_wall']:.3f}% wall)")
    if summary.get("gpu_counts"):
        lines.append("gpu_count_profile:")
        for item in summary["gpu_counts"]:
            lines.append(f"  {item['label']}=avg:{item['avg']:.6f},min:{item['min']:.6f},max:{item['max']:.6f}")
    lines.append(
        "note=gpu_compute_* uses CUDA event timing for device stages. h2d_upload/candidate_readback/"
        "contact_count_readback/contact_download are host-wall transfer timings and explain why wall time can be "
        "larger than kernel time."
    )
    return lines


def write_markdown_report(summaries, output_path):
    lines = [
        "# GPU Collision Detection Profiling Report",
        "",
        "This report is generated from benchmark CSV files. CPU rows report wall time only; GPU rows include CUDA-event stage timing, transfer timing, candidate counts, and dense-grid occupancy stats.",
        "",
    ]

    for summary in summaries:
        name = Path(summary["csv_path"]).name
        lines.extend([
            f"## {name}",
            "",
            f"- measured steps: {summary['measured_steps']}",
            f"- wall avg: {summary['wall_avg_seconds'] * 1000.0:.6f} ms",
        ])
        if not summary["has_gpu_columns"]:
            lines.extend(["- GPU stage profile: not available", ""])
            continue

        lines.extend([
            f"- GPU compute avg: {summary['gpu_compute_avg_seconds'] * 1000.0:.6f} ms",
            f"- pipeline wall avg: {summary['pipeline_wall_avg_seconds'] * 1000.0:.6f} ms",
            f"- H2D avg bytes: {summary['host_to_device_avg_bytes']}",
            f"- D2H avg bytes: {summary['device_to_host_avg_bytes']}",
            "",
            "| Stage | Avg ms | Min ms | Max ms | Total ms | % wall |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for item in summary.get("gpu_profile", []):
            lines.append(
                f"| {item['label']} | {item['avg_ms']:.6f} | {item['min_ms']:.6f} | "
                f"{item['max_ms']:.6f} | {item['total_ms']:.6f} | {item['percent_of_wall']:.2f}% |"
            )

        lines.extend(["", "Top bottlenecks:"])
        for index, item in enumerate(summary.get("top_gpu_profile", []), start=1):
            lines.append(f"{index}. {item['label']}: {item['avg_ms']:.6f} ms avg")

        if summary.get("gpu_counts"):
            lines.extend(["", "| Count | Avg | Min | Max |", "|---|---:|---:|---:|"])
            for item in summary["gpu_counts"]:
                lines.append(f"| {item['label']} | {item['avg']:.6f} | {item['min']:.6f} | {item['max']:.6f} |")
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def write_stage_svg(summaries, output_path):
    gpu_summaries = [summary for summary in summaries if summary.get("gpu_profile")]
    if not gpu_summaries:
        return
    summary = gpu_summaries[-1]
    items = [item for item in summary["gpu_profile"] if item["avg_ms"] > 0.0]
    if not items:
        return

    width = 1100
    row_height = 34
    height = 90 + row_height * len(items)
    label_width = 260
    chart_width = width - label_width - 80
    max_value = max(item["avg_ms"] for item in items)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111">GPU collision stage average time</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#444">{Path(summary["csv_path"]).name}</text>',
    ]
    y = 88
    for item in items:
        bar_width = 0 if max_value == 0 else chart_width * item["avg_ms"] / max_value
        lines.append(f'<text x="24" y="{y + 18}" font-family="Arial" font-size="14" fill="#111">{item["label"]}</text>')
        lines.append(f'<rect x="{label_width}" y="{y}" width="{bar_width:.2f}" height="22" fill="#2563eb"/>')
        lines.append(f'<text x="{label_width + bar_width + 8:.2f}" y="{y + 17}" font-family="Arial" font-size="13" fill="#111">{item["avg_ms"]:.4f} ms</text>')
        y += row_height
    lines.append("</svg>")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark wall time and compute-only CUDA timing from SOFA CSV logs."
    )
    parser.add_argument("log_dir", help="Directory containing benchmark *_timings.csv files.")
    parser.add_argument("--output", help="Optional output text file.")
    parser.add_argument("--markdown", help="Optional Markdown profiling report.")
    parser.add_argument("--svg", help="Optional SVG stage-time chart for the last GPU CSV.")
    args = parser.parse_args()

    csv_paths = find_timing_csvs(args.log_dir)
    if not csv_paths:
        raise SystemExit(f"No *_timings.csv files found in {args.log_dir}")

    output_lines = []
    output_lines.append(f"log_dir={os.path.abspath(args.log_dir)}")
    output_lines.append("")

    summaries = []
    for csv_path in csv_paths:
        summary = summarize_csv(csv_path)
        if summary is None:
            continue
        summaries.append(summary)
        output_lines.extend(format_summary(summary))
        output_lines.append("")

    output = "\n".join(output_lines).rstrip() + "\n"
    print(output, end="")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
    if args.markdown:
        write_markdown_report(summaries, args.markdown)
    if args.svg:
        write_stage_svg(summaries, args.svg)


if __name__ == "__main__":
    main()
