# Large Tissue/Blade Benchmark Summary

Date: 2026-05-08

## Scene

Two collision objects:

- Tissue: static high-resolution surface grid
- Blade: static subdivided rectangular slab intersecting the tissue plane

Geometry:

| Item | Count |
|---|---:|
| Tissue vertices | 32761 |
| Tissue triangles | 64800 |
| Blade vertices | 7990 |
| Blade triangles | 14720 |
| Total collision vertices | 40751 |
| Total collision triangles | 79520 |

## Files Added

| File | Purpose |
|---|---|
| `test_cpu_large_tissue_blade_benchmark.py` | CPU reference scene using `ParallelBruteForceBroadPhase` and `ParallelBVHNarrowPhase` |
| `test_gpu_large_tissue_blade_dense_grid_benchmark.py` | GPU dense-grid scene using `GpuCollisionBroadPhase` and `GpuCollisionNarrowPhase` |
| `scripts/run_large_tissue_blade_benchmarks_wsl.sh` | WSL runner for both scenes |
| `dense_collision_benchmark_common.py` | Added subdivided blade mesh generator |

## Run

Original command:

```bash
export SOFA_BENCHMARK_STEPS=120
export SOFA_LARGE_WARMUP_STEPS=10
export SOFA_BENCHMARK_LABEL_SUFFIX="_large_verified_20260508"
bash scripts/run_large_tissue_blade_benchmarks_wsl.sh
```

Environment:

- WSL distro: `wsl-gpu-proj`
- SOFA: `/opt/sofa/install/v25.12`
- GPU: NVIDIA GeForce GTX 1650 Ti
- Mode: batch
- Measured steps: 110
- Warmup steps: 10

Output directory:

`benchmark_logs/large_tissue_blade_20260508_005257`

Compute-breakdown rerun command:

```bash
export SOFA_BENCHMARK_STEPS=120
export SOFA_LARGE_WARMUP_STEPS=10
export SOFA_BENCHMARK_LABEL_SUFFIX="_large_compute_20260508"
bash scripts/run_large_tissue_blade_compute_breakdown_wsl.sh
```

Compute-breakdown output directory:

`benchmark_logs/large_tissue_blade_compute_20260508_010035`

## H2D Byte Check

The reported `3180800 bytes/step` H2D value is correct for the current dense-grid path.

| Item | Value |
|---|---:|
| Tissue triangles | 64800 |
| Blade triangles | 14720 |
| Total uploaded triangles | 79520 |
| `DeviceTriangle` bytes | 40 |
| Expected H2D bytes | 3180800 |

The code records this as `tissueTriangleBytes + toolTriangleBytes`, where each `DeviceTriangle`
contains three `float3` positions and one `uint32` triangle id.

## Results

Original run:

| Metric | CPU | GPU dense grid |
|---|---:|---:|
| Average step time | 0.003376584 s | 0.009307537 s |
| Minimum step time | 0.002806763 s | 0.008074294 s |
| Maximum step time | 0.005587526 s | 0.013203827 s |
| Average FPS | 296.157334 | 107.439811 |

GPU stage metrics:

| Metric | Value |
|---|---:|
| Average broad wall time | 0.008001736 ms |
| Average broad kernel time | 0.000000000 ms |
| Average narrow wall time | 5.791149382 ms |
| Average narrow kernel time | 3.599940649 ms |
| Average H2D bytes | 3180800 bytes/step |
| Average D2H bytes | 16 bytes/step |
| Average allocation after warmup | 0 bytes/step |
| Raw dense-grid candidates | 462848 |
| Unique candidates | 322560 |
| Exact contacts | 1756 |
| Grid cells | 73728 |
| Overflow count | 0 |

Compute-breakdown rerun:

| Metric | CPU | GPU dense grid |
|---|---:|---:|
| Average wall step time | 0.003296195 s | 0.008407532 s |
| Total measured wall time | 0.362581400 s | 0.924828524 s |
| Average GPU compute-only time | not applicable | 0.003352012 s |
| Total GPU compute-only time | not applicable | 0.368721344 s |
| Average pipeline wall time | not applicable | 0.005318376 s |
| Wall minus GPU compute, average | not applicable | 0.005055520 s |
| Wall minus GPU compute, total | not applicable | 0.556107180 s |
| H2D bytes | none | 3180800 bytes/step |
| D2H bytes | none | 16 bytes/step |

`gpu_compute_*` is computed from the CUDA event kernel timing columns:

```text
gpu_compute = broad_kernel_ms + narrow_kernel_ms
```

This excludes the large H2D triangle upload from the wall-clock path. The remaining difference
between full wall time and GPU compute time includes SOFA traversal/control overhead, CPU extraction,
host packing, H2D transfer time, synchronization, Python/batch overhead, and other CPU-side frame costs.

## Interpretation

This run confirms the dense-grid implementation is functioning on a much larger primitive workload:

- 79520 collision triangles are processed.
- 462848 raw grid candidates are generated.
- 322560 unique triangle pairs reach exact testing.
- 1756 exact contacts are produced.
- No bucket/candidate/contact overflow occurred.
- Contacts remain GPU-side in detection-only mode, with only 16 D2H bytes per step.

The CPU is still faster in total step time. The likely reasons are:

- The GPU path still uploads 3180800 bytes of triangle records every step.
- The dense grid launches over 73728 cells, many of which are empty.
- Deduplication sorts and uniques 462848 candidate pairs every step.
- The GTX 1650 Ti is a small mobile GPU, so launch/sync overhead is a large fraction of the frame.
- SOFA `ParallelBVHNarrowPhase` is mature and very efficient for this static two-object case.

The next benchmark that should favor the GPU needs either device-resident collision geometry or active-cell/sparse-hash compaction. Without those, larger geometry increases both useful GPU work and transitional overhead.
