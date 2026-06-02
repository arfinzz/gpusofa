# GPU Collision Remaining Optimizations And Profile Report

Date: 2026-05-17

## Summary

Implemented the remaining safe optimization work around the direct indexed SOFA CUDA collision path:

- Added async fallback uploads for non-direct indexed inputs.
- Added pinned host staging for large fallback indexed uploads.
- Added optional active-cell compaction before candidate generation.
- Added optional batched tissue/tool triangle insertion.
- Fixed the Nsight Systems setup enough to produce a full-run `.nsys-rep` fallback artifact.
- Kept the measured-fast defaults: direct SOFA CUDA positions, indexed topology, GPU hash dedupe, no CPU counter/contact readback, `compactActiveCells=false`, and `batchTriangleInsert=false`.

The important result is that the best production path is still the direct indexed no-readback path. The new compaction/batching path is useful as an experiment, but it is not a win on the current GTX 1650 Ti scene.

## Code Changes

Main files touched:

```text
SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h
SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.h
SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu
test_gpu_one_tissue_one_blade_dense_grid_benchmark.py
test_gpu_large_tissue_blade_dense_grid_benchmark.py
scripts/run_nsight_collision_profile_wsl_gpu_proj.sh
guide/setup.md
guide/plan.md
guide/architecture.md
```

### Async Fallback Staging

The direct SOFA CUDA path already passes `deviceRead()` position pointers into the backend and has no per-frame H2D position upload. For fallback indexed surfaces that only have CPU-visible positions or topology:

- Host-to-device copies now use `cudaMemcpyAsync`.
- Large fallback uploads stage through pinned host buffers when pinned allocation succeeds.
- Small uploads avoid the extra pinned copy.

This keeps the common path GPU-native while making non-direct inputs less costly.

### Optional Active-Cell Compaction

Added an experimental path that compacts mixed dense-grid cells into an active-cell list and launches candidate generation over active cells. Measured result on one-tissue scene:

```text
no compact/no batch:
avg_narrow_wall_ms=1.182022333
avg_narrow_kernel_ms=0.678519471

active compact + batch:
avg_narrow_wall_ms=5.088542000
avg_narrow_kernel_ms=3.973753616
```

Decision: keep `compactActiveCells=false` by default.

### Optional Batched Insertion

Added combined tissue/tool insert kernels for packed and indexed paths. This reduces one launch in principle, but on this scene it did not beat the existing separate indexed insertion path. It remains available through:

```text
SOFA_BATCH_TRIANGLE_INSERT=1
```

Decision: keep `batchTriangleInsert=false` by default.

### Nsight Systems

The importer binary exists at:

```text
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
```

The profiling script now exports:

```text
QUADD_INSTALL_DIR=/usr/lib/nsight-systems
```

The NVTX capture-range Systems run still exited status 0 without writing `gpu_collision_nsys.nsys-rep`, so a full-run fallback capture was added. Final artifacts:

```text
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_ncu.ncu-rep
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_nsys_full.nsys-rep
```

## Benchmark Results

Fast path environment:

```text
SOFA_GPU_DETAILED_PROFILING=0
SOFA_COPY_CONTACTS_TO_HOST=0
SOFA_DEDUPLICATE_PAIRS=1
SOFA_USE_GPU_HASH_DEDUPE=1
SOFA_USE_INDEXED_DENSE_GRID_INPUT=1
SOFA_USE_DIRECT_DEVICE_POSITIONS=1
SOFA_READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE=0
SOFA_COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE=0
SOFA_COMPACT_ACTIVE_CELLS=0
SOFA_BATCH_TRIANGLE_INSERT=0
```

### One Tissue / One Blade

Full CPU/GPU compare:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel |
| --- | ---: | ---: | ---: | ---: |
| CPU | 7.685 ms | 130.12 | n/a | n/a |
| GPU direct indexed | 5.057 ms | 197.73 | 1.523 ms | 0.900 ms |

Post-rebuild verification:

```text
GPU avg_step_seconds=0.004920647
GPU avg_fps=203.225291046
GPU avg_narrow_wall_ms=1.205574533
GPU avg_narrow_kernel_ms=0.711350395
GPU avg_host_to_device_bytes=0
GPU avg_device_to_host_bytes=0
GPU avg_kernel_launch_count=5
GPU avg_cuda_memset_count=1
```

### Large Tissue / Blade

Integrated CPU/GPU compare:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel |
| --- | ---: | ---: | ---: | ---: |
| CPU | 24.559 ms | 40.72 | n/a | n/a |
| GPU direct indexed | 34.865 ms | 28.68 | 14.005 ms | 12.721 ms |

GPU-only rerun immediately after showed better behavior:

```text
GPU avg_step_seconds=0.022658998
GPU avg_fps=44.132577425
GPU avg_narrow_wall_ms=4.416319086
GPU avg_narrow_kernel_ms=3.682984679
GPU avg_host_to_device_bytes=0
GPU avg_device_to_host_bytes=0
GPU avg_kernel_launch_count=5
GPU avg_cuda_memset_count=1
```

Interpretation: large-scene wall time is run-condition-sensitive on this laptop/WSL setup. The geometry path remains correct for the target optimization criteria: no per-frame H2D position upload, no D2H counter/contact readback, 5 launches, and one pair-hash `cudaMemset`.

## Findings

- The biggest clean win remains avoiding CPU triangle extraction/packing and H2D position copies through direct SOFA CUDA position pointers.
- Fallback async staging is now in place for non-direct input paths, but it does not affect the direct benchmark because the direct path has no position upload.
- Active-cell compaction is not automatically good. On the small scene it adds work and loses badly.
- Batched insertion is also not automatically good. The current separate indexed inserts are faster for the measured scene.
- Counter/contact readback should stay off in detection-only benchmarks. Validation runs can enable it when candidate/contact counts are needed.
- Nsight Compute is the reliable kernel artifact. Nsight Systems full-run fallback now produces a timeline artifact, while the NVTX capture-range mode still does not produce a report in this environment.

## Current Defaults

Use these unless actively experimenting:

```text
useIndexedDenseGridInput=true
useDirectDevicePositions=true
readCountersWhenContactsStayOnDevice=false
computeDeviceContactsWhenContactsStayOnDevice=false
compactActiveCells=false
batchTriangleInsert=false
copyContactsToHost=false for detection-only benchmarks
```

## Remaining Work

- Investigate large-scene variability separately from kernel changes: power state, WSL scheduling, run order, and GPU clock behavior.
- Revisit SoA/coalescing only for a measured bottleneck. The current direct indexed path already avoids storing intermediate AABBs in the fast insertion path.
- Tune exact contact later; it is still not the integrated wall-time bottleneck for detection-only runs.
