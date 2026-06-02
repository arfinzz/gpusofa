# GPU Collision Direct Indexed Implementation Report - 2026-05-17

## Scope

This report summarizes the optimization pass that made the SOFA dense-grid collision path use cached indexed topology plus direct SOFA CUDA device position buffers.

Target device:

```text
NVIDIA GeForce GTX 1650 Ti (TU117)
WSL distro: wsl-gpu-proj
SOFA root: /opt/sofa/install/v25.12
WSL checkout: /home/arfin/gpu-sofa
Windows checkout: C:\Users\arfin\Desktop\GPU SOFA
```

## Implemented

- Added direct indexed surface support for SOFA CUDA position buffers.
- Cached triangle topology in the SOFA wrapper and cached device-side index buffers in the backend.
- Avoided CPU packed triangle-array rebuilds for static-topology SOFA CUDA scenes.
- Skipped per-frame position H2D copies when `useDirectDevicePositions=true`.
- Added `computeDeviceContactsWhenContactsStayOnDevice`; fastest detection-only mode now skips exact-contact generation unless explicitly requested.
- Removed candidate/contact counter D2H reads in the fastest detection-only mode.
- Preserved a validation mode where counters can still be read for candidate counts, overflow checks, and dense-grid occupancy reporting.
- Updated one-tissue and large-tissue benchmark scenes to expose direct-position and counter policy environment flags.
- Updated `guide/` source-of-truth docs to match the current architecture and run commands.

## Fast Path Configuration

```text
SOFA_GPU_DETAILED_PROFILING=0
SOFA_COPY_CONTACTS_TO_HOST=0
SOFA_DEDUPLICATE_PAIRS=1
SOFA_USE_GPU_HASH_DEDUPE=1
SOFA_USE_INDEXED_DENSE_GRID_INPUT=1
SOFA_USE_DIRECT_DEVICE_POSITIONS=1
SOFA_READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE=0
SOFA_COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE=0
```

## Verified Benchmark Results

### One Tissue / One Blade

Artifact:

```text
output/benchmark_logs/one_tissue_direct_no_readback_20260517/
```

Key results:

```text
avg_step_seconds=0.001578430
avg_fps=633.540906374
avg_narrow_wall_ms=0.693752886
avg_narrow_kernel_ms=0.535669942
avg_narrow_sofa_triangle_extraction_ms=0.005537143
avg_narrow_backend_triangle_pack_ms=0.000000000
avg_narrow_h2d_ms=0.000237143
avg_narrow_candidate_readback_ms=0.000000000
avg_host_to_device_bytes=0
avg_device_to_host_bytes=0
avg_kernel_launch_count=5
avg_cuda_memset_count=1
```

### Large Tissue / Blade

Artifact:

```text
output/benchmark_logs/large_tissue_direct_no_readback_20260517/
```

Key results:

```text
avg_step_seconds=0.011534779
avg_fps=86.694338674
avg_narrow_wall_ms=3.027547129
avg_narrow_kernel_ms=2.721521378
avg_narrow_sofa_triangle_extraction_ms=0.012880000
avg_narrow_backend_triangle_pack_ms=0.000000000
avg_narrow_h2d_ms=0.000555714
avg_narrow_candidate_readback_ms=0.000000000
avg_host_to_device_bytes=0
avg_device_to_host_bytes=0
avg_kernel_launch_count=5
avg_cuda_memset_count=1
```

## Counter-Read Validation

The fastest path intentionally reports candidate/contact counters as zero/unknown because the CPU does not read them. Validation mode enables counter reads while keeping geometry data GPU-native.

### One Tissue / One Blade Counter Read

Artifact:

```text
output/benchmark_logs/one_tissue_direct_counter_read_20260517/
```

```text
avg_narrow_wall_ms=1.728195175
avg_narrow_candidate_readback_ms=1.055430100
avg_narrow_contact_count_readback_ms=0.239737525
avg_host_to_device_bytes=0
avg_device_to_host_bytes=44
avg_narrow_raw_candidate_count=2304
avg_narrow_unique_candidate_count=624
avg_narrow_overflow_count=0
```

### Large Tissue / Blade Counter Read

Artifact:

```text
output/benchmark_logs/large_tissue_direct_counter_read_20260517/
```

```text
avg_narrow_wall_ms=3.863485025
avg_narrow_candidate_readback_ms=3.211842525
avg_narrow_contact_count_readback_ms=0.207332500
avg_host_to_device_bytes=0
avg_device_to_host_bytes=44
avg_narrow_raw_candidate_count=462848
avg_narrow_unique_candidate_count=322560
avg_narrow_overflow_count=0
```

## Profiling

Artifact:

```text
output/benchmark_logs/nsight_direct_validation_20260517/
```

Results:

```text
nsys_status=0
ncu_status=0
ncu_report=output/benchmark_logs/nsight_direct_validation_20260517/nsight/gpu_collision_ncu.ncu-rep
```

Nsight Compute profiled the direct indexed path kernels:

```text
resetDenseGridKernel
insertIndexedTrianglesKernel
generateDenseGridUniqueCandidatePairsKernel
```

Nsight Systems exited successfully but did not produce an `.nsys-rep` because the WSL Nsight Systems importer binary/dependencies were unavailable. The helper script now records report-existence flags in future manifests so this state is explicit.

## Build Verification

Final WSL build check:

```text
cd /home/arfin/gpu-sofa/SofaGpuCollision/build-profile
make -j2
[ 80%] Built target SofaGpuCollision
[100%] Built target SofaGpuCollisionDenseGridBackendBench
```

## Findings

- The biggest integrated SOFA cost from earlier passes, CPU triangle extraction plus H2D packing/upload, is effectively removed for SOFA CUDA benchmark scenes.
- The fastest direct indexed path reaches 0 H2D bytes and 0 D2H bytes per measured frame.
- Counter readbacks remain expensive when enabled: about 1.06 ms on the one-tissue scene and 3.21 ms on the large-tissue scene for candidate counters.
- The current dense-grid wall time is now dominated by actual indexed grid insertion and candidate generation rather than wrapper-side triangle preparation.
- Exact-contact tuning remains lower priority because the fastest detection-only mode does not launch exact contact kernels, and validation runs show zero published contacts for the benchmark scene configuration.

## Remaining Optimization Work

- Add pinned/asynchronous staging only for fallback inputs that cannot expose SOFA CUDA device positions directly.
- Revisit dense-grid memory layout for large scenes: SoA AABB/cell data, fewer writes, and better coalescing.
- Batch or fuse tiny tool-side work only after kernel profiles show launch overhead dominates a specific scene.
- Repair or reinstall the Nsight Systems host importer in WSL if full `.nsys-rep` timeline reports are needed.
