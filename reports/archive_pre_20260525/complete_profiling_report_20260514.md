# Complete GPU Collision Profiling Report - 2026-05-14

## Executive Summary

Profiling completed successfully on the WSL GPU environment using the rebuilt `SofaGpuCollision` profile build. Nsight Compute reports were generated and exported for both the standalone backend and SOFA-integrated scene. Nsight Systems stream capture also succeeded after a manual `QdstrmImporter` conversion workaround.

The dense-grid backend is now stable and produces deterministic candidate/contact counts with no overflow. On the large standalone backend case, the measured backend step averages **13.728 ms**, with **3.360 ms** of GPU kernel time. On the small SOFA one-tissue/one-blade scene, the clean integrated GPU path averages **12.032 ms/step**, but only **1.668 ms** is GPU kernel time; the rest is mostly SOFA-side triangle extraction, packing, upload, and synchronization overhead.

The most important conclusion: the CUDA backend is working, but current SOFA integration overhead dominates the small integrated scene. Optimizing the wrapper path will likely matter more than optimizing individual kernels first.

## Environment

- Device: NVIDIA GeForce GTX 1650 Ti, Turing TU117, compute capability 7.5
- VRAM visible in SOFA: about 4095 MiB total, about 3300 MiB free at run start
- CUDA compiler: 12.0.140
- Nsight Systems: 2022.4.2.50
- Nsight Compute CLI: 2022.4.1.0
- SOFA root: `/opt/sofa/install/v25.12`
- Profile plugin: `/home/arfin/gpu-sofa/SofaGpuCollision/build-profile/libSofaGpuCollision.so`

## Artifacts

Primary WSL artifact roots:

- Standalone backend: `/home/arfin/gpu-sofa/benchmark_logs/full_profile_backend_20260514`
- SOFA profiler runs: `/home/arfin/gpu-sofa/benchmark_logs/full_profile_sofa_20260514`
- Clean SOFA CPU/GPU benchmark: `/home/arfin/gpu-sofa/benchmark_logs/full_profile_clean_sofa_20260514`

Key profiler files:

- Backend Nsight Compute report: `full_profile_backend_20260514/nsight/backend_dense_grid_ncu.ncu-rep`
- Backend Nsight Compute raw CSV: `full_profile_backend_20260514/nsight/backend_dense_grid_ncu_raw.csv`
- Backend Nsight Systems report: `full_profile_backend_20260514/nsight/backend_dense_grid_nsys_nocapture.nsys-rep`
- SOFA Nsight Compute report: `full_profile_sofa_20260514/nsight/gpu_collision_ncu.ncu-rep`
- SOFA Nsight Compute raw CSV: `full_profile_sofa_20260514/nsight/gpu_collision_ncu_raw.csv`
- SOFA Nsight Systems report: `full_profile_sofa_20260514/nsight/gpu_collision_nsys_nocapture.nsys-rep`

## Standalone Dense-Grid Backend

Run configuration:

- Tissue triangles: 64,800
- Blade triangles: 14,720
- Warmup steps: 2
- Measured steps: 8
- H2D per measured step: 3,180,800 bytes
- D2H per measured step: 44 bytes
- Kernel launches per step: 13
- CUDA memset calls per step: 6

Measured timing:

| Metric | Average | Min | Max |
|---|---:|---:|---:|
| Wall time | 13.728 ms | 11.796 ms | 16.657 ms |
| GPU kernel time | 3.360 ms | 3.147 ms | 3.549 ms |
| Host preparation | 8.137 ms | 6.133 ms | 10.748 ms |
| H2D upload | 0.639 ms | 0.512 ms | 0.776 ms |
| Device allocation | 0.003 ms | 0.002 ms | 0.004 ms |

Candidate/contact output:

| Metric | Value |
|---|---:|
| Raw candidates | 462,848 |
| Unique candidates | 322,560 |
| Duplicate reduction | 30.3% |
| Contacts | 1,756 |
| Dense-grid overflow | 0 |
| Hash dedupe probe overflow | 0 |
| Host-validated unique candidates | 0 |

Backend stage averages:

| Stage | Average |
|---|---:|
| Generate pairs | 1.916 ms |
| Counter clear | 0.873 ms |
| Candidate count readback | 0.473 ms |
| Contact count readback | 0.212 ms |
| Exact contact | 0.172 ms |
| Clear grid | 0.129 ms |
| Insert tissue | 0.108 ms |
| Insert tool | 0.060 ms |
| Tissue AABB | 0.056 ms |
| Tool AABB | 0.046 ms |

The backend kernels are not the only cost in the standalone step. Host preparation is the largest component at this scale. Within the GPU section, pair generation and counter clearing are larger than exact contact.

## Backend Nsight Compute Kernel Readout

| Kernel | Duration | Memory % | DRAM % | L1/TEX % | L2 % | SM % | Occupancy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clear grid | 4.01 us | 54.7 | 15.8 | 51.6 | 54.7 | 18.7 | 67.2 | Memory/latency limited |
| triangle AABB, tissue | 27.90 us | 78.5 | 72.1 | 62.6 | 78.5 | 11.5 | 86.5 | Strong DRAM pressure |
| triangle AABB, tool | 8.84 us | 54.1 | 39.4 | 39.4 | 54.1 | 8.7 | 74.6 | Small grid, 0.91 waves/SM |
| insert tissue | 81.57 us | 69.3 | 23.3 | 65.5 | 69.3 | 10.5 | 87.1 | Memory path dominates |
| insert tool | 13.56 us | 39.5 | 14.9 | 37.1 | 39.5 | 9.0 | 75.7 | Small grid, 0.91 waves/SM |
| exact contact | 102.08 us | 46.4 | 15.7 | 92.7 | 37.9 | 25.3 | 60.0 | 68 registers/thread limits occupancy |

Interpretation:

- The tissue AABB kernel is closest to a memory bandwidth ceiling, with about 72% DRAM and 79% memory pipeline utilization.
- `insert_tissue` is memory-pipeline heavy and likely sensitive to write pattern/coalescing.
- `exact_contact` has the highest per-launch duration but still low SM throughput. Its occupancy is register-limited: 68 registers/thread, theoretical occupancy 75%, achieved about 60%.
- The tool-side kernels are too small to fill the GPU well. They launch less than one full wave per SM.

## Clean SOFA Integrated Benchmark

Run configuration:

- Scene: one tissue surface and one blade mesh
- Clean run: no Nsight replay overhead
- Total iterations: 20
- Warmup steps: 10
- Measured steps: 10
- Pipeline phase: dense-grid GPU detection only
- Collision response: disabled
- Exact contact list: left device-side

GPU dense-grid integrated result:

| Metric | Average | Min | Max |
|---|---:|---:|---:|
| Full SOFA step | 12.032 ms | 9.033 ms | 14.809 ms |
| FPS | 83.1 | | |
| Narrow phase wall | 9.923 ms | 7.689 ms | 12.179 ms |
| Narrow GPU kernel time | 1.668 ms | 0.806 ms | 3.459 ms |
| Narrow host preparation | 5.335 ms | 3.422 ms | 8.744 ms |
| SOFA triangle extraction | 4.186 ms | 2.467 ms | 7.448 ms |
| Backend triangle packing | 1.148 ms | 0.954 ms | 1.647 ms |
| H2D upload | 0.519 ms | 0.429 ms | 0.657 ms |
| Candidate count readback | 0.727 ms | 0.518 ms | 0.989 ms |
| Contact count readback | 0.331 ms | 0.274 ms | 0.427 ms |

Small scene candidate/contact output:

| Metric | Value |
|---|---:|
| Narrow input primitives | 12,812 |
| Raw candidates | 2,304 |
| Unique candidates | 624 |
| Duplicate reduction | 72.9% |
| Output contacts | 104 |
| Grid cells | 32,768 |
| Active mixed cells | 56 |
| Dense-grid overflow | 0 |
| Hash dedupe probe overflow | 0 |

CPU comparison in the same clean one-tissue/one-blade benchmark:

| Path | Average Step | FPS |
|---|---:|---:|
| CPU scene | 2.967 ms | 337.0 |
| GPU dense-grid scene | 12.032 ms | 83.1 |

This CPU/GPU comparison is for the small one-tissue/one-blade detection-only scene. It does not mean the GPU backend is bad; it means the current SOFA integration overhead is too high for this small workload. The GPU path is paying fixed costs: triangle extraction, packing, uploads, kernel launch overhead, and readback synchronization.

## SOFA Nsight Compute Kernel Readout

| Kernel | Duration | Memory % | DRAM % | L1/TEX % | L2 % | SM % | Occupancy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clear grid | 2.48 us | 37.4 | 0.0 | 35.3 | 37.4 | 12.2 | 71.4 | Low DRAM, memory path active |
| AABB, 1-block mesh | 2.04 us | 1.5 | 0.2 | 6.5 | 1.5 | 0.1 | 15.4 | Too small to occupy GPU |
| AABB, 50-block mesh | 6.94 us | 48.2 | 41.2 | 35.9 | 48.2 | 9.4 | 65.9 | Memory-bound small mesh work |
| insert 1-block mesh | 29.52 us | 0.5 | 0.0 | 3.4 | 0.5 | 0.1 | 3.5 | Very under-occupied |
| insert 50-block mesh | 29.15 us | 60.4 | 20.6 | 57.1 | 60.4 | 7.5 | 71.1 | Memory path dominates |
| exact contact | 5.70 us | 3.2 | 1.3 | 10.6 | 3.2 | 3.6 | 20.5 | Small 3-block launch |

The integrated scene's kernels are mostly tiny. For this workload, kernel micro-optimization cannot overcome the CPU extraction and synchronization overhead.

## Nsight Systems Notes

Nsight Systems capture produced `.qdstrm` streams, but Nsight Systems 2022.4 in WSL did not auto-import them. Manual conversion with `QdstrmImporter` succeeded for both backend and SOFA no-capture runs.

Backend Systems no-capture report:

- Report: `full_profile_backend_20260514/nsight/backend_dense_grid_nsys_nocapture.nsys-rep`
- NVTX confirmed dense-grid narrow-phase ranges.
- CUDA API trace was present.
- GPU kernel summary was not available from Nsight Systems stats in this WSL/2022.4 setup. Nsight Compute is the authoritative kernel source for this report.

SOFA Systems no-capture report:

- Report: `full_profile_sofa_20260514/nsight/gpu_collision_nsys_nocapture.nsys-rep`
- SOFA narrow phase: 8 instances, average 28.75 ms under Systems overhead.
- Dense-grid narrow phase: 8 instances, average 23.91 ms under Systems overhead.
- CPU triangle extraction: average 4.77 ms.
- H2D triangle upload: average 2.72 ms.
- Candidate counter readback: average 0.79 ms.
- Contact counter readback: average 0.39 ms.
- One-time CUDA/API setup dominates startup, especially `cudaMemGetInfo` and library loading.

The direct SOFA no-capture Systems run used a reduced plugin path and printed optional plugin-not-found warnings for unrelated default plugins. The collision plugin loaded and the scene completed. The clean benchmark run used the standard script path and loaded the full plugin list.

## Bottlenecks

1. SOFA integration overhead dominates the small scene.
   - Triangle extraction averages 4.186 ms.
   - Triangle packing averages 1.148 ms.
   - Host preparation totals 5.335 ms.

2. Readback synchronization is expensive relative to payload size.
   - Only 44 bytes are copied D2H in the backend measured pass, but candidate/contact counter readbacks still cost about 0.68 ms combined.
   - In the clean SOFA run, candidate/contact counter readbacks cost about 1.06 ms combined.

3. Kernel launch count is high for the small scene.
   - Backend measured pass uses 13 kernel launches and 6 memsets per step.
   - Several launches have less than one wave per SM in the integrated scene.

4. Memory behavior limits several kernels.
   - Tissue AABB reaches about 72% DRAM throughput.
   - Insert kernels are memory-pipeline dominated.
   - Exact contact is not fully compute-bound; it is register-pressure and latency limited.

5. Exact contact register pressure is visible.
   - 68 registers/thread.
   - Achieved occupancy about 60%.
   - Optimize this after fixing host/SOFA overhead, because it is not the main wall-time limiter yet.

## Recommended Next Work

1. Cache SOFA triangle topology and only update vertex-position buffers.
   - Avoid rebuilding full triangle arrays every step when topology is unchanged.
   - This targets the largest clean SOFA cost: triangle extraction plus packing.

2. Keep collision surfaces GPU-native through the SOFA path.
   - The best version avoids CPU triangle extraction for already-GPU state.
   - If CPU extraction remains necessary, use pinned staging buffers and asynchronous copies.

3. Remove or defer small D2H counter readbacks.
   - Keep counters device-side when collision response is disabled.
   - If CPU needs counts for logging, read them asynchronously or every N frames.

4. Reduce launches for small scenes.
   - Fuse lightweight clear/counter operations where possible.
   - Consider combining tiny mesh AABB/insert paths or batching tool work.

5. Revisit memory layout for AABB and insertion.
   - Tissue AABB and insert kernels are memory dominated.
   - SoA layout, fewer writes, and improved coalescing are likely higher value than extra arithmetic tuning.

6. Only then tune exact contact.
   - Register pressure is real, but exact contact is not the current integrated wall-time bottleneck.
   - Possible later work: split helper logic, reduce live ranges, evaluate smaller block size if it improves occupancy without hurting memory behavior.

## Bottom Line

The fixed implementation profiles cleanly: no overflows, stable candidate/contact counts, valid Nsight Compute reports, and usable Nsight Systems timelines after manual import. The GPU backend itself is functioning, but the current SOFA wrapper path is dominated by CPU-side extraction/packing and synchronization. The next performance win should come from making the SOFA integration feed the GPU more directly, not from chasing microseconds inside the contact kernel first.
