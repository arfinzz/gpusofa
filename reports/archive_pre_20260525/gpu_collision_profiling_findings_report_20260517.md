# GPU Collision Profiling Findings Report

Date: 2026-05-17

## Scope

This report explains the latest profiling output for the GPU SOFA collision path and compares the optimized GPU path with CPU benchmark runs.

The tested GPU path is:

```text
direct indexed dense grid
direct SOFA CUDA device positions
cached device index buffers
GPU hash dedupe
no CPU contact publication
no counter readback in fast mode
compactActiveCells=false
batchTriangleInsert=false
```

Main output folders used:

```text
output/benchmark_logs/final_one_tissue_cpu_gpu_compare_20260517
output/benchmark_logs/final_one_tissue_revert_cleanup_verify_20260517
output/benchmark_logs/final_large_tissue_cpu_gpu_compare_20260517
output/benchmark_logs/final_large_gpu_only_rerun_20260517
output/benchmark_logs/one_tissue_no_compact_no_batch_20260517
output/benchmark_logs/one_tissue_active_compact_compare_20260517
output/benchmark_logs/one_tissue_direct_counter_read_20260517
output/benchmark_logs/large_tissue_direct_counter_read_20260517
output/benchmark_logs/final_nsight_profile_20260517
```

## Fast Path Configuration

Use this configuration for wall-time performance runs:

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

Important interpretation rule:

- In fast mode, `avg_narrow_raw_candidate_count` and `avg_narrow_unique_candidate_count` are zero/unknown by design because counters are not copied to CPU.
- Use counter-read validation runs to inspect candidate counts and overflow.

## One Tissue / One Blade CPU-GPU Compare

Final CPU/GPU compare:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel | H2D | D2H | Counted GPU Ops | Memsets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 7.685 ms | 130.12 | n/a | n/a | n/a | n/a | n/a | n/a |
| GPU direct indexed | 5.057 ms | 197.73 | 1.523 ms | 0.900 ms | 0 B | 0 B | 5 | 1 |

Speedup from this compare:

```text
CPU step / GPU step = 0.007685026 / 0.005057386 = 1.52x
```

Post-rebuild verification:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel | H2D | D2H | Counted GPU Ops | Memsets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU direct indexed | 4.921 ms | 203.23 | 1.206 ms | 0.711 ms | 0 B | 0 B | 5 | 1 |

Finding:

The one-tissue scene is clearly faster on the optimized GPU path. The clean win is not just kernel speed; it is the removal of CPU triangle extraction/packing, position uploads, and CPU counter readbacks from steady-state frames.

## Large Tissue / Blade CPU-GPU Compare

Integrated CPU/GPU compare:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel | H2D | D2H | Counted GPU Ops | Memsets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 24.559 ms | 40.72 | n/a | n/a | n/a | n/a | n/a | n/a |
| GPU direct indexed | 34.865 ms | 28.68 | 14.005 ms | 12.721 ms | 0 B | 0 B | 5 | 1 |

GPU-only rerun:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel | H2D | D2H | Counted GPU Ops | Memsets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU direct indexed | 22.659 ms | 44.13 | 4.416 ms | 3.683 ms | 0 B | 0 B | 5 | 1 |

Finding:

The large scene is run-condition-sensitive in this WSL/laptop setup. The same optimized geometry path produced a slower integrated CPU/GPU compare but a faster GPU-only rerun. The stable facts are still valuable:

```text
steady-state H2D bytes = 0
steady-state D2H bytes = 0
kernel/operation count = 5
cudaMemset count = 1
backend triangle pack = 0 ms
SOFA triangle extraction = about 0.017 to 0.023 ms in the latest fast runs
```

Do not use the noisy large-scene wall time alone to justify another kernel redesign. Repeat it under controlled run order and power conditions first.

## Active Compaction And Batched Insert Experiment

No compact/no batch:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel |
| --- | ---: | ---: | ---: | ---: |
| GPU direct indexed | 4.641 ms | 215.49 | 1.182 ms | 0.679 ms |

Active compaction plus batched insert:

| Path | Avg Step | FPS | Narrow Wall | Narrow Kernel |
| --- | ---: | ---: | ---: | ---: |
| GPU direct indexed experimental | 8.043 ms | 124.33 | 5.089 ms | 3.974 ms |

Finding:

The experiment regressed the one-tissue benchmark. It adds extra work and does not pay for itself on the current small scene. The flags remain implemented for future experiments, but the default is:

```text
compactActiveCells=false
batchTriangleInsert=false
```

## Counter-Read Validation

One tissue counter-read run:

| Metric | Value |
| --- | ---: |
| Narrow wall | 1.728 ms |
| Narrow kernel | 1.286 ms |
| Candidate readback | 1.055 ms |
| Contact count readback | 0.240 ms |
| H2D | 0 B |
| D2H | 44 B |
| Raw candidates | 2,304 |
| Unique candidates | 624 |
| Overflow | 0 |

Large tissue counter-read run:

| Metric | Value |
| --- | ---: |
| Narrow wall | 3.863 ms |
| Narrow kernel | 3.409 ms |
| Candidate readback | 3.212 ms |
| Contact count readback | 0.207 ms |
| H2D | 0 B |
| D2H | 44 B |
| Raw candidates | 462,848 |
| Unique candidates | 322,560 |
| Overflow | 0 |

Finding:

The validation runs prove the GPU path is generating candidate pairs without dense-grid overflow. They also show why counter reads stay off in performance mode: 44 D2H bytes can still cost milliseconds because these reads force synchronization with prior GPU work.

The duplicate reduction in validation mode:

```text
one tissue: 1 - 624 / 2304 = 72.9% duplicate reduction
large tissue: 1 - 322560 / 462848 = 30.3% duplicate reduction
```

GPU hash dedupe is useful because duplicate pairs are real and substantial.

## Nsight Output

Nsight artifacts:

```text
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_ncu.ncu-rep
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_nsys_full.nsys-rep
```

Nsight manifest:

```text
ncu_status=0
ncu_report_exists=1
nsys_status=0
nsys_report_exists=0
nsys_light_status=0
nsys_light_report_exists=0
nsys_full_status=0
nsys_full_report_exists=1
nsys_importer_available=1
```

Nsight Compute profiled these detection-only kernels:

```text
resetDenseGridKernel
insertIndexedTrianglesKernel
insertIndexedTrianglesKernel
generateDenseGridUniqueCandidatePairsKernel
```

The pair-hash clear is a `cudaMemset`, not a custom kernel, and exact contact is absent because the detection-only profile disables exact contact generation.

Finding:

The Nsight Systems NVTX capture-range and light capture paths still exited without writing their expected reports in this environment. The script now records that fact and produces the reliable full-run fallback report, `gpu_collision_nsys_full.nsys-rep`.

## What Each Profile Metric Means

`avg_step_seconds`

Total SOFA animation step duration after warmup. This is the headline user-visible metric.

`avg_fps`

`1 / avg_step_seconds`.

`avg_narrow_wall_ms`

CPU wall-clock duration of `GpuCollisionNarrowPhase::endNarrowPhase`, including wrapper work, backend calls, optional readbacks, and optional SOFA output publication.

`avg_narrow_kernel_ms`

CUDA event-timed backend GPU work. In non-detailed profiling, this is timed as one total event interval around the dense-grid GPU operations.

`avg_narrow_sofa_triangle_extraction_ms`

Wrapper-side time to recognize CUDA triangle models, update topology cache metadata, and produce `TriangleIndexedSurface`. In the direct indexed path this is tiny because it does not rebuild packed triangles.

`avg_narrow_backend_triangle_pack_ms`

CPU packing time inside the backend. It is zero for the indexed path because the backend does not receive packed `TrianglePrimitive` arrays.

`avg_narrow_h2d_ms`

Host-side measured time spent issuing H2D upload work. In direct steady state, measured H2D bytes are zero.

`avg_narrow_candidate_readback_ms`

Time spent reading candidate counters and dense-grid stats. This should be zero in performance mode.

`avg_narrow_contact_count_readback_ms`

Time spent reading contact count and overflow count. This should be zero in performance mode.

`avg_host_to_device_bytes`

Bytes copied CPU-to-GPU per measured frame. In direct steady-state benchmark frames, this is zero.

`avg_device_to_host_bytes`

Bytes copied GPU-to-CPU per measured frame. In fast mode this is zero; in counter-read validation it is 44 bytes.

`avg_kernel_launch_count`

Project-level count of GPU operations on the dense-grid path. In the current fast path, the count is 5: four kernels plus the pair-hash `cudaMemset`.

`avg_cuda_memset_count`

Number of CUDA memsets. Current fast path uses one memset to clear the GPU pair hash table to `0xff`.

`avg_narrow_raw_candidate_count`

Candidate pair count before GPU hash dedupe. Only meaningful when counters are read.

`avg_narrow_unique_candidate_count`

Candidate pair count after dedupe. Only meaningful when counters are read or exact contacts are computed.

`avg_narrow_overflow_count`

Dense-grid bucket, candidate, contact, or hash probe overflow count. The validation runs report zero overflow.

## Main Findings

1. The largest completed optimization is avoiding CPU triangle extraction/packing and per-frame H2D position copies.

2. The best default path is direct indexed, no readback, no active compaction, no batched insert.

3. Counter reads are expensive because they synchronize, not because the byte count is large.

4. The one-tissue benchmark is GPU-faster in the final compare, about 1.52x by full step time.

5. The large benchmark needs repeatability work before treating wall-time differences as stable.

6. Active-cell compaction and batched insertion are implemented but currently experimental regressions.

7. Exact contact is still not the primary detection-only bottleneck because it is skipped in the fastest mode and should be tuned later only when response-enabled runs make it dominant.

## Recommended Next Profiling Protocol

For wall-time comparison:

```text
counter reads off
detailed profiling off
same warmup and measured steps
same run order repeated at least 3 times
one fresh process per variant
record CPU and GPU clocks/power state when possible
```

For correctness/candidate validation:

```text
readCountersWhenContactsStayOnDevice=true
copyContactsToHost=false unless CPU contacts are required
check raw candidates, unique candidates, and overflow
```

For kernel investigation:

```text
use Nsight Compute report
compare insertIndexedTrianglesKernel and generateDenseGridUniqueCandidatePairsKernel
inspect memory throughput, L2/L1 hit behavior, long scoreboard stalls, and atomic pressure
```

Current decision:

Keep the optimized direct indexed no-readback path as the source-of-truth default.
