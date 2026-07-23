# Fused big-cell profiling clarification: sizes, wall time, Nsight, and internal phases

Date: 2026-07-23
Hardware: NVIDIA GeForce GTX 1650 Ti, Turing sm_75, 16 SMs
Winning configuration: factor 2, 256-thread blocks, 256-entry tool tile, CSR global table, shared-hash producer/merge, CUDA graph enabled in production

## Executive answer

The earlier 14k scene was not secretly changed into a 28k scene. Two different inputs were accidentally discussed under one scale axis:

- `hash_prefixsum_large.py` is exactly **14,368 triangles**: 12,800 tissue + 1,568 tool.
- the old NX81 standalone backend profile used 12,800 tissue + the backend's default 14,720-triangle tool = **27,520 triangles**, rounded to **28k**.

The same distinction exists at the upper scale:

- `collision_xlarge_200k.py` is exactly **200,018 triangles**: 198,450 tissue + 1,568 tool;
- the old NX316 standalone backend profile used the default 14,720-triangle tool, so it was actually **213,170 triangles**, although historical tables called it “bench 200k.”

Therefore:

- the canonical SOFA scene labels **14k** and **200k** were correct;
- the old standalone 28k result was not a new SOFA scene;
- the old standalone “200k” label was a rounded family name and should be read as 213,170 elements;
- values are only comparable when the exact geometry, graph setting, readback policy, and profiling mode are the same.

The runner now uses the historical triangle-count scales **14,368 / 79,520 / 200,018** and names them 14k / 80k / 200k. The 14k and 200k backend inputs are standalone analogues: their total contact counts match the SOFA scenes, but their VF/FV/EE subtype splits differ, so they are not claimed to be byte-identical scene geometry.

## 1. Exact input reconciliation

| Name | Execution surface | Tissue tris | Tool tris | Total tris | Status |
|---|---|---:|---:|---:|---|
| canonical 14k | `hash_prefixsum_large.py` | 12,800 | 1,568 | **14,368** | actual SOFA scene |
| old 28k | backend NX81 + default large tool | 12,800 | 14,720 | **27,520** | standalone only |
| historical 80k | backend NX181 + default large tool | 64,800 | 14,720 | **79,520** | standalone benchmark |
| canonical 200k | `collision_xlarge_200k.py` | 198,450 | 1,568 | **200,018** | actual SOFA scene |
| old “bench 200k” | backend NX316 + default large tool | 198,450 | 14,720 | **213,170** | standalone only |

Current standalone correctness totals at 14k / 80k / 200k are 2,354 / 8,018 / 12,178 contacts with zero true overflow. Separate canonical scene validation previously established 2,354 and 12,178 contacts with the scene-specific subtype splits.

## 2. What the timing columns mean

| Column | Definition | Includes | Excludes |
|---|---|---|---|
| `Wall mean` | arithmetic mean of CPU elapsed time around one backend function call | host API work, host-backed input uploads, CUDA launch calls, synchronization, counter readback, contact download | SOFA work outside that backend call |
| `± std` | population standard deviation of those wall samples | run-to-run jitter, OS scheduling, clock/thermal variation, transfer/sync variation | no percentile information |
| `GPU pipeline mean` | mean CUDA-event elapsed time around the serialized big-cell GPU pipeline | reset, count, scan, fill, mixed-list, output reset, fused consumer | CPU preparation, H2D before the start marker, D2H after the end marker, host synchronization |
| `GPU p50` | median GPU pipeline sample | the typical GPU sample; half are no slower | tail behavior |
| `GPU p95` | 95th-percentile GPU pipeline sample | a useful tail-latency bound; 95% of samples are at or below it | worst single outlier |

A large `wall - GPU` gap is not evidence that the CUDA kernel measurement is false. It means the host-visible call is doing work that is intentionally outside the event interval.

## 3. Why the standalone wall time is much larger

The standalone harness is a validation-oriented host-input path, not the SOFA zero-copy production path:

1. `TriangleIndexedSurface::devicePositions` is null, so current positions are uploaded from host memory every call.
2. `makeBenchFbpConfig()` sets `keepContactsOnDevice=false` and `readContactCounter=true`.
3. The call synchronizes for exact counters and downloads the emitted contact array.
4. CPU wrapper/API time and scheduling jitter remain in `wall_ms` but not in GPU events.

The measured transfer volume proves this:

| Scale | Wall mean ± std ms | GPU pipeline mean ms | GPU p50 ms | GPU p95 ms | H2D bytes/frame | D2H bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| 14k analogue | 0.821 ± 0.179 | 0.191 | 0.189 | 0.211 | 88,164 | 178,948 |
| 80k | 2.350 ± 0.449 | 0.543 | 0.534 | 0.646 | 481,476 | 609,412 |
| 200k analogue | 2.905 ± 0.415 | 0.795 | 0.789 | 0.841 | 1,207,704 | 925,572 |

The D2H volume grows with the contact array. The production SOFA path instead reads SofaCUDA device positions directly and keeps contacts on the GPU, so steady-state transfer counters are zero.

## 4. Does this invalidate the earlier values?

No blanket invalidation is justified. The correct interpretation is:

- a value labelled `avg_narrow_kernel_ms` or fused-kernel milliseconds is a GPU collision timing, not full-scene frame time;
- a standalone `wall_ms` value includes validation transfers and synchronization and must not be converted into deployment FPS;
- `avg_fps` and `avg_step_seconds` from a production SOFA summary are the full-scene deployment measures;
- detailed events, in-kernel clocks, and Nsight perturb execution and must not replace production measurements;
- old 28k and 213,170-element backend values remain valid for those exact backend inputs, but they must not be relabelled as the canonical 14,368 and 200,018 SOFA scenes.

## 5. Current production SOFA result

Graphs were enabled, detailed profiling was disabled, internal clocks were disabled, counter readback was disabled, direct device positions were enabled, and contacts stayed on device.

| Scene | Exact elements | Measured steps | Full step ms | Full-scene FPS | Narrow host/enqueue ms | H2D | D2H |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hash_prefixsum_large.py` | 14,368 | 150 after 10 warm-up | **0.6898** | **1,449.70** | 0.0445 | 0 | 0 |
| `collision_xlarge_200k.py` | 200,018 | 150 after 10 warm-up | **12.3011** | **81.29** | 0.1113 | 0 | 0 |

The fast path is asynchronous. With detailed timing off, `avg_narrow_wall_ms` mostly measures command enqueue and `avg_narrow_kernel_ms` is intentionally zero because the backend does not synchronize merely to obtain a timing. It is therefore wrong to compute production FPS as `1000 / fused_kernel_ms`.

The 200k scene really runs near 80 FPS end to end on this GTX 1650 Ti. That does **not** mean the fused kernel takes 12.3 ms: the standalone GPU pipeline for the same element-count scale is about 0.8 ms, and Nsight measures the production fused launch at roughly 0.345 ms. The rest of the SOFA step is outside the isolated fused launch: collision-model/bounding updates, framework traversal, host work, other GPU work, and synchronization/queue effects. A full Nsight Systems timeline would be needed to divide the remaining full-step time exactly.

## 6. CUDA-event pipeline breakdown

These graph-off runs place persistent CUDA events between all nine pipeline launches. They are for attribution, not production latency.

| Stage, ms | 14k | 80k | 200k | Meaning |
|---|---:|---:|---:|---|
| reset | 0.0276 | 0.1641 | 0.0159 | clear big-cell histogram/counters |
| marker gap | 0.0118 | 0.0354 | event boundary with no CSR clear kernel |
| tissue count | 0.0330 | 0.1029 | shared-hash producer counts tissue entries |
| tool count | 0.0171 | 0.0407 | shared-hash producer counts tool entries |
| scan | 0.0248 | 0.0805 | exclusive scan into CSR starts |
| tissue fill | 0.0199 | 0.0982 | reserve/merge/scatter tissue entries |
| tool fill | 0.0093 | 0.0349 | reserve/merge/scatter tool entries |
| mixed-list build | 0.0178 | 0.0277 | retain big cells containing both sides |
| proximity reset | 0.0195 | 0.0359 | clear contact/subtype counters |
| fused consumer | **0.1384** | **0.4806** | **0.2574** | shared staging, local sort, pair cull, FBP, emit |
| event total | **0.3191** | **1.1011** | **0.8336** | start-to-finish serialized event interval |

The 200k fused launch is faster than the 80k launch even though it has more tissue because the 200k analogue uses the smaller 1,568-triangle tool and exposes 160 mixed big cells, while the 80k benchmark uses a 14,720-triangle tool, tile chunking, and only 88 mixed big cells. Element count alone does not determine fused time.

## 7. Individual operations inside the fused kernel

CUDA events cannot be recorded from inside a running kernel. Splitting the fused kernel merely to add events would change launch overhead, shared-memory lifetime, global traffic, and the algorithm being measured. The diagnostic specialization therefore uses `clock64()` around the requested operations while the production specialization remains unchanged.

### 7.1 Mapping from the requested mental model to code

| Requested operation | Measured counter |
|---|---|
| copy packed tool entries into shared memory and build a 64-bin histogram | `tile_setup_block_cycles` |
| convert the histogram to 64 run boundaries | `bin_prefix_block_cycles` |
| scatter tool IDs into local-cell order | `tool_sort_scatter_thread_cycles` |
| load tool triangle indices/vertices, compute raw AABB, write shared tool arrays | `tool_data_load_thread_cycles` |
| test tissue/tool pairs | inflated-AABB, home-cell, and raw-AABB thread cycles |
| generate the closest-feature contact | `fbp_thread_cycles` |
| reserve/write the contact and increment its VF/FV/EE counter | `contact_emit_thread_cycles` |

`tool_gather_block_cycles` is the block wall-clock envelope around sort-scatter + tool-data load + synchronization; it overlaps the two lane-level counters and must not be added to them.

### 7.2 Average operation cost

The table uses the mean of 12 instrumented iterations. Nominal nanoseconds use the device metadata clock of 1.485 GHz. They are per-block or per-participating-thread costs, **not whole-kernel elapsed time**, because many blocks and lanes overlap.

| Operation | 14k cycles / nominal ns | 80k cycles / nominal ns | 200k cycles / nominal ns |
|---|---:|---:|---:|
| shared stage + histogram, per block tile | 514 / 346 ns | 649 / 437 ns | 647 / 435 ns |
| 64-bin prefix, per block tile | 1,023 / 689 ns | 1,062 / 715 ns | 1,091 / 735 ns |
| complete gather envelope, per block tile | 972 / 655 ns | 1,243 / 837 ns | 1,206 / 812 ns |
| sort scatter, per staged tool entry/lane | 87 / 58.7 ns | 121 / 81.7 ns | 91 / 61.6 ns |
| tool triangle+AABB+vertex load, per entry/lane | 640 / 431 ns | 967 / 651 ns | 895 / 602 ns |
| inflated-AABB test, per visited pair/lane | 62 / 41.9 ns | 58 / 39.4 ns | 69 / 46.4 ns |
| home-cell ownership, per surviving pair/lane | 67 / 45.4 ns | 68 / 45.8 ns | 75 / 50.6 ns |
| raw-AABB gap, per owned pair/lane | 43 / 29.1 ns | 44 / 29.5 ns | 50 / 33.9 ns |
| closest-feature FBP, per call/lane | 9,934 / 6,690 ns | 9,457 / 6,369 ns | 9,586 / 6,455 ns |
| contact buffer emit, per contact/lane | 539 / 363 ns | 532 / 359 ns | 550 / 371 ns |

The closest-feature computation is roughly an order of magnitude more expensive per participating lane than emitting the resulting record, and two orders of magnitude more expensive than any one cull predicate. This supports keeping all three cheap filters before FBP.

### 7.3 Work funnel

| Work counter | 14k | 80k | 200k |
|---|---:|---:|---:|
| tool entries staged | 7,424 | 29,696 | 22,144 |
| small-cell pair visits | 61,920 | 462,848 | 819,840 |
| inflated-AABB rejects | 40,416 | 370,976 | 489,792 |
| home-cell rejects | 12,672 | 48,288 | 249,792 |
| raw-AABB rejects | 5,504 | 32,992 | 64,384 |
| FBP calls | 3,328 | 10,592 | 15,872 |
| contacts emitted | 2,354 | 8,018 | 12,178 |

The conservation identity holds at every scale:

```text
pair visits - inflated rejects - home rejects
  = owned pairs
owned pairs - raw-AABB rejects
  = FBP calls
FBP calls - no-contact returns
  = emitted contacts
```

## 8. Nsight Compute for all three scale points

Nsight Compute 2022.4.1 is installed at `/usr/bin/ncu`. The previous report's “not installed” statement came from a malformed quoted shell probe and was wrong. The corrected runner captured three production fused launches per scale with graph replay disabled for per-kernel attribution.

| Scale | Fused duration mean, range | SM throughput | DRAM throughput | Achieved warp occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|
| 14k analogue | **186.6 µs**, 171.2–203.6 | 10.03% | 0.53% | 14.20% | 124 |
| 80k | **456.4 µs**, 444.9–465.3 | 13.59% | 0.88% | 30.68% | 124 |
| 200k analogue | **335.9 µs**, 333.2–338.4 | 24.19% | 1.94% | 25.95% | 124 |

`sm__warps_active...` is achieved occupancy measured by hardware counters; it is different from the occupancy API's 50% theoretical resource limit. Low DRAM percentage confirms that the fused kernel is not saturating external memory bandwidth. Its high register count and limited independent mixed-big-cell work are more relevant constraints.

Nsight replays kernels and locks/controls clocks, so its durations are for kernel attribution and same-profiler comparisons. They need not equal graph-on production event means exactly.

## 9. Resource limits and profiling perturbation

| Property | Production fused kernel | Diagnostic fused kernel |
|---|---:|---:|
| launch | 1,024 × 256 threads | 1,024 × 256 threads |
| compiler registers/thread (`cuobjdump`) | **122** | 177 after the added sub-phase clocks |
| Nsight/runtime registers/thread | 124 | 171 reported by runtime query in this run |
| static shared memory/block | 17,936 B | 17,936 B |
| local memory/thread | 0 | 0 |
| stack frame | 0 | 0 |
| theoretical occupancy | 50% | 25% |

Only the separate diagnostic kernel changed. The compiled production specialization remains 122 registers, 17,936 B shared, zero local memory, and zero stack.

## 10. Practical conclusions

Keep these interpretations:

- use actual SOFA `avg_fps`/`avg_step_seconds` for deployment frame rate;
- use graph-on production CUDA events for collision pipeline latency;
- use graph-off stage events only to divide the pipeline;
- use `clock64()` counters for relative sub-kernel attribution and per-operation lane/block cost;
- use Nsight for achieved occupancy, utilization, register count, and replayed per-kernel duration;
- keep the inflated-AABB, home-cell, and raw-AABB filters before FBP;
- keep contacts and positions device-resident in production.

Do not do these:

- do not compare the old 27,520-element backend leg to the 14,368-element SOFA scene as if they were the same input;
- do not call the 213,170-element backend leg exact 200k;
- do not convert validation-oriented standalone wall time into SOFA FPS;
- do not convert summed block/thread cycles into whole-kernel milliseconds;
- do not quote the instrumented kernel's runtime or occupancy as production performance.

## 11. Reproduction and raw data

Build/run environment is the documented WSL checkout `/home/arfin/gpu-sofa`, SOFA 25.12 at `/opt/sofa/install/v25.12`, and the GTX 1650 Ti sm_75 target.

```bash
bash /home/arfin/gpu-sofa/scripts/run_bigcell_detailed_profile_wsl.sh
bash /home/arfin/gpu-sofa/scripts/run_ncu_bigcell_wsl.sh
```

Each final Nsight leg writes both a native `${label}_profile.ncu-rep` and an imported `${label}_ncu.csv` metric table.

Raw copies are retained locally under:

- `output/bigcell_detailed_profile_20260723_085021/`
- `output/profiling_bigcell_20260723_090726/`
- WSL actual-scene production summaries under `output/benchmark_logs/current_production_20260723/`

The raw directories are intentionally gitignored; this report and the reproducible runners are source controlled.