# Winning big-cell fused algorithm: detailed CUDA profile

> Size-label, wall-time, production-FPS, corrected Nsight, and finer in-kernel timing clarification: `reports/bigcell_fused_profile_clarification_20260723.md`. The original 28k/80k/213k campaign below is retained as a valid profile of those standalone inputs.

Date: 2026-07-23
Hardware: NVIDIA GeForce GTX 1650 Ti (Turing, sm_75), 16 SMs
Configuration: big-cell factor 2, 256-entry tool tile, CSR table, shared-hash build/merge, fused raw-AABB reuse enabled

## 1. Result in one page

The profiling code is now part of the backend and is opt-in. Normal execution still launches the original production `fusedBigCellNarrowKernel`; internal profiling launches a separate `profiledFusedBigCellNarrowKernel` so clocks and counters cannot silently contaminate the winner.

The main findings are:

1. The production pipeline remains correct at every measured scale and in both canonical SOFA scenes. Production, CUDA-event, and internally instrumented legs produced identical pair/contact counts and contact subtypes, with zero true overflow.
2. The production fused kernel is unchanged at **122 compiled registers/thread, 17,936 B static shared memory, 0 local bytes, and 0 stack bytes**.
3. CUDA's runtime occupancy calculation allows **2 blocks/SM at 256 threads/block = 50% theoretical occupancy**. The instrumented kernel rises to 165 registers and allows only 1 block/SM = 25%; its elapsed time is therefore deliberately not a production number.
4. In graph-off event runs, the fused consumer is the largest single stage: **56.9% / 49.6% / 44.2%** of serialized pipeline time at the 28k / 80k / 213k backend scales. At 213k, CSR count+fill growth makes table construction nearly as important as the fused consumer.
5. Inside the fused consumer, the tissue/pair sweep accounts for **96.3-96.4%** of sampled major-phase block cycles. Tile setup, 64-bin prefix construction, and tool gather together are below 4%.
6. The closest-feature FBP call accounts for **71.1-74.0%** of sampled filter/FBP thread cycles. The inflated AABB stage accounts for 19.9-25.1%; home-cell ownership for 2.9-4.7%; the exact raw-AABB gap test for only 0.85-1.49%.
7. The raw-AABB reuse optimization is strongly validated: it rejects **59.4-75.7%** of home-cell-owned pairs while consuming under 1.5% of the sampled filter/FBP cycles.

Raw CSV/log artifacts are under `output/bigcell_detailed_profile_20260723/final3/` and are intentionally gitignored. This report and the runner script are source-controlled.

## 2. What changed

### 2.1 Pipeline CUDA events

With `DenseGridConfig::detailedProfiling=true`, eleven persistent workspace events bracket the nine winner launches:

| Boundary | Work measured after the previous boundary |
|---|---|
| 0 | pipeline start |
| 1 | histogram/counter reset |
| 2 | hash slot clear, or a no-work marker on CSR |
| 3 | tissue count/insert |
| 4 | tool count/insert |
| 5 | exclusive scan (CSR) |
| 6 | tissue CSR fill |
| 7 | tool CSR fill |
| 8 | mixed-big-cell list build |
| 9 | proximity-counter reset; also internal-profile buffer reset when enabled |
| 10 | fused generation+narrow kernel |

For the winning CSR mode, boundary 1→2 contains no kernel. Its elapsed value is exposed as `event_marker_gap_ms`, which estimates one event-boundary cost. `event_build_clear_ms` is zero on CSR and is reserved for the real clear kernel in hash-build mode.

Events are allocated once in `BigCellWorkspace`, recorded on the same stream as the kernels, and synchronized only at the final event. Detailed profiling disables CUDA graph replay because event-by-event attribution and a captured graph are different measurement modes.

### 2.2 In-kernel diagnostic specialization

`BigCellConfig::profileFusedInternals` is false by default. When true it implies detailed profiling and launches a separately compiled diagnostic kernel. It records:

- one block-level `clock64()` interval for tile setup/histogram, local-bin prefix, tool gather, and tissue/pair sweep;
- one thread-level `clock64()` interval for inflated-AABB filtering, home-cell ownership, raw-AABB gap filtering, and FBP closest-feature math;
- exact work counters for big cells, tiles, staged tool entries, visited tissue entries, small-cell pair visits, each rejection reason, FBP calls, and FBP no-contact returns.

Warp reductions turn each per-thread accumulator into one global atomic per warp at kernel exit. This avoids an atomic on every pair, but the clock reads, extra live values, reductions, and counters still perturb the kernel. The profile kernel exists for attribution, not speed measurement.

### 2.3 Occupancy and resource reporting

Detailed mode queries:

- `cudaFuncGetAttributes` for registers, static shared memory, local memory, and maximum block size;
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the resident-block limit;
- `cudaGetDeviceProperties` for SM count, threads/SM, warp size, and clock-rate metadata;
- `cuobjdump --dump-resource-usage` for the compiler's exact register/local/stack report.

The standalone benchmark writes all values into its big-cell CSV and prints a readable summary.

### 2.4 Controls

| Surface | Control |
|---|---|
| Backend C++ | `BigCellConfig::profileFusedInternals` |
| SOFA component | `bigCellProfileInternals` Data field |
| Canonical scenes | `SOFA_BIGCELL_PROFILE_INTERNALS=1` |
| Standalone bench | `SOFA_BACKEND_BENCH_BIGCELL_PROFILE_INTERNALS=1` |
| Event-only stages | `SOFA_GPU_DETAILED_PROFILING=1`, internal flag off |
| Production timing | both detailed/internal flags off; CUDA graph on |

## 3. Measurement discipline

Three separate legs were run because they answer different questions:

| Leg | Graph | Events | Internal clocks/counters | What it means |
|---|---:|---:|---:|---|
| Production | on | aggregate start/end only | off | representative winner timing |
| Events | off | every stage | off | stage attribution; event/launch overhead included |
| Internals | off | every stage | on | work funnel and relative cycle attribution; elapsed time is perturbed |

Backend scales:

| Label | Tissue triangles | Tool triangles | Total triangles | Production samples | Event samples | Internal samples |
|---|---:|---:|---:|---:|---:|---:|
| 28k | 12,800 | 14,720 | 27,520 | 80 | 30 | 12 |
| 80k | 64,800 | 14,720 | 79,520 | 80 | 30 | 12 |
| 213k | 198,450 | 14,720 | 213,170 | 80 | 30 | 12 |

Warm-up rows were excluded: 12 production, 6 event, and 3 internal iterations.

Important distinction:

- CUDA-event milliseconds are serialized GPU elapsed time between stream markers.
- Block cycles are summed over blocks; thread cycles are summed over threads. Concurrent work overlaps, so summed cycles must not be converted into wall time.
- Theoretical occupancy is a resource-limit calculation. Achieved occupancy requires a hardware-counter profiler. A corrected environment probe found Nsight Compute at `/usr/bin/ncu`; current 14k/80k/200k achieved-occupancy results are in `reports/bigcell_fused_profile_clarification_20260723.md`.

## 4. Correctness gates

### 4.1 Standalone parity matrix

The existing factor/tile/build parity matrix passed after instrumentation. The winning shared-hash leg produced:

- pairs tested: 43,584, identical to sorted-grid unique pairs;
- contacts: 8,018 = 2,828 VF + 3,469 FV + 1,721 EE;
- entry overflow: 0;
- build overflow: 0;
- total true overflow: 0.

The production, event-only, and internal legs also matched within every scale:

| Scale | Pairs tested | Contacts | VF | FV | EE | True overflow |
|---|---:|---:|---:|---:|---:|---:|
| 28k | 8,896 | 2,522 | 0 | 1,747 | 775 | 0 |
| 80k | 43,584 | 8,018 | 2,828 | 3,469 | 1,721 | 0 |
| 213k | 89,856 | 17,040 | 8,805 | 4,914 | 3,321 | 0 |

`sharedSpillCount=67,344` at 28k is not lost work: it counts shared-build entries that deliberately fell back to the correct direct-global path. It is not included in true overflow.

### 4.2 SOFA scene integration

Both scene-exposed flags loaded and ran for 20 frames with internal profiling enabled:

| Scene | Elements | Candidates after home-cell cull | Contacts | VF / FV / EE | Overflow | Instrumented total / fused ms |
|---|---:|---:|---:|---:|---:|---:|
| `hash_prefixsum_large.py` | 14,368 | 8,832 | 2,354 | 1,119 / 428 / 807 | 0 | 2.401 / 1.980 |
| `collision_xlarge_200k.py` | 200,018 | 80,256 | 12,178 | 3,615 / 4,774 / 3,789 | 0 | 2.644 / 1.781 |

These elapsed values are diagnostic-kernel values and must not replace production scene timings.

## 5. Production timing

The production leg uses graph replay and keeps internal profiling off. `gpu_kernel_ms` is the aggregate event interval already used by the benchmark when counters are read.

| Scale | Wall mean ± std (ms) | GPU pipeline mean (ms) | GPU p50 (ms) | GPU p95 (ms) |
|---|---:|---:|---:|---:|
| 28k | 0.955 ± 0.166 | 0.368 | 0.365 | 0.395 |
| 80k | 1.454 ± 0.180 | 0.494 | 0.493 | 0.515 |
| 213k | 3.194 ± 0.229 | 0.817 | 0.815 | 0.874 |

Do not subtract the event-leg stages from these production totals: the event leg disables graphs and inserts ten additional timing boundaries.

## 6. CUDA-event stage breakdown

All values below are means in milliseconds from graph-off, internal-off runs.

| Stage | 28k | 80k | 213k | Meaning |
|---|---:|---:|---:|---|
| reset | 0.0585 | 0.0933 | 0.0647 | clear histograms and counters |
| event marker gap | 0.0160 | 0.0153 | 0.0196 | no kernel; one CSR event-boundary estimate |
| tissue count | 0.0334 | 0.0600 | 0.1593 | shared hash per block, merge bin counts |
| tool count | 0.0363 | 0.0120 | 0.0120 | same count path for the fixed tool |
| scan | 0.0669 | 0.0436 | 0.0405 | one-block exclusive scan of big-cell bins |
| tissue fill | 0.0362 | 0.0678 | 0.1832 | shared hash per block, reserve and scatter CSR entries |
| tool fill | 0.0170 | 0.0182 | 0.0127 | same fill path for the fixed tool |
| mixed-cell list | 0.0049 | 0.0116 | 0.0037 | retain big cells containing both sides |
| proximity reset | 0.0042 | 0.0159 | 0.0029 | clear output class/contact counters |
| fused consumer | **0.3607** | **0.3321** | **0.3956** | stage tool tile, cull, FBP, emit contacts |
| total | **0.6341** | **0.6699** | **0.8944** | event 0→10 |
| fused share | **56.9%** | **49.6%** | **44.2%** | fused / event total |

Interpretation:

- The fused consumer is still the largest individual launch.
- Tissue count+fill scales from 0.0696 ms at 28k to 0.3425 ms at 213k, while fixed-tool count+fill stays near 0.025 ms.
- At 213k, tissue count+fill alone is 38.3% of event total. Future full-pipeline work should consider the shared-build/merge producer as well as the fused consumer.
- Small kernels are sensitive to event and launch overhead. The marker gap demonstrates why event-leg absolute time is higher than graph replay, especially at smaller scales.

## 7. Inside the fused kernel: exact work funnel

The diagnostic counters form conservation identities:

```text
pair visits
  - inflated-AABB rejects
  - home-cell rejects
= pairs tested
  - raw-AABB gap rejects
= FBP calls
  - FBP no-contact returns
= emitted contacts
```

| Funnel stage | 28k | 80k | 213k |
|---|---:|---:|---:|
| CSR entries | 210,432 | 536,304 | 1,215,744 |
| mixed big cells | 88 | 88 | 88 |
| tool-tile iterations | 168 | 168 | 168 |
| tool entries staged | 29,696 | 29,696 | 29,696 |
| small-cell pair visits | 208,128 | 462,848 | 1,085,952 |
| inflated-AABB rejects | 187,680 | 370,976 | 901,632 |
| survive inflated AABB | 20,448 | 91,872 | 184,320 |
| home-cell rejects | 11,552 | 48,288 | 94,464 |
| pairs tested | 8,896 | 43,584 | 89,856 |
| raw-AABB rejects | 5,280 | 32,992 | 68,032 |
| FBP calls | 3,616 | 10,592 | 21,824 |
| FBP no-contact | 1,094 | 2,574 | 4,784 |
| contacts | 2,522 | 8,018 | 17,040 |

Rejection effectiveness:

| Metric | 28k | 80k | 213k |
|---|---:|---:|---:|
| inflated AABB rejects of pair visits | 90.18% | 80.15% | 83.03% |
| home-cell rejects of inflated survivors | 56.49% | 52.56% | 51.25% |
| raw AABB rejects of owned pairs | 59.35% | 75.70% | 75.71% |
| FBP calls that emit a contact | 69.75% | 75.70% | 78.08% |

This is the clearest validation of the winning raw-AABB reuse: it prevents 5,280 / 32,992 / 68,032 expensive FBP calls per frame after home-cell ownership has already been resolved.

## 8. Internal cycle attribution

### 8.1 Major block phases

These percentages use the sum of block-level clock intervals. They show relative work attribution, not elapsed-time share.

| Block phase | 28k | 80k | 213k |
|---|---:|---:|---:|
| tile setup + local histogram | 0.79% | 0.85% | 0.83% |
| 64-bin prefix | 1.51% | 1.36% | 1.29% |
| tool gather | 1.44% | 1.48% | 1.47% |
| tissue/pair sweep | **96.26%** | **96.31%** | **96.41%** |

Tool staging is not the current target. Almost all block time is in the sweep that visits tissue entries and applies pair filters/FBP.

### 8.2 Pair filters and closest-feature math

These percentages use summed participating-thread cycles.

| Thread work | 28k | 80k | 213k |
|---|---:|---:|---:|
| inflated AABB | 25.11% | 19.90% | 21.48% |
| home-cell ownership | 2.91% | 4.68% | 4.53% |
| raw-AABB gap | 0.85% | 1.42% | 1.49% |
| FBP closest-feature | **71.14%** | **74.00%** | **72.50%** |

The raw-AABB check has excellent economics: under 1.5% sampled cycle share, yet it removes 59-76% of the pairs that reach it. Removing or weakening it would be a regression.

The inflated-AABB test is individually cheap but is executed for every small-cell pair visit, so its aggregate share is meaningful. If optimizing the fused consumer further, reducing duplicate pair visits before this point is more promising than shaving a few instructions from tool staging.

## 9. Resources and occupancy

| Property | Production kernel | Instrumented kernel |
|---|---:|---:|
| launch | 1,024 blocks × 256 threads | same |
| compiler registers/thread (`cuobjdump`) | **122** | 165 |
| CUDA runtime `numRegs` | 124 | 165 |
| static shared memory/block | 17,936 B | 17,936 B |
| local memory/thread | 0 B | 0 B |
| stack frame | 0 B | 0 B |
| active blocks/SM from occupancy API | **2** | 1 |
| active threads/SM | 512 | 256 |
| theoretical occupancy | **50%** | 25% |

Device limits used by the calculation: 16 SMs, 1,024 threads/SM, warp size 32, reported clock rate 1,485,000 kHz. The production kernel launches eight warps/block. Two resident blocks give 16 resident warps against the device maximum of 32 warps/SM.

The backend geometry has 88 mixed big cells at every tested scale because the tool footprint is fixed. That is about 5.5 useful blocks per SM. Larger tissue sizes increase work inside those same mixed cells rather than increasing block-level parallelism.

The instrumented kernel's 165 registers halve theoretical occupancy and make its fused elapsed time 1.66-2.54× the event-only production kernel. This slowdown is evidence that the profiling mode is isolated correctly; it is not a production regression.

## 10. What each reported metric means

| Metric | Meaning | Correct use |
|---|---|---|
| `wall_ms` | CPU elapsed time around the backend call | end-to-end backend cost including synchronization/readback |
| `gpu_kernel_ms` | aggregate CUDA-event interval for the launched pipeline | compare production runs with identical sync/graph settings |
| `event_*_ms` | elapsed stream time between adjacent persistent CUDA events | attribute graph-off pipeline stages |
| `event_marker_gap_ms` | CSR no-work event interval | estimate one event boundary; not algorithm work |
| `prod_*` resources | runtime properties of the uninstrumented kernel | theoretical launch/occupancy limits |
| `profile_*` resources | runtime properties of the diagnostic kernel | explain instrumentation perturbation |
| `profile_*_rejects` | exact number rejected at one ordered filter | reconstruct the pair funnel |
| `profile_*_block_cycles` | sum of block wall-cycle samples | relative major-phase attribution only |
| `profile_*_thread_cycles` | sum of participating-thread cycle samples | relative filter/FBP attribution only |
| `shared_spill` | entries that used correct global fallback after shared staging filled | diagnostic work-shape counter, not lost data |
| `total_overflow` | entries/contacts truly dropped due capacity | must remain zero for correctness |

## 11. Reproduction

From PowerShell, sync/build as documented in `guide/setup.md`, then run:

```powershell
wsl -d wsl-gpu-proj -- bash -lc 'SOFA_BIGCELL_PROFILE_OUT=/home/arfin/gpu-sofa/output/my_bigcell_profile bash /home/arfin/gpu-sofa/scripts/run_bigcell_detailed_profile_wsl.sh'
```

The runner produces for each 28k/80k/213k scale:

- `<scale>_production.log` and `<scale>_production_bigcell.csv`;
- `<scale>_events.log` and `<scale>_events_bigcell.csv`;
- `<scale>_internals.log` and `<scale>_internals_bigcell.csv`;
- `resource_usage.txt` from `cuobjdump`.

For a single backend diagnostic leg:

```bash
SOFA_GPU_DETAILED_PROFILING=1 \
SOFA_BACKEND_BENCH_BIGCELL_PROFILE_INTERNALS=1 \
SOFA_BIGCELL_CUDA_GRAPH=0 \
SOFA_BACKEND_BENCH_RUN_FBP=0 \
SOFA_BACKEND_BENCH_RUN_VT=0 \
SOFA_BACKEND_BENCH_RUN_HASH=0 \
SOFA_BACKEND_BENCH_RUN_SIMPLE_HASH=0 \
SOFA_BACKEND_BENCH_RUN_SORTED_GRID=0 \
SofaGpuCollisionDenseGridBackendBench
```

For a SOFA scene, add:

```bash
SOFA_USE_BIGCELL_FUSED_GENERATION=1 \
SOFA_BIGCELL_PROFILE_INTERNALS=1 \
SOFA_GPU_DETAILED_PROFILING=1
```

Always turn `SOFA_BIGCELL_PROFILE_INTERNALS` back off before measuring production performance.

## 12. Practical conclusions

Keep:

- the production kernel isolated from diagnostic code;
- the exact raw-AABB reuse prefilter;
- home-cell ownership before FBP;
- tool-side shared staging and local-bin runs;
- graph replay for production;
- event-only and internal modes as separate profiling tools.

Do not infer:

- that 25% instrumented occupancy is a production problem;
- that summed `clock64()` cycles are elapsed milliseconds;
- that event-leg absolute time should match graph replay;
- that `sharedSpillCount` is an overflow;
- that higher occupancy alone guarantees speed.

The next optimization target depends on scale. For the fused kernel itself, the FBP call dominates surviving-pair cycles, but the highest-leverage approach is still to prevent unnecessary calls cheaply. For the complete 213k pipeline, shared-hash CSR tissue count/fill is now comparable to the fused consumer and deserves equal attention.
