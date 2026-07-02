# Performance & profiling — the five broad-cull ways (canonical report)

**Date:** 2026-07-03 · **GPU:** GTX 1650 Ti (Turing sm_75, 16 SMs, ~48 KB shared/block)
**Stack:** SOFA v25.12 · CUDA 12.0 · WSL2 (`wsl-gpu-proj`) · driver 526.98
**This is the single canonical current report.** Prior reports live in
`archive_pre_20260703/` (5-way comparison, 4-way comparison, the 2026-06-18
performance report) and earlier `archive_pre_*` folders. Metric definitions:
`README_metrics_explained.md`.

All numbers below are from the fresh 2026-07-02/03 clean runs: the 12-leg full
suite, the 7-leg same-session mode comparison, the standalone backend bench, and a
per-kernel Nsight Compute capture of every broad-cull mode. Raw artifacts:
`output/benchmark_logs/full_suite_20260702_200438/`,
`mode_comparison_20260702_200412/`, `profiling_5way_20260702_202103/` (local,
`output/` is gitignored).

---

## 1. The five ways of execution

The narrow phase is shared (the FBP kernel), so every way must produce
**bit-identical contacts** — and does, in every measurement in this report.

| # | Way | Flag | One-line mechanism | Kernels/frame |
|---|-----|------|--------------------|:--:|
| 1 | **Baseline dense grid** | `useToolActiveCellGeneration=0` | 32,768-cell array, pair generation over **all** cells | 7 |
| 2 | **Optimised dense (Phase 15)** | *(default)* | dense grid, generation over **mixed cells only** (active list built during tool insert) | 7 |
| 3 | **Optimised hash** | `useHashPrefixSumGeneration=1` | open-addressing spatial hash, mark→compact→fill build, block-per-bucket generation, CUDA graph | 11 |
| 4 | **Simple hash** | `useSimpleHashGeneration=1` | direct-bucket hash: slot **is** the bucket, single-pass insert, CUDA graph | 7 |
| 5 | **Sorted grid** | `useSortedGridGeneration=1` | (cell,tri) incidence expansion → **counting sort** by cell → contiguous runs → block-per-mixed-cell generation with the tool run in **shared memory**; **home-cell** exactly-once dedup (doubles as an exact AABB pre-cull); no per-cell caps; CUDA graph | 9 |

Way 5 has two internal A/B toggles: sort engine (`sortedGridUseCubSort`:
counting sort ↔ CUB radix) and dedup (`sortedGridUsePairHashDedup`: home-cell ↔
atomicCAS pair-hash). Precedence when several flags are set: 3 > 4 > 5 > dense.

---

## 2. Headline comparison — same scene, same session (thermally fair)

`hash_prefixsum_large.py`, 14,368 elements, validation mode (counter readback on),
7 legs back-to-back (`run_mode_comparison_ab_wsl.sh`, run 20260702_200412):

| Leg | narrow **kernel** ms | FPS | launches | contacts (VF/FV/EE) | overflow |
|-----|---:|---:|:--:|:--|:--:|
| 1 dense plain | 1.694 | 335 | 7 | 2354 (1119/428/807) | 0 |
| 2 dense Phase-15 | 1.373 | 387 | 7 | 2354 | 0 |
| 3 optimised hash | 0.347 | 673 | 11 | 2354 | 0 |
| 4 simple hash | **0.336** | 599 | 7 | 2354 | 0 |
| 5 **sorted grid** | 0.352 | 640 | 9 | 2354 | 0 |
| 5a sorted, CUB engine | 0.479 | 630 | 10 | 2354 | 0 |
| 5b sorted, pair-hash dedup | 0.512 | 587 | 10 | 2354 | 0 |

**Reading it:** ways 3/4/5 are a statistical tie (~0.34–0.35 ms, ~4–5× over the
dense grid). The 5a/5b legs answer the internal design questions: the counting
sort beats CUB radix by ~0.13 ms (one pass; the scanned histogram doubles as run
starts and scatter cursors), and home-cell dedup beats the pair-hash table by
~0.16 ms (no atomicCAS dedup + the pre-cull).

### The large regime — where the ways separate

Standalone backend bench, 79,520 elements (64,800 tissue + 14,720 blade),
validation-style readback, CUDA-event kernel totals:

| Way | kernel ms (broad + FBP) | unique pairs | contacts |
|-----|---:|---:|:--|
| dense FBP | 2.647 | 322,560 | 8018 (2828/3469/1721) |
| optimised hash | 2.009 | 322,560 | 8018 |
| simple hash | 2.136 | 322,560 | 8018 |
| **sorted grid** | **0.651** | **43,584** | 8018 |

**The sorted grid is ~3× faster than every other way on the large scene.** The
sort itself roughly ties the hashes at binning; the win is **home-cell dedup
acting as an exact pre-cull** — pairs whose `contactDistance`-inflated AABBs are
disjoint cannot produce a contact (distance > contactDistance by construction),
so they are dropped before the FBP kernel: 322,560 → 43,584 pairs (−86 %). The
FBP kernel is the dominant cost at this scale, so starving it wins the frame.

---

## 3. Full suite — every scene, every run (run 20260702_200438)

| Leg | Scene / mode | FPS | narrow kernel ms | contacts (VF/FV/EE) | overflow | launches |
|-----|--------------|---:|---:|:--|:--:|:--:|
| small_fastpath | one-tissue/one-blade, detection-only | 987 | 0.284 | — | 0 | 7 |
| small_validation | one-tissue/one-blade | 560 | 0.682 | 56 (0/0/56) | 0 | 7 |
| large_fastpath | large-tissue/blade, detection-only | 140 | 2.104 | — | 0 | 7 |
| large_validation | large-tissue/blade | 126 | 2.863 | 8018 (5397/880/1741) | 0 | 7 |
| vt_self_validation | v-t self-collision | 1748 | — | 2700 (2700/0/0) | 0 | 6 |
| vt_self_fastpath | v-t self, detection-only | 7969 | — | — | 0 | 6 |
| vt_cross_validation | v-t cross-model | 1826 | — | 254 (254/0/0) | 0 | 6 |
| vt_cross_fastpath | v-t cross, detection-only | 3013 | — | — | 0 | 6 |
| hash_dense | 14,368 scene, dense Phase-15 | 367 | 1.393 | 2354 (1119/428/807) | 0 | 7 |
| hash_on | 14,368 scene, optimised hash | 633 | 0.341 | 2354 | 0 | 11 |
| hash_simple | 14,368 scene, simple hash | 671 | 0.336 | 2354 | 0 | 7 |
| hash_sorted | 14,368 scene, sorted grid | 576 | 0.356 | 2354 | 0 | 9 |

Contact counts reproduce the canonical fingerprints exactly on every leg
(56 / 8018 / 2700 / 254 / 2354). FPS is thermally noisy on this laptop —
kernel time is the robust metric.

---

## 4. FULL PROFILING — per-kernel Nsight Compute, every mode

Capture: `run_profile_5way_wsl.sh` → ncu per mode (45 launches after a 12-launch
skip; metrics: kernel time, SM throughput SOL %, DRAM %, achieved occupancy %,
registers/thread), on the 79,520-element backend bench. **Method notes:** CUDA
graphs were disabled during capture so each kernel is an individually
attributable launch (kernel *times* are unaffected; only launch overhead
differs). ncu's kernel replay isolates each launch, which inflates absolute
times slightly vs the CUDA-event totals in §2 — relative weights are the
signal. In every table, the rows marked *(exact leg)* come from the bench's
always-on exact-contact leg that precedes the profiled mode — ignore them when
summing a mode's frame. Nsight *Systems* GPU tracing is unavailable on this
WSL nsys 2022.4 (known limitation), hence ncu + CUDA events.

### 4.1 Way 2 — optimised dense grid (FBP leg)

| kernel | avg µs | SOL sm% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| `generateActiveDenseGridUniqueCandidatePairsKernel` | 1238.5 | 1.4 | 19.4 | 89.7 | 23 |
| `featureBasedProximityKernel` | 207.8 | 27.5 | 8.7 | 44.1 | 79 |
| `insertIndexedTrianglesKernel` (×2/frame) | 47.9 | 12.7 | 17.7 | 84.4 | 36 |
| `resetDenseGridKernel` | 4.1 | 24.9 | 34.5 | 65.8 | 16 |
| `resetProximityCountersKernel` | 1.9 | 0.0 | 1.0 | 3.1 | 16 |
| *(exact leg)* `generateDenseGridUniqueCandidatePairsKernel` | 1886.9 | 10.2 | 13.0 | 85.4 | 22 |
| *(exact leg)* `insertPackedTrianglesKernel` | 49.3 | 11.3 | 27.9 | 84.7 | 36 |

**Where the dense frame goes (79k scene):** generation dominates (1.24 ms —
322,560 pairs pushed through the atomicCAS dedup at 1.4 % SM = memory/atomic
latency-bound), then the FBP kernel (208 µs), then a cost invisible to ncu:
the **full per-frame `cudaMemset` of the 2M-slot pair-dedup table (~8 MB)** —
the dense path has no touched-slot clearing. That memset is the bulk of the gap
between the ~1.5 ms kernel sum here and the 2.65 ms event total in §2.

### 4.2 Way 3 — optimised hash

| kernel | avg µs | SOL sm% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| `generateMixedHashCandidatePairs32Kernel` | 1237.4 | 1.5 | 19.5 | 88.6 | 26 |
| `clearTouchedPairHash32Kernel` | 379.7 | 1.4 | 16.0 | 86.1 | 16 |
| `featureBasedProximityKernel` | 200.4 | 27.8 | 5.3 | 43.6 | 79 |
| `fillCompactHashGridTrianglesKernel` (×2/frame) | 77.3 | 12.3 | 23.2 | 85.1 | 28 |
| `markCompactHashGridCellsKernel` (×2/frame) | 51.2 | 14.8 | 15.4 | 82.1 | 23 |
| `compactHashGridSlotsKernel` | 34.4 | 27.3 | 41.7 | 75.8 | 16 |
| `resetCompactHashGridKernel` | 28.1 | 34.3 | 90.2 | 72.5 | 16 |
| `computeCompactHashPairsPerBucketKernel` | 4.4 | 20.6 | 5.8 | 71.9 | 16 |
| `resetProximityCountersKernel` | 1.9 | 0.0 | 0.9 | 3.1 | 16 |

Frame sum ≈ 2.11 ms — matches the 2.01 ms event total (§2) within ncu
inflation. Generation is the dominant cost here too (same 322,560 deduped
pairs), and with this many pairs the **touched-slot clear itself costs 380 µs**
(322k slots to reset). Note `resetCompactHashGridKernel` at 90 % DRAM —
bandwidth-bound table wipe.

### 4.3 Way 4 — simple hash

| kernel | avg µs | SOL sm% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| `generateMixedHashCandidatePairs32Kernel` | 1251.3 | 1.5 | 19.6 | 87.9 | 26 |
| `clearTouchedPairHash32Kernel` | 381.9 | 1.4 | 16.4 | 85.9 | 16 |
| `insertSimpleHashTrianglesKernel` (×2/frame) | 193.7 | 10.9 | 18.5 | 83.1 | 32 |
| `featureBasedProximityKernel` | 190.3 | 27.7 | 6.5 | 44.2 | 79 |
| `resetCompactHashGridKernel` | 58.5 | 16.0 | 94.6 | 70.5 | 16 |
| `resetProximityCountersKernel` | 1.7 | 0.0 | 1.3 | 3.2 | 16 |

The single-pass insert (194 µs ×2) replaces the hash's mark+fill (2×51 + 2×77
≈ 256 µs) — cheaper, and three build kernels fewer. Generation + touched-clear
+ FBP are identical machinery to way 3, which is why the two tie.

### 4.4 Way 5 — sorted grid (counting sort + home-cell)

| kernel | avg µs | SOL sm% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| `exclusiveScanSortedGridBinsKernel` | 413.0 | 3.8 | 1.8 | 100 | 16 |
| `scatterSortedGridIncidencesKernel` | 119.4 | 8.7 | 29.5 | 95.7 | 16 |
| `expandSortedGridIncidencesKernel` (×2/frame) | 111.1 | 6.6 | 17.7 | 77.1 | 36 |
| `featureBasedProximityKernel` | **84.7** | 37.8 | 3.5 | 55.1 | 79 |
| `generateSortedGridCandidatePairs32Kernel` | **35.5** | 56.4 | 5.1 | 92.4 | 33 |
| `buildSortedGridMixedCellsKernel` | 6.0 | 9.7 | 64.8 | 79.4 | 16 |
| `resetSortedGridKernel` | 4.2 | 27.0 | 27.7 | 60.5 | 16 |
| `resetProximityCountersKernel` | 1.9 | 0.0 | 0.9 | 3.1 | 16 |

Frame sum ≈ 0.89 ms (event total 0.65 ms, §2). Three things stand out:

1. **Generation collapsed from 1,237 µs to 35.5 µs (35×)** — no dedup table, no
   atomicCAS probing, shared-memory tool staging, and the AABB pre-cull emits
   only 43,584 pairs. At 56 % SM SOL it is the only *compute*-busy kernel in any
   broad cull.
2. **The FBP kernel drops from ~200 µs to 84.7 µs** — same kernel, 86 % fewer
   pairs — and its achieved occupancy/SOL actually *improve* (44→55 %, 27→38 %)
   because the surviving pairs are denser in real work.
3. **The new #1 cost is the scan: 413 µs at 3.8 % SM SOL on a single block.**
   One 1024-thread block sweeping 196,608 bins uses 1/16th of the GPU. This is
   the clearly identified next lever for the sorted grid: a standard multi-block
   scan (or 2-level reduce-then-scan) should cut it ~5–10×, putting the sorted
   grid near ~0.3 ms even on this large scene. Deliberately not done yet —
   single-block is trivially correct and deterministic; optimise after it wins
   on merit (it already does).

### 4.5 The FBP narrow kernel across ways (same kernel, different diets)

| Fed by | pairs | avg µs | SOL sm% | occ% |
|---|---:|---:|---:|---:|
| dense / hash / simple | 322,560 | 190–208 | ~27.7 | ~44 |
| sorted (home-cell pre-cull) | 43,584 | 84.7 | 37.8 | 55.1 |

The 79-register, load-throughput-bound profile (long_scoreboard + lg_throttle,
§5.21 of `guide/plan.md`) is unchanged — the sorted grid doesn't make the kernel
faster, it makes it *do less*.

---

## 5. Correctness (every way, every harness)

| Harness | dense | Phase-15 | opt hash | simple | sorted (all 4 toggle combos) |
|---|---|---|---|---|---|
| 14,368 SOFA scene | 2354 (1119/428/807) | 2354 | 2354 | 2354 | 2354 |
| 79,520 backend bench | 8018 (2828/3469/1721) | 8018 | 8018 | 8018 | 8018 |
| overflow / drops | 0 | 0 | 0 | 0 | 0 |

- Home-cell mode reports `uniquePairCount` 43,584 (the pre-cull is *supposed* to
  shrink it); pair-hash mode reproduces 322,560 exactly. Contacts identical
  either way.
- Sorted grid has **no per-cell caps**; its only capacity is the incidence
  buffer (16× triangle count — the SOFA scene's triangles span ~8.6 cells on
  average, which overflowed the initial 8× default by 8,080 before the fix;
  `incidenceOverflowCount` is asserted 0 in all reported runs).
- Robustness gauntlet: 24 consecutive processes across all sorted-grid toggle
  combos, every one 8018.

## 6. Environment gotchas (documented so they are never re-debugged)

1. **CUB `DeviceRadixSort` intermittently returns `cudaSuccess` with completely
   unsorted output on this WSL2 stack** — ~20–25 % of processes, decided per
   process, constant within it. Proven with a device-side verifier (536,295 of
   536,304 entries violated ordering/run-boundary checks in failing processes;
   0 in passing ones — same binary). **Shipped mitigation:** a frame-0 health
   probe (verify kernel + one 4-byte sync, once per process) detects it, prints
   a notice, permanently falls back to the counting sort for the process, and
   redoes the frame. `SOFA_SORTED_GRID_VERIFY=1` enables continuous per-frame
   verification. This is a second reason (besides speed) that the counting sort
   is the default engine.
2. **Never pad a buffer with `cudaMemsetAsync` before a same-frame kernel
   rewrite** — async memsets may execute on a copy/DMA engine. The sorted grid
   pads the key tail with a **fill kernel that runs after the writer** instead.
3. Nsight Systems 2022.4 under WSL2 does not capture GPU kernel timelines; use
   ncu (per-kernel metrics) + CUDA-event totals, as in §4.

## 7. Which way should you run?

| Scene | Use | Why |
|---|---|---|
| Small tool, local footprint (surgical default) | **dense + Phase 15** (default) | fixed build stages of any alternative aren't worth it; 987 FPS fast path |
| Mid-size, both meshes large-ish (≈14k elements) | opt hash / simple hash / sorted grid — **tied** | simple hash = fewest kernels; sorted grid = no caps + best correctness story |
| Large + pair-dense (≈80k elements) | **sorted grid** | ~3× over everything (home-cell pre-cull); next lever identified (multi-block scan) |
| Self-collision / point-cloud | dense (v-t path) | alternative culls are tri-tri-only |

## 8. Optimization history in one table (details: `guide/plan.md` §5)

| Milestone | Effect | Ref |
|---|---|---|
| Phase 15 tool-active-cell generation (default ON) | generation 300 µs → 8 µs small scene; 4.3× FPS | §5.15 |
| Grid-stride correctness fix | large scenes stopped silently dropping pairs | §5.17 |
| Optimised hash (compact buckets, mixed-only, no binary search, touched-clear, 32-bit pairs) | 2.5–3× over dense on large | §5.19 |
| Scan drop + FBP AABB pre-reject + CUDA graphs | hash → ~0.35 ms, CUB removed | §5.20 |
| FBP occupancy raise — **failed, reverted** | LSU-bound, not occupancy-bound | §5.21 |
| Simple hash (4th way) | ties opt hash with 7 kernels — compaction wasn't the win | §5.22 |
| **Sorted grid (5th way)** | ties on mid, **~3× on large** via home-cell pre-cull | §5.23 |

**Methods that DO NOT work (do not retry):** FBP `__launch_bounds__`/register
reduction/`__ldg` (measured regression — LSU-bound); `compactActiveCells`
(regressed, replaced by Phase 15); `batchTriangleInsert` (breaks
tissue-before-tool ordering); warp-aggregated atomics (atomics < 2 % of time);
CUB radix as sorted-grid default (slower + unreliable on this stack).

## 9. How to reproduce every number here

```bash
# 7-leg mode comparison (§2):
scripts/run_mode_comparison_ab_wsl.sh
# Full suite (§3):
scripts/run_full_benchmark_suite_wsl.sh
# Backend bench + parity (§2 large table, §5):
./SofaGpuCollisionDenseGridBackendBench          # all legs incl. sorted
scripts/run_sorted_grid_parity_wsl.sh            # sorted-grid 4-combo parity
# Full per-kernel profiling (§4):
scripts/run_profile_5way_wsl.sh
# Build (Windows -> WSL sync + cmake):
scripts/sync_and_build_wsl.sh
```
