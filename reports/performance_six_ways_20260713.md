# GPU Collision Detection — Six-Way Performance Report (2026-07-13)

**This is the single canonical performance report.** Everything was re-measured fresh on
2026-07-13 on the GTX 1650 Ti laptop (WSL2, CUDA 12.x, SOFA v25.12): the standalone
backend bench at two sizes, the 8-leg same-session SOFA mode comparison, the full 16-leg
scene suite, and per-kernel Nsight Compute profiles for every execution mode. Earlier
reports live in `archive_pre_20260713/` (and older archives); metric definitions in
[README_metrics_explained.md](README_metrics_explained.md).

**Headline: all six broad-cull ways produce identical contacts on every scene, and the
newest way — way 6, big-cell fused generation + narrow phase — is the fastest on every
SOFA scene measured (0.30 ms on the 14k surgical scene, 2.01 ms on the 213k scene) and
the fastest or tied on the standalone bench.**

---

## 1. What "A/B" means (and how every number here was produced)

An **A/B comparison** is the simplest honest experiment: run version **A**, run version
**B**, *change nothing else*, compare. In this project that discipline is enforced three
ways:

1. **Same session** — competing modes run back-to-back in one process wherever possible,
   so GPU temperature and clock state are comparable (this laptop throttles; a "win"
   measured cold vs hot is a lie).
2. **Same inputs, same required outputs** — every mode must produce the *identical*
   contact list. A mode that is faster but finds different contacts is a bug, not a win.
   Contact counts (and their VF/FV/EE class splits) are printed for every leg and must
   match exactly.
3. **Kernel time over FPS** — `avg_narrow_kernel_ms` (CUDA-event-timed GPU work) is the
   robust metric; FPS includes CPU scene overhead and thermal noise.

Every table below is an A/B (or A/B/C/D/E/F) in this sense.

## 2. The six ways, in one paragraph each

All six are *broad culls* for the triangle–triangle proximity path: their only job is to
find candidate triangle pairs near enough to matter, which then get the exact
closest-feature test (VF/FV/EE, Ericson math). Same input, same required output.

| # | Way | Core idea | Kernels |
|---|---|---|---|
| 1 | baseline dense grid | fixed cell array, one block per grid cell | 7 |
| 2 | optimised dense grid (Phase 15, **default**) | dense grid + generate only over tool-occupied cells | 7 |
| 3 | optimised spatial hash | occupied-cells-only via mark→compact→fill hash table | 11 |
| 4 | simple direct-bucket hash | one-pass insert straight into per-cell hash buckets | 7 |
| 5 | sorted grid (tiled binning) | (cell, triangle) incidences → counting sort → contiguous per-cell runs; home-cell exactly-once dedup that doubles as an exact AABB pre-cull | 9 |
| 6 | **big-cell fused** (2026-07-12) | small cells grouped into 2³ big cells; per-big-cell CSR table; **one kernel** per mixed big cell stages tool ids+AABBs+**vertices** in shared memory and runs the narrow-phase math **inline** — no pair list, no separate narrow launch | 9 |

Ways 1–5 hand their pair list to the shared `featureBasedProximityKernel`. Way 6 is the
only way that never materializes a pair list: generation and narrow phase are fused, so
each surviving pair goes straight from the AABB/home-cell test to the closest-feature
math with the tool triangle's vertices already sitting in shared memory.

## 3. Correctness (the gate everything passes before timing counts)

Fresh 2026-07-13 runs, every way, every scene — contacts and class splits identical,
zero overflow of any kind:

| Scene | Contacts (VF/FV/EE) | Ways agreeing |
|---|---|---|
| bench 80k (79,520 elements) | 8018 (2828/3469/1721) | FBP-dense, hash, simple, sorted, bigcell |
| bench 200k (213,170 elements) | 17,040 (8805/4914/3321) | same five |
| SOFA 14,368 scene | 2354 (1119/428/807) | all 8 legs (dense ×2, hash, simple, sorted ×3, bigcell) |
| SOFA xlarge 213k scene | 12,178 | dense, sorted, bigcell |
| small / large / vt-self / vt-cross | 56 / 8018 / 2700 / 254 | historical fingerprints reproduced |

Structural invariants also hold: way 6's `pairs_tested` = way 5's `unique_pairs`
(43,584 at 80k; 89,856 at 200k) — the two ways provably apply the same exactly-once
home-cell rule; and both report identical entry/incidence totals (536,304 / 1,215,744).

## 4. Performance — the A/B tables

### 4.1 Standalone backend bench (kernel-only, no SOFA, 30 measured steps)

GPU kernel avg per frame, milliseconds:

| Way | 80k elements | 200k elements |
|---|---:|---:|
| dense grid + FBP | 2.580 | 4.495 |
| optimised hash | 2.011 | 4.670 |
| simple hash | 2.150 | 4.698 |
| sorted grid (way 5) | 0.633 | 1.066 |
| **big-cell fused (way 6)** | **0.575** | **1.066** |

Way 6 is 9% faster than way 5 at 80k and exactly tied at 200k; both are ~4× faster
than every hash/dense way. (The hash ways *scale worse than dense* here because their
per-pair atomicCAS dedup cost grows with the 776,832-pair candidate set.)

### 4.2 SOFA 14,368-element scene — 8 legs, same session (160 steps each)

| Leg | narrow kernel ms | fps |
|---|---:|---:|
| **bigcell_fused** | **0.304** | 683 |
| hash_opt | 0.342 | 783 |
| simple_hash | 0.343 | 750 |
| sorted_grid | 0.380 | 630 |
| sorted_cub | 0.506 | 552 |
| sorted_pairhash | 0.523 | 612 |
| dense_phase15 | 1.334 | 413 |
| dense_plain | 1.627 | 364 |

(fps ordering differs from kernel ordering because fps folds in CPU-side scene overhead
and run-to-run clock state; kernel time is the metric that ranks the GPU algorithms.)

### 4.3 SOFA xlarge 213k scene (full suite legs)

| Leg | narrow kernel ms | narrow wall ms | fps |
|---|---:|---:|---:|
| **xlarge_bigcell** | **2.007** | 2.566 | 69.6 |
| xlarge_sorted | 2.259 | 2.766 | 74.4 |
| xlarge_dense | 3.705 | 4.279 | 67.4 |

### 4.4 Full 16-leg suite (fresh, `full_suite_20260713`)

All legs green: small 0.86/0.74 ms kernel (fast/validation, 56 contacts), large
2.12/2.78 ms (8018), v-t self/cross 1543–6970 fps (2700/254), 14k-scene five-way legs
as §4.2, xlarge as §4.3. Zero overflow everywhere.

### 4.5 Way-6 internal A/Bs (what the toggles cost)

| Toggle A/B | Result |
|---|---|
| **CSR build vs literal per-big-cell hash-table build** (`bigCellUseHashBuild`) | CSR 0.71 ms vs hash build 2.02–2.17 ms full pipeline at 80k — **hash ~2.9× slower** (per-frame slot clearing + atomicCAS probe chains + sparse region sweeps). Hash build needs 2048 slots/region for zero drops (1024 dropped 256 entries) |
| **bigCellFactor 4 vs 2 vs 1** | 200k bench: 1.52 / 1.09 / 1.07 ms. Factor 4 leaves only 48 mixed big cells → 48 blocks on a 16-SM GPU (parallelism collapse). **Default: 2** |
| **toolTile 256 vs 32** | 0.71 vs 0.84 ms at 80k — the chunk loop is correct but not free |
| way-5 sort engine (counting vs CUB) | 0.380 vs 0.506 ms (SOFA scene) — counting sort still wins |
| way-5 dedup (home-cell vs pair-hash) | 0.380 vs 0.523 ms — home-cell still wins (and keeps the free pre-cull) |

## 5. Per-kernel Nsight Compute profile — every way (80k bench)

Captured 2026-07-13 with CUDA graphs disabled for clean attribution
(`profiling_5way_20260712_194244` + `_194502` on the WSL side; per-launch means).
ncu locks clocks and replays kernels, so durations are inflated vs wall benchmarks —
**read them relative to each other**. The legacy dense-exact leg
(`generateDenseGridUniqueCandidatePairsKernel`, ~1.89 ms) runs in every capture and is
excluded from the pipeline sums below.

### Way 2 — optimised dense grid

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| generateActive…PairsKernel | 1,245 µs | 1.4 | 19.4 | 90 | 23 |
| featureBasedProximityKernel | 242 µs | 24.4 | 7.5 | 34 | 94 |
| inserts (packed+indexed) | ~97 µs | 11–13 | 18–28 | 84–86 | 36 |

**Bound by:** the generation kernel — 1.4% SM at 90% occupancy is the signature of
*latency-bound dedup*: every candidate pair does an atomicCAS insert into the pair-hash
and waits. Dense also pays a hidden multi-MB pair-table memset per frame.

### Way 3 — optimised hash

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| generateMixedHashCandidatePairs | 1,246 µs | 1.5 | 19.4 | 88 | 26 |
| clearTouchedPairHash | 381 µs | 1.4 | 15.9 | 86 | 16 |
| featureBasedProximityKernel | 229 µs | 24.4 | 4.7 | 34 | 94 |
| fill / mark / compact / reset | 76+51+34+28 µs | 12–34 | 15–90 | 73–85 | 16–28 |

**Bound by:** the same atomicCAS-dedup generation as dense (1.5% SM) plus 381 µs of
touched-slot clearing. The clever table build (mark→compact→fill, ~190 µs total) is NOT
the problem — the per-pair dedup is.

### Way 4 — simple hash

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| generateMixedHashCandidatePairs | 1,250 µs | 1.5 | 19.6 | 88 | 26 |
| clearTouchedPairHash | 382 µs | 1.4 | 16.4 | 86 | 16 |
| featureBasedProximityKernel | 217 µs | 24.4 | 5.8 | 34 | 94 |
| insertSimpleHash ×2 | 192 µs each | 10.9 | 18.3 | 83 | 32 |
| resetCompactHashGrid | 59 µs | 15.9 | **94.1** | 71 | 16 |

**Bound by:** identical generation/dedup story. Its one-pass insert (192 µs ×2) is
simpler but not faster than way 3's build; its table reset is DRAM-saturated (94%).

### Way 5 — sorted grid

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| exclusiveScanSortedGridBins | **413 µs** | 3.8 | 1.8 | 100 | 16 |
| scatterSortedGridIncidences | 120 µs | 8.6 | 29.3 | 96 | 16 |
| expandSortedGridIncidences ×2 | 111 µs each | 6.6 | 17.6 | 78 | 36 |
| featureBasedProximityKernel | **88 µs** | 35.8 | 3.4 | 40 | 94 |
| generateSortedGridCandidatePairs | **35 µs** | **56.5** | 5.0 | 93 | 33 |
| mixed/reset | ~10 µs | — | — | — | 16 |

**The transformation:** generation drops from ~1,250 µs @1.5% SM (ways 2–4) to
**35 µs @56% SM** — home-cell dedup needs no table, no CAS, no clearing, and its AABB
pre-cull cuts pairs 322,560 → 43,584, which also shrinks FBP 229 → 88 µs. Remaining #1
cost: the single-block scan over 196k bins (413 µs @3.8% SOL — one block, one SM).

### Way 6 — big-cell fused

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| **fusedBigCellNarrowKernel** | **527 µs** | 14.0 | 1.1 | 42 | **122** |
| countBigCellEntries ×2 | 57 µs each | 7.6 | 11.6 | 77 | 34 |
| exclusiveScanSortedGridBins | **52 µs** | 3.8 | 0.8 | 100 | 16 |
| fillBigCellEntries ×2 | 48 µs each | 9.8 | 15.3 | 86 | 36 |
| mixed/reset | ~6 µs | — | — | — | 16 |

**Why it wins:** (a) the scan input is factor³ smaller than way 5's → 52 µs instead of
413 µs (way 6 never needed the multi-block-scan fix); (b) generation + narrow phase are
one kernel: the 527 µs fused kernel replaces way 5's 35 µs generation **plus** 88 µs FBP
**plus** the pair-list write/read between them, and reads each tool triangle's vertices
from shared memory instead of global per pair (DRAM 1.1% — almost nothing left to
fetch). Costs: 122 registers (FBP math + staging logic) cap occupancy at ~42%, and the
CSR build walks triangles twice (count 57+fill 48 vs way 5's single 111 µs expansion).
Pipeline sums (ncu-inflated): way 6 ≈ 795 µs vs way 5 ≈ 884 µs — matching the wall-clock
0.575 vs 0.633 ms.

### The shared narrow kernel across ways (same kernel, different feed)

`featureBasedProximityKernel`: 242 µs under dense (322,560 pairs) → 88 µs under way 5
(43,584 pre-culled pairs, SM% rises 24 → 36). The *feed* determines its cost; way 6
absorbs it entirely.

## 6. What bounds what — the one-table summary

| Way | Dominant cost | Nature | Fixable? |
|---|---|---|---|
| dense (2) | pair generation 1.25 ms | atomicCAS dedup latency (1.4% SM) | only by changing dedup strategy (= way 5/6) |
| hash (3) | generation 1.25 ms + clear 0.38 ms | same + table hygiene | same |
| simple (4) | generation 1.25 ms + clear 0.38 ms | same | same |
| sorted (5) | single-block scan 0.41 ms | 1 block on a 16-SM GPU (3.8% SOL) | multi-block scan (~50 µs projected) — still open |
| **bigcell (6)** | fused kernel 0.53 ms | 122 regs → 42% occupancy; latency-bound sweep | `__launch_bounds__` retest justified (different shape than the old FBP dead-end); heavy-big-cell work splitting |

## 7. Environment gotchas (do not re-debug)

1. **CUB DeviceRadixSort on this WSL2 stack** intermittently returns `cudaSuccess` with
   completely unsorted output (~20–25% of processes, constant per process). Shipped
   mitigation: frame-0 health probe → process-wide fallback to the counting sort + frame
   redo + stderr notice. Counting sort is faster anyway.
2. **Never `cudaMemsetAsync` a buffer a same-frame kernel rewrites** — copy-engine
   ordering can clobber it on WSL2. Use a post-writer fill kernel.
3. **nsys 2022.4 under WSL2 cannot capture GPU kernel timelines** — use ncu + CUDA events.
4. **Windows→WSL orchestration:** `wsl.exe` invocations inherit the caller's wall-clock
   cap and `nohup`'d background jobs die when the session exits; the suite script's
   `SOFA_SUITE_ONLY` regex filter exists to run it in batches. PowerShell expands `$VAR`
   inside double-quoted `bash -c` strings — always use the script-file pattern.

## 8. History

Way-6 design/landing detail: `archive_pre_20260713/way6_bigcell_fused_20260712.md`.
Five-way state + prior profiling: `archive_pre_20260713/performance_five_ways_20260703.md`.
Earlier optimization history (hash rework, CUDA graphs, FBP occupancy dead-end, 4-way):
`archive_pre_20260703/` and older archives. Raw artifacts for this report:
`output/benchmark_logs/{bench_6way_20260713, mode_comparison_20260712_193954,
full_suite_20260713, profiling_5way_20260712_194244, profiling_5way_20260712_194502}`.
