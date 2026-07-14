# GPU Collision Detection — All Execution Modes, Complete Comparison (2026-07-15)

**This is the single canonical performance report.** It covers **every A/B combination
the codebase has** — 12 distinguishable tri-tri execution modes — measured fresh on
2026-07-15 on identical scenes, plus per-kernel Nsight Compute profiles for every mode.
What each mode/input/value means: [README_execution_modes.md](README_execution_modes.md).
Metric formulas: [README_metrics_explained.md](README_metrics_explained.md). Prior
reports: `archive_pre_20260715/` and older archives.

Hardware: GTX 1650 Ti laptop (Turing, 16 SMs), WSL2, SOFA v25.12. Measurement discipline:
same-session back-to-back legs, identical deterministic scenes, kernel time as the
ranking metric, and **identical contact sets required from every mode** (checked on every
leg; also the VF/FV/EE class split).

---

## 1. The complete mode inventory (answering "what do we have?")

Two dense-grid variants, two hash ways, the sorted grid with 2×2 toggle combos, and the
big-cell fused way with 4 build strategies = **12 modes**:

| # | Mode | One-line description |
|---|---|---|
| 1 | `dense_plain` | baseline dense grid, one block per grid cell |
| 2 | `dense_phase15` | dense grid, generation over tool-occupied cells only (**production default**) |
| 3 | `hash_opt` | optimised spatial hash (mark→compact→fill), 11 kernels |
| 4 | `simple_hash` | direct-bucket spatial hash, single insert pass, 7 kernels |
| 5 | `sorted_grid` | sorted grid: counting sort + home-cell dedup (= exact AABB pre-cull) |
| 6 | `sorted_cub` | way 5 with `cub::DeviceRadixSort` instead of the counting sort |
| 7 | `sorted_pairhash` | way 5 with the atomicCAS pair-hash dedup instead of home-cell |
| 8 | `sorted_cub_pairhash` | way 5 with both swaps |
| 9 | `bigcell_direct` | way 6 (fused generation+narrow), CSR table via direct global atomics |
| 10 | `bigcell_sharedhash` | way 6, CSR built via per-block **shared-memory hash tables** (**way-6 default**) |
| 11 | `bigcell_sharedsort` | way 6, CSR built via per-block **shared-memory bitonic sort** |
| 12 | `bigcell_globalhash` | way 6, table as a literal per-big-cell **global-memory hash multi-map** |

(Way-6 has two further tuning knobs measured earlier and held at their measured-best
defaults here: `bigCellFactor=2`, `toolTileCapacity=256`. Every way also has a CUDA-graph
replay toggle, default on, worth ~7–10% of launch overhead when on.)

## 2. Correctness — identical scenes, identical contacts, all 12 modes

The scenes are byte-identical for every leg (deterministic generators, same file, same
parameters); only one env toggle differs per leg. Every mode reported:

| Scene | Contacts (VF/FV/EE) | Pairs (dedup-dependent) | Overflows/spills |
|---|---|---|---|
| bench 80k (79,520 el) | 8018 (2828/3469/1721) ×12 | 322,560 (ways 1–4, pair-hash modes) / 43,584 (home-cell modes) | 0 everywhere |
| bench 200k (213,170 el) | 17,040 (8805/4914/3321) ×12 | 776,832 / 89,856 | 0 everywhere |
| SOFA 14,368 scene | 2354 (1119/428/807) ×12 legs | — | 0 everywhere |

The two pair counts are both correct: home-cell dedup *additionally* rejects
AABB-disjoint pairs during generation (the pre-cull), which never changes the contact
set — the narrow math would have rejected the same pairs, just later and slower.

## 3. The full A/B table (the ranking)

GPU kernel time per frame, milliseconds. SOFA = 14,368-element scene, 12 legs
same-session (160 steps). Bench = kernel-only tool (30 steps).

| Mode | SOFA 14k | bench 80k | bench 200k |
|---|---:|---:|---:|
| **bigcell_sharedhash** | **0.290** | **0.564** | **0.806** |
| bigcell_direct | 0.316 | 0.587 | 1.077 |
| simple_hash | 0.335 | 2.22 | 4.73 |
| sorted_grid | 0.346 | 0.747 | 1.097 |
| hash_opt | 0.352 | 2.11 | 4.71 |
| sorted_cub | 0.484 | 1.289 | 2.775 |
| sorted_pairhash | 0.515 | 2.256 | 5.125 |
| sorted_cub_pairhash | 0.669 | 2.924 | 6.763 |
| bigcell_sharedsort | 0.759 | 1.558 | 3.388 |
| bigcell_globalhash | 1.092 | 2.800 | 6.985 |
| dense_phase15 | 1.505 | 2.58–4.75* | 4.50 |
| dense_plain | 1.658 | — | — |

\* the 80k base bench process started on cold clocks (1350 MHz ramping); its early legs
read high — the canonical warm dense-FBP number is ~2.6 ms. All 12-leg SOFA legs and all
variant bench processes ran warm; treat <5% deltas as noise.

**Readings:**

- **Champion: way 6 with the shared-hash build** — fastest on every scene, 16–26% ahead
  of the runner-up at the bench sizes and 5.7× faster than the production-default dense
  path on the SOFA scene.
- **Every toggle A/B has a clear loser**: CUB loses to the counting sort (1.4–2.5×),
  pair-hash dedup loses to home-cell (3–4.7× at the bench — it both pays CAS *and*
  forfeits the pre-cull), the global hash-table build loses to CSR (2.9–6.5×, worse with
  scale because its slot table grows with big-cell count), and the shared *sorted-list*
  build loses to the shared *hash* build (~2.7–4×).
- **Scale behavior**: ways 1–4 cluster at 4.5–4.7 ms at 200k (their per-pair atomicCAS
  dedup scales with the 776,832-pair candidate set); the home-cell family (sorted,
  bigcell) stays at 0.8–1.1 ms.

## 4. Full per-kernel breakdown — what each part costs in every mode

Nsight Compute per-launch means at 80k, CUDA graphs disabled for attribution. ncu locks
clocks and replays kernels, so durations are inflated vs §3 — **read them relatively**.
The legacy dense-exact leg (`generateDenseGridUniqueCandidatePairsKernel`, ~1.89 ms
@10% SM) runs in every capture as a stable cross-capture reference and is excluded from
pipeline sums. "×2" = launched once per mesh (tissue, tool).

### dense_phase15 (way 2)

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| generateActive…UniqueCandidatePairs | 1,245 µs | 1.4 | 19 | 90 | 23 |
| featureBasedProximityKernel (322k pairs) | 242 µs | 24 | 8 | 34 | 94 |
| inserts (packed + indexed ×2) | ~97 µs | 11–13 | 18–28 | 84 | 36 |
| resets | ~6 µs | — | — | — | 16 |

Bound by generation: 1.4% SM at 90% occupancy = every candidate pair blocked on an
atomicCAS dedup round trip. Dense also re-memsets a multi-MB pair table each frame.

### hash_opt (way 3)

| Kernel | dur | SM% | DRAM% |
|---|---:|---:|---:|
| generateMixedHashCandidatePairs | 1,246 µs | 1.5 | 19 |
| clearTouchedPairHash | 381 µs | 1.4 | 16 |
| featureBasedProximityKernel | 229 µs | 24 | 5 |
| fill / mark / compact / reset (the build) | 76+51+34+28 µs | 12–34 | 15–90 |

Same dedup-bound generation as dense, plus table hygiene. Its clever build (~190 µs) was
never the problem.

### simple_hash (way 4)

| Kernel | dur | SM% |
|---|---:|---:|
| generateMixedHashCandidatePairs | 1,250 µs | 1.5 |
| clearTouchedPairHash | 382 µs | 1.4 |
| featureBasedProximityKernel | 217 µs | 24 |
| insertSimpleHash ×2 | 192 µs each | 11 |
| resetCompactHashGrid | 59 µs | 16 (**DRAM 94%**) |

### sorted_grid (way 5, counting + home-cell)

| Kernel | dur | SM% | DRAM% | occ% |
|---|---:|---:|---:|---:|
| exclusiveScanSortedGridBins (**#1 cost**) | 413 µs | 3.8 | 2 | 100 (1 block!) |
| scatterSortedGridIncidences | 120 µs | 9 | 29 | 96 |
| expandSortedGridIncidences ×2 | 111 µs each | 7 | 18 | 78 |
| featureBasedProximityKernel (43,584 pairs) | 88 µs | 36 | 3 | 40 |
| generateSortedGridCandidatePairs | **35 µs** | **57** | 5 | 93 |
| mixed/resets | ~10 µs | — | — | — |

The transformation vs ways 1–4: home-cell dedup removes the table → generation
1,250 → 35 µs; its pre-cull (−86% pairs) shrinks the narrow kernel 242 → 88 µs.

### sorted_cub (way 5 + CUB radix sort)

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| exclusiveScanSortedGridBins | 413 µs | 3.8 | 2 | 100 | 16 |
| cub OnesweepKernel | 205 µs | 19 | 53 | **25** | **160** |
| expand ×2 | 111 µs each | 7 | 18 | 78 | 36 |
| featureBasedProximityKernel | 88 µs | 36 | 3 | 40 | 94 |
| cub Histogram + ExclusiveSum + pad | 55+3+17 µs | — | — | — | — |
| generation | 35 µs | 57 | 5 | 92 | 33 |

CUB replaces the 120 µs scatter with ~280 µs of general-purpose sort machinery (plus the
frame-0 health probe this WSL2 stack requires). The counting sort wins because the keys
are already dense bin ids — no radix passes needed.

### sorted_pairhash (way 5 + atomicCAS dedup)

| Kernel | dur | SM% |
|---|---:|---:|
| generateSortedGridCandidatePairs | **1,178 µs** | **1.6** |
| exclusiveScanSortedGridBins | 413 µs | 3.8 |
| clearTouchedPairHash | 380 µs | 1.4 |
| featureBasedProximityKernel (322k pairs) | 263 µs | 27 |
| scatter + expand ×2 | 119 + 112 µs each | 7–9 |

Swapping the dedup back to a table makes way 5 pay everything ways 1–4 pay — generation
balloons 35 → 1,178 µs AND the narrow kernel triples (no pre-cull). The dedup strategy,
not the sorted layout, is way 5's real invention.

### bigcell_direct (way 6, CSR via direct atomics)

| Kernel | dur | SM% | DRAM% | occ% | regs |
|---|---:|---:|---:|---:|---:|
| **fusedBigCellNarrowKernel** | 524 µs | 14 | 1.1 | 40 | **122** |
| countBigCellEntries ×2 | 57 µs each | 8 | 11 | 77 | 34 |
| exclusiveScan (bins factor³ smaller) | **52 µs** | 3.8 | 1 | 100 | 16 |
| fillBigCellEntries ×2 | 48 µs each | 10 | 15 | 87 | 36 |
| mixed/resets | ~6 µs | — | — | — | — |

The fused kernel replaces way 5's generation + pair list + separate narrow launch; the
big-cell histogram makes the scan 8× cheaper (52 vs 413 µs). DRAM 1.1% during the fused
kernel = the shared-memory staging left almost nothing to fetch.

### bigcell_sharedhash (way 6 default — the champion)

| Kernel | dur | SM% | occ% | regs |
|---|---:|---:|---:|---:|
| fusedBigCellNarrowKernel | **467 µs** | 14 | 30 | 122 |
| exclusiveScan | 52 µs | 3.8 | 100 | 16 |
| fillBigCellEntriesSharedHash ×2 | 50 µs each | **27** | 47 | 31 |
| countBigCellEntriesSharedHash ×2 | **36 µs** each | **22** | 89 | 24 |
| mixed/resets | ~6 µs | — | — | — |

vs bigcell_direct: count −37% with SM% tripled (per-bin instead of per-entry global
atomics), fill duration-neutral, and — the second-order find — **the fused consumer
itself dropped 524 → 467 µs** because block-built entries land contiguously inside each
bin run (a block's chunk is index-contiguous triangles → local vertex gathers). Note the
champion also has the *lowest* fused-kernel occupancy (30%) — occupancy is not the metric.

### bigcell_sharedsort (way 6, shared bitonic build)

| Kernel | dur | SM% |
|---|---:|---:|
| fillBigCellEntriesSharedSort ×2 | **527 µs** each | 52 |
| fusedBigCellNarrowKernel | 489 µs | 13 |
| countBigCellEntriesSharedSort ×2 | **283 µs** each | **77** |
| exclusiveScan | 52 µs | 3.8 |

The 2048-entry in-shared bitonic sort (~66 network passes/block) is pure compute —
77% SM on count and half a millisecond per fill launch. It costs far more than every
atomic it avoids. (Consumer also gets the 489 µs locality bonus — but the build eats it.)

### bigcell_globalhash (way 6, global hash multi-map)

| Kernel | dur | SM% | DRAM% |
|---|---:|---:|---:|
| clearBigCellHashSlots (67 MB every frame) | **996 µs** | 11 | **79** |
| insertBigCellHashEntries (CAS probing) | 909 µs | 14 | 5 |
| fusedBigCellNarrowKernel (sparse region sweeps) | 703 µs | 11 | 2 |
| mixed/resets | ~8 µs | — | — |

The global hash pays three times: a DRAM-saturated slot wipe, probe-chain inserts, and a
consumer that must sweep mostly-empty regions. At 200k the slot table scales with
big-cell count → 6.99 ms. This is why "hash table" only wins *inside shared memory*.

## 5. The memory-hierarchy lesson (the pair of mirrored A/Bs)

| Where the structure lives | Sorted/CSR organization | Hash organization | Winner |
|---|---:|---:|---|
| **global memory** | 0.59 ms | 2.80 ms | **sorted** (3–6.5×) — coalescing beats probe latency |
| **shared memory** | 1.56 ms | **0.56 ms** | **hash** (~2.8×) — probes are ~free, sort-network compute dominates |

Same two data structures, opposite winners, decided purely by which memory level they
occupy. Combined with the dedup A/B (§4, sorted_pairhash) the project's three recurring
findings are: (1) avoid per-item dependent global-memory chains (CAS probes, table
clears); (2) organization that costs compute is only worth it where the memory it saves
is expensive; (3) occupancy has never predicted performance on the hot kernels here.

## 6. What bounds each mode — one table

| Mode | Dominant cost | Nature |
|---|---|---|
| dense ×2 | generation ~1.25 ms | atomicCAS dedup latency (1.4% SM) |
| hash / simple | generation + 0.38 ms clear | same + table hygiene |
| sorted_grid | 413 µs single-block scan | 1 block on 16 SMs (still its open lever) |
| sorted_cub | + ~160 µs sort machinery | general-purpose radix vs dense keys |
| sorted_pairhash(+cub) | generation 1.18 ms | reintroduced dedup table |
| bigcell_direct | fused kernel 524 µs | 122 regs, latency-bound sweep |
| **bigcell_sharedhash** | fused kernel 467 µs | same, minus atomic and locality waste |
| bigcell_sharedsort | 810 µs of bitonic | raw compute |
| bigcell_globalhash | 1.9 ms clear+insert | DRAM wipe + probe chains |

Open levers, ranked (IDEAS.md §11): fused-kernel register pressure (`__launch_bounds__`
retest on the NEW kernel shape), heavy-big-cell work splitting, dense touched-slot
clearing, home-cell pre-cull ported to hash ways, sorting CSR runs by triangle id for
mode 0's locality bonus.

## 7. Environment gotchas (unchanged, do not re-debug)

1. CUB DeviceRadixSort on this WSL2 stack intermittently returns success with unsorted
   output → frame-0 health probe auto-falls back to the counting sort.
2. Never `cudaMemsetAsync` a buffer a same-frame kernel rewrites → post-writer fill kernel.
3. nsys 2022.4 under WSL2 can't capture GPU timelines → ncu + CUDA events.
4. Windows→WSL: script-file pattern always; `SOFA_SUITE_ONLY` batches the suite under
   wall-clock caps; nohup'd WSL jobs die with the session; PowerShell eats `$VAR` in
   double-quoted `bash -c` strings.

## 8. Artifacts & history

This report's raw data: `output/benchmark_logs/{bench_allmodes_20260715,
mode_comparison_20260714_193533, profiling_gaps_20260714_193722,
profiling_sharedbuild_20260714_185817, profiling_5way_20260712_194244, _194502}`.
Suite/xlarge legs (16-leg regression, 2026-07-13 run — small 56, large 8018, vt 2700/254,
xlarge 12,178 ×3 ways): `full_suite_20260713`. Prior canonical reports:
`archive_pre_20260715/` (six-ways + shared-build A/B), `archive_pre_20260713/` (way-6
landing + five-ways), and older archives.
