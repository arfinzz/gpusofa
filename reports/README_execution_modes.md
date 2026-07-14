# Execution Modes, Inputs, and How to Read the Tables

Companion to [README_metrics_explained.md](README_metrics_explained.md) (per-metric
formulas) and the canonical performance report
([performance_all_modes_20260715.md](performance_all_modes_20260715.md)). This file
answers three questions: **what are the execution modes, what inputs do they run on, and
what does every value in the tables mean?**

---

## 1. The pipeline in one paragraph

Every frame, the GPU collision pipeline answers: *which pairs of triangles are close
enough to touch, and where exactly?* It runs in two stages. The **broad cull** sorts
triangles into spatial cells and shortlists candidate pairs that share a cell. The
**narrow phase** runs exact closest-feature math (vertex–face and edge–edge distances,
Ericson's algorithms) on each shortlisted pair and emits a `ProximityContact` (points,
normal, signed distance, barycentric weights) for every pair within `contactDistance`.
The broad cull is the part with six interchangeable implementations ("ways"); the narrow
math is identical everywhere — which is why every mode must produce identical contacts.

## 2. The 12 execution modes (every A/B combination)

A **mode** = a way + its toggle settings. All 12 produce identical contacts on identical
scenes; only speed differs.

| # | Mode (leg name) | Way | What is different | Selecting envs |
|---|---|---|---|---|
| 1 | `dense_plain` | 1 baseline dense grid | fixed cell array; generation launches one block per grid cell (mostly empty) | `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=0` |
| 2 | `dense_phase15` | 2 optimised dense grid | same grid; generation only over tool-occupied cells (**the production default**) | *(all toggles 0)* |
| 3 | `hash_opt` | 3 optimised spatial hash | occupied-cells-only storage via mark→compact→fill hash table; 11 kernels | `SOFA_USE_HASH_PREFIXSUM_GENERATION=1` |
| 4 | `simple_hash` | 4 simple direct-bucket hash | triangles inserted straight into per-cell hash buckets in one pass; 7 kernels | `SOFA_USE_SIMPLE_HASH_GENERATION=1` |
| 5 | `sorted_grid` | 5 sorted grid | (cell, triangle) incidences → **counting sort** → contiguous runs; **home-cell** exactly-once dedup that doubles as an exact AABB pre-cull | `SOFA_USE_SORTED_GRID_GENERATION=1` |
| 6 | `sorted_cub` | 5 | sort engine swapped to `cub::DeviceRadixSort` | + `SOFA_SORTED_GRID_CUB_SORT=1` |
| 7 | `sorted_pairhash` | 5 | dedup swapped to the atomicCAS pair-hash (like ways 3/4; forfeits the pre-cull) | + `SOFA_SORTED_GRID_PAIRHASH_DEDUP=1` |
| 8 | `sorted_cub_pairhash` | 5 | both swaps at once | + both |
| 9 | `bigcell_direct` | 6 big-cell fused | per-big-cell CSR table built with **direct global atomics**; ONE kernel per mixed big cell stages tool ids+AABBs+vertices in shared memory and runs the narrow math **inline** (no pair list, no separate narrow launch) | `SOFA_USE_BIGCELL_FUSED_GENERATION=1 SOFA_BIGCELL_SHARED_BUILD=0` |
| 10 | `bigcell_sharedhash` | 6 | CSR build privatized: each block populates a **shared-memory hash table** for its triangle chunk, merges per-bin (**the way-6 default**) | `SOFA_USE_BIGCELL_FUSED_GENERATION=1` *(default shared build)* |
| 11 | `bigcell_sharedsort` | 6 | CSR build privatized via a **shared-memory bitonic sort** + run-writer merge | + `SOFA_BIGCELL_SHARED_BUILD=2` |
| 12 | `bigcell_globalhash` | 6 | the table is a literal **per-big-cell open-addressing hash multi-map in global memory** (fixed slot regions, cleared per frame) | + `SOFA_BIGCELL_HASH_BUILD=1` (`SOFA_BIGCELL_HASH_SLOTS=2048` for zero drops) |

Precedence when several way-flags are set: **bigcell > hash > simple > sorted > dense**.
Way-6 extras: `SOFA_BIGCELL_FACTOR` (big-cell edge in small cells, default 2 — 4 collapses
block parallelism), `SOFA_BIGCELL_TOOL_TILE` (shared staging chunk, default 256). Every
way has a CUDA-graph replay toggle (`SOFA_{HASH,SIMPLE_HASH,SORTED_GRID,BIGCELL}_CUDA_GRAPH`,
default on) — replays the frame's kernel sequence without per-launch CPU overhead.

**What "A/B" means:** run configuration A, run configuration B, change nothing else
(same scene, back-to-back for equal GPU thermals), compare kernel time, and require both
to emit the identical contact set. Every mode pair above is such an A/B.

## 3. The inputs (scenes) — identical for every mode

| Input | Geometry | Elements | Used for |
|---|---|---|---|
| bench "80k" | 181×181 tissue grid (64,800 tris) + subdivided blade (14,720 tris), generated in-process | 79,520 | kernel-only timing, ncu profiling |
| bench "200k" | 316×316 tissue grid (198,450) + same blade | 213,170 | scale behavior |
| `testscenes/hash_prefixsum_large.py` | 81×81 tissue + blade, static, in SOFA | 14,368 | end-to-end SOFA timing (the surgical-scale scene) |
| `testscenes/collision_xlarge_200k.py` | 316×316 tissue + blade in SOFA | ~213k | end-to-end at scale |
| `one_tissue_one_blade.py` / `large_tissue_blade.py` / v-t scenes | see guide/setup.md §4 | 12,812 / 79,520 / — | full-suite regression legs |

The geometry generators are deterministic (no randomness), and a comparison run launches
every leg on the *same scene file with the same parameters* — only the mode env differs.
That is what makes contact-count equality a meaningful correctness gate.

## 4. What each reported value means

### Correctness values (must match across modes on the same scene)

| Value | Meaning |
|---|---|
| `contacts` (+ `vf/fv/ee`) | Emitted proximity contacts, split by closest-feature class (Vertex–Face / Face–Vertex / Edge–Edge). **The gate: identical across all 12 modes.** 8018 on the 80k bench, 17,040 at 200k, 2354 on the 14k scene, 12,178 on the xlarge scene |
| `unique_pairs` / `pairs_tested` | Candidate pairs surviving the broad cull. Ways 1–4 and pair-hash dedup: 322,560 (80k). Home-cell modes (sorted default, all bigcell modes): 43,584 — lower because home-cell dedup pre-culls AABB-disjoint pairs; contacts are unaffected |
| `incidences` / `entries` | (cell, triangle) records built by ways 5/6 — 536,304 at 80k; way 5 and way 6 must agree |
| `*_overflow` | Dropped items due to a capacity limit. **Must be 0** in a valid measurement; any overflow invalidates the contact-parity claim (best-effort modes report honestly) |
| `shared_spill` | Way-6 shared-build entries that fell back to the direct global path (staging full). Correct either way; 0 on all measured scenes |

### Performance values

| Value | Meaning | Trust level |
|---|---|---|
| `*_gpu_kernel_avg_ms` / `nkern` | CUDA-event-timed GPU work per frame | **the robust metric** — ranks algorithms |
| `wall_avg_ms` / `nwall` | CPU-side orchestration time incl. sync | includes host noise |
| `fps` | whole-app frame rate | noisiest: CPU scene overhead + laptop thermals; use for sanity only |
| ncu `dur` | per-kernel duration under the profiler | clocks locked + replay ⇒ inflated absolutes; **read relatively** |
| ncu `SM%` | achieved compute throughput vs peak | ~1–2% = latency-bound (waiting), >50% = compute-busy |
| ncu `DRAM%` | memory bandwidth used vs peak | ~90% = memory-bound; ~1% = data already on-chip |
| ncu `occ%` | resident warps vs hardware max | **not a goal** — measured twice on this project that raising it makes the hot kernels slower |
| ncu `regs` | registers per thread | >64 starts limiting occupancy; the fused kernel runs 122 by design |

### Reading a comparison table

1. Check every leg's `contacts` match — otherwise stop, it's a bug hunt not a benchmark.
2. Check overflows/spills are 0.
3. Rank by kernel ms; treat <5% differences as noise unless reproduced.
4. Cold-clock caveat: the first leg of a fresh process can read high (idle 300 MHz →
   ramp); same-session back-to-back legs are the fair comparison.

## 5. Where things live

- Canonical numbers + per-kernel profiles: `performance_all_modes_20260715.md`
- Metric formulas: `README_metrics_explained.md`
- How to run everything: `guide/setup.md` (§5 scripts, §6 env tables, §7 walkthroughs)
- Design/why: `guide/architecture.md`; history: `guide/plan.md` §5
- Raw artifacts: `output/benchmark_logs/` (gitignored; mirrored from WSL)
