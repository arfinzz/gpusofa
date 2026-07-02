# Five-way broad-cull comparison — dense / optimised dense / optimised hash / simple hash / sorted grid

**Date:** 2026-07-03 · **GPU:** GTX 1650 Ti (Turing sm_75, 16 SM) · **SOFA** v25.12 / CUDA 12.0 / WSL2
**Supersedes nothing** — extends `four_way_broadcull_comparison_20260626.md` with the 5th way.

This report adds the **sorted-grid broad cull** ("5th way") — Green's sorted-particle-grid
/ tiled-binning method adapted to triangles — and measures all five ways head-to-head on
identical geometry, plus the 5th way's two internal A/B toggles (sort engine, dedup
strategy). The narrow phase (FBP kernel) is shared by every way, so **contacts must be
identical**; only candidate-pair production differs.

---

## The five ways

| # | Way | Flag | Broad cull |
|---|-----|------|-----------|
| 1 | Baseline dense grid | `useToolActiveCellGeneration=0` | 32,768-cell dense grid, all-cells generation |
| 2 | Optimised dense grid | (default) | dense grid + Phase-15 mixed-cell generation |
| 3 | Optimised hash | `useHashPrefixSumGeneration=1` | spatial hash, mark→compact→fill, CUDA graph |
| 4 | Simple hash | `useSimpleHashGeneration=1` | direct-bucket spatial hash, one-pass insert, CUDA graph |
| 5 | **Sorted grid (NEW)** | `useSortedGridGeneration=1` | expand (cell,tri) incidences → **sort by cell** → contiguous per-cell runs → block-per-mixed-cell generation with the tool run staged in **shared memory**, CUDA graph |

### The 5th way, in one paragraph

Each triangle is expanded into one `(cellKey, triId)` incidence per overlapped cell (key =
`cellId*2 + meshTag`, so tissue sorts before tool inside a cell; the expansion also stores
each triangle's inflated AABB and bumps a per-key histogram in the same pass). The
incidences are **sorted by key** — default engine is a hand-rolled **one-pass counting
sort** (single-block chained scan of the histogram doubles as the run boundaries `starts[]`
AND the scatter cursors; **no CUB dependency**), with `cub::DeviceRadixSort` behind a
toggle. Generation runs one block per mixed cell over the now-contiguous runs, staging the
tool run (ids + AABBs) in shared memory — the tiled-binning + shared-memory-privatization
pattern, made possible by the sort. **No per-cell capacity caps** → no best-effort triangle
drops (unlike the simple hash). Steady state replays as a CUDA graph. **9 kernels** in the
default configuration.

### Home-cell dedup — the surprise win

The default dedup is **home-cell exactly-once emission** (Le Grand / GPU Gems 3): a pair is
emitted only from the cell containing the min corner of its two AABBs' intersection — no
dedup hash table at all. Two consequences:
1. Exactly-once by construction (each pair has one owner cell).
2. **It doubles as an exact pre-cull**: pairs whose inflated AABBs don't overlap are
   skipped outright. That is safe — both AABBs are inflated by `contactDistance`, so
   disjoint inflated boxes ⇒ triangle distance > contactDistance ⇒ the FBP kernel would
   reject them anyway. On the large bench geometry this collapses **322,560 candidate
   pairs → 43,584 (−86%)**, and since the FBP narrow kernel is the dominant cost there,
   the whole pipeline gets dramatically faster (below).

---

## Headline results (all bit-identical, overflow 0)

### In-SOFA 7-leg comparison (14,368 elements, validation, back-to-back / thermally fair)

| Leg | narrow **kernel** ms | FPS | launches | contacts (VF/FV/EE) |
|-----|---:|---:|:--:|:--|
| dense plain | 1.694 | 335 | 7 | 2354 (1119/428/807) |
| dense Phase-15 | 1.373 | 387 | 7 | 2354 |
| optimised hash | 0.347 | 673 | 11 | 2354 |
| simple hash | 0.336 | 599 | 7 | 2354 |
| **sorted grid** (counting + home-cell) | **0.352** | 640 | 9 | 2354 |
| sorted grid (CUB radix engine) | 0.479 | 630 | 10 | 2354 |
| sorted grid (pair-hash dedup) | 0.512 | 587 | 10 | 2354 |

On this scene the sorted grid **ties the hash cluster** (0.34–0.35 ms, ~4–5× over dense).
The toggle A/Bs answer the design questions directly: the counting sort beats CUB radix
(−0.13 ms: one pass + free `starts[]` vs multi-pass over the padded buffer), and home-cell
beats the pair-hash table (−0.16 ms: no atomicCAS dedup, plus the pre-cull).

### Standalone backend bench (79,520 elements — the large regime)

| Path | kernel ms (broad + FBP) | unique pairs | contacts |
|------|---:|---:|:--|
| dense FBP | 2.647 | 322,560 | 8018 (2828/3469/1721) |
| optimised hash | 2.009 | 322,560 | 8018 |
| simple hash | 2.136 | 322,560 | 8018 |
| **sorted grid** | **0.651** | **43,584** | **8018** |

**On the large scene the sorted grid is ~3× faster than every other way end-to-end** — not
because sorting beats hashing at binning (it roughly ties), but because home-cell's exact
AABB pre-cull removes 86% of the candidate pairs before the load-bound FBP kernel ever sees
them. The prediction in the design discussion ("sorting will likely lose on this sparse
workload") was **wrong** — measured and gladly corrected. The win comes from the dedup
strategy the sorted representation enables, not from the sort itself.

---

## Correctness

- All five ways, all toggle combos: **2354 (1119/428/807)** on the SOFA scene and
  **8018 (2828/3469/1721)** on the bench — identical, overflow 0.
- Home-cell mode intentionally reports a **smaller uniquePairCount** (the pre-cull);
  pair-hash mode reproduces the other ways' 322,560 exactly.
- No per-cell caps: `sortedgrid_incidence_overflow=0` (capacity = 16× triangle count; the
  SOFA scene's triangles overlap ~8.6 cells on average, which overflowed the initial 8×
  default — fixed, and the stat is watched).
- 24-process gauntlet + repeated parity runs: every run 8018.

## Two hard-won environment findings (documented so they're never re-debugged)

1. **CUB `DeviceRadixSort` intermittently returns `cudaSuccess` with garbage output on
   this WSL2 stack** (~20–25% of processes, decided per process, stable within the
   process). Proven by a device-side verifier: 536,295/536,304 ordering violations in
   failing processes, 0 in passing ones — same binary, same inputs. Mitigation shipped:
   a **frame-0 health probe** (verify kernel + one 4-byte sync, once per process) that
   detects the condition, **falls back to the counting sort for the process, and redoes
   the frame**. `SOFA_SORTED_GRID_VERIFY=1` enables continuous per-frame verification.
   This is also why the counting sort is the default engine.
2. **Never pad a buffer with `cudaMemsetAsync` before a kernel that rewrites it in the
   same frame.** Async memsets can execute on a copy/DMA engine; the ordering held in
   practice here, but the replacement — a **tail-pad fill kernel that runs after the
   writer** — is unambiguous, cheaper (pads only the unused tail), and CUDA-graph-friendly.

---

## Which way wins, and when? (updated)

| Scene | Winner | Why |
|---|---|---|
| Small tool, local footprint (surgical default) | **dense + Phase 15** | unchanged — fixed build stages aren't worth it |
| Mixed / moderately large (14k elements) | optimised hash ≈ simple hash ≈ **sorted grid** (tied ~0.34–0.35 ms) | pick by taste: fewest kernels = simple hash; no-caps correctness = sorted grid |
| **Large + dense pairs** (79k elements, heavy overlap) | **sorted grid** (~3×) | home-cell's exact AABB pre-cull starves the FBP kernel of redundant pairs |
| Self-collision / point-cloud | dense (v-t path) | hash/sorted culls are tri-tri only |

All alternative culls remain **opt-in, default-off**; precedence when multiple flags are
set: optimised hash > simple hash > sorted grid.

## How to run

```bash
# 7-leg comparison (all five ways + sorted-grid toggle A/Bs, thermally fair):
scripts/run_mode_comparison_ab_wsl.sh
# Full suite (the hash scene runs dense / hash / simple / sorted legs):
scripts/run_full_benchmark_suite_wsl.sh
# Backend parity incl. the sorted leg (fbp == hash == simple == sorted contacts):
./SofaGpuCollisionDenseGridBackendBench
# Sorted-grid toggle parity across all 4 engine/dedup combos:
scripts/run_sorted_grid_parity_wsl.sh
# Single scene:
SOFA_USE_SORTED_GRID_GENERATION=1 runSofa ... testscenes/hash_prefixsum_large.py
```

New flags: Data fields `useSortedGridGeneration`, `sortedGridUseCubSort`,
`sortedGridUsePairHashDedup`; envs `SOFA_USE_SORTED_GRID_GENERATION`,
`SOFA_SORTED_GRID_CUB_SORT`, `SOFA_SORTED_GRID_PAIRHASH_DEDUP`,
`SOFA_SORTED_GRID_CUDA_GRAPH` (default 1), `SOFA_SORTED_GRID_VERIFY` (default 0),
bench envs `SOFA_BACKEND_BENCH_RUN_SORTED_GRID` / `_SORTED_CUB` / `_SORTED_PAIRHASH`.

## Conclusion

The tiled-binning + shared-memory pattern, enabled by a sort, earns its place as the 5th
way — **tied with the hashes on the mid-size scene and ~3× ahead of everything on the
large one**, with the strongest correctness story of all five (no per-cell caps, no
best-effort drops, exactly-once dedup by construction). The decisive ingredient is
**home-cell emission acting as an exact pre-cull for the narrow phase**, which is only
practical because the sorted layout keeps per-triangle AABBs streaming coalesced. The
counting sort (not CUB) is the right engine here: faster, dependency-free, and immune to
the WSL2 CUB flakiness this work uncovered and fenced off.
