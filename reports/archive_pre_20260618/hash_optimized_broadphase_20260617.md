# Optimized spatial-hash broad cull — verification & 4-mode comparison

**Date:** 2026-06-17
**Hardware:** NVIDIA GeForce GTX 1650 Ti (Turing sm_75), WSL2, SOFA 25.12.
**Build:** `SofaGpuCollision/build-profile/libSofaGpuCollision.so` (clean rebuild, this session).
**Status:** Working, **independently correctness-verified**, and **2.5–3× faster
kernel than the dense grid** on the large-tissue + large-tool regime — with
**bit-identical contacts**.

> This report covers a second round of optimization applied to the spatial-hash +
> prefix-sum broad cull (the `useHashPrefixSumGeneration` path). The narrow phase
> is unchanged, so contacts remain identical to the dense grid. Only the broad
> cull's internals changed. Supersedes the perf numbers in
> [hash_prefixsum_broadphase_experiment_20260609.md](hash_prefixsum_broadphase_experiment_20260609.md)
> (which documents the *pre-optimization* hash design).

---

## 1. The seven optimizations

All live inside `cuda/GpuCollisionBackend.cu` + small profiling plumbing in the
controller. **No new files or directories were added**; CUB is header-only from
the CUDA toolkit (no build-system change).

| # | Optimization | What changed | Why it helps |
|---|---|---|---|
| 1 | **Avoid full pair-hash `cudaMemset` every frame** | The dedup table is cleared by `clearTouchedPairHash{32,64}Kernel`, which resets **only the slots touched last frame** (recorded in `touchedPairHashSlots`). First frame / capacity growth still does one full `cudaMemset` (guarded by `*PairHashKeysInitialized`). | The dedup table is sized to ~2× `maxCandidatePairs` (millions of slots); memsetting it all every frame was pure bandwidth waste when only a few thousand slots are used. |
| 2 | **Compact hash bucket storage** | Two-pass build: `markCompactHashGridCellsKernel` claims a slot per occupied cell (atomicCAS), then `compactHashGridSlotsKernel` assigns each occupied slot a dense **bucket index** `0..occupiedBucketCount`. Per-bucket arrays (`tissueCount`, `toolCount`, `tissueIds`, `toolIds`) are indexed by that compact id, not by the full `tableSize`. | Storage and per-bucket work scale with **occupied** cells, not the whole table. |
| 3 | **Generate only mixed occupied cells** | `fillCompactHashGridTrianglesKernel` appends a bucket to `mixedBucketIds` the moment it holds **both** tissue and tool. Pair counting and generation iterate only that list. | A bucket with only tissue (or only tool) yields zero candidate pairs — skipping them removes empty-bucket overhead. (The hash analogue of the Phase-15 tool-active-cell trick.) |
| 4 | **Remove per-pair binary search** | `generateMixedHashCandidatePairs{32,64}Kernel` launches **one block per mixed bucket**; the block's threads split that bucket's `t×u` local pairs via plain `localPair/u`, `localPair%u`. The old design launched one thread per *global* raw-pair index and binary-searched `pairOffsets` to recover `(bucket, localPair)`. | Eliminates a `log(buckets)` binary search per candidate pair and gives clean per-block locality. |
| 5 | **Persistent CUB scan replaces `thrust::exclusive_scan`** | `cub::DeviceScan::ExclusiveSum` with temp storage held in the workspace (`scanTempStorage`, grown once). | No per-frame Thrust temp-allocation; the scan reuses one persistent buffer. |
| 6 | **Compact pair encoding (32-bit)** | When both triangle counts ≤ 65535, candidate pairs and the dedup table use **`uint32`** (`encodeCompactCandidatePair`, `generate…32Kernel`) instead of `uint64`. Gated by `useCompactCandidatePairs`; falls back to 64-bit otherwise. | Halves the candidate-array and dedup-table bandwidth in the common case. |
| 7 | **Hash-stage profiling** | 8 new CUDA-event timers (`hashGrid{Reset,PairHashClear,InsertTissue,InsertTool,PairCount,Scan,GeneratePairs,ProximityCounterClear}Milliseconds`) flow `BackendExecutionStats → StageSnapshot → controller`, emitted as `avg_narrow_hash_*_ms` summary keys. | Per-stage attribution of the hash broad cull (used to confirm no single stage dominates). |

### Resulting kernel sequence (13 launches vs the old 8, vs dense 7)
> **Update (2026-06-17b):** the **CUB scan** + `setCompactHashRawTotal` shown below
> were later removed (their offsets are unused by the block-per-bucket generator),
> dropping this to **11 launches**. See §5.

`clear-touched-pair-hash` → `resetCompactHashGrid` → `markCompactHashGridCells`
(tissue) → `markCompactHashGridCells` (tool) → `compactHashGridSlots` →
`fillCompactHashGridTriangles` (tissue) → `fillCompactHashGridTriangles` (tool) →
`computeCompactHashPairsPerBucket` → **CUB scan** → `setCompactHashRawTotal` →
`generateMixedHashCandidatePairs{32,64}` → `resetProximityCounters` →
`featureBasedProximityKernel`. **More, smaller kernels** — but each does far less
work, and the per-frame full memset + Thrust alloc + binary search are gone.

---

## 2. Correctness — verified four independent ways

The invariant: the optimized hash cull must feed the **same** candidate set to the
**same** FBP narrow kernel, so contacts must be **bit-identical** to the dense
grid. Confirmed:

| Evidence | Result |
|---|---|
| **Standalone backend bench** (no SOFA, this session's clean rebuild) | `hash_contacts = fbp_contacts = 8018`; `hash_unique_pairs = 322,560 = dense unique_candidates`; bucket-overflow 0, probe-overflow 0 |
| **Independent 3-mode SOFA A/B** (my rebuild, large+large scene) | plain dense, Phase-15 dense, and optimized hash all emit **2354 (1119/428/807)**, overflow 0 |
| **Full suite** (`full_suite_hashopt`, all 5 scenes) | small **56** EE, large **8018** (5397/880/1741), v-t self **2700**, v-t cross **254**, hash **2354** — every count matches the documented baseline; overflow 0 everywhere |
| **Code review** of the two riskiest changes | *Touched-clear*: every claimed dedup slot is recorded and cleared next frame, with a first-frame full-init guard → no stale entries. *No-binary-search*: block-per-bucket + divide/modulo indexing is exact. |

**Conclusion: correctness preserved.** The optimization is purely a speed change.

---

## 3. Performance — the 4-mode comparison

Same-session, back-to-back, thermally fair, on the large-tissue + large-tool scene
(`testscenes/hash_prefixsum_large.py`, 12,800 + 1,568 tris = 14,368 elements),
counter readback on. **Trust the kernel time** (GPU-event measured, low noise);
FPS is laptop-thermal-noisy.

| Mode | narrow **kernel** (ms) | narrow wall (ms) | FPS | launches | contacts |
|---|---:|---:|---:|---:|---|
| Plain dense grid (Phase 15 **off**) | 2.126 | 2.866 | 263.8 | 7 | 2354 |
| Optimised dense grid (Phase 15 **on**) | 1.787 | 2.445 | 294.7 | 7 | 2354 |
| **Earlier** hash path (2026-06-09, prior build)¹ | ~1.27 | ~1.70 | ~405 | 8 | 2354 |
| **Optimised hash path (this work)** | **0.700** | **1.296** | **408.1** | 13 | 2354 |

¹ Earlier-hash is from a *different build/session* (cool GPU); use its **kernel
time** (~1.27 ms) as the cross-session reference, not its FPS.

**Kernel-time speedups of the optimized hash:**
- vs **plain dense**: 2.126 → 0.700 ms = **3.0× faster**
- vs **optimised dense** (Phase 15): 1.787 → 0.700 ms = **2.6× faster**
- vs **earlier hash**: ~1.27 → 0.700 ms = **~1.8× faster**

A second, separate same-session A/B (the user's `hash_optimized_ab3` run) reproduced
the ranking even more strongly on a cooler GPU: dense **1.748 ms / 238 FPS** →
optimized hash **0.508 ms / 565 FPS** — **3.4× faster kernel, +137 % FPS**, contacts
identical.

### Where the time goes now (detailed per-stage, warm)
`generate 0.167` · `scan 0.128` · `insert-tool 0.126` · `pair-hash-clear 0.101` ·
`pair-count 0.096` · `insert-tissue 0.094` · `counter-clear 0.054` · `reset 0.047`
(ms). No single stage dominates — the old bottlenecks (full memset, Thrust scan,
binary-search generate) are gone and the work is now balanced across small stages.

---

## 4. When to use which (updated guidance)

| Scene regime | Best mode | Why |
|---|---|---|
| Small tool, local footprint (surgical default) | **dense + Phase 15** | culls to ~30 cells; the hash build's fixed stages aren't worth it |
| **Large tissue and/or large tool** | **optimised hash** (`useHashPrefixSumGeneration=True`) | 2.5–3× faster kernel than dense, identical contacts |
| Self-collision / point-cloud-vs-mesh (v-t) | **dense (v-t path)** | the hash cull is wired only for the tri-tri FBP path |

Still **opt-in, default-off**: with the flag off, the runtime is byte-identical to
the dense path, so these changes cost the default surgical scene nothing.

---

## 5. Follow-ups

- **Drop the scan on the critical path — DONE (2026-06-17b).** The CUB
  `ExclusiveSum` + `setCompactHashRawTotalKernel` (2 launches, ~0.13 ms) have been
  removed: the block-per-bucket generator (`kGenBlocks` fixed at 1024) needs no
  global offsets, and the raw-pair-count statistic is now folded into
  `computeCompactHashPairsPerBucketKernel` via a single `atomicAdd` into
  `rawCandidateCount`. The cleanup also removed the now-inert `pairOffsets` /
  `rawTotal` / `scanTempStorage` workspace buffers and the (now-unused)
  `<cub/cub.cuh>` include — **no CUB dependency remains**. **Launches 13 → 11**;
  contacts bit-identical (2354 / hash = fbp = 8018, overflow 0), re-verified by the
  3-mode A/B (`hash_opt` 0.349 ms / 514 FPS, launches 11) and the standalone bench.
  Documented in
  [hash_micro_optimizations_20260617.md](hash_micro_optimizations_20260617.md) §1
  (#3) and `guide/plan.md` §5.20.

### Still open

- **Route the v-t paths through the optimized hash cull** (currently tri-tri FBP only).
- **Auto-tune the hash table size** from a measured occupancy histogram instead of
  the fixed `4×triangles` heuristic.

---

## 6. Reproduce

```bash
# Standalone backend parity (no SOFA): prints hash_contacts == fbp_contacts
SOFA_BACKEND_BENCH_RUN_HASH=1 ./SofaGpuCollisionDenseGridBackendBench

# 3-mode same-session A/B (plain dense | Phase-15 dense | optimised hash):
scripts/run_mode_comparison_ab_wsl.sh        # large tissue + large tool

# Full suite + hash A/B with the current build:
scripts/run_full_benchmark_suite_wsl.sh
```

Per-stage hash timers appear in each summary as `avg_narrow_hash_*_ms` (define
in [README_metrics_explained.md](README_metrics_explained.md)). Artifacts for this
report: `output/benchmark_logs/hash_optimized_ab3_20260617_094259`,
`…/hash_optimized_detailed_20260617_094403`,
`…/full_suite_hashopt_20260617_094508`, and the independent re-run
`…/hashopt_modecmp_*`.
