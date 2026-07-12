# Way 6 — Big-Cell / Small-Cell Fused Generation + Narrow Phase (2026-07-12)

The sixth broad-cull way, built from the user's two-level-grid brainstorm: group the
existing dense-grid cells ("small cells") into **big cells**, build a per-big-cell table of
which triangle touches which small cell, then launch **one block per mixed big cell** that
pulls its table *and the tool triangles' vertices* into shared memory and runs the
narrow-phase closest-feature math **inline in the same kernel** — no intermediate candidate
pair list, no separate FBP launch, one contact list out.

Companion docs: design plan in `PLAN_WAY6.md` (temporary, deleted after this report),
idea history in [IDEAS.md](../IDEAS.md) §3, 5-way baseline in
[performance_five_ways_20260703.md](performance_five_ways_20260703.md).

---

## 1. What was built

### Pipeline (CSR build, the default)

| # | Kernel | Notes |
|---|---|---|
| 1 | `resetSortedGridKernel` | reused from way 5 (bins = bigCell × side) |
| 2,3 | `countBigCellEntriesKernel` ×2 | fused per-triangle AABB store + per-(bigCell, side) histogram |
| 4 | `exclusiveScanSortedGridBinsKernel` | reused from way 5 — input is factor³ smaller, so the single-block scan is cheap here (way 6 does NOT need the multi-block-scan work) |
| 5,6 | `fillBigCellEntriesKernel` ×2 | exact CSR scatter of packed entries `(triId << 6) \| localSmallCellId` |
| 7 | `buildSortedGridMixedCellsKernel` | reused from way 5 |
| 8 | `resetProximityCountersKernel` | reused |
| 9 | `fusedBigCellNarrowKernel` | **the centerpiece** (below) |

### The fused kernel

One block per mixed big cell (grid-stride over the active list), 256 threads:

1. Stage a tile (≤ `toolTileCapacity`, default 256) of the big cell's **tool** entries into
   shared memory; oversized big cells loop over chunks (each tool entry lives in exactly
   one chunk, so no pair is visited twice).
2. **In-shared 64-bin counting sort** by local small-cell id → per-small-cell runs directly
   indexable by local id (the "per-big-cell table in shared memory", organized for O(1) lookup).
3. Gather tool AABBs **and all three vertices** (9 floats/tri) from global **once per big
   cell** instead of once per pair. Shared budget ≈ 17.9 KB of 48 KB.
4. Sweep the big cell's tissue entries from global; for each same-small-cell (tissue, tool)
   pair: inflated-AABB overlap test → **home-cell ownership at small-cell granularity**
   (identical exactly-once rule as way 5 ⇒ provably the same surviving pair set) → the
   **identical FBP VF/FV/EE math** → atomic append to the single global contact list.

The FBP per-pair math was extracted verbatim into shared device helpers
(`fbpComputeClosestFeatureContact` / `fbpEmitContact` in `FbpKernels.cuh`) so ways 1–5 and
way 6 execute the same instructions — verified by contact parity below.

### The build-strategy A/B (user decision: "both, as a toggle")

`bigCellUseHashBuild=1` (env `SOFA_BIGCELL_HASH_BUILD`) replaces count/scan/fill with a
**literal per-big-cell open-addressing hash multi-map**: each (big cell, side) owns a fixed
slot region (`bigCellHashSlots`, default 1024, pow2, ≤4096); entries probe linearly from
`hash(localSmallCellId)` claiming slots with atomicCAS; a fill kernel clears all slots each
frame. The fused kernel reads either source into the identical shared layout, so **only the
build differs**.

### Flags / envs

`useBigCellFusedGeneration` (`SOFA_USE_BIGCELL_FUSED_GENERATION`), `bigCellFactor`
(`SOFA_BIGCELL_FACTOR`, default 2 — see §4), `bigCellToolTile` (`SOFA_BIGCELL_TOOL_TILE`),
`bigCellUseHashBuild`, `bigCellHashSlots`, `SOFA_BIGCELL_CUDA_GRAPH` (graph replay default
on, same `CudaGraphReplayer` pattern). Dispatch precedence: bigcell > hash > simple >
sorted > dense. Bench leg: `SOFA_BACKEND_BENCH_RUN_BIGCELL` (+ `_BIGCELL_FACTOR`, `_BIGCELL_TILE`,
`_BIGCELL_HASH_BUILD`, `_BIGCELL_HASH_SLOTS`). Parity script:
`scripts/run_bigcell_parity_wsl.sh`.

---

## 2. Correctness gates — ALL PASSED

Standalone bench, 80k geometry (64,800 tissue + 14,720 blade):

| Combo | Contacts | Pairs tested | Overflows |
|---|---|---|---|
| FBP / hash / simple / sorted baselines | 8018 (vf 2828 / fv 3469 / ee 1721) | 322,560 (43,584 sorted home-cell) | 0 |
| way 6 CSR, factor 4, tile 256 | **8018 (identical classes)** | **43,584** | 0 |
| way 6 CSR, factor 4, tile 32 (forced chunk loop) | 8018 | 43,584 | 0 |
| way 6 CSR, factor 2 | 8018 | 43,584 | 0 |
| way 6 CSR, factor 1 | 8018 | 43,584 | 0 |
| way 6 hash build, 1024 slots | 8018 | 43,584 | build_overflow=256 (see note) |
| way 6 hash build, **2048 slots** | 8018 | 43,584 | **0** |

`pairs_tested` equals way 5's `unique_pairs` exactly — the home-cell rule reproduced at the
same small-cell granularity. 200k bench (198,450 + 14,720): all six ways emit
**17,040 contacts (vf 8805 / fv 4914 / ee 3321)**, way 6 pairs = way 5 pairs = 89,856, zero
overflows. SOFA end-to-end (14,368-tri scene): all 8 legs report **2354 contacts
(1119/428/807)**.

Note: at 1024 slots the hash build dropped 256 entries yet contacts still matched — luck
(the dropped entries owned no surviving pair), not a guarantee. The hash build is
best-effort like way 4; `buildOverflowCount` reports drops honestly. 2048 slots is the
clean setting for this geometry.

## 3. Performance

### SOFA end-to-end, 14,368-tri scene (160 steps, same session, counter readback on)

| Leg | avg narrow kernel ms | fps |
|---|---:|---:|
| **bigcell_fused (way 6, factor 2)** | **0.309** | 829 |
| simple_hash | 0.332 | 839 |
| sorted_grid (way 5) | 0.337 | 856 |
| hash_opt | 0.339 | 793 |
| sorted_cub | 0.473 | 707 |
| sorted_pairhash | 0.511 | 698 |
| dense_phase15 | 1.304 | 434 |
| dense_plain | 1.541 | 405 |

**Way 6 posts the fastest narrow-kernel time of all eight legs** (−8% vs way 5, −7% vs
simple hash). fps differences at these speeds are dominated by CPU-side scene overhead;
kernel time is the robust metric (script's standing note).

### Standalone bench, GPU kernel avg (steps 30)

| Size | way 5 sorted | way 6 f1 | way 6 f2 | way 6 f4 | dense/hash ways |
|---|---:|---:|---:|---:|---:|
| 80k (79,520 el) | 0.651–0.893* | 0.789 | **0.641** | 0.713 | 2.0–3.4 |
| 200k (213,170 el) | 1.082–1.090 | 1.070 | **1.093** | 1.520 | 4.69–5.03 |

\* way-5 80k numbers vary with co-run legs; canonical 2026-07-03 value 0.651 ms.
At 200k, way 5 vs way 6 (f1/f2) is a statistical tie; factor 4 loses 40%.

### Full 15-leg suite (160 steps/leg, shared run dir `full_suite_way6_20260712`)

Every historical fingerprint reproduced (small 56, large 8018, vt_self 2700, vt_cross 254,
hash scene 2354 across all five ways there), zero overflow everywhere. New legs:

| Leg (scene) | narrow kernel ms | fps | contacts |
|---|---:|---:|---:|
| hash_bigcell (14,368) | **0.299** | 707 | 2354 |
| xlarge_bigcell (213k SOFA) | **2.011** | 76.5 | 12,178 |
| xlarge_sorted (213k SOFA) | 2.247 | 75.4 | 12,178 |
| xlarge_dense (213k SOFA) | 4.125 | 66.6 | 12,178 |

On the SOFA 200k scene (finer 128×8×128 grid → plenty of mixed big cells) **way 6 beats
way 5 by ~10%** — the tie at the bench's coarser grid becomes a win when the fused kernel
has enough blocks to fill the GPU. (The scene's 12,178 contacts differ from the bench's
17,040 because grid config and contact distance differ; the gate is identity *across ways
on the same scene*, which holds.)

### Build-strategy A/B (the user's Q1, answered with numbers)

Full-pipeline GPU kernel time, 80k bench, factor 4: **CSR 0.713 ms vs literal hash-table
build 2.02–2.17 ms (~2.9× slower)**. The hash build pays: (a) clearing every slot region
each frame (9–19 MB of writes), (b) atomicCAS probe chains on insert, (c) sweeping sparse
slot regions (mostly empty slots) in the fused kernel instead of dense CSR runs. This is
the measured confirmation of the brainstorm-phase prediction — and of the batch-1 paper
review's conclusion (scattered hash layouts survive into every consumer of the table).

### Why factor 4 loses: parallelism collapse

At factor 4 the 200k scene has only **48 mixed big cells** → 48 blocks for the whole fused
phase on a 16-SM GPU, with the blade region concentrated in few, heavy blocks (and multiple
tool chunks each re-sweeping the tissue entries). Factor 2 gives 88 blocks, factor 1 gives
168. Bigger big cells amortize staging better per block but starve the GPU; **factor 2 is
the measured sweet spot** (best at 80k, tied-best at 200k) and is now the default
everywhere. This confirms the load-imbalance risk called out in the design plan (§6).

## 4. Defaults set from measurement

- `bigCellFactor` default **2** (was 4 in the initial design).
- CSR build default; hash build kept as A/B toggle.
- `toolTileCapacity` 256 (tile 32 costs ~17% at 80k — chunk loop verified correct but not free).

## 5. New assets

- `cuda/detail/BigCellGrid.cuh` (module 8 of the single-TU umbrella, included last)
- `computeBigCellFusedProximityContacts` API + `BigCellConfig`/`BigCellStats`
- Shared FBP helpers `fbpComputeClosestFeatureContact`/`fbpEmitContact` (extracted, zero
  behavior change to ways 1–5 — parity re-verified this session)
- `testscenes/collision_xlarge_200k.py` (~213k elements, same construction as the other
  scenes, all six way toggles)
- Scripts: `run_bigcell_parity_wsl.sh`, `run_ncu_bigcell_wsl.sh`; mode comparison extended
  to 8 legs; full suite extended with `hash_bigcell` + 3 × `xlarge_*` legs
- 200k bench size via `SOFA_LARGE_TISSUE_NX=316` (no code change needed)

## 6. ncu per-kernel profile (way 6)

Captured with graphs off, same metric set as the 5-way canonical profile
(`profiling_bigcell_20260712_031825/` on the WSL side). ncu locks clocks and replays
kernels, so absolute durations are inflated vs the wall benchmarks — read them relatively.

Per-launch averages, 200k geometry:

| Kernel | f2 dur | f2 SM% | f2 occ% | f4 dur | regs |
|---|---:|---:|---:|---:|---:|
| `fusedBigCellNarrowKernel` | 717 µs | 18.7 | 43.5 | 828 µs | **122** |
| `countBigCellEntriesKernel` (per launch avg) | 190 µs | 7.7 | 78.9 | 316 µs | 34 |
| `fillBigCellEntriesKernel` (per launch avg) | 157 µs | 8.8 | 84.6 | 282 µs | 36 |
| `exclusiveScanSortedGridBinsKernel` | **52 µs** | 3.8 | 100 | 10 µs | 16 |
| reset/mixed/counters | 2–3 µs each | — | — | — | 16 |

Readings:

- **The scan-shrink works as designed**: way 6's scan input (2 × bigCellCount bins) is
  factor³ smaller than way 5's, so the single-block scan costs 52 µs at factor 2 (10 µs at
  factor 4) vs way 5's measured 413 µs — way 6 sidesteps the multi-block-scan lever
  entirely.
- **The fused kernel carries 122 registers** (FBP alone was 79; staging + sweep logic adds
  the rest), capping occupancy at ~43% — yet it still beats the separate
  generation-then-FBP pipeline on the SOFA scene because it never writes/reads a pair
  list, never launches a second kernel, and reads each tool triangle's vertices once per
  big cell. SM 18.7% / DRAM 1.5% says latency-bound but far healthier than the hash ways'
  1.5% SM generation.
- **Factor 4's collapse is visible everywhere**: fused 717 → 828 µs (fewer, heavier
  blocks), count/fill nearly double (more same-bin atomic contention per big cell), and
  occupancy drops (39.7%).
- Register pressure is the top follow-up lever: `__launch_bounds__` tuning was a measured
  regression on the *load-bound* standalone FBP kernel, but the fused kernel has a
  different profile (fewer in-flight loads, more arithmetic per byte), so a re-test on
  this shape is justified despite the old dead-end note.

## 7. Verdict

The user's big-cell idea, bent to CSR + shared staging + fusion, produced a way that
**wins outright on the surgical-scale SOFA scene** (0.309 ms narrow kernel, fastest of all
eight configurations) and **ties way 5 at 200k**. The two-level grouping itself is only
worth a small factor (2); the real wins are the **fusion** (no pair list, no separate FBP
launch) and the **shared-memory staging of tool vertices** (fetched once per big cell, not
once per pair). The literal hash-table build — measured at ~2.9× the CSR build — closes the
brainstorm question with numbers.

Follow-ups worth considering (not started): grid-stride work-splitting inside heavy big
cells (attacks the factor-4 collapse directly), and porting the fused-generation shape to
the vertex-triangle path.
