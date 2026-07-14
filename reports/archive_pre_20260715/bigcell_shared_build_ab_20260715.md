# Big-Cell Shared-Memory Build A/B — Hash Table vs Sorted List (2026-07-15)

The second half of the two-level-grid brainstorm, implemented and measured: instead of
every entry paying its own global atomic during the way-6 table build ("count" and "fill"
kernels), each block takes a **fixed chunk of triangles, populates its contribution in
shared memory first, and merges to global later** — with the shared organization kept as
an explicit A/B:

- **mode 1 — shared HASH TABLE**: insert-or-count keyed by (bigCell, side) bin; the merge
  does ONE global atomic per distinct bin the block touched instead of one per entry;
  fill stages entries with their within-bin offsets and scatters into ranges reserved
  per-bin.
- **mode 2 — shared SORTED LIST**: stage the chunk's (bin, entry) pairs, **bitonic-sort**
  them by bin entirely in shared memory, then one global reservation per run and a
  run-writer scatter.

Baseline (mode 0) is the existing direct-global-atomics CSR build. Anything overflowing
the shared staging falls back to the direct path (`sharedSpillCount`) — the entry multiset
is identical in every mode, so **contacts are bit-identical by construction and by test**.
Flag: `bigCellSharedBuild` / env `SOFA_BIGCELL_SHARED_BUILD` (bench:
`SOFA_BACKEND_BENCH_BIGCELL_SHARED_BUILD`). Canonical 6-way baseline:
[performance_six_ways_20260713.md](performance_six_ways_20260713.md).

## 1. Verdict

**The shared hash table wins everywhere; the shared sorted list loses everywhere — the
exact mirror image of the global-memory build A/B.**

| | global memory build (2026-07-12 A/B) | shared memory build (this A/B) |
|---|---|---|
| sorted/CSR organization | **0.71 ms — winner** | 1.54 ms — loser (+158%) |
| hash organization | 2.02 ms — loser (2.9×) | **0.525 ms — winner (−12%)** |

Why the flip: in **global** memory, a hash table pays long-latency probe round trips and
scattered reads, while sorting buys coalesced layout — sorting wins. In **shared** memory,
probes cost ~nothing (single-cycle-ish, no DRAM), while a bitonic sort of 2048 staged
entries burns ~66 compare-exchange passes of raw compute per block — hashing wins. Same
two data structures, opposite winners, decided by which memory level they live in.

## 2. Correctness gate — PASSED

8-leg parity matrix (bench 80k): every leg 8018 contacts (vf 2828 / fv 3469 / ee 1721),
`pairs_tested` = 43,584 = way-5's `unique_pairs`, `entry_overflow` = 0,
**`shared_spill` = 0** (everything fit in the shared staging on both scenes) — including
the new `sharedhash_f2` and `sharedsort_f2` legs. 200k: 17,040 contacts, 89,856 pairs,
zero everything, all three modes. SOFA 10-leg comparison: 2354 (1119/428/807) on every leg.

## 3. Results

### Standalone bench, way-6 full-pipeline GPU kernel avg (way-5 same-session reference)

| Build mode | 80k | 200k |
|---|---:|---:|
| mode 0 — direct global atomics | 0.598 ms | 1.084 ms |
| **mode 1 — shared hash (new DEFAULT)** | **0.525 ms (−12%)** | **0.807 ms (−26%)** |
| mode 2 — shared sorted list | 1.543 ms (+158%) | 3.339 ms (+208%) |
| way 5 sorted grid (reference) | 0.677 ms | 1.087 ms |

With the shared-hash build, **way 6 now beats way 5 by 22% at 80k and 26% at 200k** — the
first configuration to decisively lead at the large size (previous state was a tie).

### SOFA 14,368-element scene — 10 legs, same session (160 steps)

| Leg | narrow kernel ms |
|---|---:|
| **bigcell_sharedhash** | **0.290** ← new overall record |
| bigcell_fused (mode 0) | 0.299 |
| simple_hash | 0.333 |
| hash_opt | 0.351 |
| sorted_grid | 0.353 |
| sorted_cub / sorted_pairhash | 0.473 / 0.515 |
| bigcell_sharedsort | 0.752 |
| dense_phase15 / dense_plain | 1.329 / 1.617 |

## 4. Per-kernel Nsight Compute (80k, graphs off, per-launch means — read relatively)

| Kernel | mode 0 | mode 1 (hash) | mode 2 (sort) |
|---|---:|---:|---:|
| count | 57 µs @ 7.5% SM | **36 µs @ 21.6% SM** | 283 µs @ **77.4% SM** |
| fill | 48 µs @ 9.7% SM | 50 µs @ 26.5% SM | 527 µs @ 52.3% SM |
| fused consumer | 524 µs | **467 µs** | 489 µs |

Three readings:

1. **Count is where the shared hash pays off directly**: 37% faster, and SM utilization
   triples (7.5 → 21.6%) — the kernel stops waiting on per-entry global atomics and does
   real work. Fill is duration-neutral (the staging bookkeeping offsets the atomic
   savings) but its global-atomic traffic drops the same way.
2. **The unexpected second-order win: the fused consumer kernel itself speeds up
   524 → 467 µs** under the privatized builds. Cause: entries written per-block land
   contiguously within each bin's run, and a block's chunk is index-contiguous triangles —
   so the consumer's tissue sweep touches consecutive triangle ids back-to-back, improving
   the vertex-gather locality. Build layout quality is consumer performance.
3. **The bitonic sort is pure compute cost**: 77% SM on count and half a millisecond on
   fill — sorting 2048 staged entries per block (~66 network passes) costs far more than
   *all* the global atomics it avoids. Confirms (again, from the other side) the Phase-13
   finding: atomics were never expensive enough here to justify heavy machinery to avoid
   them; the shared hash wins because its own overhead is nearly free, not because the
   atomics were crushing.

Occupancy footnote: the fused kernel's occupancy *dropped* (39.7 → 30.3%) in the fastest
configuration — one more data point that occupancy is not the metric to chase on this
kernel family.

## 5. Defaults changed

`bigCellSharedBuild` **defaults to 1 (shared hash)** in the header, the SOFA component,
both scenes, and the bench — measured better at every size and scene, zero spill, safe
fallback path for pathological chunks. Modes 0 and 2 remain selectable for A/B.

## 6. Where this leaves the scoreboard (narrow kernel, contacts identical)

| Scene | way 6 (shared-hash build) | way 5 | best hash way | dense default |
|---|---:|---:|---:|---:|
| 14k SOFA | **0.290 ms** | 0.353 | 0.333 | 1.329 |
| 80k bench | **0.525 ms** | 0.677 | ~2.0 | 2.58 |
| 200k bench | **0.807 ms** | 1.087 | ~4.7 | 4.50 |

## 7. Artifacts

`output/benchmark_logs/{bench_sharedbuild_20260715, mode_comparison_20260714_185706,
profiling_sharedbuild_20260714_185817}` (WSL + mirrored to Windows). Parity:
`scripts/run_bigcell_parity_wsl.sh` (now 8 legs). Follow-ups unchanged from the canonical
report (fused-kernel register pressure; heavy-big-cell work splitting), plus one new
question worth a cheap test someday: whether sorting the *global* CSR runs by triangle id
(cheap post-pass) buys the same consumer-locality bonus for mode 0.
