# Ideas Log — Broad/Narrow Phase Optimization Brainstorms

Running record of every optimization idea we have discussed — the good, the bad, and the
already-buried — so nothing gets re-litigated or re-invented from scratch. Measured numbers
come from the canonical report ([reports/performance_five_ways_20260703.md](reports/performance_five_ways_20260703.md));
architecture background lives in [guide/architecture.md](guide/architecture.md).

Last updated: 2026-07-12.

**Status legend**

| Tag | Meaning |
|---|---|
| `DONE` | Implemented, verified, measured |
| `QUEUED` | Agreed next step, not started |
| `PROMISING` | Designed on paper, worth prototyping |
| `FUTURE` | Only pays under conditions we don't have yet; revisit trigger noted |
| `NOT WORTH IT` | Analyzed, the math says no at current scale |
| `DEAD END` | Measured or proven bad — do not retry |
| `SETTLED` | A question/misconception we resolved with evidence |

---

## 1. Where we stand (context for everything below)

Five broad-cull ways produce bit-identical contacts and feed the same FBP narrow kernel.
On the 79,520-element bench the sorted grid (way 5) wins at ~0.651 ms vs 2.0–2.6 ms for the
others, mostly because its home-cell dedup doubles as an exact AABB pre-cull
(322,560 → 43,584 emitted pairs, −86%; FBP 200 → 85 µs). Its remaining #1 cost is the
single-block scan: 413 µs at 3.8% SOL. Hash-way generation is 1,237 µs at 1.5% SM —
latency-bound on atomicCAS dedup chains, not on raw atomic throughput.

---

## 2. QUEUED — Multi-block scan / multi-block counting sort

**What:** Replace the single-block Hillis-Steele scan (one 1024-thread block crawling the
whole histogram, 413 µs at 3.8% SOL) with the standard multi-block scheme: per-block
shared-memory histograms → merge counts → device-wide scan → scatter. Expected ~413 → ~50 µs,
bringing the sorted way to roughly ~0.3 ms.

**Origin — independently re-invented in brainstorm:** the idea "launch a block per big cell,
give each block a *fixed chunk* of triangles (not spatially grouped — impossible before the
structure exists), build something small in shared memory, then merge into global" is exactly
this algorithm. Key design insight from that discussion:

> The merge step decides everything. Mini **hash tables** merge badly (variable-length
> scattered per-cell lists from every block, recombined with more global atomics — the shared
> detour buys nothing). Mini **histograms** merge perfectly (counts just add). When designing
> any "build locally, combine globally" scheme, ask *how do the partial results combine?* first.

Bonus available in the same rework: per-block shared-histogram aggregation in expansion —
one global atomicAdd per (block, non-empty cell) instead of per incidence. Shared atomics
are fast on Turing; expansion is only 111 µs ×2 today, so this is a side benefit, not a goal.

## 3. PROMISING — Way 5b: big-cell tiled generation + FBP fusion

**Origin (user idea):** two-level grid — small cells grouped into big cells; build a
per-big-cell table of triangle→small-cell; launch one block per big cell; pull the table
into shared memory.

**Bent design (what we'd actually build):**

- No per-big-cell hash table. Reuse way 5's sort with a remapped key
  `(bigCellId << k) | localSmallCellId` — the existing sort then delivers each big cell's
  triangles as one **contiguous slice** for free. A contiguous slice beats a shared-memory
  hash table because we only ever *iterate* a bucket's contents, never query it by key —
  and staging a slice is one coalesced sweep, while a hash table is scattered and needs
  probing even once it sits in shared memory.
- One block per big cell (e.g. 4×4×4 small cells). Stage the **tool side only** in shared:
  ids (~1 KB) + AABBs (~6 KB) + vertex coordinates (~9 KB) + 64 local run offsets
  ≈ **16.5 KB at 256 tool triangles** — fits in the GTX 1650 Ti's 48 KB. Staging both sides
  (~72 KB) does not fit; the tissue side streams from global/L2, which is fine because
  consecutive pair-tests reuse the same tissue triangle back-to-back.
- Fuse candidate generation and the FBP narrow test in the same kernel, so staged tool
  vertices are fetched from global once per big cell instead of once per pair. This is the
  direct attack on FBP's measured LSU/long-scoreboard load-boundedness (79 regs, occupancy
  ceiling measured — see dead ends).
- Oversized big cells (tool tris > tile capacity) need a fallback path (loop tiles or defer
  to plain way 5). Home-cell dedup is order-independent, so bit-identical contacts are
  preserved.
- Ship behind a flag (`useSortedGridBigCellFusion`), A/B'd bit-identically against way 5.

**External validation:** Alcantara et al. (SIGGRAPH Asia 2009) converged on the same
two-tier shape — partition into buckets globally, then process per-bucket in shared memory,
one block per bucket. Their Tier 1 = our sort; their Tier 2 = our big-cell block. We skip
their per-bucket cuckoo table because our access pattern is iterate-only.

**Sequencing:** after the multi-block scan (§2), since 5b builds on the same sort.

## 4. SETTLED — External paper reviews

### Batch 1 (2026-07-12): five GPU-hashing references

Verdict: **no technique beats our measured bottlenecks — way 5 already won by removing hash
tables from the hot path; these papers optimize the thing we optimized away.** Two still matter.

| Paper | What it offers | Verdict for us |
|---|---|---|
| NVIDIA blog: massively parallel hash maps (cuCollections) | Open addressing + linear probing; **cooperative probing** (group of 4 threads loads 4 slots in one coalesced read): +13% insert / +40% find, *at high load factor* | Wrong bottleneck today: our tables are deliberately oversized (short chains), and hash-gen pain is one mandatory CAS round trip per pair, not chain length. Becomes the right toolbox if tables are ever shrunk (memory pressure) or made persistent (§5) |
| Alcantara et al., *Real-Time Parallel Hashing on the GPU* (poster + SIGGRAPH Asia 2009) | Two-tier: bucket ≤512 items globally, then per-bucket **cuckoo hash built in shared memory**; lookups ≤3 probes | Build ≈ radix-sort cost *by their own numbers* — we already pay the sort, cuckoo would be additive. Their win is retrieval vs binary search; our scanned histogram already gives O(1) run starts, so no gap to close. Insert evictions can cascade into full rebuilds → frame spikes. **But their architecture independently validates way 5b (§3)** |
| Wolfson & Rigoutsos, *Geometric Hashing: An Overview* (1997) | Vision technique: transformation-invariant feature indexing + voting for object recognition | Different problem domain (unknown transformations; we know exact coordinates). Nothing to import |
| Patel et al., *GPU based geometric hashing for space partitioning* (IEEE) | Same vision technique GPU-ified for SURF/palm-print recognition; 14–61× speedups | Speedups are vs a *sequential CPU* baseline, not vs any GPU method. Not our domain |

Their motivation section is worth keeping: hashing parallelizes poorly because of
(a) dependent sequential operations, (b) variable probe counts forcing worst-case waits,
(c) scattered storage defeating caches — the literal anatomy of our measured
1,237 µs @ 1.5% SM hash generation.

### Batch 2 (2026-07-12): eight hashing / collision-detection papers

Verdict: **much closer to home than batch 1 — three genuine takeaways, still no change to
the queued plan.** (1) I-Cloth proves persistent coarse candidate structures are a real,
published technique — at 1M-triangle scale, on BVHs; (2) PSCC's normal-cone culling is the
one genuinely *new* technique → §7; (3) voxel hashing + SlabHash + ASH form the proven
engineering playbook for the §5 incremental idea.

| Paper | What it offers | Verdict for us |
|---|---|---|
| Nießner et al., *Real-time 3D reconstruction at scale using voxel hashing* (TOG 2013) | THE persistent GPU spatial hash: sparse voxel blocks in a bucketed hash, per-frame localized inserts where the sensor sees, garbage-collection pass, GPU↔host streaming | The blueprint for §5. "Update only where the camera looks" maps to "update only where the tool deforms." Proves persistent spatial hashes run at real-time rates. No change to today's plan — same three traps (§5) apply |
| Tang et al., *I-Cloth: Incremental collision handling* (SIGGRAPH Asia 2018) | Persists the **BVTT front** (the frontier where BVH-vs-BVH traversal stopped) across frames and *repairs* it — descend where boxes newly overlap, retract where they stopped; incremental impact zones. 7–10× vs prior GPU cloth on ~1M-triangle self-colliding meshes | Literature proof that the *coarse* candidate structure can persist across frames (nuances our §8 analysis — concept is sound). Needs a hierarchy to have a "front"; grids re-derive instead. At our scale generation is 35.5 µs, so the repair machinery cannot pay. Revisit if scenes grow ~10× or the legacy BVH path is ever revived |
| *PSCC: Parallel Self-Collision Culling with Spatial Hashing on GPUs* (PACMCGIT/i3D 2018) | **Normal-cone culling**: a smooth surface patch whose triangle normals fit in a cone (plus a planarity test) provably cannot self-collide — whole patches skipped before any broad phase; survivors go to spatial hashing | The one new technique in the batch → §7. Self-collision only; does nothing for tool-vs-tissue. Orthogonal to all five ways |
| Li, Tang et al., *P-Cloth* (SIGGRAPH Asia 2020) | Cloth at 0.5–1.65M triangles on 4–8 GPUs; spatial hashing for DCD+CCD; work-queue generation for load balancing; non-linear impact zones | Validates spatial hashing for collision at huge scale. Multi-GPU distribution is irrelevant to a single-GPU GTX 1650 Ti laptop target |
| Ashkiani et al., *A Dynamic Hash Table for the GPU* (SlabHash, IPDPS 2018) | Chaining via warp-width "slab" linked lists + SlabAlloc; concurrent insert/**delete**/search; warp-cooperative work sharing; 512M updates/s on a K40c | Reference design for a *mutable* table. Deletes unlink a slab — **no tombstone drift**, removing the main caveat on persistent open addressing. Preferred table shape if §5 ever happens |
| Jünger et al., *WarpCore* (2020) | Warp-level cooperative probing library; 1.6B inserts/s (GV100); advantage grows at >90% load factor | Same verdict as the cuCollections blog: pays at high load factors / persistent tables. Our tables are oversized by design, and way 5 removed them from the hot path |
| Dong et al., *ASH* (Open3D, 2021) | Framework: decouples the hash map from value storage via an index heap — stable integer indices into flat buffers instead of pointers | Engineering pattern, not an algorithm — and it matches our existing workspace-array style. Adopt the pattern if §5 ever happens |
| Shao et al., *H-CNN: Spatial Hashing Based CNN* (2018) | Hierarchical, offline-built (perfect-spatial-hashing style) tables to run CNNs on sparse 3D shapes | ML shape analysis on static data with offline construction. Nothing for per-frame collision |

## 5. FUTURE — Incremental / persistent table across frames

**Origin (user idea):** don't rebuild the table every frame; only update triangles that moved.

**The three traps found in analysis:**

1. *"Moved" isn't what matters — "changed cells" is.* Deforming tissue moves nearly every
   vertex slightly every frame; only cell-boundary crossings (~1–10% of triangles at
   surgical speeds) invalidate table entries. Good news: surgical deformation is local,
   so the truly-static fraction is large.
2. *The table isn't the only per-frame rebuild.* AABBs must be refreshed for deformed
   triangles, and the emitted pair list + narrow phase must re-run from scratch every frame
   regardless — the home-cell pre-cull is position-dependent (boxes shift even without cell
   changes), and contacts come from exact positions. Incremental attacks only the build
   portion: after the multi-block scan lands, roughly 0.2–0.3 ms of saveable budget, which
   the incremental machinery (change detection, delete/insert, compaction, periodic
   rebuilds) eats into.
3. *Verification gets harder.* A persistent table's layout depends on insert/delete history —
   bugs become "only reproduces after 500 frames of this exact motion." Conditional
   rebuild-on-overflow also breaks CUDA-graph capture.

**If/when we do it, the agreed shape:**

- Prefer keeping the **sorted grid and making the update incremental** — compact out
  cell-crossing triangles' entries from last frame's sorted incidence array, merge in their
  new entries. No tombstones, no spikes, generation unchanged.
- If a hash table is used anyway, the proven recipes are now on file (§4 batch 2):
  **SlabHash**-style chaining for clean deletes (no tombstone drift), **voxel hashing**'s
  localized-update + periodic garbage-collection pattern, **ASH**'s index-heap decoupling
  (stable indices into flat buffers — matches our workspace style). Open addressing +
  tombstones + cooperative probing (cuCollections/WarpCore) is the fallback shape.
- **Never cuckoo** for this: deletes are clean (≤3 known slots) but inserts can cascade/evict
  into a full rebuild on a random frame — "usually fast, occasionally 10×" is worse than
  "always the same cost" for a surgical sim with a haptics loop.

**Revisit trigger:** scene sizes ~10× larger, or scenes with a big static fraction
(mostly-rigid anatomy + small deforming region). At today's sizes rebuild-per-frame is
cheaper than the bookkeeping — which is why the deformable-body literature mostly rebuilds
(and where it doesn't — I-Cloth — the persistent structure is a BVH front at 1M-triangle
scale, not a grid).

## 6. FUTURE — Skip AABB recomputation for untouched triangles (FEM dirty-region list)

Every frame we recompute every triangle's AABB from its corners (fused in expansion). If a
triangle's corners didn't move, the result is bit-identical to last frame's — wasted work.
The FEM solver *already knows* which vertices it deformed (it wrote them), so a
"this frame I touched vertices X–Y" list is free — no detection pass. Update boxes and cell
entries only for triangles using those vertices.

Pairs naturally with §5, but helps even the current rebuild-everything pipeline. Only pays
when deformation is local; in current test scenes gravity moves everything, so today it
saves nothing. Same revisit trigger as §5.

## 7. FUTURE — Normal-cone self-collision culling (from PSCC)

For **self-collision only** (tissue folding onto itself; useless for tool-vs-tissue):
a smooth surface patch — one whose triangle normals all fit inside a cone narrower than a
half-space, plus a cheap planarity test — provably cannot collide with itself. So whole
patches are skipped *before any grid or hash work*; only folded/creased regions reach the
broad phase. Requires a patch hierarchy with per-frame cone refits (cheap, AABB-refit-like).

Today our self-collision usage (vertex-triangle path) is small, so there is nothing to cull
profitably. **Revisit trigger:** self-collision becomes a meaningful slice of frame time
(bigger deformations, draping/folding tissue).

## 8. NOT WORTH IT — Caching the coarse candidate-pair list across frames

The coarse same-cell pair list depends only on cell membership, so it *is* stable across
frames with no cell crossings (user was right on the dependency). But:

- The whole generation kernel costs **35.5 µs** — the hard ceiling on what caching saves.
- The position-dependent AABB pre-cull woven into generation is worth far more than
  generation costs: skipping it inflates FBP 85 → 200 µs.

Caching would trade ≤35 µs of savings for >100 µs of narrow-phase regression plus cache
machinery. No.

*Nuance from batch-2 papers:* I-Cloth (§4) shows persistent coarse candidate structures do
work — via BVH traversal fronts, at ~1M-triangle self-collision scale where the coarse phase
is the bottleneck. The concept is sound; the 35.5 µs ceiling is what kills it *here*.

## 9. SETTLED — Misconceptions and corrected predictions

- **"atomicCAS is faster than atomicAdd" — no.** Both are global atomics on the same L2
  hardware. atomicAdd is one always-succeeds operation (ticket dispenser); atomicCAS needs
  read-compare-swap and retries on collision (claiming lockers). Per entry, hash inserts do
  *more* global atomics than counting-sort scatter, not fewer. Measured: hash generation
  (CAS dedup) 1,237 µs @ 1.5% SM vs counting-sort scatter 119 µs; sorted generation
  (no table at all) 35.5 µs @ 56% SM.
- **"Atomics are very slow" — folklore at our scale.** Phase-13 measurement: atomic cycles
  < 2% of dominant kernel time. The expensive thing is *chains* of dependent memory
  operations (probe loops), not the atomic instruction.
- **"Sorting will lose to hashing" — my own wrong prediction.** The 5-way bake-off proved
  the opposite (0.651 ms vs 2.0–2.6 ms) because the sort's home-cell dedup doubles as the
  −86% pre-cull, and counting sort even beats CUB radix here (0.352 vs 0.479 ms).

## 10. DEAD ENDS — measured, do not retry

- **FBP occupancy raise** (`__launch_bounds__`/register squeeze): measured regression.
  The kernel is LSU/load-bound (long_scoreboard + lg_throttle, 79 regs); more resident warps
  just queue more loads. The fix is reuse (shared staging, §3), not occupancy.
- **CUB DeviceRadixSort on WSL2**: intermittently returns cudaSuccess with fully unsorted
  output (~20–25% of processes, per-process constant). Mitigated permanently: frame-0 health
  probe → process-wide counting-sort fallback + frame redo. Counting sort is faster anyway.
- **`cudaMemsetAsync` on a buffer a same-frame kernel rewrites**: copy-engine ordering
  hazard; use a post-writer tail-pad fill kernel instead.
- **Merging the dense grid's twin drivers**: 72% identical but 42 interleaved hunks —
  a merge trades duplication for unreadability. Left as twins deliberately.
- **CUDA graph for the dense way**: compute-bound path, ~1.6% projected gain. Deferred.
- **Pair-hash dedup as sorted-grid default**: works, but home-cell beats it
  (0.352 vs 0.512 ms) *and* keeps the free pre-cull. Kept only as a toggle
  (`SOFA_SORTED_GRID_PAIRHASH_DEDUP`).

## 11. Ranked pending list (snapshot)

1. **Multi-block scan** (§2) — certain, low-risk, ~413 → ~50 µs. `QUEUED`
2. **Way 5b big-cell fusion** (§3) — highest ceiling, attacks FBP load-boundedness. `PROMISING`
3. **Dense touched-slot clearing** — helps the default surgical path (~25% of its kernel
   budget; dense also pays a hidden ~8 MB pair-table memset/frame). `QUEUED`
4. **Home-cell AABB pre-cull ported to hash ways** — strictly bigger lever than any probing
   optimization (−86% of inserts vs ~13% faster inserts). `PROMISING`
5. Cooperative probing / incremental tables / dirty-region AABBs / normal-cone self-collision
   culling — `FUTURE`, triggers in §4–7.
