# Spatial-Hash + Prefix-Sum Broad Cull — Experiment Report

**Branch:** `experiment/hash-prefixsum-broadphase` (additive, opt-in, default-off)
**Date:** 2026-06-09
**Status:** Working, correctness-verified, faster than the dense grid in the large+large regime.
**Independently re-verified 2026-06-09 (second pass):** clean rebuild, backend
bench `hash_contacts=488=fbp_contacts` (vf 0 / fv 133 / ee 355, occupied_slots
16,536, unique_pairs 6,720, overflow 0); SOFA A/B `dense=hash=2354` contacts
(1119/428/807), raw 61,920 / unique 33,408 identical on both legs, overflow 0,
hash +10.9 % FPS / −17 % narrow wall on a cool GPU. All §3 and §4 numbers
reproduce.

> This is an **experimental alternative broad cull** for the tri–tri feature-based
> proximity (FBP) path. It does **not** touch the existing dense-grid implementation.
> Everything here is gated behind a default-off flag. With the flag off, the runtime
> path is byte-identical to `main`.

---

## 1. Motivation

The current default broad cull is a **dense uniform grid** (64×8×64 = 32,768 fixed
cells) plus the Phase 15 *tool-active-cell* optimization. That optimization assumes
an **asymmetry**: a small tool sweeping through a large tissue, so we only need to
emit candidate pairs for the handful of cells the tool actually touches.

When **both** sides are large (the user's target: ~1500-triangle tool against a
large tissue), that asymmetry weakens — the tool lights up many cells, the grid is a
fixed size regardless of where the geometry actually is, and a lot of grid memory is
spent on empty space.

This experiment replaces the dense grid with two ideas that scale with *occupancy*
rather than with a fixed cell count:

1. **Spatial hash** — only *occupied* cells consume a table slot. An open-addressing
   hash table keyed by linear cell id, slots claimed lock-free with `atomicCAS`,
   per-slot tissue/tool triangle counts bumped with `atomicAdd`.
2. **Prefix-sum work expansion** — for each occupied slot, the number of candidate
   pairs is `tissueCount × toolCount`. An `exclusive_scan` over those per-slot counts
   produces global offsets; then we launch **one thread per candidate pair**. Each
   thread binary-searches the offset array to recover its `(slot, localPair)` and
   emits exactly one pair. Perfect load balancing, no per-cell serial loops.

Both then feed the **identical** FBP narrow kernel (6 VF + 9 EE feature tests,
Ericson closest-feature math), so the produced contacts must be bit-identical.

---

## 2. What was added (all additive)

| File | Addition |
|---|---|
| `GpuCollisionBackend.h` | `HashPrefixSumConfig`, `HashPrefixSumStats`, entry point `computeHashPrefixSumProximityContacts(...)` |
| `GpuCollisionBackendStub.cpp` | No-CUDA stub returning `false` with diagnostic |
| `cuda/GpuCollisionBackend.cu` | `HashGridWorkspace` (persistent, grow-on-demand), hash kernels (`resetHashGridKernel`, `insertHashGridTrianglesKernel`, `computeHashPairsPerSlotKernel`, `setHashRawTotalKernel`, `generateHashPrefixSumCandidatePairsKernel`), and the public `computeHashPrefixSumProximityContacts` 9-step pipeline reusing `featureBasedProximityKernel` |
| `GpuCollisionNarrowPhase.{h,cpp}` | `Data<bool> useHashPrefixSumGeneration` (default **false**), `Data<uint> hashTableSize` (default **0** = auto). Dispatch sub-branch inside the existing FBP branch |
| `tools/DenseGridBackendBench.cpp` | Standalone hash phase gated by `SOFA_BACKEND_BENCH_RUN_HASH`, writes `<label>_hash.csv`, prints a correctness assertion |
| `testscenes/hash_prefixsum_large.py` | Large tissue (12,800 tris) + large blade tool (1,568 tris) scene, hash flag from env |
| `scripts/run_hash_prefixsum_large_ab_wsl.sh` | A/B launcher: same scene, dense leg then hash leg |

**Pipeline (9 steps)** in `computeHashPrefixSumProximityContacts`:
reset table → insert tissue triangles → insert tool triangles → compute
`pairsPerSlot` per occupied slot → `thrust::exclusive_scan` to `pairOffsets` →
set `rawTotal` → generate candidate pairs (one thread per pair, binary search,
dedup via the existing `insertUniqueCandidatePair` open-addressing hash) →
FBP narrow kernel → batched pinned readback of counters.

**Auto table size:** `nextPow2((firstTris + secondTris) * 4)`, min 1024. Override
with `hashTableSize` Data / `SOFA_HASH_TABLE_SIZE`.

---

## 3. Correctness

Verified at two levels.

### 3a. Standalone backend bench (`SofaGpuCollisionDenseGridBackendBench`)
Same geometry through the dense FBP path and the hash path:

```
fbp_contacts  = 488  (vf=0  fv=133  ee=355)
hash_contacts = 488  (vf=0  fv=133  ee=355)   ← identical
hash_table_size=65536  occupied_slots=16536  unique_pairs=6720  overflow=0
```

### 3b. Real SOFA scene (`testscenes/hash_prefixsum_large.py`, 12,800 + 1,568 tris)
A/B run, 190 measured steps each. **Every** broad-/narrow-phase count is identical:

| metric | dense_baseline | hash_prefixsum |
|---|---:|---:|
| output_contact_count | **2354** | **2354** |
| vf / fv / ee | 1119 / 428 / 807 | 1119 / 428 / 807 |
| raw_candidate_count | 61,920 | 61,920 |
| unique_candidate_count | 33,408 | 33,408 |
| overflow_count | 0 | 0 |
| probe_overflow_count | 0 | 0 |

The hash broad cull reproduces the dense grid's candidate set exactly (same 33,408
unique pairs out of 61,920 raw) and therefore the same 2,354 contacts.

---

## 4. Performance (large tissue + large tool, 190 measured steps)

| metric | dense_baseline | hash_prefixsum | delta |
|---|---:|---:|---:|
| avg whole-pipeline FPS | 359.5 | **405.2** | **+12.7 %** |
| avg step time (ms) | 2.782 | **2.468** | −11.3 % |
| avg narrow wall (ms) | 1.974 | **1.698** | −14.0 % |
| avg narrow kernel (ms) | 1.436 | **1.268** | −11.7 % |
| avg host-sync (ms) | 0.538 | 0.430 | −20.0 % |
| avg device-alloc (ms) | 0.00176 | 0.00073 | −58 % |
| kernel launches / frame | 7 | 8 | +1 (the scan) |

A short 20-step confirmation run showed the same ranking (457 → 545 FPS).

**Why it wins here:** the dense grid pays for all 32,768 cells and inserts both
large meshes into a fixed structure; the hash table only materializes occupied
slots and the prefix-sum generation gives perfectly balanced one-thread-per-pair
work with no serial per-cell inner loop. In the large+large regime that more than
pays for the extra `exclusive_scan` launch.

> ⚠️ This advantage is **regime-specific**. For the original small-tool/large-tissue
> scenes, the Phase 15 tool-active-cell path is extremely cheap and likely still
> wins. This experiment is meant for the *both-large* case, which is why it ships
> default-off and per-scene opt-in.

---

## 5. How to use

In a scene (Python), on the `GpuCollisionNarrowPhase`:

```python
useFeatureBasedProximity=True,        # required: hash path feeds the FBP kernel
useHashPrefixSumGeneration=True,      # opt in to the hash + prefix-sum broad cull
hashTableSize=0,                      # 0 = auto (~4 slots / input triangle, pow2)
```

A/B from WSL:

```bash
scripts/run_hash_prefixsum_large_ab_wsl.sh     # dense leg, then hash leg
# env knobs: SOFA_BENCHMARK_STEPS, SOFA_HASH_TISSUE_NX/NZ, SOFA_HASH_BLADE_SEGMENTS_*
```

Standalone backend correctness (no SOFA):

```bash
SOFA_BACKEND_BENCH_RUN_HASH=1 ./SofaGpuCollisionDenseGridBackendBench
```

---

## 6. Guarantees / scope

- **Dense path untouched.** With `useHashPrefixSumGeneration=false` (the default)
  the dispatch falls through to the unchanged `computeFeatureBasedProximityContacts`
  call. `git diff` on the narrow phase shows only the two new `Data` fields and the
  `if/else` wrapper around the original (now `else`-branch) call.
- **Self-collision / vertex-triangle path** is not routed through the hash cull yet
  — only the cross-model tri–tri FBP path. (The v-t kernels are unchanged.)
- **Workspace** is a persistent singleton that grows on demand; table-sized arrays
  use the growing allocator so increasing `hashTableSize` between frames is safe.
- Zero overflow observed at auto table size in both test scenes.

---

## 7. Possible follow-ups (not done)

- Route self-collision (v-t) candidate generation through the same hash + prefix-sum
  expansion.
- Compute the hash table size from a measured occupancy histogram instead of a fixed
  4×-triangles heuristic.
- Fuse the `pairsPerSlot` compute into the insert kernel to drop one launch.
- A scaling sweep (tissue/tool size grid) to find the crossover where hash overtakes
  the dense tool-active-cell path.
