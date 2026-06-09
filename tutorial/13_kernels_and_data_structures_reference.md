# 13 — Kernels & Data Structures: the complete reference

This is the **engineering reference** for the GPU collision plugin. Where the
rest of the tutorial *explains*, this file *enumerates*: every CUDA kernel that
can launch, what data goes in, what comes out, the data structures on both
sides, how many kernels run per frame, and exactly how many threads each one
asks the GPU for and why.

Everything here is grounded in
`SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu` and
`SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h`. Line numbers are
given so you can open the source and read along; they may drift by a few lines
as the file changes, but the function/kernel names are stable.

If you have not read [07_phase3_kernels.md](07_phase3_kernels.md) yet, read it
first — it teaches the *idea* of each kernel with pictures. This file is the
dense lookup table you come back to.

---

## 0. The one-paragraph orientation

Every collision frame is **two stages**: a **broad cull** that throws away
triangle pairs that can't possibly touch, and a **narrow phase** that does exact
geometry on the survivors. There are **four narrow-phase paths** (you pick one
per scene with flags), but they **all share the same broad cull machinery** — a
uniform grid of cells, into which triangles are bucketed, from which candidate
pairs are emitted. The narrow kernel then runs one thread per candidate pair.
The whole thing is designed so the **CPU never waits for the GPU** (see
[10_phase4_sync_and_output.md](10_phase4_sync_and_output.md)).

```text
                 INPUT: two triangle meshes (or a point cloud + a mesh),
                        already living in GPU memory (zero-copy)
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        │                       BROAD CULL                          │
        │  reset grid → insert mesh A → insert mesh B →             │
        │  generate candidate pairs (dedup)                         │
        └────────────────────────────┼─────────────────────────────┘
                                     │  a list of candidate pairs (uint64s)
        ┌────────────────────────────┼─────────────────────────────┐
        │                      NARROW PHASE                         │
        │  reset counters → one thread per pair → exact geometry →  │
        │  write ProximityContact + bump counters                  │
        └────────────────────────────┼─────────────────────────────┘
                                     │
                 OUTPUT: a flat array of ProximityContact on the GPU
                         + per-class counters (VF / FV / EE / overflow)
```

---

## 1. Conventions you need to read the tables

- **Thread.** One GPU worker. Runs the kernel body once for its index.
- **Block.** A group of threads (always **256** here). Threads in a block can
  share fast memory and synchronize; the plugin mostly uses blocks just as a
  scheduling unit.
- **Grid.** All the blocks of one launch. Written `<<<blocks, threads>>>`.
- **`tid`.** The global thread index, `blockIdx.x * blockDim.x + threadIdx.x`.
- **Over-launch + grid-stride.** Several kernels are launched with a *fixed*
  number of blocks that does **not** depend on the exact work size, and each
  thread then loops `for (i = tid; i < count; i += totalThreads)`. This avoids a
  CPU↔GPU round-trip to read the work count before launching. The grid is a
  **saturation target** (enough threads to fill the GPU), not a correctness
  bound. See §6.
- **`threadCount = 256`** everywhere in this plugin. The 256 is a Turing-friendly
  block size (8 warps); it is `constexpr` in the launch sites.
- The reference scenes set the grid to **64 × 8 × 64 = 32,768 cells**
  (`gridResolutionX/Y/Z`). The header *defaults* are 64×24×64, but every shipped
  scene overrides Y to 8 because the surgical geometry is thin in Y.

---

## 2. Master kernel table

The plugin defines ~25 `__global__` kernels. The ones that actually run in the
**production collision paths** are below, grouped by stage. (`exact*` kernels are
the legacy SAT boolean-intersection path, default-off; the `broadPhaseCompact` /
`treePairOverlap` kernels belong to the old AABB-tree broad phase and are not on
the dense-grid path.)

| # | Kernel (`.cu` line) | Stage / path | Launch `<<<blocks, threads>>>` | One thread = | In → Out |
|---|---|---|---|---|---|
| 1 | `resetDenseGridKernel` (1044) | broad, all dense paths | `<<<ceil(cells/256), 256>>>` = **128×256** | one **grid cell** (+ thread 0 zeros the counters) | grid buckets → all zeroed |
| 2 | `insertIndexedTrianglesKernel` (1322) | broad, tri-tri | `<<<ceil(triCount/256), 256>>>` | one **triangle** | triangle → appended to every overlapped cell's bucket |
| 3 | `insertIndexedPointsKernel` (2270) | broad, vertex-triangle | `<<<ceil(pointCount/256), 256>>>` | one **point** | point → appended to its cell's tool bucket |
| 4 | `generateActiveDenseGridUniqueCandidatePairsKernel` (1803) | broad, **default** (Phase 15) | `<<<min(cells,1024), 256>>>` = **1024×256** | one **active (mixed) cell**, block-strided | cell's tissue×tool → deduped candidate pairs |
| 4b | `generateDenseGridUniqueCandidatePairsKernel` (1701) | broad, fallback / v-t | `<<<cells, 256>>>` = **32768×256** | one **grid cell** (one block per cell) | cell's tissue×tool → deduped candidate pairs |
| 5 | `resetProximityCountersKernel` (2239) | narrow, all FBP paths | `<<<1, 1>>>` | a single thread zeros 5 counters | counters → 0 |
| 6 | `featureBasedProximityKernel` (2080) | narrow, **tri-tri** | `<<<1024, 256>>>`, grid-strided | one **candidate pair** | triangle pair → ≤1 `ProximityContact` (6 VF + 9 EE tests) |
| 7 | `featureBasedVertexTriangleProximityKernel` (2323) | narrow, **vertex-triangle** | `<<<1024, 256>>>`, grid-strided | one **(point, triangle) pair** | pair → ≤1 `ProximityContact` (closest-point-on-triangle) |
| H1 | `resetHashGridKernel` (2607) | broad, **hash** (experiment) | `<<<ceil(table/256), 256>>>` | one **hash slot** | slots → emptied (`0xffffffff`) |
| H2 | `insertHashGridTrianglesKernel` (2637) | broad, hash | `<<<ceil(triCount/256), 256>>>` | one **triangle** | triangle → open-addressing `atomicCAS` into slots |
| H3 | `computeHashPairsPerSlotKernel` (2703) | broad, hash | `<<<ceil(table/256), 256>>>` | one **hash slot** | slot → `pairsPerSlot = tissue×tool` |
| H4 | `thrust::exclusive_scan` (library) | broad, hash | (Thrust-chosen) | prefix-sum over `pairsPerSlot` | per-slot counts → global `pairOffsets` |
| H5 | `setHashRawTotalKernel` (2719) | broad, hash | `<<<1, 1>>>` | one thread | offsets → `rawTotal` (grand total of pairs) |
| H6 | `generateHashPrefixSumCandidatePairsKernel` (2746) | broad, hash | `<<<1024, 256>>>`, grid-strided | one **raw candidate pair** | binary-search `(slot,localPair)` → deduped candidate pair |
| 6 | `featureBasedProximityKernel` (2080) | narrow, hash | `<<<1024, 256>>>`, grid-strided | one **candidate pair** | **same kernel as the dense tri-tri path** |

> **Why the hash path produces bit-identical contacts to the dense path:** it
> changes only *how candidate pairs are stored and generated* (rows H1–H6
> replace rows 1–4). The set of candidate pairs it emits is the same, and it
> hands them to the **same** `featureBasedProximityKernel` (row 6). Different
> broad cull, identical narrow phase ⇒ identical contacts.

---

## 3. How many kernels run per frame, per path

Measured as `avg_kernel_launch_count` in the benchmark summaries (excludes the
`cudaMemset` that clears the dedup table, which is counted separately).

| Path | Kernel launches / frame | The sequence |
|---|---:|---|
| **Tri-tri FBP, dense grid** (default) | **7** | reset → insert tissue → insert tool → generate pairs → reset counters → FBP narrow → (counter readback is a copy, not a kernel) |
| **Tri-tri FBP, hash + prefix-sum** (experiment) | **8** | reset → insert A → insert B → pairs-per-slot → **scan** → raw-total → generate pairs → reset counters → FBP narrow *(the scan is the “+1” vs dense)* |
| **Vertex-triangle** (self or cross) | **6** | reset → insert triangles → insert points → generate pairs → reset counters → v-t narrow |

The "+1" for the hash path is the `thrust::exclusive_scan`. That single extra
launch is what the prefix-sum work-expansion costs, and in the both-large regime
it more than pays for itself (see
[reports/branch_comparison_20260609.md](../reports/branch_comparison_20260609.md)).

---

## 4. INPUT data structures (host → backend)

These are the C++ structs the SOFA component fills in and hands to the backend.
Defined in `GpuCollisionBackend.h`.

### 4.1 `TriangleIndexedSurface` — a triangle mesh
```cpp
struct TriangleIndexedSurface {
    const TriangleVertex* positions;        // host vertex array (used if device* is null)
    const TriangleVertex* devicePositions;  // GPU vertex array — the zero-copy fast path
    std::uint32_t         vertexCount;
    const std::uint32_t*  triangleIndices;  // flat: 3 indices per triangle
    std::uint32_t         triangleCount;
    std::uint64_t         surfaceId;         // identity, for the topology cache + self-corner exclusion
    std::uint64_t         topologyVersion;   // bump to invalidate the cached index upload
};
```
- **`devicePositions`** is the whole speed story: when non-null it points
  straight at SofaCUDA's `CudaVec3f` device buffer, so positions are **never
  copied** — the kernels read the simulation's own GPU memory. `positions`
  (host) is only the fallback for the standalone bench.
- **`triangleIndices`** is `3 × triangleCount` `uint32`s. It *is* uploaded once
  and cached, keyed by `(surfaceId, topologyVersion)` — topology rarely changes,
  positions change every frame, so only positions need to be fresh.
- One `TriangleVertex` is `{float x, y, z}` = **12 bytes**.

### 4.2 `PointCloudSurface` — a vertex cloud (vertex-triangle path)
```cpp
struct PointCloudSurface {
    const TriangleVertex* positions;
    const TriangleVertex* devicePositions;  // zero-copy, same idea
    std::uint32_t         pointCount;
    std::uint64_t         surfaceId;
    std::uint64_t         topologyVersion;
};
```
Used two ways: (a) self-collision — the cloud is a mesh's own vertex set tested
against its own triangles; (b) cross-model — a `CudaPointCollisionModel` vs a
`CudaTriangleCollisionModel`.

### 4.3 Config structs
```cpp
struct DenseGridConfig {            // drives the broad cull for ALL paths
    float gridMin{X,Y,Z}, gridMax{X,Y,Z};        // world-space box the grid covers
    uint  gridResolution{X,Y,Z};                 // cells per axis (scenes: 64,8,64)
    float contactDistance;                        // AABBs are inflated by this
    uint  maxTissueTrianglesPerCell;              // bucket capacity (scenes: 128)
    uint  maxToolTrianglesPerCell;                // bucket capacity (scenes: 128)
    uint  maxCandidatePairs;                       // candidate array capacity
    bool  useGpuHashDedupe;                        // GPU open-addressing pair dedup
    bool  useToolActiveCellGeneration{true};       // Phase 15: generate over mixed cells only
    ...
};
struct FeatureBasedProximityConfig { // narrow-phase knobs
    float contactDistance;            // drop pairs farther apart than this
    bool  computeBarycentrics;        // fill bary weights for the solver
    bool  keepContactsOnDevice;       // detection-only: skip the D2H of contacts
    bool  readContactCounter;         // copy back just the counts (for validation)
    uint  maxContacts;                // output capacity
};
struct HashPrefixSumConfig {         // experiment-only
    uint  hashTableSize;              // 0 = auto = nextPow2((firstTris+secondTris)*4)
    uint  maxProbe;                   // linear-probe cap before "overflow" (default 64)
};
```

---

## 5. INTERNAL data structures (the GPU workspace)

These never leave the GPU. They live in a **persistent singleton workspace** that
grows on demand and is reused every frame (so steady-state frames do **zero**
`cudaMalloc`). For sizes below, `C` = cell count (32,768), `T` = hash table size.

### 5.1 Dense-grid workspace (`DenseGridWorkspace`, `.cu` line 191)
| Field | Type & size | What it holds |
|---|---|---|
| `grid` | `DeviceCellBucket[C]` | per cell: `{tissueCount, toolCount}` (8 bytes/cell) |
| `cellTissueIds` | `uint32[C × maxTissuePerCell]` | the tissue-triangle ids in each cell's bucket |
| `cellToolIds` | `uint32[C × maxToolPerCell]` | the tool-triangle (or point) ids in each cell |
| `activeCellIds` | `uint32[C]` | list of *mixed* cells, built during the tool insert (Phase 15) |
| `activeCellCount` | `uint32` | how many entries `activeCellIds` actually has this frame |
| `candidatePairs` | `uint64[maxCandidatePairs]` | the survivors: each is `(firstTriIdx << 32) | secondTriIdx` |
| `pairHashKeys` | `uint64[pow2]` | open-addressing table used to **dedup** candidate pairs |
| `candidateCount` / `rawCandidateCount` | `uint32` each | deduped count / pre-dedup count |
| `proximityContacts` | `DeviceProximityContact[maxContacts]` | the narrow-phase output (the twin of `ProximityContact`) |
| `proximity{Contact,Overflow,Vf,Fv,Ee}Count` | `uint32` each | the five output counters |

`DeviceCellBucket` (`.cu` 102): just `{uint32 tissueCount; uint32 toolCount;}`.
The actual triangle ids live in the separate `cellTissueIds` / `cellToolIds`
arrays — the bucket only stores how many and where (`cellId * cap + localIndex`).

### 5.2 Hash-grid workspace (`HashGridWorkspace`, `.cu` line 2430, experiment)
The hash path uses its **own** workspace (independent of the dense one):
| Field | Type & size | What it holds |
|---|---|---|
| `cellKeys` | `uint32[T]` | each slot stores the **linear cell id** it was claimed for, or `0xffffffff` (empty) |
| `tissueCount` / `toolCount` | `uint32[T]` each | per-slot triangle counts |
| `tissueIds` / `toolIds` | `uint32[T × maxPerCell]` each | per-slot buckets |
| `pairsPerSlot` | `uint32[T]` | `min(tissue,cap) × min(tool,cap)` for each slot |
| `pairOffsets` | `uint32[T]` | exclusive prefix-sum of `pairsPerSlot` (global start offset per slot) |
| `rawTotal` | `uint32` | grand total of candidate pairs = `pairOffsets[T-1] + pairsPerSlot[T-1]` |
| `candidatePairs`, `pairHashKeys`, counters, `proximityContacts` | (same as dense) | shared design; the narrow output is identical |

The key difference from the dense grid: **only occupied cells take a slot**. A
dense grid always materializes all 32,768 cells; the hash table only ever touches
the handful of slots for cells that actually contain geometry.

---

## 6. OUTPUT data structure (the contacts)

Every FBP path writes the **same** output type. Defined in `GpuCollisionBackend.h`
as `ProximityContact` (its on-device twin is `DeviceProximityContact`, `.cu`
1942). One per accepted pair:

```cpp
struct ProximityContact {
    uint32  firstPrimitiveIndex;    // triangle id on side A (or vertex id for v-t)
    uint32  secondPrimitiveIndex;   // triangle id on side B
    ProximityFeatureKind featureKind;   // VertexFace=0, FaceVertex=1, EdgeEdge=2
    uint8   firstFeatureLocalIndex;     // which vertex/edge (0..2) on A
    uint8   secondFeatureLocalIndex;    // which vertex/edge (0..2) on B
    float   firstBarycentrics[3];       // weights reconstructing the closest point on A
    float   secondBarycentrics[3];      // weights reconstructing the closest point on B
    TriangleVertex pointOnFirst;        // world-space closest point on A
    TriangleVertex pointOnSecond;       // world-space closest point on B
    TriangleVertex normal;              // unit A→B when separated, flipped when penetrating
    float   signedDistance;             // negative ⇒ interpenetrating
};
```

Plus **five `uint32` counters** that the narrow kernel bumps with `atomicAdd`:
`contactCount`, `vfContactCount`, `fvContactCount`, `eeContactCount`,
`overflowCount`. These are what the benchmark reports as
`vf/fv/ee=1119/428/807`, etc. The contact *array* normally stays on the GPU
(detection-only); only the counters are optionally copied back (one tiny sync)
when `readContactCounter` is on.

**Feature kinds** (`ProximityFeatureKind`): `VertexFace` = a vertex of A is
closest to a face of B; `FaceVertex` = a vertex of B closest to a face of A;
`EdgeEdge` = the closest points lie on an edge of each. The barycentrics are
always written in `(a0,a1,a2,b0,b1,b2)` order; unused weights are zero. This is
exactly what a constraint solver needs to apply a response without re-deriving
the geometry. The math behind it is in [08_the_math.md](08_the_math.md).

---

## 7. Thread allocation — worked examples

"How many threads?" depends on the kernel's role. There are two families:

### 7.1 One-thread-per-element kernels (size known on the CPU)
For inserts and resets, the CPU already knows the element count, so it sizes the
grid exactly: `blocks = ceil(count / 256)`.

*Example — the small surgical scene (one-tissue/one-blade, 12,800 + 12 tris):*
- `resetDenseGridKernel`: `ceil(32768/256)=128` blocks → **32,768 threads**, one per cell.
- `insertIndexedTrianglesKernel` (tissue): `ceil(12800/256)=50` blocks → **12,800 threads**, one per tissue triangle.
- `insertIndexedTrianglesKernel` (tool): `ceil(12/256)=1` block → 256 threads, 12 active, the rest exit at `if (tid >= triangleCount) return`.

### 7.2 Over-launched grid-stride kernels (size known only on the GPU)
The candidate-pair count is computed *on the GPU* during generation. Reading it
back to size the narrow launch would force a CPU↔GPU sync and kill the
[no-wait fast path](10_phase4_sync_and_output.md). So instead the narrow kernel
launches a **fixed** `<<<1024, 256>>>` = **262,144 threads** and each thread
grid-strides:

```cpp
for (uint i = tid; i < *candidatePairCount; i += gridDim.x * blockDim.x)
    ... test pair i ...   // early-outs use `continue`, never `return`
```

- If there are 56 candidate pairs (small scene), 56 threads do one test each and
  the other ~262k exit immediately — cheap, ~1 instruction each.
- If there are 322,560 candidate pairs (large scene), each of the 262,144 threads
  loops ~1.2 times. **No pair is dropped** because the loop covers everything past
  the launch size.

> **The Phase 17 bug this design once had:** the early-outs inside that loop used
> `return` instead of `continue`. A `return` abandons a thread's *remaining*
> strided pairs, so on the large scene ~80% of pairs silently vanished. The fix
> (every in-loop early-out is `continue`) is why the large scene now reports the
> correct **8018** contacts instead of a buggy ~3700. **Lesson: any
> over-launched kernel that grid-strides a device-counted list must `continue`,
> never `return`, on a per-item early-out.**

### 7.3 The Phase 15 active-cell trick (why generation isn't 32,768 blocks)
The naive candidate generator launches **one block per grid cell** — 32,768
blocks — but in a surgical scene only ~30 cells are *mixed* (contain both tissue
and tool). That's 99% empty blocks: pure scheduler overhead (~300 µs, which was
80% of GPU time). Phase 15 builds a list of the mixed cells **during the tool
insert** (no extra scan) and launches
`generateActiveDenseGridUniqueCandidatePairsKernel` with `min(cells,1024)=1024`
blocks that **block-stride over just the active list**. Generation dropped
300 µs → 7.9 µs and the small scene went 221 → 943 FPS, with bit-identical
output. Default-on (`useToolActiveCellGeneration`).

---

## 8. The hash + prefix-sum broad cull, step by step (experiment)

This is the alternative broad cull (rows H1–H6 of §2), default-off, for the
**both-large** regime. Pipeline in `computeHashPrefixSumProximityContacts`
(`.cu` ~5983–6029):

1. **`resetHashGridKernel`** — one thread per slot sets `cellKeys[slot] = 0xffffffff` (empty), zeros counts.
2. **`insertHashGridTrianglesKernel` ×2** (mesh A then mesh B) — one thread per triangle. For each cell the triangle's inflated AABB overlaps, it `atomicCAS`-claims a slot for that cell id (linear probing on collision), then `atomicAdd`s into that slot's tissue/tool bucket. Only occupied cells ever consume a slot.
3. **`computeHashPairsPerSlotKernel`** — one thread per slot writes `pairsPerSlot = min(tissue,cap) × min(tool,cap)` (the number of candidate pairs that slot will produce).
4. **`thrust::exclusive_scan`** — turns the per-slot counts into global start offsets `pairOffsets` (slot *k*'s pairs occupy `[pairOffsets[k], pairOffsets[k]+pairsPerSlot[k])`). *This is the one extra kernel vs the dense path.*
5. **`setHashRawTotalKernel`** — one thread computes `rawTotal = pairOffsets[T-1] + pairsPerSlot[T-1]`, the grand total of candidate pairs.
6. `cudaMemset` clears the dedup hash table.
7. **`generateHashPrefixSumCandidatePairsKernel`** — launches one thread per *raw pair index* `i ∈ [0, rawTotal)`, grid-strided. Each thread **binary-searches** `pairOffsets` to find which slot owns `i` and the local pair index within it, decodes that into `(tissueId, toolId)`, and `atomicCAS`-dedups it into the candidate array. **Perfect load balance** — every thread does exactly one pair's worth of work, no per-cell serial inner loop.
8. **`resetProximityCountersKernel`** — `<<<1,1>>>`.
9. **`featureBasedProximityKernel`** — the *same* narrow kernel as the dense path, over the *same* candidate pairs ⇒ the *same* contacts.

Why it can win: the dense grid pays to reset and scan all 32,768 fixed cells and
load-balances poorly (block-per-cell), while the hash table only materializes
occupied slots and the prefix sum gives every thread exactly one pair. Why it's
off by default: in the small-tool regime the Phase 15 active-cell path is already
~8 µs, so the scan is pure overhead there. Numbers:
[reports/branch_comparison_20260609.md](../reports/branch_comparison_20260609.md).

---

## 9. Quick map: flag → path → kernels

| Scene flags (on `GpuCollisionNarrowPhase`) | Path | Broad-cull kernels | Narrow kernel |
|---|---|---|---|
| `useFeatureBasedProximity=True` (default broad cull) | tri-tri FBP, dense | reset, insert×2, generate(active) | `featureBasedProximityKernel` |
| `+ useHashPrefixSumGeneration=True` | tri-tri FBP, hash | resetHash, insertHash×2, pairsPerSlot, scan, rawTotal, generateHash | `featureBasedProximityKernel` (same) |
| `+ useVertexTriangleProximity=True`, self-collision pair | v-t self | reset, insertTris, insertPoints, generate | `featureBasedVertexTriangleProximityKernel` |
| `+ useVertexTriangleProximity=True`, point-model vs tri-model | v-t cross | reset, insertTris, insertPoints, generate | `featureBasedVertexTriangleProximityKernel` |

---

### See also
- [07_phase3_kernels.md](07_phase3_kernels.md) — the kernels with pictures and intuition.
- [08_the_math.md](08_the_math.md) — the VF/FV/EE closest-feature geometry.
- [06_the_dense_grid.md](06_the_dense_grid.md) — the grid data structure with a worked numeric example.
- [00_high_level_flow.md](00_high_level_flow.md) — the easy top-to-bottom story.
- `guide/architecture.md` — the canonical reference doc.
