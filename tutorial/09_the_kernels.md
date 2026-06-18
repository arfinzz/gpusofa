# 09 — The CUDA kernel cascade (building the grid & resolving contacts)

This is the GPU's actual work. The narrow phase dispatches to a backend function
in `cuda/GpuCollisionBackend.cu`, which launches a sequence of kernels. We'll
trace the **feature-based proximity (FBP)** path, which runs 7 kernel launches +
1 `cudaMemset`. (The legacy exact-contact path is the same minus the last two
launches; the vertex-triangle paths are file 11.)

Remember from file 01: a kernel launch runs thousands of threads in parallel,
and it's **non-blocking** — the CPU fires it and moves on.

The 7 operations, in order:

```text
1. resetDenseGridKernel              clear the grid and counters
2. cudaMemset(pairHashKeys, 0xff)    clear the dedupe hash table
3. insertIndexedTrianglesKernel      insert tissue triangles into cells
4. insertIndexedTrianglesKernel      insert blade triangles + build active-cell list
5. generateActiveDenseGridUniqueCandidatePairsKernel   find + dedupe pairs (over ~30 active cells)
6. resetProximityCountersKernel      zero the contact counters
7. featureBasedProximityKernel       the actual geometry math
```

(Operation 5 is `generateActive...` when `useToolActiveCellGeneration` is on,
which is the default. With it off, the same slot runs the all-cells
`generateDenseGridUniqueCandidatePairsKernel` instead — see §9.4.)

Let's take them one at a time.

---

## 9.1 Kernel 1 — `resetDenseGridKernel`

**Job:** wipe the grid clean so this frame starts fresh. Last frame's triangle
counts must be zeroed.

**Launch shape:** one thread per cell. 32,768 cells / 256 threads per block =
128 blocks.

```cpp
resetDenseGridKernel<<<128, 256>>>(grid, cellCount, pairHashKeys, ...);
```

**What each thread does:**

```cpp
const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < cellCount) {
    grid[id] = DeviceCellBucket { 0u, 0u };   // tissueCount = 0, toolCount = 0
}
// the first thread also zeroes the small global counters:
// candidateCount, contactCount, overflowCount, etc.
```

Thread 0 handles cell 0, thread 1 handles cell 1, ... thread 32,767 handles the
last cell. Each just writes zeros. It's memory-bound (mostly writing) and takes
about **2.5 microseconds** — basically free.

---

## 9.2 Operation 2 — `cudaMemset` on the hash table

**Job:** clear the deduplication hash table to "all empty."

```cpp
cudaMemset(pairHashKeys, 0xff, pairHashCount * sizeof(unsigned long long));
```

`cudaMemset` is a hardware-accelerated "fill memory with a byte value." Filling
with `0xff` bytes makes every 64-bit slot equal to `0xffffffffffffffff` — the
sentinel value `kEmptyPairSlot` that means "this slot is empty."

Why a `cudaMemset` instead of a kernel? The team measured that `cudaMemset`
(a dedicated hardware path) is faster than a custom clear-kernel for this on the
GTX 1650 Ti. It's counted separately in the CSV as `cuda_memset_count = 1`.

The hash table size is `nextPowerOfTwo(maxCandidatePairs * 2)` =
`nextPowerOfTwo(4,000,000)` = **4,194,304** slots (2²²). Sizing it to 2× the
max candidate count keeps it sparse so hash collisions are rare.

---

## 9.3 Kernels 3 & 4 — `insertIndexedTrianglesKernel`

**Job:** put every triangle into the cells it overlaps (the process from file
06). Run once for tissue, once for the blade.

**Launch shape:** one thread per triangle.

- Tissue: 12,800 triangles / 256 = **50 blocks**.
- Blade: 12 triangles / 256 = **1 block** (with 244 idle threads).

```cpp
insertIndexedTrianglesKernel<<<50, 256>>>(tissuePositions, tissueIndices,
    12800, /*insertTissue=*/true, config, grid, cellTissueIds, ...);
insertIndexedTrianglesKernel<<<1, 256>>>(bladePositions, bladeIndices,
    12, /*insertTissue=*/false, config, grid, cellToolIds, ...);
```

**What each thread does:**

```cpp
const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
if (triangleId >= triangleCount) return;   // idle threads exit

// read the triangle's 3 vertices THROUGH the index buffer (zero-copy positions)
const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);

// compute its inflated AABB, find the cell span, insert into each cell
insertTriangleAabbIntoGrid(triangleAabb(triangle, config.contactDistance),
                           triangleId, insertTissue, config, grid, cellIds, ...);
```

The key line is `indexedTriangleAt`:

```cpp
__device__ DeviceTriangle indexedTriangleAt(positions, triangleIndices, triangleId) {
    const std::uint32_t i0 = triangleIndices[3*triangleId + 0];
    const std::uint32_t i1 = triangleIndices[3*triangleId + 1];
    const std::uint32_t i2 = triangleIndices[3*triangleId + 2];
    return { positions[i0], positions[i1], positions[i2], triangleId };
}
```

This reads the three indices for the triangle, then looks up those vertices in
the position array. **`positions` here is the zero-copy VRAM pointer from
`deviceRead()`** — the thread reads the live tissue positions directly, no copy.
This "indexed" approach is why we never repack the triangle data: the kernel
follows indices into the shared vertex array at the moment it needs them.

Each thread then walks its triangle's cell span and does the `atomicAdd`
insertion from file 06. The atomics are what let 12,800 threads insert
simultaneously without corrupting the per-cell counts.

**Timing note:** the tissue insert (~28 µs) does real work. The blade insert
(~30 µs) is almost all *launch overhead* — 12 triangles is far too few to fill
the GPU, so the kernel spends its time starting up, not computing. This is a
known small inefficiency for tiny meshes (see `guide/plan.md` §5.13), but it's
not worth fixing because it's tiny in absolute terms.

### The blade insert also builds the active-cell list (Phase 15)

The blade insert (kernel 4, `insertTissue=false`) does one extra thing when
`useToolActiveCellGeneration` is on (the default). Right after a blade triangle
claims a slot in a cell, it checks: *am I the first blade triangle in this cell,
and does this cell already have tissue?* If so, it appends the cell's ID to an
"active list":

```cpp
const std::uint32_t localIndex = atomicAdd(&grid[cellId].toolCount, 1u);
if (localIndex < bucketCapacity) {
    cellIds[cellId * bucketCapacity + localIndex] = triangleId;
    // NEW: register this as a mixed cell, exactly once
    if (buildToolActiveList && !insertTissue && localIndex == 0u
        && grid[cellId].tissueCount > 0u) {
        const std::uint32_t a = atomicAdd(activeCellCount, 1u);
        activeCellIds[a] = cellId;
    }
}
```

Two facts make this correct:
- `localIndex == 0` means "first blade triangle in this cell," so each cell is
  appended **at most once** (automatic dedup).
- `grid[cellId].tissueCount > 0` is safe to read because the tissue insert
  (kernel 3) fully finishes before the blade insert (kernel 4) starts — CUDA
  runs the kernels in queue order. So this captures exactly the **mixed** cells.

The list (~30 entries) is the input to kernel 5. Building it here costs ~30
extra `atomicAdd`s piggybacked on a kernel that was already running and mostly
idle — effectively free. This is the trick that lets kernel 5 skip the 32,738
empty cells.

---

## 9.4 Kernel 5 — candidate generation (active-cell, by default)

**Job:** for each mixed cell, form candidate pairs and deduplicate them.

This kernel has two variants, and which one runs depends on
`useToolActiveCellGeneration`:

- **Default (flag on): `generateActiveDenseGridUniqueCandidatePairsKernel`** —
  iterates only the ~30 cells in the active list. Launched with a small fixed
  grid (1,024 blocks) that grid-strides over the list. **~8 µs.**
- **Fallback (flag off): `generateDenseGridUniqueCandidatePairsKernel`** — one
  block per cell, all 32,768 cells. Most blocks find an empty cell and exit, but
  just *visiting* 32,768 cells costs **~300 µs**. This used to be the single
  most expensive kernel in the whole pipeline.

**Default launch shape:** a fixed 1,024 blocks, 256 threads each. The kernel
reads the active-cell count *from GPU memory* and grid-strides over it, so the
CPU never needs to know how many active cells there are (no readback, no sync —
same idea as the over-launch in §9.6).

```cpp
generateActiveDenseGridUniqueCandidatePairsKernel<<<1024, 256>>>(grid,
    activeCellIds, activeCellCount, cellTissueIds, cellToolIds, config,
    candidatePairs, pairHashKeys, ...);
```

**What each block does:**

```cpp
const std::uint32_t activeCount = *activeCellCount;     // read from GPU memory
// grid-stride: block b handles active cells b, b+1024, b+2048, ...
for (uint32_t i = blockIdx.x; i < activeCount; i += gridDim.x) {
    const std::uint32_t cellId = activeCellIds[i];      // a guaranteed-mixed cell
    const auto bucket = grid[cellId];
    const std::uint32_t tissueCount = min(bucket.tissueCount, maxTissueTrianglesPerCell);
    const std::uint32_t toolCount   = min(bucket.toolCount,   maxToolTrianglesPerCell);
    const std::uint64_t totalPairs  = tissueCount * toolCount;

    for (localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x) {
        const std::uint32_t tissueLocal = localPair / toolCount;
        const std::uint32_t toolLocal   = localPair % toolCount;
        const std::uint32_t tissueTriangleId = cellTissueIds[cellId*128 + tissueLocal];
        const std::uint32_t toolTriangleId   = cellToolIds  [cellId*64  + toolLocal];

        atomicAdd(rawCandidateCount, 1u);
        insertUniqueCandidatePair(encodeCandidatePair(tissueTriangleId, toolTriangleId), ...);
    }
}
```

The all-cells fallback is identical *except* its outer line is
`const cellId = blockIdx.x;` (each block owns one cell, no active list, all
32,768 cells). The inner pairing + dedupe logic is the same in both.

Because the active variant only ever visits cells that are guaranteed to be
mixed, it produces **exactly the same candidate pairs** as the all-cells
variant — it just skips the ~32,738 cells that could never contribute a pair.
That's why the contact output is bit-identical between the two (verified by
A/B; `guide/plan.md` §5.15).

> **Advanced sidebar — a third, experimental broad cull.** The two variants
> above both use the *dense* grid (a fixed 32,768-cell array). There is a third,
> **optional and default-off** broad cull on the `experiment/hash-prefixsum-broadphase`
> branch: a **spatial hash** (only occupied cells take a table slot, claimed with
> `atomicCAS`) plus **prefix-sum work-expansion** (count `tissue×tool` pairs per
> occupied cell, `exclusive_scan` them, then launch *one thread per candidate
> pair*). It feeds the **same** kernel 7 below, so contacts are bit-identical to
> the dense path. It's aimed at the *both-large* case (large tissue **and** large
> tool, ~1,500+ triangles each), where it measured **+11.8 % FPS / −15 % narrow
> wall**. Turn it on with `useHashPrefixSumGeneration=True`. Beginners can skip
> this — the dense active-cell path above is the default. Details:
> `reports/hash_prefixsum_broadphase_experiment_20260609.md`.

> **A correctness note from the field.** When this optimization was added, a
> large-scene A/B (a subdivided blade producing 322,560 candidate pairs)
> revealed a *separate* latent bug in kernel 7 — see §9.6. The active-cell
> change was correct; it just exposed something else.

### Encoding a pair into one number

```cpp
__device__ std::uint64_t encodeCandidatePair(uint32 tissueId, uint32 toolId) {
    return (uint64(tissueId) << 32) | uint64(toolId);
}
```

The pair (tissue 4200, blade 7) becomes a single 64-bit number: tissue ID in the
top 32 bits, tool ID in the bottom 32. `(4200 << 32) | 7`. In hex that's
`0x0000106800000007` (4200 = 0x1068). Packing both IDs into one number makes
the dedupe hash table simple — it stores one number per pair.

### The dedupe — `insertUniqueCandidatePair`

This is the lock-free hash table. Walk through it:

```cpp
__device__ bool insertUniqueCandidatePair(pair, pairHashKeys, capacity, ...) {
    const std::uint32_t mask = capacity - 1;       // capacity is a power of two
    std::uint32_t slot = mixCandidatePairHash(pair) & mask;   // starting slot

    for (probe = 0; probe < 256; ++probe) {
        const std::uint64_t previous = atomicCAS(&pairHashKeys[slot],
                                                  kEmptyPairSlot,   // expected: empty
                                                  pair);            // new: this pair
        if (previous == kEmptyPairSlot) {
            // WE claimed an empty slot → this pair is new → output it
            const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            candidatePairs[outputIndex] = pair;
            return true;
        }
        if (previous == pair) {
            return false;     // someone already inserted this exact pair → skip
        }
        slot = (slot + 1) & mask;   // collision with a DIFFERENT pair → try next slot
    }
    // 256 probes failed → table too full → count overflow
}
```

How it works:

1. **Hash the pair** to pick a starting slot. `mixCandidatePairHash` is the
   MurmurHash3 finalizer — a sequence of XOR-shifts and multiplies that scrambles
   the 64-bit pair into a well-distributed slot index. `& mask` keeps it in range.

2. **`atomicCAS`** tries to claim the slot: "if this slot is empty, put my pair
   in it." Returns whatever was there before.
   - If it was empty → **I won** → this is the first time anyone saw this pair →
     append it to the output list (via another `atomicAdd` for the output slot).
   - If it already held *my exact pair* → a duplicate → skip silently.
   - If it held a *different* pair (hash collision) → move to the next slot and
     try again (this is "linear probing").

The beauty: thousands of threads insert pairs simultaneously, duplicates vanish
automatically, and there's **no CPU involvement and no lock** — just atomic
hardware instructions. This replaced an older approach that copied all pairs to
the CPU, sorted them, and removed duplicates there (slow PCIe round trips).

After this kernel: `candidatePairs` holds the ~624 unique pairs, and
`candidateCount` says how many.

---

## 9.5 Kernel 6 — `resetProximityCountersKernel`

**Job:** zero the contact output counters before the math kernel writes to them.

**Launch shape:** a single thread (`<<<1, 1>>>`) — there are only 5 counters to
zero.

```cpp
__global__ void resetProximityCountersKernel(contactCount, overflowCount,
                                             vfCount, fvCount, eeCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *contactCount = 0; *overflowCount = 0;
        *vfCount = 0; *fvCount = 0; *eeCount = 0;
    }
}
```

`vfCount`, `fvCount`, `eeCount` track how many contacts of each *feature type*
were found (Vertex-Face, Face-Vertex, Edge-Edge — explained in file 10).
Trivial cost.

---

## 9.6 Kernel 7 — `featureBasedProximityKernel` (the math)

**Job:** for each candidate pair, compute the exact closest geometry and emit a
contact if they're within `contactDistance`. This is the mathematical core; file
08 explains the geometry. Here we cover the kernel structure.

**Launch shape — over-launch + grid-stride:**

```cpp
constexpr std::uint32_t kFbpMaxBlocks = 1024;
const std::uint32_t fbpUpperPairs = min(maxCandidatePairs, kFbpMaxBlocks * 256);
const std::uint32_t fbpBlocks = max(1u, fbpUpperPairs / 256);   // up to 1024 blocks
featureBasedProximityKernel<<<1024, 256>>>(...);
```

Here's the subtle part. The CPU doesn't *know* how many candidate pairs there
are — that count (`candidateCount`) lives on the GPU. To find out, the CPU would
have to copy it back (a synchronizing `cudaMemcpy`), which would stall the
pipeline. So instead of launching exactly the right number of threads, we launch
a **fixed, generous grid** and let the kernel figure out the work count itself —
on the GPU, with no readback:

```cpp
const std::uint32_t pairCount = *candidatePairCount;            // read from GPU memory
const std::uint32_t stride = gridDim.x * blockDim.x;            // total threads launched
for (uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < pairCount; idx += stride) {
    const std::uint64_t pair = candidatePairs[idx];
    ... process this pair ...
}
```

This is a **grid-stride loop**, and it does two jobs at once:

1. **No sync needed.** The CPU launches a fixed grid; the kernel reads
   `pairCount` from GPU memory and only the threads with `idx < pairCount` do
   work. The CPU never learns the count, so there's no readback and no stall.
   This is what took the FBP path from 109 FPS (with the readback) to ~940 FPS
   (without) — together with the active-cell generation of §9.4.

2. **Correctness at any scale.** This is the important part, and it was a real
   bug for a while. The grid is a fixed 1,024 blocks × 256 = 262,144 threads.
   A small scene has 624 pairs (fewer than the threads), so each thread does at
   most one pair. But a **large** scene can produce *more* pairs than threads —
   the large-tissue benchmark has **322,560 pairs**. Without the `for` loop,
   each thread would do exactly one pair and the kernel would silently process
   only the first 262,144 and **drop the rest**. The grid-stride `for` makes
   each thread loop back and pick up `idx + stride`, `idx + 2·stride`, … until
   every pair is covered. (The earlier version had no loop — it just did
   `if (tid >= pairCount) return; ... candidatePairs[tid]` — and dropped ~80% of
   pairs on the large scene. That was Phase 17's fix; see `guide/plan.md` §5.17.)

**What each loop iteration does:**

```cpp
const std::uint64_t pair = candidatePairs[idx];
const std::uint32_t aIdx = pair >> 32;          // tissue triangle ID
const std::uint32_t bIdx = pair & 0xffffffff;   // blade triangle ID

const DeviceTriangle ta = indexedTriangleAt(firstPositions, firstIndices, aIdx);
const DeviceTriangle tb = indexedTriangleAt(secondPositions, secondIndices, bIdx);

// 6 vertex-face tests + 9 edge-edge tests (file 10), keep the closest
// ... find bestDistSq, bestKind, bestPoints, bestBarycentrics ...

if (bestDistSq > contactDistance * contactDistance) continue;   // too far → next pair

// emit a contact
const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
contacts[outIdx] = { aIdx, bIdx, featureKind, barycentrics, points, normal, distance };
atomicAdd(/* the matching vf/fv/ee counter */, 1u);
```

Each iteration unpacks the pair back into the two triangle IDs, looks up both
triangles (again through the zero-copy position pointers), runs the closest-
feature math, and — if the triangles are within 0.03 — writes a
`DeviceProximityContact` to the output buffer. The `atomicAdd(contactCount, 1)`
hands each emitting thread a unique output slot. Note the `continue` (not
`return`): a thread that finds "too far" moves on to its next strided pair
rather than exiting.

---

## 9.7 The whole cascade, with timings

Putting it together for the one-tissue/one-blade FBP run (validation mode,
where the kernel time is actually measured), **with the default active-cell
generation**:

```text
Op                                       Launch        ~Time     What
1. resetDenseGridKernel                  32,768 thr    2.5 µs    clear grid
2. cudaMemset (hash table)               —             ~1 µs     clear dedupe table
3. insertIndexedTrianglesKernel (tissue) 12,800 thr    28 µs     fill tissue buckets
4. insertIndexedTrianglesKernel (blade)  12 thr        30 µs     fill tool buckets + active list (launch-bound)
5. generateActiveUniqueCandidatePairs    1,024 blks     8 µs     pairs + dedupe over ~30 active cells
6. resetProximityCountersKernel          1 thr         ~0 µs     zero counters
7. featureBasedProximityKernel           1,024 blks    17 µs     the geometry math
                                                       ──────
                                          total        ~87 µs
```

Compare to the **old all-cells path** (flag off), where op 5 was a 32,768-block
kernel taking **~300 µs** — about 80% of the whole frame. Phase 15 collapsed it
to ~8 µs, and now no single kernel dominates: the tissue/blade inserts and the
FBP math are the largest pieces, all in the tens of microseconds.

Two beginner-surprising facts survive:
- The *math* kernel (7) is cheap (17 µs) because there are only 624 pairs — the
  geometry was never the bottleneck.
- The *spatial bookkeeping* used to be the whole cost, and the fix wasn't a
  faster algorithm — it was *not visiting cells that can't matter*.

(In the production fast path with no readback, the CPU never waits for any of
this. It fires all 7 operations and returns. The numbers above are from a
validation run that does sync, so the per-kernel times are observable.)

---

## 9.8 Summary of Phase 3

```text
The backend launches 7 GPU operations, all non-blocking:
  reset grid → clear hash → insert tissue → insert blade (+ build active list) →
  generate+dedupe pairs (over ~30 active cells) → reset counters → FBP math.
Atomics make the parallel inserts and dedupe safe.
The blade insert builds the mixed-cell list for free, so generation skips the
  ~32,738 empty cells (Phase 15) — a 38× win on that kernel.
The over-launch + grid-stride lets the math kernel run without a sync to learn
  the pair count AND stay correct when there are more pairs than threads (Phase 17).
Result: ~624 contacts' worth of geometry evaluated, contacts written to VRAM.
```

The one piece we deferred is *what the math kernel actually computes* — the
closest-feature geometry and barycentric weights. That's the next file. Go to
[10_the_math.md](10_the_math.md).
