# 07 — Phase 3: The CUDA Kernel Cascade

This is the GPU's actual work. The narrow phase dispatches to a backend function
in `cuda/GpuCollisionBackend.cu`, which launches a sequence of kernels. We'll
trace the **feature-based proximity (FBP)** path, which runs 7 kernel launches +
1 `cudaMemset`. (The legacy exact-contact path is the same minus the last two
launches; the vertex-triangle paths are file 09.)

Remember from file 01: a kernel launch runs thousands of threads in parallel,
and it's **non-blocking** — the CPU fires it and moves on.

The 7 operations, in order:

```text
1. resetDenseGridKernel              clear the grid and counters
2. cudaMemset(pairHashKeys, 0xff)    clear the dedupe hash table
3. insertIndexedTrianglesKernel      insert tissue triangles into cells
4. insertIndexedTrianglesKernel      insert blade triangles into cells
5. generateDenseGridUniqueCandidatePairsKernel   find + dedupe candidate pairs
6. resetProximityCountersKernel      zero the contact counters
7. featureBasedProximityKernel       the actual geometry math
```

Let's take them one at a time.

---

## 7.1 Kernel 1 — `resetDenseGridKernel`

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

## 7.2 Operation 2 — `cudaMemset` on the hash table

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

## 7.3 Kernels 3 & 4 — `insertIndexedTrianglesKernel`

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

---

## 7.4 Kernel 5 — `generateDenseGridUniqueCandidatePairsKernel`

**Job:** scan the cells, form candidate pairs, and deduplicate them. This is the
**most expensive kernel** (~300 µs) because it touches all 32,768 cells.

**Launch shape:** one block per cell (32,768 blocks), 256 threads each.

```cpp
generateDenseGridUniqueCandidatePairsKernel<<<32768, 256>>>(grid, cellTissueIds,
    cellToolIds, config, candidatePairs, pairHashKeys, ...);
```

**What each block does:**

```cpp
const std::uint32_t cellId = blockIdx.x;       // this block owns one cell
const auto bucket = grid[cellId];
const std::uint32_t tissueCount = min(bucket.tissueCount, maxTissueTrianglesPerCell);
const std::uint32_t toolCount   = min(bucket.toolCount,   maxToolTrianglesPerCell);
const std::uint64_t totalPairs  = tissueCount * toolCount;

// each thread handles some of the pairs in this cell
for (localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x) {
    const std::uint32_t tissueLocal = localPair / toolCount;
    const std::uint32_t toolLocal   = localPair % toolCount;
    const std::uint32_t tissueTriangleId = cellTissueIds[cellId*128 + tissueLocal];
    const std::uint32_t toolTriangleId   = cellToolIds  [cellId*64  + toolLocal];

    atomicAdd(rawCandidateCount, 1u);
    insertUniqueCandidatePair(encodeCandidatePair(tissueTriangleId, toolTriangleId), ...);
}
```

For a cell with no tissue or no tool triangles, `totalPairs = 0` and the block
does nothing (it exits immediately). This is why most of the 32,768 blocks are
idle — only the few "mixed" cells near the blade have work. That's also why SM
throughput on this kernel is only ~26%: most blocks are empty. (It's still the
slowest kernel because just *visiting* 32,768 cells takes time.)

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

## 7.5 Kernel 6 — `resetProximityCountersKernel`

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
were found (Vertex-Face, Face-Vertex, Edge-Edge — explained in file 08).
Trivial cost.

---

## 7.6 Kernel 7 — `featureBasedProximityKernel` (the math)

**Job:** for each candidate pair, compute the exact closest geometry and emit a
contact if they're within `contactDistance`. This is the mathematical core; file
08 explains the geometry. Here we cover the kernel structure.

**Launch shape — the over-launch trick:**

```cpp
constexpr std::uint32_t kFbpMaxBlocks = 256;
const std::uint32_t fbpUpperPairs = min(maxCandidatePairs, kFbpMaxBlocks * 256);  // 65,536
const std::uint32_t fbpBlocks = max(1u, fbpUpperPairs / 256);   // 256 blocks
featureBasedProximityKernel<<<256, 256>>>(...);   // 65,536 threads
```

Here's the subtle part. We have only ~624 candidate pairs, but we launch
**65,536 threads** (256 blocks × 256). Why so many? Because the CPU doesn't
*know* how many pairs there are — that count (`candidateCount`) lives on the GPU.
To find out, the CPU would have to copy it back (a synchronizing `cudaMemcpy`),
which would stall the pipeline.

Instead, we **over-launch** a fixed, generous number of threads and let each
thread check whether it actually has work:

```cpp
const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
const std::uint32_t pairCount = *candidatePairCount;   // read from GPU memory
if (tid >= pairCount) return;     // ← 99% of threads exit here, in 1 instruction
```

Thread 0 through 623 do real work; threads 624 through 65,535 immediately exit.
The wasted threads cost almost nothing (one comparison each), and in exchange we
**eliminate a synchronous readback**. This single design choice is what took the
FBP path from 109 FPS (with the readback) to 775 FPS (without). The trade-off
shows up in Nsight as low SM throughput (~5%) — most threads do nothing — but it
buys a huge latency win.

**What each working thread does:**

```cpp
const std::uint64_t pair = candidatePairs[tid];
const std::uint32_t aIdx = pair >> 32;          // tissue triangle ID
const std::uint32_t bIdx = pair & 0xffffffff;   // blade triangle ID

const DeviceTriangle ta = indexedTriangleAt(firstPositions, firstIndices, aIdx);
const DeviceTriangle tb = indexedTriangleAt(secondPositions, secondIndices, bIdx);

// 6 vertex-face tests + 9 edge-edge tests (file 08), keep the closest
// ... find bestDistSq, bestKind, bestPoints, bestBarycentrics ...

if (bestDistSq > contactDistance * contactDistance) return;   // too far → no contact

// emit a contact
const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
contacts[outIdx] = { aIdx, bIdx, featureKind, barycentrics, points, normal, distance };
atomicAdd(/* the matching vf/fv/ee counter */, 1u);
```

Each thread unpacks its pair back into the two triangle IDs, looks up both
triangles (again through the zero-copy position pointers), runs the closest-
feature math, and — if the triangles are within 0.03 — writes a
`DeviceProximityContact` to the output buffer. The `atomicAdd(contactCount, 1)`
hands each emitting thread a unique output slot.

---

## 7.7 The whole cascade, with timings

Putting it together for the one-tissue/one-blade FBP run (validation mode,
where the kernel time is actually measured):

```text
Op                                       Threads      ~Time     What
1. resetDenseGridKernel                  32,768       2.5 µs    clear grid
2. cudaMemset (hash table)               —            ~1 µs     clear dedupe table
3. insertIndexedTrianglesKernel (tissue) 12,800       28 µs     fill tissue buckets
4. insertIndexedTrianglesKernel (blade)  12           30 µs     fill tool buckets (launch-bound)
5. generateUniqueCandidatePairs          32,768 blks  300 µs    pairs + dedupe (the heavy one)
6. resetProximityCountersKernel          1            ~0 µs     zero counters
7. featureBasedProximityKernel           65,536       17 µs     the geometry math
```

The candidate-generation kernel (5) dominates because it visits every cell. The
math kernel (7) is cheap because there are only 624 pairs. This is the opposite
of what beginners expect — "the math must be the slow part" — but it isn't; the
*spatial bookkeeping* is.

(In the production fast path with no readback, the CPU never waits for any of
this. It fires all 7 operations and returns. The numbers above are from a
validation run that does sync, so the per-kernel times are observable.)

---

## 7.8 Summary of Phase 3

```text
The backend launches 7 GPU operations, all non-blocking:
  reset grid → clear hash → insert tissue → insert blade →
  generate+dedupe pairs → reset counters → feature-based proximity math.
Atomics make the parallel inserts and dedupe safe.
The over-launch trick lets the math kernel run without a sync to learn the pair count.
Result: ~624 contacts' worth of geometry evaluated, contacts written to VRAM.
```

The one piece we deferred is *what the math kernel actually computes* — the
closest-feature geometry and barycentric weights. That's the next file. Go to
[08_the_math.md](08_the_math.md).
