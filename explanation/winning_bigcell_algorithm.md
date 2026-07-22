# The winning big-cell collision algorithm, from start to finish

## 1. The problem in plain language

We have two triangle meshes:

- the **tissue** mesh;
- the **tool** mesh.

If the tissue has 80,000 triangles and the tool has hundreds or thousands, testing every tissue triangle against every tool triangle would create millions or billions of tests.

Most triangle pairs are far apart. The algorithm therefore works in two levels:

1. **Broad cull:** quickly find triangle pairs that might be close.
2. **Narrow phase:** perform accurate geometry only for those surviving pairs.

The winning path also fuses the last part of broad culling with the narrow phase. It does not first create a huge global list of triangle pairs. It finds local candidates inside one CUDA block and immediately tests them.

## 2. Important words

### Small cell

The world is divided into a regular 3D grid. One box in this grid is a **small cell**.

### Big cell

Several neighboring small cells are grouped into one **big cell**.

The winning configuration uses factor 2, so one big cell contains:

```text
2 cells in X × 2 cells in Y × 2 cells in Z = 8 small cells
```

The code supports factor 4 too, which has 64 small cells per big cell. That is why a packed entry reserves six bits for the local small-cell number.

### AABB

AABB means axis-aligned bounding box. It is the smallest box, aligned with X/Y/Z, that contains a triangle.

Two forms are used:

- **raw AABB:** exactly contains the triangle;
- **inflated AABB:** raw AABB expanded by `contactDistance` in every direction.

Inflated boxes answer “could these objects be close enough?” Raw boxes answer “is their minimum box-to-box gap already larger than the allowed contact distance?”

### Mixed big cell

A big cell is **mixed** when it contains at least one tissue entry and at least one tool entry. Only mixed cells can produce tissue-tool contacts.

### Home-cell rule

The same triangle can overlap multiple cells. Without a rule, the same triangle pair might be tested several times.

The algorithm calculates a deterministic overlap point from the two inflated boxes and assigns the pair to the small cell containing that point. Only that cell is allowed to test the pair. This is the **home-cell rule**.

## 3. Input arrays

For a mesh with `N` triangles, the important inputs are:

| Input | Shape | Meaning |
|---|---:|---|
| positions | vertex array | XYZ coordinates |
| triangle indices | `3 × N` indices | three vertex references per triangle |
| triangle count | 1 integer | number of triangles |
| grid configuration | one structure | grid origin, cell size, resolution, contact distance |
| big-cell configuration | one structure | factor, factor shift, big-grid resolution |
| max contacts | 1 integer | output capacity |

The algorithm creates these important work arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `hist` | `2 × bigCellCount` | entry count for each big cell and side |
| `starts` | `2 × bigCellCount + 1` | CSR start positions after prefix sum |
| `cursors` | `2 × bigCellCount` | temporary write positions |
| `entriesPacked` | `entryCapacity` | triangle ID plus local small-cell ID |
| `mixedBigCellIds` | up to `bigCellCount` | big cells containing both sides |
| `mixedBigCellCount` | 1 integer | length of that list |
| tissue inflated AABBs | tissue triangle count | reused by the fused broad test |
| contacts | `maxContacts` | final accurate contact records |

There are two histogram bins per big cell:

```text
bin = 2 × bigCellId + side
side = 0 for tissue
side = 1 for tool
```

## 4. The nine logical CUDA launches

The champion uses CSR construction with shared-memory hash building. Its logical frame sequence is:

| # | Kernel | Grid and block shape | Main input | Main output | Why needed |
|---:|---|---|---|---|---|
| 1 | `resetSortedGridKernel` | up to 1,024 blocks × 256 threads | previous counters/histogram | zeroed histogram and counters | remove last frame's data |
| 2 | `countBigCellEntriesSharedHashKernel` for tissue | `ceil(tissueTriangles/256)` × 256 | tissue triangles | tissue histogram contribution, inflated AABBs, total entries | count where tissue entries will go |
| 3 | same count kernel for tool | `ceil(toolTriangles/256)` × 256 | tool triangles | tool histogram contribution and total entries | count tool entries |
| 4 | `exclusiveScanSortedGridBinsKernel` | 1 block × 1,024 threads | histogram | `starts` and initial `cursors` | convert counts into contiguous CSR ranges |
| 5 | `fillBigCellEntriesSharedHashKernel` for tissue | `ceil(tissueTriangles/256)` × 256 | tissue triangles and cursors | packed tissue entries | populate tissue CSR ranges |
| 6 | same fill kernel for tool | `ceil(toolTriangles/256)` × 256 | tool triangles and cursors | packed tool entries | populate tool CSR ranges |
| 7 | `buildSortedGridMixedCellsKernel` | `ceil(bigCellCount/256)` × 256 | tissue/tool counts | mixed-cell ID list | discard cells containing only one side |
| 8 | `resetProximityCountersKernel` | 1 block × 1 thread | contact counters | zero contact/subtype/overflow counters | start a clean output list |
| 9 | `fusedBigCellNarrowKernel` | 1,024 blocks × 256 threads | CSR, mixed list, vertices, AABBs | final contacts and statistics | local candidate generation + exact collision work |

When CUDA graphs are enabled, the first frame captures this sequence and later frames replay it. The logical kernels and their data flow do not change.

## 5. Kernel 1: reset

The reset kernel clears:

- every histogram bin;
- total-entry count;
- entry-overflow count;
- mixed-cell count;
- shared-builder spill count.

Threads use a grid-stride loop, so a bounded number of blocks can clear arrays larger than the launched thread count.

## 6. Kernels 2 and 3: count entries using a shared hash table

One CUDA thread owns one input triangle.

For its triangle, a thread:

1. loads the three indexed vertices;
2. builds an inflated AABB using `contactDistance`;
3. finds all small cells overlapped by that box;
4. converts every small-cell coordinate to a big-cell ID and a local small-cell ID;
5. adds counts to a block-local shared hash table.

The shared hash table has 1,024 slots. Instead of every entry immediately performing a global histogram atomic, entries targeting the same bin accumulate in shared memory. At the end of the block, the block merges one total per occupied shared slot into global `hist`.

If the bounded shared hash cannot hold an entry, that entry safely falls back to the global path. `sharedSpillCount` records this event; it does not change the result.

Why two launches? Tissue and tool need separate histogram bins, and each mesh has a different number of triangles.

## 7. Kernel 4: exclusive prefix sum

The histogram contains counts, for example:

```text
bin 24 tissue count = 3
bin 25 tool count   = 2
bin 26 tissue count = 5
```

An exclusive scan converts these counts to start offsets:

```text
starts[24] = 500
starts[25] = 503
starts[26] = 505
starts[27] = 510
```

Now the ranges are known:

```text
bin 24 owns entriesPacked[500..502]
bin 25 owns entriesPacked[503..504]
bin 26 owns entriesPacked[505..509]
```

This data layout is CSR: one large entry array plus a small start-offset array.

## 8. Kernels 5 and 6: fill the CSR entry array

The fill pass walks the triangle AABBs over the grid again. This time it writes actual entries.

Each entry is packed into 32 bits:

```text
packed = (triangleId << 6) | localSmallCellId
```

The lower six bits store local cell 0-63. The upper bits store the triangle ID.

Like the count pass, the champion fill pass stages work in shared memory:

- up to 2,048 entries per block;
- a 1,024-slot shared hash groups entries by bin;
- one global reservation is made per occupied shared bin;
- threads scatter staged entries into that reserved contiguous range.

If staging or shared hashing is full, the entry takes the correct global fallback path.

## 9. Kernel 7: build the mixed-cell list

One thread examines one big cell.

It checks:

```text
hist[2 × bigCell]     > 0   // tissue exists
hist[2 × bigCell + 1] > 0   // tool exists
```

If both are true, the thread appends the big-cell ID to `mixedBigCellIds` using an atomic counter.

This is an important broad cull. Empty cells and one-sided cells never reach the expensive fused kernel.

## 10. Kernel 8: reset contact counters

One thread sets these values to zero:

- total contacts;
- contact overflow;
- VF contacts;
- FV contacts;
- EE contacts.

The tested-pair counter is reset by the broader grid reset path.

## 11. Kernel 9: the fused broad/narrow kernel

### Launch shape

```text
1,024 blocks
256 threads per block
```

The kernel uses a grid-stride loop over mixed big cells:

```text
mixed index = blockIdx.x
next mixed index = current + gridDim.x
```

For the measured scenes, the mixed list is smaller than 1,024, so the first active blocks normally own one mixed big cell each. Extra blocks see no work and exit.

### Shared memory owned by one block

Each active block has about 17.5 KiB of shared memory:

| Shared item | Size/purpose |
|---|---|
| 256 tool triangle IDs | identify output primitives |
| 256 exact tool raw AABBs | winning reusable reject data |
| `256 × 3` tool vertices | avoid reloading vertices for every pair |
| 256 packed scratch entries | unsorted tool tile |
| 65 run starts | begin/end of 64 local-cell runs |
| 64 bin cursors | histogram then scatter positions |
| one tile count | number of valid staged entries |

### Step 11.1: choose the tissue and tool ranges

For CSR mode, the block reads `starts` to find:

- tissue range for this big cell;
- tool range for this big cell.

The tool range is processed in chunks of at most 256 because the shared tile contains 256 tool entries.

### Step 11.2: stage one tool tile

For a tile of size `T ≤ 256`:

- thread 0 loads entry 0;
- thread 1 loads entry 1;
- …;
- thread `T-1` loads entry `T-1`.

Each thread puts its packed entry into shared scratch and increments the local-small-cell histogram.

### Step 11.3: counting-sort the tool tile by local cell

The first 64 threads clear the 64 bin counters. Thread 0 calculates prefix starts for the bins. All threads then scatter packed entries into runs.

After this, tools belonging to local cell 5 are contiguous, tools for local cell 6 are contiguous, and so on.

This is crucial: a tissue entry in local cell 5 only scans the tool run for local cell 5. It does not scan the whole big cell.

### Step 11.4: load reusable tool data

For each sorted tool entry, one thread:

1. decodes `toolTriangleId = packed >> 6`;
2. loads the triangle's three vertices once;
3. stores all three vertices in shared memory;
4. computes the exact raw AABB once;
5. stores that raw AABB in shared memory.

The raw AABB is the campaign winner. Every tissue triangle paired with this tool triangle reuses the same box.

### Step 11.5: distribute tissue entries among threads

Thread `i` handles tissue entry indices:

```text
i, i + 256, i + 512, ...
```

For one tissue entry, the thread:

1. decodes the tissue triangle ID and local cell;
2. finds the matching tool run using `sRunStart[local]`;
3. loads the tissue triangle vertices once;
4. computes its exact raw AABB once;
5. walks only the matching tool run.

### Step 11.6: inflated-AABB overlap

The tissue inflated box was already created during the count pass.

The tool's inflated box is reconstructed from its shared raw box:

```text
inflated min = raw min - contactDistance
inflated max = raw max + contactDistance
```

If the inflated boxes do not overlap, the pair is rejected immediately.

### Step 11.7: home-cell ownership

The kernel calculates the maximum of the two inflated minimum corners:

```text
mx = max(tissue.minX, tool.minX)
my = max(tissue.minY, tool.minY)
mz = max(tissue.minZ, tool.minZ)
```

That overlap point maps to one deterministic small cell. If the current local cell is not that cell, this duplicate entry is rejected. Another cell owns the pair.

Only after this rule passes does `pairsTestedCount` increase.

### Step 11.8: exact raw-AABB gap rejection

For each axis, the code calculates the empty gap between the raw boxes. Overlapping axes have gap zero.

```text
gapSquared = gx² + gy² + gz²
```

If:

```text
gapSquared > contactDistance²
```

then no points on the two triangles can be close enough, so the pair is rejected before the 15 feature tests.

The previous code rebuilt both raw boxes here for every pair. The winning code reuses the boxes computed during tool staging and tissue-entry setup.

### Step 11.9: accurate closest-feature tests

For a surviving pair, the helper checks 15 feature combinations:

```text
3 × tissue vertex against tool face  = 3 VF
3 × tool vertex against tissue face  = 3 FV
3 tissue edges × 3 tool edges        = 9 EE
total                                  15
```

It keeps the feature pair with the smallest squared distance.

If the best distance is larger than `contactDistance`, there is no contact.

If it is close enough, the helper produces:

- feature type VF/FV/EE;
- local feature indices;
- barycentric coordinates;
- point on tissue;
- point on tool;
- normal;
- signed distance.

### Step 11.10: append the contact

A successful thread performs:

```text
outputIndex = atomicAdd(contactCount, 1)
```

If `outputIndex < maxContacts`, it writes the contact and increments the correct VF/FV/EE counter.

If capacity is exceeded, it increments `overflowCount` instead of writing outside the array.

## 12. Complete worked trace

Use a simplified example:

```text
big-cell factor = 2
bigCellId = 12
local small-cell coordinate = (1, 0, 1)
factor shift = 1
```

The local ID is:

```text
local = x | (y << 1) | (z << 2)
      = 1 | 0 | 4
      = 5
```

Suppose tissue triangle 101 and tool triangle 9 both overlap this local cell.

Their packed entries are:

```text
tissue packed = (101 << 6) | 5 = 6469
tool packed   = (9 << 6) | 5   = 581
```

Their global histogram bins are:

```text
tissue bin = (12 << 1) | 0 = 24
tool bin   = (12 << 1) | 1 = 25
```

The count kernels add to bins 24 and 25. The prefix scan gives each bin a range. The fill kernels place 6469 in the tissue range and 581 in the tool range. Because both counts are nonzero, big cell 12 enters the mixed list.

In the fused kernel:

1. one block takes big cell 12;
2. a tool-tile thread reads packed value 581;
3. it decodes tool triangle 9 and local cell 5;
4. counting sort places it in the run for local cell 5;
5. the thread loads tool triangle 9's vertices and stores its raw AABB;
6. a tissue thread reads packed value 6469;
7. it decodes tissue triangle 101 and local cell 5;
8. it loads tissue triangle 101 and computes its raw AABB once;
9. it scans only the tool run for cell 5;
10. inflated boxes overlap and cell 5 wins the home-cell rule;
11. the exact raw-box gap is tested;
12. if the gap passes, 15 closest-feature combinations are evaluated;
13. if the closest distance is within the threshold, one contact is appended.

### A raw-gap rejection example

Let `contactDistance = 0.10`.

Along X:

```text
tissue raw box = [0.00, 0.04]
tool raw box   = [0.25, 0.30]
X gap          = 0.21
```

Assume Y and Z overlap, so their gaps are zero:

```text
gapSquared       = 0.21² = 0.0441
thresholdSquared = 0.10² = 0.0100
```

`0.0441 > 0.0100`, so the pair cannot produce a contact. The kernel skips all 15 feature tests.

If the tool box instead started at X=0.08, the X gap would be 0.04:

```text
0.04² = 0.0016 <= 0.0100
```

The pair survives and proceeds to exact VF/FV/EE testing.

## 13. Why this algorithm is fast

It combines several levels of reuse and rejection:

1. grid cells reject distant triangles;
2. mixed-cell filtering rejects one-sided big cells;
3. shared-memory counting groups entries before global merging;
4. local-cell runs avoid an all-to-all comparison inside a big cell;
5. the home-cell rule eliminates duplicates;
6. inflated AABB overlap is cheap;
7. reusable exact raw AABBs reject more pairs before feature math;
8. tool vertices and raw boxes are loaded once per tile, not once per pair;
9. contacts are written directly, avoiding a large global candidate-pair array.

## 14. What the algorithm does not guarantee

- It does not make every warp coherent. Tissue entries can have different run lengths.
- It is not bandwidth-bound on the tested GTX 1650 Ti; divergence and dependency latency dominate.
- It does not guarantee that 128 threads are always faster; the production launch remains 256 threads.
- It does not use warp-aggregated output because measured hit lanes were too sparse.
- It does not use a survivor queue. That would be a separate two-kernel architecture with extra memory traffic.

## 15. Where to read the code

- Broad build and fused kernel: `SofaGpuCollision/src/SofaGpuCollision/cuda/detail/BigCellGrid.cuh`
- Closest-feature geometry and contact append: `SofaGpuCollision/src/SofaGpuCollision/cuda/detail/FbpKernels.cuh`
- Backend benchmark and counters: `SofaGpuCollision/src/tools/DenseGridBackendBench.cpp`
- Permanent optimization decisions: `findings/fused_kernel_optimization_findings.md`
- Reproducible final validation: `scripts/run_fused_winner_validation_wsl.sh`
