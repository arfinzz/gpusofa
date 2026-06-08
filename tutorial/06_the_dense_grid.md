# 06 — The Dense Grid (with a Fully Worked Example)

This is the central data structure. Everything in Phase 3 operates on it. Once
you understand the dense grid, the kernels are easy. We'll build it up slowly
and then work a real numeric example using the benchmark's actual grid
settings.

---

## 6.1 The problem it solves

We have 12,800 tissue triangles and 12 blade triangles. We want to find which
tissue triangles are close to which blade triangles.

The naive approach: check every tissue triangle against every blade triangle.
That's 12,800 × 12 = 153,600 checks. Doable here, but it explodes for bigger
scenes (two 50,000-triangle meshes = 2.5 *billion* checks).

The dense grid is the trick that avoids most of those checks. The insight:
**two triangles can only touch if they're in the same region of space.** So
chop space into a grid of boxes ("cells"), drop each triangle into the cells it
overlaps, and then only check triangles that share a cell.

---

## 6.2 What the grid is

Imagine the 3D space around the tissue divided into a regular grid of small
boxes, like a 3D spreadsheet or a Rubik's cube with many subdivisions.

From the benchmark scene:

```python
gridMinX=-4.5, gridMinY=-0.5, gridMinZ=-4.5
gridMaxX=4.5,  gridMaxY=0.5,  gridMaxZ=4.5
gridResolutionX=64, gridResolutionY=8, gridResolutionZ=64
```

This means:

- The grid covers a box from (−4.5, −0.5, −4.5) to (4.5, 0.5, 4.5). That's 9
  units wide in X, 1 unit tall in Y, 9 units deep in Z. (Slightly bigger than
  the 8×8 tissue so everything fits.)
- It's chopped into **64 cells** along X, **8** along Y, **64** along Z.
- Total cells = 64 × 8 × 64 = **32,768 cells**.

### The size of one cell

- X: 9.0 units / 64 cells = **0.140625 units** per cell.
- Y: 1.0 unit / 8 cells = **0.125 units** per cell.
- Z: 9.0 units / 64 cells = **0.140625 units** per cell.

The code stores the *inverse* of these (the "inverse cell size") because
multiplying is faster than dividing on a GPU:

```text
inverseCellSize.x = resolutionX / (gridMaxX - gridMinX) = 64 / 9.0 = 7.1111...
inverseCellSize.y = 8 / 1.0 = 8.0
inverseCellSize.z = 64 / 9.0 = 7.1111...
```

So "how many cells fit in one unit" is 7.11 in X and Z, 8.0 in Y.

---

## 6.3 Each cell has two buckets

A cell is not just empty space — it holds two lists ("buckets"):

```cpp
struct DeviceCellBucket {
    std::uint32_t tissueCount;   // how many tissue triangles are in this cell
    std::uint32_t toolCount;     // how many blade triangles are in this cell
};
```

- The **tissue bucket** lists tissue triangles that overlap this cell.
- The **tool bucket** lists blade (tool) triangles that overlap this cell.

The actual triangle IDs go into two giant flat arrays:

```text
cellTissueIds[cellId * maxTissueTrianglesPerCell + slot] = tissueTriangleId
cellToolIds  [cellId * maxToolTrianglesPerCell   + slot] = toolTriangleId
```

`maxTissueTrianglesPerCell = 128` and `maxToolTrianglesPerCell = 64` are the
**capacities** — each cell can hold up to 128 tissue triangles and 64 blade
triangles. If more than that try to fit, the extras are counted as "overflow"
(a signal that the grid is too coarse for the scene). In the benchmark, cells
hold far fewer than the cap, so no overflow.

So the total memory for the tissue bucket array is 32,768 cells × 128 slots ×
4 bytes = 16 MB. The workspace allocates this once (Phase 0) and reuses it.

---

## 6.4 Linear cell IDs — turning (x, y, z) into one number

A cell has a 3D coordinate like (31, 3, 31) — column 31, layer 3, row 31. But
arrays are 1D, so we flatten that to a single index. From `denseCellId`:

```cpp
cellId = x + y * resolutionX + z * (resolutionX * resolutionY)
       = x + y * 64 + z * (64 * 8)
       = x + y * 64 + z * 512
```

This is exactly like flattening a 3D array into 1D. X is the fastest-varying,
then Y, then Z.

Example: cell (31, 3, 31) → `31 + 3*64 + 31*512 = 31 + 192 + 15872 = 16095`. So
that cell's bucket lives at `grid[16095]`.

---

## 6.5 Which cells does a triangle land in?

A triangle isn't a point — it has extent. So it might overlap several cells. The
process (from `triangleAabb` + `denseGridCellSpan`):

### Step 1 — compute the triangle's bounding box (AABB)

Take the min and max corner of the triangle, then inflate by `contactDistance`:

```cpp
minPoint = min(p0, p1, p2) - contactDistance   // 0.03
maxPoint = max(p0, p1, p2) + contactDistance
```

The inflation by `contactDistance` is important: we want to catch triangles
that are *close* (within 0.03), not just overlapping. Inflating the box by 0.03
means two boxes overlap whenever the triangles are within 0.06 of each other,
which guarantees we don't miss a near-contact.

### Step 2 — convert the box corners to cell coordinates

```cpp
cellMin.x = floor((aabbMin.x - gridMin.x) * inverseCellSize.x)
cellMax.x = floor((aabbMax.x - gridMin.x) * inverseCellSize.x)
// same for y and z
```

Then clamp each to `[0, resolution-1]` so triangles near the grid edge don't
index out of bounds.

### Step 3 — the triangle occupies every cell in that span

The triangle goes into all cells from `cellMin` to `cellMax` (a small box of
cells).

---

## 6.6 WORKED EXAMPLE — inserting one tissue triangle

Let's do this with real numbers. Take a tissue triangle near the origin. The
tissue is flat at y = 0 with vertices 0.1 apart. Say a triangle with vertices:

```text
p0 = (0.0, 0.0, 0.0)
p1 = (0.1, 0.0, 0.0)
p2 = (0.1, 0.0, 0.1)
```

(All at y = 0 because the sheet is flat.)

### Bounding box

```text
minPoint = (0.0, 0.0, 0.0) - 0.03 = (-0.03, -0.03, -0.03)
maxPoint = (0.1, 0.0, 0.1) + 0.03 = ( 0.13,  0.03,  0.13)
```

### Cell coordinates

Using gridMin = (−4.5, −0.5, −4.5) and inverseCellSize = (7.111, 8.0, 7.111):

```text
cellMin.x = floor((-0.03 - (-4.5)) * 7.111) = floor(4.47 * 7.111) = floor(31.78) = 31
cellMin.y = floor((-0.03 - (-0.5)) * 8.0)   = floor(0.47 * 8.0)   = floor(3.76)  = 3
cellMin.z = floor((-0.03 - (-4.5)) * 7.111) = floor(4.47 * 7.111) = floor(31.78) = 31

cellMax.x = floor(( 0.13 - (-4.5)) * 7.111) = floor(4.63 * 7.111) = floor(32.92) = 32
cellMax.y = floor(( 0.03 - (-0.5)) * 8.0)   = floor(0.53 * 8.0)   = floor(4.24)  = 4
cellMax.z = floor(( 0.13 - (-4.5)) * 7.111) = floor(4.63 * 7.111) = floor(32.92) = 32
```

So this triangle spans cells X∈[31,32], Y∈[3,4], Z∈[31,32] — a 2×2×2 block of
8 cells. (Most tissue triangles fall in fewer, but edge-straddling ones hit a
few.)

Notice the Y cells are 3 and 4. The grid's Y range is [−0.5, 0.5] split into 8
layers; the middle is at y = 0, which lands on the boundary between layers 3 and
4. Since the tissue is flat at y = 0, **all tissue triangles land in Y-layers 3
and 4** — the middle of the grid. The top and bottom Y-layers (0–2 and 5–7) are
empty of tissue.

### Writing into the grid

For each of those 8 cells, the insertion does (from `insertTriangleAabbIntoGrid`):

```cpp
const std::uint32_t localIndex = atomicAdd(&grid[cellId].tissueCount, 1u);
if (localIndex < bucketCapacity)   // 128
    cellTissueIds[cellId * 128 + localIndex] = triangleId;
else
    atomicAdd(overflowCount, 1u);
```

`atomicAdd` returns the slot number to use and increments the count in one safe
step. If this is the first tissue triangle in cell 16095, `localIndex = 0`, and
the triangle's ID is written to `cellTissueIds[16095*128 + 0]`. The next
triangle to land there gets `localIndex = 1`, and so on. The `atomic` part
guarantees that even if 5 threads insert into cell 16095 at the same instant,
they get slots 0, 1, 2, 3, 4 — no collision.

---

## 6.7 The blade lands in the middle too

The blade box spans X∈[−0.9, 0.9], Y∈[−0.3, 0.3], Z∈[−0.06, 0.06], centered at
the origin. Its triangles, run through the same math, land in cells around the
center of the grid — overlapping exactly the Y-layers 3 and 4 where the tissue
is, in the X and Z range near the middle.

**This overlap is the whole point.** Cells near the origin end up with *both*
tissue triangles (in their tissue bucket) *and* blade triangles (in their tool
bucket). Those are the "mixed" cells where collisions can happen.

A cell far from the origin — say cell (5, 3, 5) out near the edge — has tissue
triangles but no blade triangles. Its tool bucket is empty, so it produces no
candidate pairs. The grid has automatically pruned away the 99% of the tissue
that's nowhere near the blade.

---

## 6.8 From cells to candidate pairs

Once every triangle is inserted, the grid is scanned and for each cell that has
*both* tissue and tool triangles, we form all combinations:

```text
candidate pairs from a cell = (its tissue triangles) × (its tool triangles)
```

Example: a cell holding tissue triangles {4200, 4201, 4280} and blade triangle
{7} produces 3 × 1 = 3 candidate pairs:
`(4200, 7)`, `(4201, 7)`, `(4280, 7)`.

These pairs are the candidates that the expensive narrow-phase math will
actually evaluate. For the benchmark, the grid reduces 153,600 brute-force
checks down to about **2,304 raw candidate pairs** (and 624 after removing
duplicates — next section).

### Which cells get scanned? (the Phase 15 optimization)

There are two ways to do this scan, and the project uses the smarter one by
default:

1. **All-cells scan (the original, now the fallback).** Visit *every* one of the
   32,768 cells, check each for mixed tissue+tool content. The blade only
   touches ~30 cells, so ~32,738 cells are visited for nothing. This is the
   single most expensive kernel in the original pipeline (~300 µs — file 07).

2. **Tool-active-cell scan (Phase 15, the default since 2026-05-25).** Notice the
   asymmetry: the *tissue* is huge (covers thousands of cells), but the *tool*
   is tiny (touches ~30 cells). A pair can only exist in a cell that has a tool
   triangle. So instead of scanning all cells, scan only the **tool-occupied
   cells** — about 30 of them.

   The clever part: that list of tool-occupied cells is built **for free during
   the blade insert** (kernel 4). When the first blade triangle lands in a cell
   that already contains tissue, that cell's ID is appended to an "active list."
   No separate scan pass is needed — the list is a side effect of an insert
   that already runs. (Details and the correctness argument are in file 07 §7.4
   and `guide/plan.md` §5.15.)

   Result: candidate generation visits ~30 cells instead of 32,768. On the
   one-tissue scene this dropped the generation kernel from ~300 µs to ~8 µs —
   a 38× speedup — and the whole frame from ~221 to ~943 FPS. It produces the
   **exact same candidate pairs** (it just skips cells that could never
   contribute one), so the contacts are bit-identical.

The flag is `useToolActiveCellGeneration` (default `true`). The all-cells scan
is still there as a fallback and for the rare scene where the tool is not small.

---

## 6.9 The duplicate problem

A triangle that spans 8 cells gets inserted into all 8. If both a tissue and a
blade triangle span an overlapping set of cells, the *same* pair (4200, 7) gets
generated from *every* shared cell — maybe 4 times. We only want it once.

The fix is the **pair hash table** (`pairHashKeys`), covered in detail in file
07. The short version: each pair is packed into a single 64-bit number and
inserted into a hash table using `atomicCAS`. The first thread to insert a given
pair "wins" and adds it to the output; later threads trying the same pair see
it's already there and silently skip. This deduplication happens entirely on
the GPU with no CPU involvement.

In the benchmark: 2,304 raw candidates → 624 unique after dedupe. That
"duplicate reduction ratio" of ~0.73 is logged in the CSV.

---

## 6.10 Why a *dense* grid (vs alternatives)

The grid is called "dense" because it allocates *every* cell up front (all
32,768), whether or not they contain anything. The alternative — a "sparse"
structure or a tree (BVH) — only stores occupied regions.

- **Dense grid pro:** dead simple, perfectly parallel. Every thread computes a
  cell index with a little arithmetic — no tree traversal, no pointer chasing.
  GPUs love this.
- **Dense grid con:** wastes memory on empty cells, and if the scene is huge
  the cell count explodes.

For near-axis-aligned surgical geometry of this size, the dense grid is a great
fit, which is why the project chose it. The CPU SOFA path uses a BVH tree
instead, which requires sequential traversal — exactly the thing GPUs are bad
at. Replacing the tree with the dense grid is what unlocks the parallelism.

---

## 6.11 Summary

```text
The dense grid = a 64×8×64 box of cells covering scene space.
Each cell has a tissue bucket and a tool bucket.
A triangle is inserted into every cell its (inflated) bounding box overlaps.
Cells with BOTH tissue and tool triangles generate candidate pairs.
By default, only the ~30 tool-occupied (mixed) cells are scanned, not all
  32,768 — the list of those cells is built for free during the blade insert.
Duplicate pairs are removed with a GPU hash table.
Result: ~624 unique candidate pairs instead of 153,600 brute-force checks.
```

Now you understand the structure. Next we walk the seven CUDA kernels that
build and use it. Go to [07_phase3_kernels.md](07_phase3_kernels.md).
