# 12 — Glossary & Cheat-Sheet

Every term used in this tutorial, defined in one place. Skim it once, then come
back when a word trips you up.

---

## Core SOFA terms

**SOFA** — Simulation Open Framework Architecture. The C++ physics-simulation
engine this project plugs into.

**Scene** — a tree of nodes and components describing a simulation, written in a
Python `createScene(root)` function.

**Node** — a branch in the scene tree. Holds components and child nodes.

**Component** — a unit of behavior (positions, topology, a collision detector, a
solver, etc.). You assemble components like LEGO.

**`MechanicalObject`** — the component that stores vertex positions (and
velocities/forces). It's "dumb" — just points, no connectivity. Holds the
"degrees of freedom."

**Degrees of freedom (DoF)** — the numbers the simulation can change. One 3D
vertex = 3 DoFs (x, y, z).

**`MeshTopology`** — the component that stores connectivity: which vertex indices
form each triangle. Static in this project (never changes during a run).

**Topology** — the connectivity of a mesh (the index lists), as opposed to the
geometry (the positions).

**`TriangleCollisionModel`** — a component flagging an object as collidable via
its triangles. With a `CudaVec3f` `MechanicalObject` underneath, SOFA upgrades it
to a `CudaTriangleCollisionModel`.

**`CudaPointCollisionModel`** — a collision model representing a set of loose
points (a point cloud) rather than triangles. Used in the cross-model
vertex-triangle path.

**Data field** — a named, configurable value on a component (e.g.
`contactDistance`). Set from Python, read in C++ via `.getValue()`. Declared with
`initData`.

**`DefaultAnimationLoop`** — the clock; its `step()` drives one frame.

**`CollisionPipeline`** — SOFA's collision orchestrator; calls broad phase, then
narrow phase, then response each frame.

**`RequiredPlugin`** — loads a shared library, making its components available by
name.

**`ObjectFactory`** — SOFA's registry mapping component-name strings to C++
classes. Populated by `RegisterObject` when a plugin loads.

**`DetectionOutput`** — SOFA's standard contact record: two points, a normal, a
distance value.

**`TDetectionOutputVector<A, B>`** — a typed list of `DetectionOutput`s for a
specific pair of collision-model types. Templated, so it works for any pair
(e.g. point-vs-triangle).

**`BarycentricMapping`** — a SOFA component that transfers forces from a surface
mesh to an underlying volume mesh using barycentric weights. (Relevant to the
future hybrid-physics use case, not the benchmark.)

---

## Geometry terms

**Vertex** — a point in 3D space, `(x, y, z)`.

**Triangle** — a flat face defined by 3 vertex indices.

**Mesh** — vertices + topology (connectivity).

**Surface mesh** — hollow; just the outer triangle skin.

**Volume mesh** — solid; the interior is filled with tetrahedrons.

**Tetrahedron** — a 3D pyramid with 4 vertices; the building block of volume
meshes (and the basis for FEM physics, which the benchmark omits).

**AABB** — Axis-Aligned Bounding Box: the smallest box (aligned to the X/Y/Z
axes) that contains a shape. Cheap to compute and compare.

**Feature** — a vertex, edge, or face of a triangle.

**VF / FV / EE** — Vertex-Face / Face-Vertex / Edge-Edge: the kinds of
closest-feature pairings between two triangles.

**Barycentric coordinates** — three weights `(u, v, w)` with `u+v+w=1` that
express a point inside a triangle as a blend of its corners. Let a solver
distribute a contact force to the triangle's vertices.

**Contact** — a record that two primitives are touching/close: points, normal,
distance (and, in this project, barycentric weights).

**`ProximityContact` / `DeviceProximityContact`** — this project's rich contact
record (primitive IDs, feature kind, barycentrics, points, normal, signed
distance). The `Device` version is the GPU-memory layout.

**Normal** — a unit vector giving the direction to push two objects apart.

**Signed distance** — how far apart (positive) or how deep into each other
(negative) two features are.

**Contact distance** — the proximity threshold. Triangles within this distance
count as a contact. `0.03` in the benchmark.

---

## CPU/GPU/memory terms

**CPU** — a few powerful cores; fast at branchy sequential logic.

**GPU** — thousands of weak cores; fast at the same simple task done in parallel.
This project's GTX 1650 Ti has 1024 CUDA cores across 16 SMs.

**SM (Streaming Multiprocessor)** — a cluster of GPU cores. The 1650 Ti has 16.

**VRAM** — the GPU's own memory (video RAM).

**System RAM** — the CPU's normal memory.

**PCIe bus** — the motherboard cable connecting CPU and GPU. Moving data across
it is slow relative to compute.

**H2D (Host-to-Device)** — copying CPU RAM → GPU VRAM.

**D2H (Device-to-Host)** — copying GPU VRAM → CPU RAM.

**Zero-copy** — reading data where it already lives (in VRAM) via a device
pointer, instead of copying it. The core optimization of this project.

**`CudaVec3f`** — a `MechanicalObject` template that allocates positions in VRAM
as 32-bit floats. Enables zero-copy.

**`Vec3d`** — the CPU template: positions in System RAM as 64-bit doubles.

**`deviceRead()`** — a SofaCUDA function returning a raw pointer into the VRAM
position buffer. The zero-copy entry point.

**`reinterpret_cast`** — reinterpret the same bytes as a different C++ type
without converting. Used to view SOFA's VRAM positions as the backend's
`TriangleVertex` (the layouts are asserted identical).

---

## CUDA terms

**CUDA** — NVIDIA's framework for GPU programming.

**Kernel** — a function (`__global__`) that runs on the GPU, executed by many
threads in parallel.

**Thread** — one running copy of a kernel. Gets a unique ID to pick its data.

**Block** — a group of threads. **Grid** — a group of blocks. Launch shape is
`kernel<<<numBlocks, threadsPerBlock>>>`. This project uses 256 threads/block.

**Launch** — starting a kernel. **Non-blocking / asynchronous** — the CPU fires
the launch and continues without waiting.

**`atomicAdd`** — safely add to a shared counter when many threads hit it at
once; returns the pre-add value (a unique slot number).

**`atomicCAS` (Compare-And-Swap)** — atomically: if a slot equals `expected`,
set it to `new`; return the old value. Used for lock-free hash insertion.

**`cudaMalloc`** — allocate VRAM. Slow; this project does it once (Phase 0) and
reuses.

**`cudaMemcpy` / `cudaMemcpyAsync`** — copy across PCIe (blocking / non-blocking).

**`cudaMemset`** — hardware-fast "fill VRAM with a byte value." Used to clear the
hash table.

**`cudaDeviceSynchronize`** — make the CPU wait until the GPU finishes queued
work. Avoiding unnecessary syncs is a major theme.

**CUDA event** — a timestamp marker dropped into the GPU command stream; used to
measure real GPU kernel time (`cudaEventElapsedTime`).

**Pinned (page-locked) host memory** — host memory that transfers across PCIe
faster. Used for the batched counter readback.

**Occupancy** — how many warps are active on an SM vs the max. Limited by
registers/thread (the FBP kernel's 68 registers cap it at ~50%).

**Register** — fast per-thread storage. More registers/thread = fewer threads
fit = lower occupancy.

---

## This project's specific terms

**Broad phase** — the cheap pass that finds which object pairs might collide.
Here: `GpuCollisionBroadPhase`.

**Narrow phase** — the expensive pass that finds exact contacts. Here:
`GpuCollisionNarrowPhase`.

**Dense grid** — a regular 3D grid of cells covering scene space. Triangles are
binned into cells; only same-cell triangles are checked. 64×8×64 = 32,768 cells
in the benchmark.

**Cell** — one box of the dense grid. Holds a tissue bucket and a tool bucket.

**Bucket** — a per-cell list of triangle (or vertex) IDs, one for "tissue", one
for "tool."

**Tissue / tool** — the two sides of a collision. "Tissue" is the first surface,
"tool" is the second (the blade, or a point cloud). Naming is by role, not
material.

**Candidate pair** — a `(tissue triangle, tool triangle)` combination generated
from a shared cell; a pair worth checking precisely.

**Dedupe (deduplication)** — removing duplicate candidate pairs (generated when
triangles span multiple shared cells) via a GPU hash table.

**`pairHashKeys`** — the GPU hash table used for lock-free dedupe.

**`encodeCandidatePair`** — packs `(tissueId, toolId)` into one 64-bit number:
`(tissueId << 32) | toolId`.

**Topology cache (`m_triangleTopologyCache`)** — a CPU-side cache of the flat
triangle-index array per model, so it isn't rebuilt every frame. Keyed by model
pointer, validated by an FNV-1a hash.

**FNV-1a hash** — the algorithm that fingerprints a topology into one 64-bit
number, so unchanged topology can be detected and re-upload skipped.

**`surfaceId`** — a unique ID per collision model (its pointer cast to uint64).
Used to cache device-side indices and to detect self-collision (matching IDs).

**`topologyVersion`** — the topology hash; the device-index cache re-uploads only
when it changes.

**`TriangleIndexedSurface`** — the struct passed to the backend: a device
position pointer + a host index pointer + counts + surfaceId + version.

**`PointCloudSurface`** — the point-cloud equivalent for the vertex-triangle
path: just device positions + count + surfaceId.

**`DenseGridWorkspace`** — the singleton owning all reusable GPU buffers (grid,
buckets, hash table, candidate list, contacts, events). Allocated once, reused
every frame.

**`DenseGridConfig`** — the bundle of grid parameters (bounds, resolution,
contact distance, capacities, flags) passed from the scene to the backend.

**Over-launch** — launching a fixed, generous number of threads (65,536) and
having each exit early if it has no work, to avoid a synchronous readback of the
exact count.

**Own-corner exclusion** — in self-collision, skipping (triangle, vertex) pairs
where the vertex is one of the triangle's own corners (would falsely report
distance 0).

**Detection-only mode** — finding contacts but not responding to or drawing them
(`copyContactsToHost=False`, no response components). Used for clean benchmarks.

**Feature-based proximity (FBP)** — the closest-feature collision method
(6 VF + 9 EE tests, barycentric output) that replaced the older SAT yes/no test.

**SAT (Separating Axis Theorem)** — the legacy boolean "do these triangles
cross?" test. Superseded by FBP for the surgical use case.

**Ericson 5.1.5 / 5.1.9** — the textbook algorithms for closest-point-on-triangle
and closest-points-between-segments, implemented as device functions.

---

## Per-frame phases (this tutorial's structure)

| Phase | Name | File | What |
|---|---|---|---|
| 0 | Setup | 03 | Plugin load, VRAM allocation, caches, workspace (once) |
| 1 | Broad phase | 04 | Find object pairs to check |
| 2 | Narrow prep | 05 | Zero-copy extraction, build config |
| 3 | Kernel cascade | 07–09 | The 7 GPU operations |
| 4 | Sync & output | 10 | Don't wait; log timing |
| — | Profiling | 11 | How the stopwatch works |

---

## The four collision paths

| Path | When | Tests per pair | All contacts are |
|---|---|---|---|
| Exact-contact (SAT, legacy) | default | boolean | (yes/no) |
| Tri-tri FBP | `useFeatureBasedProximity` | 15 (6 VF + 9 EE) | VF, FV, or EE |
| V-t self-collision | + `useVertexTriangleProximity` + `(cm,cm)` pair | 1 | VertexFace |
| V-t cross-model | + `useVertexTriangleProximity` + (point, triangle) pair | 1 | VertexFace |

---

## Key numbers for the one-tissue/one-blade benchmark

| Quantity | Value |
|---|---|
| Tissue triangles | 12,800 |
| Tissue vertices | 6,561 |
| Blade triangles | 12 |
| Grid cells | 32,768 (64×8×64) |
| Contact distance | 0.03 |
| Raw candidate pairs | ~2,304 |
| Unique candidate pairs | ~624 |
| Contacts (FBP) | ~56 (all edge-edge) |
| Fast-path FPS | ~775 |
| Fast-path H2D / D2H bytes | 0 / 0 |
| Kernel launches / memsets | 7 / 1 (FBP path) |

---

## Where to go deeper

- `guide/architecture.md` — the full reference (every struct, every kernel,
  every dispatch branch, with source line references).
- `guide/plan.md` — the project's build history, why each decision was made.
- `guide/setup.md` — how to build, run, and profile on your machine.
- `reports/gpu_collision_phase11_12_kernel_profile_20260525.md` — the Nsight
  kernel-level findings.

You've now traced the entire pipeline from the Python scene to the GPU contacts
and back to the CSV. Re-read any file that's still fuzzy — the concepts compound.
