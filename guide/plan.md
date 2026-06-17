# Implementation Plan

This is the implementation roadmap and historical log for the GPU SOFA
collision pipeline. Every phase has a status, a "why this was needed"
paragraph, the measured outcome, and (where rejected) an explanation of the
alternative that was considered. New work appends; finished phases stay so
their rationale remains discoverable.

Last refreshed: 2026-06-09 (full-suite re-benchmark + experimental hash +
prefix-sum broad cull, §5.18).

---

## 1. Mission

Make SOFA collision detection fast enough for surgical simulation on the
target hardware (NVIDIA GeForce GTX 1650 Ti laptop, WSL2). Specifically:

- **One-tissue / one-blade scene**: 10⁴ triangles per side, simulating a
  blade cutting tissue.
- **Large tissue / blade scene**: 5×10⁴+ triangles per side.

Primary optimization metric: integrated SOFA wall time. Secondary: emit
contact data in a form that a CUDA constraint solver can consume directly
(barycentric weights + signed distance + normal, kept on device).

The 2026-05-25 production fast path (with Phase 15 tool-active-cell
generation now default-on) delivers:

| Scene | FPS | Narrow wall | H2D / D2H bytes per frame |
|---|---:|---:|---:|
| one-tissue / one-blade, **tri-tri FBP** | **~940** | **~0.5 ms** | **0 / 0** |
| 2-layer slab, **v-t self-collision** | ~1400 | ~0.5 ms | 0 / 16 |
| tissue + tool cloud, **v-t cross-model** | ~1900 | ~0.4 ms | 0 / 16 |
| large-tissue / subdivided-blade, tri-tri FBP | ~116 | ~4.1 ms | 0 / 16 |

(FPS figures are thermally sensitive on this laptop; ranges given. The
one-tissue tri-tri FBP path was ~775 FPS before Phase 15 and ~940 after.)

All four paths share the same dense-grid broad cull. The differences are in
the narrow-pass kernel and the host-side dispatch. **Phase 15 (§5.15)
collapsed the candidate-generation kernel from ~300 µs to ~8 µs, making it
no longer the bottleneck.**

**2026-06-09 re-benchmark (`reports/benchmark_suite_20260609.md`).** The full
suite was re-run on the same hardware. Contact counts are **bit-identical** to
the numbers above on every scene (56 EE small, 8018 large, 2700 VF self,
254 VF cross). Narrow-phase wall and kernel times are **at or better than**
the 2026-05-25 figures (e.g. small fast path 0.47 ms wall / 0.28 ms kernel vs
0.56 / 0.35); whole-pipeline FPS varies within this laptop's known
thermal/power-state band (§7.2). A new **experimental** spatial-hash +
prefix-sum broad cull (§5.18, opt-in, default-off) adds +11.8 % FPS / −15 %
narrow wall on a large-tissue + large-tool scene with bit-identical contacts.

---

## 2. Current source of truth

The production narrow-phase architecture (2026-05-25) is the **direct
indexed dense-grid broad cull** with one of four narrow-pass kernels chosen
by Data fields on `GpuCollisionNarrowPhase`:

```text
CudaTriangleCollisionModel
  └─► GpuCollisionNarrowPhase::extractCudaIndexedSurface
       (and/or extractCudaPointCloud for v-t cross-model)
       └─► TriangleIndexedSurface / PointCloudSurface
            └─► indexed dense-grid broad cull (cached device topology + deviceRead() positions)
                 └─► one of:
                      computeDenseGridIndexedTriangleContacts   (SAT exact-contact, legacy)
                      computeFeatureBasedProximityContacts      (tri-tri FBP, Phase 11)
                      computeFeatureBasedVertexTriangleContacts (v-t, Phase 12, self + cross-model)
```

Default Data field configuration for production runs:

```text
useDenseGrid                              = true
useIndexedDenseGridInput                  = true
useDirectDevicePositions                  = true
cacheTriangleTopology                     = true
useGpuHashDedupe                          = true
useFeatureBasedProximity                  = (opt in for FBP)
useVertexTriangleProximity                = (opt in with FBP for v-t)
copyContactsToHost                        = false       ← detection-only
proximityReadContactCounter               = false       ← unless validating
proximityKeepContactsOnDevice             = true        ← constraint-solver-ready
readCountersWhenContactsStayOnDevice      = false       ← unless validating
computeDeviceContactsWhenContactsStayOnDevice = false   ← unless using legacy exact-contact device-side
useToolActiveCellGeneration               = true        ← DEFAULT ON (Phase 15, §5.15)
compactActiveCells                        = false       ← regressed on GTX 1650 Ti, superseded by the above
batchTriangleInsert                       = false       ← regressed; mutually exclusive with useToolActiveCellGeneration
```

`useToolActiveCellGeneration` is the Phase 15 optimization and is **default
on** (4.3× on one-tissue, 1.08× on large-tissue, bit-identical output,
never a regression — §5.15).

The older `compactActiveCells` and `batchTriangleInsert` remain runtime
switches but stay **off**: the 2026-05-14 profile showed both regressed
(one-tissue narrow wall ~1.18 → ~5.09 ms). `compactActiveCells` is
effectively superseded by `useToolActiveCellGeneration`, which achieves the
active-cell idea without the separate full-grid scan that made
`compactActiveCells` slow.

---

## 3. Goals and non-goals

**In scope:**

- GPU-first collision detection for SOFA CUDA triangle and point models.
- Detection-only mode with zero per-frame H2D/D2H traffic.
- Optional barycentric proximity output for a constraint solver.
- Backward-compatible exact-contact path so legacy scenes keep working.

**Not in scope (deliberate):**

- CPU collision response. We hand off to SOFA's existing `BVHNarrowPhase` /
  contact-manager pipeline when `copyContactsToHost=true`, but we do not
  optimize that side.
- Continuous collision detection (CCD). The current path is discrete.
- Cutting / tearing. The narrow phase produces contact data; topology
  edits are the responsibility of a downstream simulator (Carving plugin,
  etc.).
- Constraint solving. We produce solver-ready contact data; the solver
  itself is separate.

---

## 4. Phase status

Long-form phase history. Use the table to navigate; each row points to a
section below for the "why" and the measured outcome.

| # | Phase | Status | Outcome anchor |
|---|---|---|---|
| 0 | Baseline + profiling | Done | §5.0 |
| 1 | Dense-grid CUDA backend | Done | §5.1 |
| 2 | Direct GPU collision scene path | Done | §5.2 |
| 3 | GPU hash dedupe | Done | §5.3 |
| 4 | Topology cache | Done | §5.4 |
| 5 | Counter/readback fast path | Done | §5.5 |
| 6 | Backend pack elimination | Done | §5.6 |
| 7 | Launch/stat trimming | Done | §5.7 |
| 8 | True GPU-native indexed topology | Done | §5.8 |
| 9 | Memory-layout redesign | Partly done | §5.9 |
| 10 | Exact-contact tuning | Deferred | §5.10 |
| 11 | Feature-based proximity (VF + EE) | Done end-to-end | §5.11 |
| 12 | Vertex-triangle proximity | Done end-to-end (self + cross-model) | §5.12 |
| 13 | Atomic / lock optimization | Decided against | §5.13 |
| 14 | Profiling reshape | Done | §5.14 |
| 15 | Tool-active-cell candidate generation | **Done — DEFAULT ON 2026-05-25** | §5.15 |
| 16 | Workspace-cached broad-cull CUDA events | **Done 2026-05-25** | §5.16 |
| 17 | FBP/v-t grid-stride correctness fix | **Done 2026-05-25** | §5.17 |
| 18 | Spatial-hash + prefix-sum broad cull | **Experimental, opt-in, default-off (branch)** | §5.18 |

---

## 5. Phase notes

### 5.0  Baseline and profiling

**Why this was needed.** Before any GPU work, we needed to know what the
unmodified CPU SOFA pipeline cost on the target hardware. This established
the budget for the rest of the project.

**What was done.** Captured per-frame timings, kernel launch counts, and
H2D/D2H byte counters for unmodified SOFA collision pipelines on the
benchmark scenes. Logs are in `output/benchmark_logs/<early-dated-dirs>`.

**Outcome.** Verified that the CPU narrow phase was the integrated bottleneck
and worth GPU porting.

### 5.1  Dense-grid CUDA backend

**Why.** The first cut needed any GPU narrow phase at all — a place to put
the kernel code, an API boundary, and a workspace abstraction. We chose a
dense uniform 3D grid because (a) it is simple, (b) it works well for the
near-axis-aligned surgical geometry, and (c) cells naturally produce
candidate pairs without an explicit BVH traversal.

**What was done.** Added `cuda/GpuCollisionBackend.cu` with the public
entries declared in `GpuCollisionBackend.h`. Introduced `DenseGridWorkspace`
as the singleton owner of reusable GPU allocations.

**Outcome.** Functional GPU narrow phase behind a stable C++ ABI. CPU
fallback path retained.

### 5.2  Direct GPU collision scene path

**Why.** Per-frame `cudaMemcpy(H2D)` of tissue vertex positions was the
single biggest wall-time cost in the early bring-up — ~5 ms out of a 7 ms
narrow wall. The tissue positions already live on the GPU inside a SOFA
`MechanicalObject<CudaVec3f>`; we just had to stop round-tripping them.

**What was done.** Benchmark scenes use `template='CudaVec3f'` for the
mechanical object. The `extractCudaIndexedSurface` helper now reads the
SOFA CUDA vector's `deviceRead()` pointer and passes it straight through
as `TriangleIndexedSurface.devicePositions`. The backend uses that pointer
verbatim.

**Outcome.** `avg_host_to_device_bytes = 0` for position uploads in the
benchmark scenes. The reuploads only happen for fallback paths where the
collision model is not CUDA-templated.

### 5.3  GPU hash dedupe

**Why.** Candidate pair generation can emit the same pair from multiple
cells. The original implementation collected all candidate pairs to host,
sorted, and de-duplicated. That added a D2H+H2D round-trip per frame. We
needed dedupe to stay on the GPU.

**What was done.** A power-of-two `pairHashKeys` table sized to
`nextPowerOfTwo(maxCandidatePairs * 2)`. The
`generateDenseGridUniqueCandidatePairsKernel` does open-addressing
`atomicCAS` insertion — first inserter wins. Duplicate inserts skip the
output append.

**Outcome.** No host sort/unique. Probe overflow is tracked
(`hashDedupeProbeOverflowCount`) but does not occur in measured scenes
because the table is sized 2× larger than the maximum candidate count.

### 5.4  Topology cache

**Why.** The SOFA triangle vector returns `Triangle` objects with three
indices; we need a flat `(v0, v1, v2, v0, v1, v2, ...)` `uint32_t` array
to feed the backend. Building this every frame from a stable mesh is
wasted work.

**What was done.** `m_triangleTopologyCache` in `GpuCollisionNarrowPhase`,
keyed by `CudaTriangleCollisionModel*`. Each cache entry holds the flat
index array, vertex/triangle counts, and an FNV-1a topology hash. The cache
is invalidated when (a) size changes or (b) `validateTriangleTopologyCache=true`
and the hash changes. The `seenThisFrame` flag drives a per-frame prune so
removed collision models don't leak.

**Outcome.** `avg_narrow_backend_triangle_pack_ms ≈ 0` in steady-state
scenes. Position scratch is similarly cached but not used in the production
direct-device path (`useDirectDevicePositions=true`).

### 5.5  Counter / readback fast path

**Why.** Even after positions stayed on GPU, the backend was reading back
small counters every frame (`rawCandidateCount`, `uniqueCandidateCount`,
`overflowCount`, contact counts). Each `cudaMemcpy` synchronizes the host
with pending GPU work — five small reads at ~50 µs each was adding ~250 µs
of pure sync per frame for benchmark scenes that needed nothing of the
data.

**What was done.** Added explicit Data fields:

- `readCountersWhenContactsStayOnDevice` (default false in detection-only)
- `computeDeviceContactsWhenContactsStayOnDevice` (default false)
- `copyContactsToHost` (default true, set false in detection-only scenes)

When all three are false, the backend skips every counter read and every
contact-array download. Counters and contacts stay device-side.

**Outcome.** `avg_device_to_host_bytes = 0` in the detection-only fast path.

### 5.6  Backend pack elimination

**Why.** The early backend received a `std::vector<TrianglePrimitive>` —
each triangle was packed with three full `TriangleVertex` copies (9 floats
+ 1 index = 40 bytes per triangle, vs 12 bytes for just the indices). For
12 800 triangles that's 470 KB of redundant work per frame.

**What was done.** Added `TriangleIndexedSurface` (positions + indices) and
`computeDenseGridIndexedTriangleContacts(...)` as the indexed-input
counterpart. The CUDA `indexedTriangleAt(positions, indices, triId)` helper
reads through the index buffer at the load site. Asserted
`sizeof(TrianglePrimitive) == sizeof(DeviceTriangle)` so the legacy packed
path remains a zero-copy upload.

**Outcome.** `avg_narrow_backend_triangle_pack_ms ≈ 0.001 ms` (it's down to
the cost of the static asserts and bounds checks; basically free).

### 5.7  Launch / stat trimming

**Why.** The detection-only narrow path used to make ~12 kernel launches
per frame, including separate AABB-build kernels, separate insert kernels,
and separate stat-reset kernels. On the GTX 1650 Ti each launch costs
~5-10 µs of latency, so reducing launch count was worth real time.

**What was done.**

- Fused per-cell `DeviceCellBucket` reset + counter reset + (optional)
  pair-hash reset into one `resetDenseGridKernel`. Then measured that
  `cudaMemset(pairHashKeys, 0xff)` was actually faster for the pair hash
  on this GPU, so we left that as a memset.
- Fused AABB build + cell insertion in
  `insertIndexedTrianglesKernel` (no intermediate AABB array on the fast
  path).
- Skipped `DeviceDenseGridStats` writes when stats are not read back (the
  kernel takes a nullable stats pointer).

**Outcome.** Fastest detection-only indexed path = 5 kernel launches + 1
`cudaMemset`. FBP adds 2 more launches (resetProximityCounters + FBP
kernel) for a total of 7.

### 5.8  True GPU-native indexed topology

**Why.** Even after Phase 5.6 reduced per-triangle pack overhead, the
indexed path was still uploading positions to the workspace's
`indexedTissuePositions` buffer on every frame for any non-direct input.
We wanted to skip that upload entirely for `CudaVec3f`-backed scenes.

**What was done.** Added `useDirectDevicePositions=true` Data field. When
set, `extractCudaIndexedSurface` puts the SOFA CUDA vector's `deviceRead()`
pointer directly into `TriangleIndexedSurface.devicePositions` and leaves
`positions=nullptr`. The backend then uses that pointer in place of the
workspace's `indexedTissuePositions`. Index buffers stay device-cached via
`(surfaceId, topologyVersion)` matching.

**Outcome.** Fastest detection-only path achieves **0 H2D bytes per frame**
in steady state. Topology indices are uploaded once at scene start.

### 5.9  Memory-layout redesign

**Why.** The AABB-then-insert split kept full AABB arrays in device memory
that were only read once. Memory traffic was a measurable cost in the
candidate-generation kernel.

**What was done (and what didn't work).**

- AABB compute + cell insertion are now fused inside
  `insertIndexedTrianglesKernel` — no intermediate AABB array on the fast
  path.
- `compactActiveCells` was implemented to scan the grid, build a compact
  array of "mixed" cells (cells with both tissue and tool present), and
  launch candidate generation only over those cells. **Measured: worse**.
  On the GTX 1650 Ti one-tissue scene, the compact pass took 0.4 ms vs the
  full scan it replaced taking 0.3 ms because the scene already has high
  cell utilization. Disabled by default.
- `batchTriangleInsert` combined the two insertion launches into one. Same
  story: measured regression because the SM array (16 SMs) is small enough
  that launch latency for two small kernels is dominated by the larger
  first one anyway.
- SoA AABB layout (separate `minX`, `minY`, `minZ`, `maxX`, `maxY`, `maxZ`
  arrays) — deferred. Would help only on a workload where AABBs are read
  multiple times, which the fused-path no longer does.

**Outcome.** Direct indexed dense-grid is the source of truth. Optional
flags exist for the experimental layouts but the defaults are unchanged.

### 5.10  Exact-contact tuning

**Why and why-not.** The legacy `exactTriangleContactKernel` runs SAT-style
boolean intersection. Its register pressure pushes occupancy low. We
considered splitting it into smaller helper kernels to reduce live ranges.

**Status.** Deferred. The kernel is rarely the integrated wall-time
bottleneck (a few hundred µs on the worst scene), and Phase 11 (FBP)
replaced it with something fundamentally different for the constraint-
solver use case. Tuning the SAT kernel further is low priority.

### 5.11  Feature-based proximity (VF + EE)

**Why.** The SAT exact-contact kernel produces "are these two triangles
intersecting?" + a contact point on the deepest overlap axis. A constraint
solver needs more: the **closest pair of features** (vertex-face or
edge-edge) with **barycentric weights** so the solver can distribute
contact forces back to the underlying DOFs. The SAT kernel cannot produce
barycentrics. It also lacks gradient continuity — a sliding contact will
"jump" between SAT axes as the geometry moves.

**What was done.** Implemented `computeFeatureBasedProximityContacts` in
the CUDA backend:

- Three `__device__` math helpers: `closestPointOnTriangleBary` (Ericson
  5.1.5), `closestPointSegmentSegment` (Ericson 5.1.9),
  `reconstructFromBary`.
- `featureBasedProximityKernel`: one thread per candidate pair, runs 6 VF
  tests (3 verts of A vs face B, 3 verts of B vs face A) and 9 EE tests
  (3×3 edges). Keeps the closest feature pair if distance ≤
  `contactDistance`. Emits one `ProximityContact` per pair with feature
  kind, barycentric weights, world-space contact points, normal, and signed
  distance.
- Gated by `useFeatureBasedProximity` Data field (default false).

**The sync-overhead lesson.** The first cut measured 109 FPS on the
one-tissue scene — much slower than the 633 FPS baseline. Root cause was
not the kernel (0.54 ms) but the **synchronous counter readbacks**: the
broad cull was forced to read `uniqueCandidateCount` so the wrapper could
size the FBP launch, and the validation path was issuing 5 separate
`cudaMemcpy` calls for the per-class counters. Total sync overhead:
~7-8 ms per frame.

The fix (2026-05-25):

- **Removed the broad-cull counter readback** by switching the FBP kernel
  to over-launch (`gridDim = min(maxCandidatePairs, 65536) / 256` blocks,
  each thread guards `if (tid >= *candidatePairCount) return`). The kernel
  reads the candidate count from L2 at the first warp's load. Empty blocks
  exit in ~1 instruction.
- **Batched the 5 counter reads** into 5 `cudaMemcpyAsync` into a single
  pinned host buffer + one `cudaDeviceSynchronize`.
- **Made the FBP `cudaEvent` pair persistent on the workspace**, created
  lazily once. No per-frame `cudaEventCreate`/`Destroy`.
- **Skipped the FBP `cudaEvent` pair entirely** when no counter readback
  is needed (the production fast path).

**Outcome.** Production fast path: **775 FPS**, 0.67 ms narrow wall,
0.51 ms narrow kernel, 0.19 ms host sync. Matches the pre-FBP detection-
only baseline of 633 FPS — proving that FBP work overlaps the next frame's
host scheduling.

Validation mode (counter readback on): 449 FPS, 1.49 ms narrow wall, FBP
kernel itself 0.07 ms, 56 contacts emitted, all EdgeEdge — the closest
feature for a flat tissue against a small blade box is always edge-edge
at 0.03 m distance.

### 5.12  Vertex-triangle proximity (self-collision + cross-model)

**Why.** Surgical-simulation use cases beyond tri-tri:

1. **Tissue self-collision** during folding, cutting, or tearing — a
   tissue mesh's own vertices come close to its own non-adjacent triangles.
2. **Cross-model tool puncture** — a rigid tool whose tip is a point
   cloud (or a soft-body whose surface is sampled as points) coming into
   contact with a tissue triangle mesh.

A general vertex-triangle kernel handles both with the same math:
`closestPointOnTriangleBary` per (vertex, triangle) candidate pair.

**What was done.**

- New CUDA kernels:
  - `insertIndexedPointsKernel` inserts each vertex as a 1-point AABB into
    the tool-side cell buckets, using the same dense-grid cell-walk as the
    triangle insertion.
  - `featureBasedVertexTriangleProximityKernel` runs
    `closestPointOnTriangleBary` per (triangleId, vertexId) candidate. An
    optional `selfCollisionVertexExclusionStride` argument skips pairs
    where the vertex is one of the triangle's three corner indices.
- New backend entry `computeFeatureBasedVertexTriangleContacts(point, tri, ...)`
  that runs the broad cull and the v-t narrow pass end to end. Reuses the
  workspace, the proximity buffer, the pinned counter buffer, and the
  persistent CUDA events.
- New extractor `extractCudaPointCloud` that pulls device positions from a
  `CudaPointCollisionModel`'s `MechanicalState`.
- Dispatch in `GpuCollisionNarrowPhase::endNarrowPhase`:
  - Self-collision branch: when `pair.first == pair.second` and
    `useVertexTriangleProximity=true`, mirror the cached
    `TriangleIndexedSurface` into a `PointCloudSurface` sharing the same
    `surfaceId` (which trips own-corner exclusion).
  - Cross-model branch (at the top of the per-pair loop): when one side
    `dynamic_cast`s to `CudaPointCollisionModel` and the other to
    `CudaTriangleCollisionModel`, extract both and call the v-t backend.
    SurfaceIds differ so own-corner exclusion stays off.
- Two new SOFA scenes:
  - `testscenes/self_collision_vertex_triangle.py`: a two-layer slab
    with `selfCollision=True`. Each top vertex is 0.05 above a bottom
    triangle; `contactDistance=0.06` makes every top vertex emit one or
    more contacts.
  - `testscenes/cross_model_vertex_triangle.py`: a tissue triangle
    grid + a small tool point cloud hovering above it.

**Outcome.** Verified end-to-end 2026-05-25:

| Scene | FPS | narrow_wall | contacts | VF / FV / EE |
|---|---:|---:|---:|---:|
| v-t self-collision (2-layer slab, 512 verts, 900 tris) | 1385 | 0.66 ms | 2700 | 2700 / 0 / 0 |
| v-t cross-model (tissue 41×41, tool 8×8) | 1968 | 0.37 ms | 254 | 254 / 0 / 0 |

All emitted contacts are `VertexFace` by construction (the v-t kernel only
emits that class). 2700 contacts ≈ 5.3 per vertex — each vertex's inflated
AABB spans multiple grid cells, the kernel emits one contact per candidate
that passes the distance filter, and own-corner exclusion suppresses
adjacent-corner false positives. 254 contacts ≈ 4 per tool vertex,
matching a tool grid that doesn't align with the tissue grid (each tool
point projects onto multiple proximate tissue triangles).

The cross-model path uses `surfaceId` distinctness (point-model pointer vs
triangle-model pointer) to keep own-corner exclusion OFF, so a tool vertex
is never wrongly filtered out by accident.

### 5.13  Atomic / lock optimization

**Status.** Decided against on 2026-05-24 after measuring.

**Why considered.** Intuitively, the insertion kernels use `atomicAdd` on
per-cell bucket counters, and the pair-hash uses `atomicCAS`. Atomics are
the canonical CUDA scaling concern. We considered:

1. Warp-aggregated atomics (CUB style) — one atomic per warp per cell.
2. Two-pass prefix-sum scatter — eliminate atomics entirely.

**Why rejected.** Re-imported the 2026-05-17 Nsight Compute report with
focused metrics:

| Kernel | Duration | L2 atomic cycles | What dominates |
|---|---:|---:|---|
| `resetDenseGridKernel` | 2.5 µs | 0.0% | trivial |
| `insertIndexedTrianglesKernel` (blade, 12 tris) | 30.4 µs | 0.19% | launch latency |
| `insertIndexedTrianglesKernel` (tissue, 12 800 tris) | 28.6 µs | 19.1% | memory + atomics |
| `generateDenseGridUniqueCandidatePairsKernel` | 299.6 µs | 0.55% | compute, low SM throughput |

The kernel with appreciable atomic pressure (tissue insert, 19.1%) is
already only 28.6 µs out of a 360 µs broad-cull total. Replacing atomics
with warp-aggregated or prefix-sum would save at most ~10-14 µs out of a
~700 µs narrow path — a 2% gain at the cost of significant rewrite. The
dominant kernel (candidate generation, 300 µs) has only 0.55% atomic
cycles; atomics aren't its bottleneck either.

Optimization budget redirected to Phase 11 (FBP) and Phase 12 (v-t).

Report: `reports/gpu_collision_atomic_profile_finding_20260524.md`.

### 5.14  Profiling reshape

**Why.** After Phase 11 the CSV was misleading: the FBP kernel's time was
lumped into `denseGridExactContactMilliseconds` (despite no SAT kernel
running), the VF/FV/EE per-class breakdown was kernel-side but never
reached the CSV, and CUDA events were created/destroyed per frame.

**What was done.**

- New `featureBasedProximityKernelMilliseconds` field in
  `BackendExecutionStats` and `StageSnapshot`. FBP wrapper routes its
  `cudaEventElapsedTime` here.
- VF/FV/EE counters: kernel-side atomic adds → `BackendExecutionStats`
  fields → forwarded by `accumulateBackendStats` into `StageSnapshot` →
  serialized as 3 new CSV columns + 3 `avg_*` summary lines.
- Derived `hostSynchronizationMilliseconds = max(0, wall - kernel)` computed
  at the end of `endNarrowPhase`, written as a new CSV column.
- Persistent workspace-owned `fbpStartEvent` / `fbpEndEvent`. Created lazily
  in `ensureFbpEvents()`. Released in workspace destructor.
- Persistent pinned host counter buffer (`proximityCountersHostPinned`, 5×
  uint32). Five `cudaMemcpyAsync` + one `cudaDeviceSynchronize` replaces
  the previous five synchronous `cudaMemcpy` calls.
- New `proximityCounterReadbackInterval` Data field on
  `GpuCollisionNarrowPhase`. When > 0, the narrow phase reads counters
  every Nth frame; otherwise honors the explicit
  `proximityReadContactCounter` flag. Enables long profiling runs without
  paying sync every frame.

**Outcome.** CSV/summary now cleanly separate broad-cull time from FBP
kernel time, expose VF/FV/EE breakdowns, and report a derived host-sync
column for diagnosing the wall-vs-kernel gap.

### 5.15  Tool-active-cell candidate generation (DONE, DEFAULT ON, 2026-05-25)

**Status: implemented, verified correct on both small AND large scenes,
default ON.** Controlled by `useToolActiveCellGeneration` (now default
true). Scoped to the **indexed** contact path
(`computeDenseGridIndexedTriangleContacts`), which is the production
FBP/exact-contact broad cull. The packed fallback path ignores the flag
and stays correct (just unoptimized).

**Measured (one-tissue/one-blade, 2026-05-25 A/B, same cool GPU
back-to-back):**

```text
Fast path (production, no counter readback):
  baseline (off):  221 FPS, 4.00 ms narrow wall, 3.83 ms narrow kernel
  active  (on):    943 FPS, 0.56 ms narrow wall, 0.35 ms narrow kernel
  -> 4.3x FPS, 10.9x lower kernel time

Validation (counter readback on):
  baseline (off):  119 FPS, 56 contacts (all EE), overflow 0
  active  (on):    475 FPS, 56 contacts (all EE), overflow 0
  -> CONTACT COUNTS BIT-IDENTICAL

Nsight, generation kernel only:
  baseline: generateDenseGridUniqueCandidatePairsKernel       grid=32768, 300 us
  active:   generateActiveDenseGridUniqueCandidatePairsKernel grid= 1024, 7.9 us
  -> 32x fewer launched blocks, 38x faster generation
```

**Measured (large-tissue/blade, 79 520 collision elements, subdivided
blade, after the §5.17 grid-stride fix):**

```text
Validation (counter readback on):
  baseline (off):  108 FPS, 4.63 ms narrow wall, 3.91 ms narrow kernel
  active  (on):    116 FPS, 4.09 ms narrow wall, 3.33 ms narrow kernel
  -> 1.08x FPS; 8018 contacts (5397 VF / 880 FV / 1741 EE) BIT-IDENTICAL
     between off/on; unique_candidate_count=322560 identical; overflow 0
```

**Verdict and default decision.** The win is large on the one-tissue
scene (4.3×, where the tiny 12-triangle blade touches ~30 cells) and
modest on the large scene (1.08×, where the subdivided blade touches many
cells so the tool/tissue asymmetry is weaker and the FBP kernel itself
dominates). **It is never a regression** — contact output is bit-identical
on both scenes and the active list is structurally ≤ cellCount (grid-
strided), so the path can never do more work than the all-cells generator.
On that evidence the default was flipped to **ON** (`DenseGridConfig` and
the `GpuCollisionNarrowPhase` Data field both default true; the benchmark
scenes' env-flag defaults are now true).

Generation dropped from ~80% of GPU time to ~13% on the small scene. Total
GPU kernel time per frame ~380 us -> ~61 us. Report:
`reports/gpu_collision_phase15_16_optimization_20260525.md`.

**The `compactActiveCells` precedent (why this needed measuring).** The
2026-05-14 `compactActiveCells` experiment regressed because it ran a
*separate full-grid scan*. Phase 15 avoids that scan entirely (it builds
the list during the tool insert), which is why it wins where the older
approach lost. The large-scene A/B was the gate for flipping the default,
and it cleared.

---

**Original design notes (as-built):** This is the highest-value
optimization. It attacks the single dominant cost in the narrow phase.

**The problem.** Nsight (`reports/end_to_end_verification_20260525.md`
§7) shows `generateDenseGridUniqueCandidatePairsKernel` consuming
**~300 µs = ~80% of all GPU time** in the tri-tri FBP path. The cause is
structural: the kernel launches **one block per grid cell**:

```cpp
generateDenseGridUniqueCandidatePairsKernel<<<cellCount, 256>>>   // cellCount = 32 768
...
const std::uint32_t cellId = blockIdx.x;   // ONE block per cell, ALL cells
```

The blade (tool) only occupies ~30 cells. So ~32 738 blocks read an empty
or single-class bucket, find `toolCount == 0` → `totalPairs == 0` → exit
having done nothing. The 300 µs is **block-scheduling overhead for 32 768
mostly-empty blocks across 16 SMs (~512 dispatch waves)**, measured at
only 26 % SM throughput. The actual useful pair work (624 unique pairs)
is trivial.

**Why this is NOT the failed `compactActiveCells` (Phase 9).** The
existing `compactActiveCells` experiment ran a separate
`compactActiveDenseGridCellsKernel` that **scans all 32 768 cells** (one
thread per cell) to build a mixed-cell list. That scan reads the entire
256 KB grid — the same memory traffic as the generation it replaces — so
it regressed (1.18 → 5.09 ms on the GTX 1650 Ti). The new approach
**never scans all cells**.

**The design: build the active-cell list for free during the tool insert.**

The tool insert (`insertIndexedTrianglesKernel` with `insertTissue=false`)
is already running, already launch-bound (1 block, 244 idle threads for a
12-triangle blade). We piggyback the active-list build onto it. In
`insertTriangleAabbIntoGrid`, when inserting a TOOL triangle:

```cpp
const std::uint32_t localIndex = atomicAdd(&grid[cellId].toolCount, 1u);
// NEW: register this cell as active the first time a tool triangle lands
// in it, but only when tissue is already present in the cell.
if (buildToolActiveList && localIndex == 0u && grid[cellId].tissueCount > 0u) {
    const std::uint32_t a = atomicAdd(activeCellCount, 1u);
    activeCellIds[a] = cellId;   // activeCellIds is sized to cellCount, cannot overflow
}
```

Correctness arguments:

- `localIndex == 0` ⇒ "first tool triangle to claim a slot in this cell"
  ⇒ each cell is appended **at most once** (natural dedupe, no extra
  pass).
- `grid[cellId].tissueCount > 0` is valid because **tissue insert
  (kernel 2) completes before tool insert (kernel 3)** on the serialized
  default stream. So the list contains exactly the **mixed** cells
  (~30 or fewer), not just tool cells. Tool-only cells contribute zero
  pairs anyway, so excluding them is a pure win.
- `resetDenseGridKernel` already zeroes `activeCellCount` (line 1036),
  so no extra reset is needed.
- `activeCellIds` is already allocated to `cellCount` entries
  (`ensure(...)` line 515), so the append can never overflow.

**Generation reuses an existing kernel.** The repo **already has**
`generateActiveDenseGridUniqueCandidatePairsKernel` (added for the
`compactActiveCells` experiment). It device-reads `*activeCellCount` and
grid-strides:

```cpp
const std::uint32_t activeCount = *activeCellCount;
for (uint32_t activeIndex = blockIdx.x; activeIndex < activeCount; activeIndex += gridDim.x) { ... }
```

So we launch a **fixed modest grid** — e.g. `<<<256, 256>>>` — with **no
host readback**. The ~30 active cells are handled by the first ~30 blocks;
the remaining ~226 blocks read the device count, see they're out of
range, and exit in ~1 instruction. **256 blocks vs 32 768 = 128× fewer
launched blocks**, and only ~30 do real work.

**New opt-in flag.** Add `useToolActiveCellGeneration`:

- `DenseGridConfig.useToolActiveCellGeneration` (backend)
- `GpuCollisionNarrowPhase` Data field `useToolActiveCellGeneration`
- env var `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION`

Kept **distinct from `compactActiveCells`** so the regressed experiment
stays off and the two are independently measurable.

**Incompatibilities.** This path depends on tissue being inserted before
tool. It is therefore **mutually exclusive with `batchTriangleInsert`**
(which fuses both inserts and breaks the ordering). The backend must
reject the combination with a diagnostic (same pattern as
`canonicalPairEmission` being rejected for indexed mode).

**Expected outcome.** Generation: ~300 µs → target < 30 µs. Narrow wall
~0.83 ms → target ~0.5 ms. FPS ~672 → target ~900-1000 on the one-tissue
scene. The insert kernels, FBP kernel, and contact output are unchanged.

**Risk.** On the 1650 Ti, even empty-block early-exit is cheap, so the
real win may be smaller than the theoretical 10×. The 2026-05-14
`compactActiveCells` regression is the cautionary precedent — but that
failure was the *separate scan*, which this design eliminates. **Must be
A/B measured** before becoming a default. If it wins, consider making it
the default for the indexed path (it is never worse than the dense path:
the active list is always ≤ the number of cells).

**Implementation checklist.**

1. Add `useToolActiveCellGeneration` to `DenseGridConfig` (header + stub).
2. Add `buildToolActiveList`, `activeCellIds`, `activeCellCount` params to
   `insertTriangleAabbIntoGrid` and the indexed/packed insert kernels.
3. Append-on-first-tool-insert logic as above.
4. In `computeDenseGridIndexedTriangleContacts` (and the packed variant),
   when `useToolActiveCellGeneration`, launch the tool insert with the
   active-list outputs, then launch
   `generateActiveDenseGridUniqueCandidatePairsKernel<<<kActiveGenBlocks, 256>>>`
   instead of the all-cells generator.
5. Reject `useToolActiveCellGeneration && batchTriangleInsert`.
6. Wire the Data field + env var in `GpuCollisionNarrowPhase` and the
   benchmark scenes.
7. A/B benchmark one-tissue + large-tissue, capture Nsight on the new
   generation launch, write a finding report.

### 5.16  Workspace-cached broad-cull CUDA events (DONE, 2026-05-25)

**Status: implemented, always on.** The four broad-cull timing events
(`broadStageStart/End`, `broadTotalStart/End`) now live on
`DenseGridWorkspace`, lazy-created once via `ensureBroadEvents()` and
freed in the workspace destructor. Both contact functions alias the
cached handles; `destroyStageEvents()` is now a no-op. Verified: v-t
self (2700 VF) and v-t cross (254 VF) contact counts unchanged, no
correctness change. Removes ~40-80 us/frame of `cudaEventCreate` /
`cudaEventDestroy` driver churn (small relative to the Phase 15 win but
free and always on).

---

**Original design notes (as-built):** Small, low-risk CPU-side win.

**The problem.** `computeDenseGridIndexedTriangleContacts` (line ~3219)
and `computeDenseGridTriangleContacts` (line ~4160) each call
`cudaEventCreate` 2-4× and `cudaEventDestroy` 2-4× **every frame** for
their `stageStart`/`stageEnd`/`totalStart`/`totalEnd` timing events. At
~5-10 µs per create/destroy on the GTX 1650 Ti, that is **~40-80 µs of
pure driver overhead per frame** — a measurable chunk of the
`host_synchronization_ms` gap.

**The design.** Mirror exactly what Phase 5.14 already did for the FBP
events (`fbpStartEvent` / `fbpEndEvent`). Add to `DenseGridWorkspace`:

```cpp
cudaEvent_t broadStageStart{}, broadStageEnd{}, broadTotalStart{}, broadTotalEnd{};
bool broadEventsReady{false};
cudaError_t ensureBroadEvents() {
    if (broadEventsReady) return cudaSuccess;
    cudaError_t err = cudaEventCreate(&broadStageStart);
    if (err == cudaSuccess) err = cudaEventCreate(&broadStageEnd);
    if (err == cudaSuccess) err = cudaEventCreate(&broadTotalStart);
    if (err == cudaSuccess) err = cudaEventCreate(&broadTotalEnd);
    broadEventsReady = (err == cudaSuccess);
    return err;
}
```

Replace the per-frame `cudaEventCreate`/`cudaEventDestroy` in both
functions with `workspace.ensureBroadEvents()` + the cached handles. Drop
the `destroyStageEvents()` calls (events live for plugin lifetime, freed
in the workspace destructor). `cudaEventRecord` overwrites the timestamp
each frame, so reuse is safe.

**Expected outcome.** ~40-80 µs/frame off `host_synchronization_ms`.
Narrow wall ~0.83 ms → ~0.75 ms, FPS ~672 → ~750 on the one-tissue scene
(stacks additively with Phase 5.15).

**Risk.** Minimal — this is a mechanical mirror of a change that already
shipped for the FBP events. The only subtlety is that `detailedProfiling`
mode currently creates the stage events conditionally; with caching we
create all four once regardless (unused events cost nothing).

**Implementation checklist.**

1. Add the four `broad*` events + `broadEventsReady` + `ensureBroadEvents()`
   to `DenseGridWorkspace`.
2. Free them in `release()` (guarded by `broadEventsReady`).
3. Swap both functions to use the cached handles; delete the per-frame
   create/destroy and the `destroyStageEvents()` lambda.
4. Verify with a fast-path run that `host_synchronization_ms` drops and no
   correctness change.

### 5.17  FBP / v-t grid-stride correctness fix (DONE, 2026-05-25)

**Status: fixed.** A real correctness bug, discovered during the Phase 15
large-tissue A/B, affecting all FBP and v-t output (not introduced by
Phase 15 — Phase 15 exposed it).

**The bug.** `featureBasedProximityKernel` and
`featureBasedVertexTriangleProximityKernel` processed exactly one candidate
pair per thread (`candidatePairs[tid]`) with **no grid-stride loop**. They
launched a fixed over-launch grid of 256 blocks × 256 threads = **65 536
threads**. On the one-tissue scene (624 candidate pairs) this was fine, but
the large-tissue scene produces **322 560 candidate pairs** — so the kernel
silently processed only the first 65 536 and **dropped ~80% of pairs**.

The bug was invisible until the large-tissue A/B because (a) all prior
scenes had < 65 536 pairs, and (b) it produced a *plausible* contact count.
It surfaced as a discrepancy: baseline emitted 3691 contacts, active 3790,
even though both had identical `unique_candidate_count` (322 560). The
cause was that the two paths order `candidatePairs[]` differently (cell
order vs active-list order), so each processed a different first-65 536
subset.

**The fix.** Wrap both kernel bodies in a grid-stride loop:

```cpp
const uint32_t pairCount = *candidatePairCount;
const uint32_t stride = gridDim.x * blockDim.x;
for (uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < pairCount; idx += stride) {
    ... process candidatePairs[idx] ...   // returns became continues
}
```

Now every pair is processed regardless of launch grid size. The launch
grid is purely a GPU-saturation target (bumped to 1024 blocks). After the
fix, the large-tissue scene emits **8018 contacts** (5397 VF / 880 FV /
1741 EE) — bit-identical between baseline and active, and the *correct*
count (the old 3691/3790 were both wrong).

**Impact.** This makes FBP and v-t correct for arbitrarily large candidate-
pair counts. Small scenes (< 65 536 pairs) are unaffected (one-tissue still
56 EE, v-t self 2700 VF, v-t cross 254 VF). The fix is independent of
Phase 15 but was a prerequisite for trusting the large-scene A/B.

**Lesson recorded:** the over-launch pattern (introduced in the Phase 11
sync fix, §5.11) needs a grid-stride loop to be correct, not just a
per-thread bound check. Any future over-launched kernel must grid-stride.

### 5.18  Spatial-hash + prefix-sum broad cull (EXPERIMENTAL, opt-in, default-off)

**Status: working, correctness-verified, and MERGED to `main` (2026-06-09,
fast-forward). Opt-in, default-off; with the flag off the dense path is
byte-identical to the pre-merge behaviour, so the merge cost the default path
nothing. `main` now carries both the stable FBP/v-t engine and this hash cull.**

**Why.** The dense grid + Phase 15 tool-active-cell path assumes an
*asymmetry* — a small tool sweeping a large tissue, so only ~30 cells are
mixed. When **both** sides are large (the target: ~1,500-triangle tool against
a large tissue), that asymmetry weakens: the tool lights up many cells and the
fixed 32,768-cell grid spends memory on empty space.

**What was done.** A self-contained alternative broad cull that scales with
*occupancy* instead of a fixed cell count, feeding the **same** FBP narrow
kernel:

1. **Spatial hash** — open-addressing table keyed by linear cell id, slots
   claimed lock-free with `atomicCAS`; per-slot tissue/tool triangle counts
   bumped with `atomicAdd`. Only occupied cells consume a slot.
2. **Prefix-sum work expansion** — per occupied slot, candidate pairs =
   `tissueCount × toolCount`; `thrust::exclusive_scan` over those counts gives
   global offsets; then **one thread per candidate pair**, each binary-searching
   the offset array to recover `(slot, localPair)`. Perfect load balance, no
   serial per-cell loop.

Auto table size = `nextPow2((firstTris + secondTris) × 4)`, min 1024. New
backend entry `computeHashPrefixSumProximityContacts`; new Data fields
`useHashPrefixSumGeneration` (default false) + `hashTableSize` (default 0 =
auto); dispatch sub-branch inside the existing FBP branch (the original call
moves verbatim into the `else`).

**Measured (2026-06-09, large tissue + large tool, 14,368 elements, same-session
A/B).**

```text
                 FPS     narrow_wall  narrow_kernel  contacts (VF/FV/EE)
dense grid       343.6   1.997 ms     1.439 ms       2354 (1119/428/807)
hash+prefix-sum  384.1   1.695 ms     1.254 ms       2354 (1119/428/807)
                 +11.8%  -15.1%       -12.9%         BIT-IDENTICAL, overflow 0
```

Standalone backend bench also bit-identical (488 = 488). Reports:
`reports/hash_prefixsum_broadphase_experiment_20260609.md` (design) and
`reports/benchmark_suite_20260609.md` (suite).

**Scope / caveats.**
- Regime-specific: for small-tool/large-tissue the Phase 15 path is cheaper and
  still wins. This is for the both-large case.
- Only the cross-model tri–tri FBP path is routed through the hash cull; the
  v-t (self + cross) kernels are unchanged.

### 5.19  Optimised hash broad cull (2026-06-17, ~2.5–3× faster kernel than dense)

The §5.18 hash design was reworked with **seven optimizations**, all inside
`cuda/GpuCollisionBackend.cu` (no new files; CUB is header-only):

1. **Clear only touched dedup slots** (`clearTouchedPairHash{32,64}Kernel`) instead
   of a full `cudaMemset` of the multi-million-slot pair-hash table every frame
   (first frame / growth still does one full memset, guarded).
2. **Compact bucket storage** — two-pass `markCompactHashGridCellsKernel` →
   `compactHashGridSlotsKernel` assigns each occupied slot a dense bucket index;
   per-bucket arrays scale with occupancy, not `tableSize`.
3. **Generate only mixed buckets** — `fillCompactHashGridTrianglesKernel` builds a
   `mixedBucketIds` list (buckets with both tissue and tool); the hash analogue of
   Phase 15.
4. **No per-pair binary search** — `generateMixedHashCandidatePairs{32,64}Kernel`
   runs one block per mixed bucket and splits its `t×u` pairs by divide/modulo.
5. **Persistent CUB scan** (`cub::DeviceScan::ExclusiveSum` with workspace temp)
   replaces `thrust::exclusive_scan` (no per-frame Thrust alloc).
6. **32-bit compact pair encoding** when both triangle counts ≤ 65535.
7. **Per-stage hash profiling** — 8 `avg_narrow_hash_*_ms` summary keys.

Kernel launches: 7 (dense) → **13** (optimised hash) — more, smaller kernels.

**Measured (2026-06-17, large tissue + large tool, 14,368 elements, same-session,
independently re-verified).** Trust the kernel time.

```text
                          narrow_kernel   FPS     contacts (VF/FV/EE)
plain dense (Phase15 off) 2.126 ms        263.8   2354 (1119/428/807)
optimised dense (Ph15 on) 1.787 ms        294.7   2354 (1119/428/807)
earlier hash (2026-06-09) ~1.27 ms        ~405    2354  (prior build)
optimised hash (this)     0.700 ms        408.1   2354 (1119/428/807)
                          3.0x vs plain dense / 2.6x vs optimised dense / 1.8x vs earlier hash
```

A second same-session A/B reproduced 0.508 ms / 565 FPS for the optimised hash
(dense 1.748 ms / 238 FPS). Standalone backend bench bit-identical
(`hash = fbp = 8018`, unique 322,560 = 322,560, overflow 0). Full suite: every
scene's contacts match the documented baseline, overflow 0. Report:
`reports/hash_optimized_broadphase_20260617.md`.

**Open follow-ups.** The CUB scan's `pairOffsets`/`rawTotal` now feed only the
`rawPairCount` stat (the block-per-bucket generator needs no global offsets) — the
scan could be dropped/folded. Route v-t through the hash cull; auto-tune table size
from a measured occupancy histogram.

---

## 6. Phase 11/12 follow-ups (landed 2026-05-25)

After the cross-model wiring landed, three opportunistic follow-ups were
identified at the end of §5.12. All three landed in the same session.

### 6.0a  Nsight Compute capture on the new kernels

Script: `scripts/run_nsight_fbp_profile_wsl.sh`. Profiles three scenes
(tri-tri FBP, v-t self-collision, v-t cross-model) with a metric set
focused on "what bounds this kernel?".

Headline finding: **none of the new kernels are bottlenecks**. The most
notable observations:

- `featureBasedProximityKernel` uses 68 registers/thread, limiting
  occupancy to ~50% of theoretical peak. SM throughput is 4.8% because the
  over-launch wastes most threads (only 624 candidate pairs out of 65 536
  threads). Cheap in absolute time (~17 µs).
- `featureBasedVertexTriangleProximityKernel` uses only 32 registers
  (single closest-point test per pair vs 15 for FBP). 53-62% occupancy.
- `insertIndexedPointsKernel` is launch-overhead-bound for small point
  clouds (~6 µs for 64 points). Expected behavior.
- All FBP atomic pressure is < 0.5%. The atomic decision from §5.13 stands.

Full report: `reports/gpu_collision_phase11_12_kernel_profile_20260525.md`.
Per-scene reports at `output/benchmark_logs/fbp_nsight_20260525/<scene>/`.

### 6.0b  Backend bench extension

`SofaGpuCollision/src/tools/DenseGridBackendBench.cpp` now runs three
phases per invocation: legacy exact-contact, tri-tri FBP, and v-t cross-
model. The two FBP phases share the same synthetic geometry (12 800
tissue triangles + 224 blade triangles by default) and emit separate
CSVs:

```text
output/benchmark_logs/<bench-run>/backend_dense_grid_benchmark.csv
output/benchmark_logs/<bench-run>/backend_dense_grid_benchmark_fbp.csv
output/benchmark_logs/<bench-run>/backend_dense_grid_benchmark_vt.csv
```

Toggle with environment variables:

```text
SOFA_BACKEND_BENCH_RUN_FBP=1    (default true)
SOFA_BACKEND_BENCH_RUN_VT=1     (default true)
SOFA_BACKEND_BENCH_FBP_MAX_CONTACTS=1000000
SOFA_BACKEND_BENCH_VT_MAX_CONTACTS=1000000
```

Reference numbers from 2026-05-25 run (small workload, 12 800 + 224
triangles, contact_distance=0.03):

```text
exact-contact: wall=4.36 ms  kernel=1.63 ms  contacts=0     (no intersection)
tri-tri FBP:   wall=2.43 ms  kernel=1.89 ms  contacts=488   (133 FV + 355 EE)
v-t x-model:   wall=1.63 ms  kernel=n/a      contacts=38    (38 VF)
```

The FBP path finds proximity (488 contacts) where SAT finds intersection
(0 contacts). By design.

### 6.0c  Cross-model `DetectionOutput` publication

New helper `publishCudaPointTriangleContacts` in `GpuCollisionNarrowPhase.cpp`.
Uses `TDetectionOutputVector<CudaPointCollisionModel, CudaTriangleCollisionModel>`
— SOFA's template accepts any two model types, no specialization needed.

Wired into the cross-model dispatch: when `copyContactsToHost=true`, the
narrow phase calls the helper. To make the contacts actually arrive on
the host, the dispatch now couples flags:

```cpp
proximityConfig.keepContactsOnDevice =
    d_proximityKeepContactsOnDevice.getValue() && !d_copyContactsToHost.getValue();
```

i.e. `copyContactsToHost=true` forces `keepContactsOnDevice=false` regardless
of the user's other setting, ensuring the proximity contacts are
downloaded before publication.

Verified with the cross-model smoke scene at `copyContactsToHost=true`:

```text
xm_publication_final (SOFA_COPY_CONTACTS_TO_HOST=1, SOFA_PROXIMITY_KEEP_CONTACTS_ON_DEVICE=0):
avg_fps=332.9
avg_narrow_wall_ms=2.74
avg_device_to_host_bytes=19320       # = 254 contacts × ~76 bytes + 16 byte counters
avg_narrow_output_contact_count=254
avg_narrow_vf_contact_count=254
```

The 332 FPS is the cost of moving 19 KB of contact data per frame. Users
who don't need CPU contacts should leave `copyContactsToHost=false` for
the 1968-FPS detection-only path.

### 6.0d  Regression sweep after the follow-ups

| Scene | FPS (after follow-ups, 2026-05-25 late) | Pre-follow-up baseline |
|---|---:|---:|
| Tri-tri FBP (fast path) | 700 | 775 |
| V-t self-collision | 1589 | 1385 |
| V-t cross-model (detection-only) | 1380 | 1968 |
| V-t cross-model (with CPU publication) | 332 | N/A (new path) |

Variation across runs is within GTX 1650 Ti thermal/power-state noise.
No structural regression.

---

## 7. Remaining work (priority order)

### 7.0  Generation + event optimizations — LANDED + DEFAULT ON 2026-05-25

Fully closed out. Both optimizations are implemented, verified correct on
small *and* large scenes, and Phase 15 is now **default ON**. A latent
correctness bug exposed during the large-scene A/B was also fixed (§5.17).

1. **Tool-active-cell candidate generation (§5.15)** — DONE, default ON.
   Generation kernel 300 µs → 7.9 µs (38×), one-tissue fast path
   221 → 943 FPS (4.3×), large-tissue 108 → 116 FPS (1.08×), contact
   counts bit-identical on both. Generation grid dropped 32 768 → 1 024
   blocks (Nsight-confirmed).
2. **Workspace-cached broad-cull events (§5.16)** — DONE, always on.
3. **FBP/v-t grid-stride correctness fix (§5.17)** — DONE. Both proximity
   kernels now grid-stride over all candidate pairs; large scenes no
   longer silently drop ~80% of pairs.

Nothing remains open here. The next-tier opportunities (extend active-cell
generation to the v-t point-insert path; warp-per-pair FBP for register
pressure) are tracked in §7.4 and §7.5.

### 7.1  Profiling polish (small, opportunistic)

- Move `recordBroadPhase` / `recordNarrowPhase` from a `std::mutex` to a
  thread-local + relaxed-atomic pattern. Single-threaded SOFA step today
  so this is cosmetic; bumps to "do it" priority if a multithreaded
  broad phase ever lands.
- Document the assumption that record functions overwrite rather than
  accumulate per step. (Currently true because each runs exactly once per
  step.) Add an inline comment in `GpuPipelineProfiling.h`.

### 7.2  Large-scene repeatability

Large-scene benchmark numbers are run-condition-sensitive on the GTX
1650 Ti laptop. The 2026-05-17 verification showed:

- `final_large_tissue_cpu_gpu_compare_20260517`: GPU slower than CPU
  (33.8 / 24.5 ms = 1.4× slower).
- `final_large_gpu_only_rerun_20260517`: GPU recovered to 4.4 ms narrow
  wall, 44 FPS.

The geometry facts are stable (0 H2D, 0 D2H, 5 launches, 1 memset). The
wall-time variance is from thermal / power-state effects. Before making
another kernel-level decision on large scenes:

```text
- Run with fixed-power-state CPU governor where possible
- Same run order between CPU and GPU comparisons
- Same warmup / measured step counts
- Fresh process per variant
- Counter reads disabled for wall-time runs
- Counter reads enabled only for validation runs
```

### 7.3  Direct constraint-solver consumption

The FBP path emits barycentrics + signed distance + normal in device
memory. A constraint solver written for CUDA can read these directly with
no copy. **Not in scope** for this project; flag it for the downstream
team.

### 7.4  FBP kernel register pressure

`featureBasedProximityKernel` uses 68 registers per thread, capping
occupancy at ~50% of peak on sm_75. Mitigation paths (warp-per-pair,
register-to-shared spill, kernel split) documented in
`reports/gpu_collision_phase11_12_kernel_profile_20260525.md` §6.0a.

Not urgent — the kernel costs only ~17 µs per launch and the production
fast path already meets the FPS target.

### 7.5  Other deferred items

- SoA AABB memory layout (see §5.9).
- Per-active-cell candidate generation that scales with active-cell count
  rather than total cell count. Worth it only if a new workload makes the
  current 300-µs candidate-generation kernel dominate.

---

## 8. Git tracking policy

Tracked:

- All `SofaGpuCollision/src/...` source.
- All `test_*.py` scene files.
- All `scripts/run_*_wsl.sh` launchers and helper scripts.
- `guide/` (this directory).
- `reports/` (generated reports and presentation decks that the team
  intentionally pins to a commit).

Not tracked (`.gitignore`):

- `output/` (raw benchmark + profile artifacts, except `output/README.md`).
- `SofaGpuCollision/build*/` (local build directories).

Before committing:

```powershell
git status --short
git diff --stat
git diff -- SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
git diff -- SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu
git diff -- guide/
```

Stage intentionally. Do not stage `output/` content (the
`.gitignore` already excludes it; staged-by-accident files happen).
