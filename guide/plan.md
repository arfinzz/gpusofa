# Implementation Plan

This is the implementation roadmap and historical log for the GPU SOFA
collision pipeline. Every phase has a status, a "why this was needed"
paragraph, the measured outcome, and (where rejected) an explanation of the
alternative that was considered. New work appends; finished phases stay so
their rationale remains discoverable.

Last refreshed: 2026-05-25.

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

The 2026-05-25 production fast path delivers:

| Scene | FPS | Narrow wall | H2D / D2H bytes per frame |
|---|---:|---:|---:|
| one-tissue / one-blade, exact-contact detection | 633 | 0.69 ms | 0 / 0 |
| one-tissue / one-blade, **tri-tri FBP** | **775** | **0.67 ms** | **0 / 0** |
| 2-layer slab, **v-t self-collision** | 1385 | 0.66 ms | 0 / 16 |
| tissue + tool cloud, **v-t cross-model** | 1968 | 0.37 ms | 0 / 16 |

All four paths share the same dense-grid broad cull. The differences are in
the narrow-pass kernel and the host-side dispatch.

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
compactActiveCells                        = false       ← regressed on GTX 1650 Ti
batchTriangleInsert                       = false       ← regressed on GTX 1650 Ti
```

The optional `compactActiveCells` and `batchTriangleInsert` are runtime
switches kept for experimentation. The 2026-05-14 profile showed that
enabling either regressed one-tissue narrow wall from ~1.18 ms to ~5.09 ms,
so the default keeps them off. The architecture leaves them in place because
a future larger-scene workload could flip the conclusion.

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
  - `test_gpu_self_collision_vertex_triangle_smoke.py`: a two-layer slab
    with `selfCollision=True`. Each top vertex is 0.05 above a bottom
    triangle; `contactDistance=0.06` makes every top vertex emit one or
    more contacts.
  - `test_gpu_cross_model_vertex_triangle_smoke.py`: a tissue triangle
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
