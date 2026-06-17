# Architecture

This document is the canonical source-of-truth for how `SofaGpuCollision`
works end to end. It is intended to be readable cold — if you have never seen
the codebase before, start here. Sections build on each other; you can skim
the headings to navigate.

Last refreshed: 2026-06-09 (added the experimental spatial-hash + prefix-sum
broad-cull Data flags, §5.18 in plan.md; built on the 2026-05-25 Phase 12
cross-model architecture).

---

## 1. What this project is solving

Surgical simulation in SOFA needs **collision detection between a deformable
tissue mesh and rigid surgical tools** every frame, fast enough for haptic
loops (~500-1000 Hz) and at minimum interactive rates (~60-100 Hz) for full
simulation steps. The CPU collision pipeline in SOFA is general but slow on
the kind of dense meshes (~10⁴ triangles, ~5×10² candidate pairs) used in
surgical scenes.

This project ports the narrow phase to CUDA, hosted as a SOFA plugin so
existing scene files keep working. Target hardware: an **NVIDIA GeForce
GTX 1650 Ti laptop GPU** running under WSL2 (`wsl-gpu-proj`). The constraint
is real: optimizations that win on data-centre GPUs often regress on a
laptop SM count of 16, so every "obvious" perf win is measured before it
becomes the default.

The plugin provides four narrow-phase output modes, each opt-in:

1. **Exact-contact** (SAT-style triangle-triangle intersection, legacy default)
2. **Feature-based proximity (FBP) tri-tri** (Ericson closest-feature VF + EE)
3. **FBP vertex-triangle self-collision** (vertex set against its own triangles)
4. **FBP vertex-triangle cross-model** (CudaPointCollisionModel against CudaTriangleCollisionModel)

All four share the same dense-grid broad cull. The choice between them is
driven by Data fields on `GpuCollisionNarrowPhase` and the type of the
collision-model pair the SOFA broad phase emits.

---

## 2. The 30-second mental model

```text
SOFA scene  ──►  RequiredPlugin loads libSofaGpuCollision.so
                  │
                  ├─►  registers GpuCollisionBroadPhase
                  └─►  registers GpuCollisionNarrowPhase

per simulation step (DefaultAnimationLoop):
  CollisionPipeline::computeCollisionDetection()
    GpuCollisionBroadPhase::beginBroadPhase()
    for each CollisionModel:
        GpuCollisionBroadPhase::addCollisionModel(cm)
    GpuCollisionBroadPhase::endBroadPhase()        ─► cmPairs

    GpuCollisionNarrowPhase::beginNarrowPhase()
    for each pair in cmPairs:
        GpuCollisionNarrowPhase::addCollisionPair(pair)
    GpuCollisionNarrowPhase::endNarrowPhase()      ─► DetectionOutput

  CollisionPipeline::computeCollisionResponse()    (no-op in benchmark mode)
```

Inside `endNarrowPhase`, the dispatch picks one of the four output modes per
pair based on:

| Condition | Path |
|---|---|
| Cross-model (`CudaPoint`, `CudaTri`) and `useFeatureBasedProximity && useVertexTriangleProximity` | **v-t cross-model** |
| Self-collision (`pair.first == pair.second`) and `useFeatureBasedProximity && useVertexTriangleProximity` | **v-t self-collision** |
| Both sides `CudaTri` and `useFeatureBasedProximity` | **tri-tri FBP** |
| Both sides `CudaTri` (default) | **exact-contact (SAT)** |
| Anything else | legacy CPU SOFA fallback |

All four GPU paths share the dense-grid broad cull. They differ only in the
narrow-pass kernel that runs after the candidate pairs have been generated.

---

## 3. Source-file map

```text
SofaGpuCollision/
├── CMakeLists.txt                       Build (CUDA-aware), produces libSofaGpuCollision.so
└── src/
    ├── SofaGpuCollision/
    │   ├── config.h                     SOFA_GPU_COLLISION_API export macro + CUDA toggle
    │   ├── init.cpp                     Plugin entry: initExternalModule(), metadata
    │   ├── GpuCollisionBackend.h        Stable C++ API boundary; only this header crosses the .cu line
    │   ├── GpuCollisionBackendStub.cpp  Fallback when CUDA disabled (returns false + diagnostic)
    │   ├── GpuCollisionBroadPhase.h     SOFA component header (inherits BruteForceBroadPhase)
    │   ├── GpuCollisionBroadPhase.cpp   Broad-phase implementation
    │   ├── GpuCollisionNarrowPhase.h    SOFA component header (inherits BVHNarrowPhase)
    │   ├── GpuCollisionNarrowPhase.cpp  Narrow-phase implementation: dispatch + extractors + cache
    │   ├── GpuKinematicRigidController.{h,cpp}  Deterministic rigid-body controller for benchmark scenes
    │   ├── GpuPipelineBenchmarkController.{h,cpp}  Per-frame CSV + summary writer
    │   ├── GpuPipelineProfiling.{h,cpp}  Thread-safe StepSnapshot accumulator
    │   └── cuda/
    │       └── GpuCollisionBackend.cu    ALL CUDA kernels + public function bodies live here
    └── tools/
        └── DenseGridBackendBench.cpp    Standalone bench harness (no SOFA scene required)
```

Outside the plugin:

```text
testscenes/                                          ALL benchmark scenes live here
  one_tissue_one_blade.py                            Tri-tri FBP, small tool (surgical default)
  large_tissue_blade.py                              Tri-tri FBP, large mesh
  self_collision_vertex_triangle.py                  V-t self-collision scene
  cross_model_vertex_triangle.py                     V-t cross-model scene
  hash_prefixsum_large.py                            Hash + prefix-sum broad-cull A/B scene
  dense_collision_benchmark_common.py                Shared geometry generators (imported by all scenes)
scripts/                                             WSL launchers + verification tools
guide/                                               This documentation
tutorial/                                            Beginner course
reports/                                             Generated reports and decks
output/                                              Raw benchmark/profile artifacts (gitignored)
```

---

## 4. Plugin lifecycle (scene load → first frame)

```text
runSofa scene.py
  │
  ├─► Python loads scene module, calls createScene(root)
  │
  ├─► root.addObject("RequiredPlugin", pluginName=["SofaCUDA", ...])
  │     SOFA dlopens each named .so
  │     For SofaGpuCollision: initExternalModule() runs
  │     Static ::RegisterObject in GpuCollisionBroadPhase.cpp / NarrowPhase.cpp
  │       fires, putting "GpuCollisionBroadPhase" and "GpuCollisionNarrowPhase"
  │       into the ObjectFactory registry.
  │
  ├─► root.addObject("GpuCollisionBroadPhase", ...)
  │     Factory creates the component, init() runs:
  │       calls backend::probe()
  │       caches m_backendAvailable
  │
  ├─► root.addObject("GpuCollisionNarrowPhase", ...)
  │     Same probe path; cache m_backendAvailable
  │
  ├─► Each child node adds a MechanicalObject<CudaVec3f> + MeshTopology +
  │     TriangleCollisionModel. Because the MechanicalObject is templated on
  │     CudaVec3f, SOFA auto-specializes the TriangleCollisionModel to
  │     sofa::gpu::cuda::CudaTriangleCollisionModel.
  │
  └─► Sofa.Simulation.init(root) walks the tree, calling init() on every
        component. The plugin is now hot.

Per frame:
  DefaultAnimationLoop::step()
    CollisionPipeline::computeCollisionDetection()
      [broad + narrow phase as above]
    CollisionPipeline::computeCollisionResponse()
      No-op for detection-only scenes (no ContactManager registered).
```

`backend::probe()` is the single gate for "is CUDA available?". If it returns
`false`, every backend entry point returns `false` with a diagnostic and
`GpuCollisionNarrowPhase` falls back to the inherited `BVHNarrowPhase` CPU
path when `allowCPUFallback=True`.

---

## 5. Broad phase in detail

File: `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBroadPhase.cpp`

`GpuCollisionBroadPhase` inherits SOFA's `BruteForceBroadPhase`. It exposes
five Data fields:

| Field | Default | Purpose |
|---|---|---|
| `enableGPU` | true | Try GPU AABB culling; if false, delegate to BruteForceBroadPhase |
| `allowCPUFallback` | true | When GPU is unavailable, hand pairs to the CPU broad phase |
| `logBackendStatus` | true | One-shot init log of CUDA backend availability |
| `logBoxesOnce` | false | Dump the first-frame root AABBs for debugging |
| `useObjectAabbCulling` | false | Run a GPU AABB-vs-AABB cull before narrow phase. **Off by default** because one-tissue scenes only have 2 collision models — pair enumeration is free, GPU overhead is loss |

The per-frame flow:

```text
beginBroadPhase()
  m_pendingModels.clear()

for each CollisionModel cm in the scene:
    addCollisionModel(cm)        ─► m_pendingModels.push_back(cm)

endBroadPhase()
  if !enableGPU: delegate to BruteForceBroadPhase
  else if !useObjectAabbCulling (default):
    O(n²) enumeration of pairs (n ≤ ~4 in surgical scenes)
    for each pair (i, j):
        if doesSelfCollide(cm): emplace_back(cm, cm)       ← self-collision!
        if intersector->canIntersect(a, b): cmPairs.emplace_back(a, b)
  else (useObjectAabbCulling = true):
    extract one AABB per model
    backend::computeBroadPhasePairs(boxes, gpu_pairs, ...)
    fold gpu_pairs back into cmPairs

  profiling::recordBroadPhase(stageSnapshot)
```

The `doesSelfCollide(cm)` branch is the one that triggers `(cm, cm)` pairs
when a `TriangleCollisionModel` has `selfCollision=True`. The narrow phase
detects this pair shape and routes it to the v-t self-collision backend.

---

## 6. Narrow phase in detail

File: `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp`

`GpuCollisionNarrowPhase` inherits SOFA's `BVHNarrowPhase`. It exposes a
larger set of Data fields than the broad phase because it owns the four
output-mode switches and the dense-grid configuration.

### 6.1 The full Data field matrix

| Field | Default | Purpose |
|---|---|---|
| `enableGPU` | true | Master switch for GPU narrow phase |
| `allowCPUFallback` | true | When CUDA fails or rejects, hand pairs to BVHNarrowPhase |
| `useDenseGrid` | true | Use the dense-grid broad cull (always true in practice) |
| `useIndexedDenseGridInput` | true | Use cached indexed topology + device positions; the source of the 0-H2D fast path |
| `useDirectDevicePositions` | true | Read SOFA CUDA `deviceRead()` pointer directly; avoid CPU position rebuild |
| `cacheTriangleTopology` | true | Cache stable triangle indices across frames |
| `validateTriangleTopologyCache` | false | Hash topology every frame to detect same-size edits (off for performance) |
| `useGpuHashDedupe` | true | GPU-side `atomicCAS` hash table for candidate-pair dedupe |
| `canonicalPairEmission` | false | Emit each pair only from canonical overlap cell (off for indexed path) |
| `deduplicatePairs` | true | Master dedupe toggle |
| `usePinnedHostStaging` | true | Pinned host buffers for fallback (non-direct) uploads |
| `copyContactsToHost` | true | Publish contacts to SOFA DetectionOutput. **Set false for detection-only** |
| `detailedProfiling` | false | Per-stage CUDA event timings + NVTX ranges (adds syncs) |
| `readCountersWhenContactsStayOnDevice` | false | D2H read counters in detection-only mode (for validation) |
| `computeDeviceContactsWhenContactsStayOnDevice` | false | Run exact-contact kernel even when contacts stay device-side |
| `compactActiveCells` | **false** | Experimental: compact mixed cells via a separate full-grid scan. Regressed on GTX 1650 Ti, superseded by `useToolActiveCellGeneration` |
| `batchTriangleInsert` | **false** | Experimental: combine tissue + tool insert into one launch. Regressed; mutually exclusive with `useToolActiveCellGeneration` |
| `useToolActiveCellGeneration` | **true** | **Phase 15, DEFAULT ON.** Generate candidate pairs over tool-occupied (mixed) cells only — active list built during the tool insert (no scan), generation grid-strides over it. 4.3× one-tissue, 1.08× large-tissue, bit-identical output, never a regression |
| `useHashPrefixSumGeneration` | **false** | **EXPERIMENTAL (branch `experiment/hash-prefixsum-broadphase`), DEFAULT OFF.** Replace the dense grid with a spatial-hash cell structure + prefix-sum work-expansion broad cull for the tri–tri FBP path. Targets the *large-tissue + large-tool* regime where the tool-active-cell asymmetry weakens. Bit-identical contacts; **optimised 2026-06-17 to ~2.5–3× faster kernel than dense** (0.5–0.7 vs 1.8–2.1 ms on 12.8k+1.6k tris; compact buckets, mixed-bucket gen, no binary search, touched-slot clear, CUB scan, 32-bit pairs). See `reports/hash_optimized_broadphase_20260617.md`. Requires `useFeatureBasedProximity=true` |
| `hashTableSize` | **0** | Slot count for `useHashPrefixSumGeneration`. 0 = auto (~4 slots per input triangle, rounded to a power of two) |
| `useFeatureBasedProximity` | **false** | Phase 11+. Replace SAT exact-contact with VF + EE closest-feature kernel |
| `useVertexTriangleProximity` | **false** | Phase 12. Route self-collision and (point-model, triangle-model) pairs to v-t |
| `proximityComputeBarycentrics` | true | Populate ProximityContact barycentric weights |
| `proximityReadContactCounter` | false | Read back contact counters every frame (slower but visible) |
| `proximityKeepContactsOnDevice` | true | Skip ProximityContact host copy (constraint-solver-ready) |
| `proximityMaxContacts` | 1000000 | Device buffer capacity for proximity output |
| `proximityCounterReadbackInterval` | 0 | Sampled-readback mode: 0=off, N=read every Nth frame |
| `minGPUPairCount` | 8 | Below this many candidate pairs, fall through to CPU |
| `gridMinX/Y/Z`, `gridMaxX/Y/Z` | scene-tuned | World-space dense grid bounds |
| `gridResolutionX/Y/Z` | scene-tuned | Cell counts on each axis |
| `contactDistance` | 0.03 | Inflation for triangle AABBs; controls proximity threshold |
| `maxTissueTrianglesPerCell` | 64 | Bucket capacity for the "tissue" side of each cell |
| `maxToolTrianglesPerCell` | 32 | Bucket capacity for the "tool" side (or vertex side, in v-t) |
| `maxCandidatePairs` | 1000000 | Candidate pair output buffer capacity |

### 6.2 Frame lifecycle

```text
beginNarrowPhase()
  m_pendingPairs.clear()
  ++m_frameCounter                              ← drives sampled readback
  markTriangleTopologyCacheUnused()             ← LRU-ish cache invalidation

for each pair in cmPairs:
    addCollisionPair(pair)
      ─► m_pendingPairs.push_back(pair)         ← deferred to endNarrowPhase

endNarrowPhase()
  stageSnapshot = StageSnapshot{}
  phaseStart = chrono::steady_clock::now()

  for each pair in m_pendingPairs:
    1. Cross-model v-t check (Phase 12 cross-model)
       if (useFeatureBasedProximity && useVertexTriangleProximity
           && pair.first != pair.second
           && one side is CudaPoint, other is CudaTri):
         extractCudaPointCloud + extractCudaIndexedSurface
         backend::computeFeatureBasedVertexTriangleContacts(...)
         convert ProximityContact ─► ExactContact for SOFA publication
         continue                ─── skip the rest for this pair

    2. Indexed extraction
       extractCudaIndexedSurface(pair.first / pair.second)
       (extractCudaTriangles fallback for non-indexed path)

    3. DenseGridConfig + FeatureBasedProximityConfig assembled from Data fields

    4. Dispatch (mutually exclusive):
       if (self-collision pair + useVertexTriangleProximity):
         backend::computeFeatureBasedVertexTriangleContacts (v-t self)
       else if (useFeatureBasedProximity):
         backend::computeFeatureBasedProximityContacts      (tri-tri FBP)
       else if (indexed inputs ok):
         backend::computeDenseGridIndexedTriangleContacts   (exact-contact)
       else:
         backend::computeDenseGridTriangleContacts          (packed fallback)
         or backend::computeExactTriangleContacts           (no-grid fallback)

    5. accumulateBackendStats(stageSnapshot, backendStats)

    6. If contacts produced: publishCudaTriangleContacts(...) into DetectionOutput

  pruneTriangleTopologyCache()                  ← evict entries not seen this frame
  stageSnapshot.wallMilliseconds = chrono now - phaseStart
  stageSnapshot.hostSynchronizationMilliseconds = max(0, wall - kernel)
  profiling::recordNarrowPhase(stageSnapshot)
```

### 6.3 Topology cache

`m_triangleTopologyCache` is a `std::unordered_map<const void*, CachedTriangleTopology>`
keyed by `CudaTriangleCollisionModel*`. The cache entry holds:

```cpp
struct CachedTriangleTopology {
    std::size_t triangleCount;
    std::size_t vertexCount;
    std::uint64_t topologyHash;
    std::uint64_t hitCount;
    std::uint64_t missCount;
    bool seenThisFrame;
    std::vector<std::uint32_t> triangleIndices;   // flat (v0,v1,v2,v0,v1,v2,...)
    std::vector<backend::TriangleVertex> positionScratch;     // fallback only
    std::vector<backend::TrianglePrimitive> triangleScratch;  // fallback only
};
```

Why this matters: SOFA surgical scenes have static topology. Without caching,
every frame would re-walk the SOFA triangle vector and rebuild the flat index
array. With caching, the FNV-1a-style topology hash is checked once at scene
load; thereafter the hash is reused and the index buffer is uploaded to GPU
only once via `surfaceId` + `topologyVersion` matching in
`DenseGridWorkspace`.

### 6.4 Extractors

| Helper | Returns | Used for |
|---|---|---|
| `extractCudaIndexedSurface(cm, ...)` | `TriangleIndexedSurface` with `devicePositions` from SOFA CUDA `deviceRead()` and `triangleIndices` from the topology cache | Tri-tri FBP, exact-contact, v-t self-collision |
| `extractCudaSelfCollisionSurfaces(cm, ...)` | Both a `TriangleIndexedSurface` AND a `PointCloudSurface` mirroring the same device positions and `surfaceId` | (Helper available; in practice the dispatch builds the PointCloudSurface inline from the existing indexed surface to avoid a redundant extract.) |
| `extractCudaPointCloud(cm, ...)` | `PointCloudSurface` from a `CudaPointCollisionModel`, reading positions from `getMechanicalState()->read(position)->getValue().deviceRead()`. `surfaceId` is the point-model pointer cast to uint64 | V-t cross-model |
| `extractCudaTriangles(cm, ...)` | Legacy: a packed `std::vector<TrianglePrimitive>` with positions baked in | Packed-fallback path when topology cache or direct device positions are unavailable |

### 6.5 Surface-id semantics

The kernel's own-corner exclusion (the line in `featureBasedVertexTriangleProximityKernel`
that skips pairs where the vertex id equals one of the triangle's corner
indices) is **gated by a `selfCollisionVertexExclusionStride != 0` runtime
argument**. The C++ wrapper sets that stride to 1 when
`pointCloud.surfaceId == triangleSurface.surfaceId`, and to 0 otherwise.
This is the only safety net preventing the v-t cross-model case from being
incorrectly affected by the exclusion that the self-collision case needs.

Result: surfaceIds must be **unique per collision model**. We use the model
pointer cast to `uint64_t`. Two different `CudaTriangleCollisionModel`
instances never collide ids (different heap pointers), and a point model
pointer is also distinct from any triangle model pointer.

---

## 7. Backend API boundary

File: `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h`

The header is the only file shared between the SOFA-side `.cpp` and the
CUDA-side `.cu`. No `__device__` types leak across this line; everything in
the header is plain C++ and includes nothing CUDA-specific. This keeps the
CUDA toolchain firewall narrow and makes the stub backend trivial to write.

### 7.1 Public entry points

```cpp
BackendStatus probe();                                              // "is CUDA ready?"
bool computeBroadPhasePairs(...);                                   // optional GPU AABB cull
bool prefilterNarrowPhasePairs(...);                                // legacy tree-pair prefilter
bool computeExactTriangleContacts(...);                             // legacy all-pairs SAT
bool computeDenseGridTriangleContacts(...);                         // packed dense-grid
bool computeDenseGridIndexedTriangleContacts(...);                  // indexed dense-grid (default)
bool computeFeatureBasedProximityContacts(...);                     // Phase 11: tri-tri FBP
bool computeFeatureBasedVertexTriangleContacts(...);                // Phase 12: v-t FBP (self + cross)
```

Each entry point fills a `BackendExecutionStats` describing per-stage timings,
H2D/D2H byte counts, kernel launch count, contact counts, and (for FBP)
per-class VF/FV/EE breakdowns.

### 7.2 Data types

```cpp
struct TriangleVertex { float x, y, z; };     // matches float3 layout, asserted at compile time
struct TrianglePrimitive { TriangleVertex p0, p1, p2; std::uint32_t triangleIndex; };

struct TriangleIndexedSurface {
    const TriangleVertex* positions;          // host pointer; null when devicePositions provided
    const TriangleVertex* devicePositions;    // GPU pointer from SOFA CUDA deviceRead()
    std::uint32_t vertexCount;
    const std::uint32_t* triangleIndices;     // flat (v0,v1,v2,v0,v1,v2,...)
    std::uint32_t triangleCount;
    std::uint64_t surfaceId;                  // identity for caching + exclusion
    std::uint64_t topologyVersion;            // bumps when triangleIndices change
};

struct PointCloudSurface {
    const TriangleVertex* positions;          // optional host pointer
    const TriangleVertex* devicePositions;    // GPU pointer (preferred)
    std::uint32_t pointCount;
    std::uint64_t surfaceId;                  // distinct from any triangle surface
    std::uint64_t topologyVersion;
};

enum class ProximityFeatureKind : std::uint8_t {
    VertexFace = 0,    // vertex of "first" against face of "second"
    FaceVertex = 1,    // face of "first" against vertex of "second"
    EdgeEdge   = 2,    // closest points on two edges
};

struct ProximityContact {
    std::uint32_t firstPrimitiveIndex;        // triangle id, or vertex id for v-t
    std::uint32_t secondPrimitiveIndex;
    ProximityFeatureKind featureKind;
    std::uint8_t firstFeatureLocalIndex;      // 0..2 vertex/edge index on first triangle
    std::uint8_t secondFeatureLocalIndex;
    std::uint8_t reserved;
    float firstBarycentrics[3];               // (1,0,0) selects a vertex; (1-s, s, 0) selects an edge endpoint
    float secondBarycentrics[3];
    TriangleVertex pointOnFirst;
    TriangleVertex pointOnSecond;
    TriangleVertex normal;                    // unit, from pointOnFirst to pointOnSecond
    float signedDistance;                     // ≥ 0 for proximity; the constraint solver assigns sign
};

struct ExactContact {                         // legacy SAT output, also receives FBP repack
    std::uint32_t firstTriangleIndex;
    std::uint32_t secondTriangleIndex;
    TriangleVertex pointOnFirst;
    TriangleVertex pointOnSecond;
    TriangleVertex normal;
    float signedDistance;
};

struct DenseGridConfig { /* grid bounds, resolution, contact distance, capacities, mode flags */ };
struct FeatureBasedProximityConfig { /* contactDistance, emitOnePerPair, computeBarycentrics, ... */ };

struct BackendExecutionStats { /* timings + byte counts + per-class contact counts */ };
struct FeatureBasedProximityStats { /* candidatePairCount, emittedContactCount, vf/fv/ee */ };
```

### 7.3 Design rules

- **No CUDA types in this header.** `cuda_runtime.h` is included only from
  the `.cu`.
- **No SOFA types either, except `sofa::core::CollisionModel*`** as an opaque
  identifier in the broad-phase entry.
- **Stable ABI.** Adding a field to `BackendExecutionStats` does not break
  callers — they read by name. Adding a new entry point is additive.
- **No exceptions.** Backend returns `bool` + writes a `std::string`
  diagnostic. The narrow phase decides whether to log/fall back.

---

## 8. CUDA backend internals

File: `SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu`
(~4500 lines).

### 8.1 The persistent workspace

A single thread-local `DenseGridWorkspace` lives behind
`denseGridWorkspace()`. It owns all reusable GPU allocations:

```text
// triangle path
tissueTriangles, toolTriangles, tissueAabbs, toolAabbs   (packed-fallback only)
grid (DeviceCellBucket per cell)
cellTissueIds, cellToolIds                               (per-bucket triangle/vertex IDs)
candidatePairs (uint64 packed (high << 32 | low))
pairHashKeys                                             (unsigned long long for atomicCAS)
contacts (DeviceExactContact)

// counters
rawCandidateCount, candidateCount, contactCount, overflowCount, activeCellCount
denseGridStats

// indexed inputs
indexedTissuePositions, indexedToolPositions
indexedTissueIndices, indexedToolIndices
pinned* counterparts for host staging

// FBP additions (Phase 11)
proximityContacts (typed externally as DeviceProximityContact*)
proximityContactCount, proximityOverflowCount
proximityVfCount, proximityFvCount, proximityEeCount
proximityCountersHostPinned (5x uint32, pinned, for one batched D2H)
fbpStartEvent, fbpEndEvent (persistent cudaEvents, lazy-created)

// stable identity / cache invalidation
indexedTissueSurfaceId, indexedToolSurfaceId
indexedTissueTopologyVersion, indexedToolTopologyVersion
frameCounter
```

`workspace.ensure(...)` reallocates only when capacities grow.
`workspace.ensureIndexedInput(...)` does the same for indexed buffers.
`workspace.ensureProximityStorage(...)` adds Phase 11 buffers.
`workspace.ensureFbpEvents()` creates `fbpStartEvent` / `fbpEndEvent` once.
`workspace.release()` is called from the destructor on plugin unload.

This is why steady-state runs report `avg_workspace_resize_count=0`: the
first frame allocates, every subsequent frame reuses.

### 8.2 Device-side math (FBP additions)

Three `__device__ __forceinline__` helpers underpin both FBP paths:

```cpp
// Ericson 5.1.5 — closest point on triangle to point p, as barycentrics.
__device__ float3 closestPointOnTriangleBary(float3 p, float3 a, float3 b, float3 c);

// Reconstruct a world-space point from (a, b, c, barys).
__device__ float3 reconstructFromBary(float3 a, float3 b, float3 c, float3 bary);

// Ericson 5.1.9 — closest points of two line segments p1q1 and p2q2.
__device__ void closestPointSegmentSegment(
    float3 p1, float3 q1, float3 p2, float3 q2,
    float& s, float& t, float3& c1, float3& c2);
```

Both Ericson routines are branchless except for region selection in
`closestPointOnTriangleBary` (7 region checks for vertex / edge / interior).
They produce continuous gradients — a moving vertex's `closestPoint` slides
smoothly across triangle features as the geometry changes. This is the
property a soft-body constraint solver needs.

### 8.3 The seven dense-grid kernels

```cpp
resetDenseGridKernel
    Clears DeviceCellBucket per cell + small counters (rawCandidateCount, candidateCount,
    contactCount, overflowCount, activeCellCount) + optionally the pair hash (we use
    cudaMemset for that instead because it benchmarked faster on the 1650 Ti).

insertIndexedTrianglesKernel
    One thread per triangle. Loads vertices through indexedTriangleAt(positions, indices, triId).
    Inflates by contactDistance to form AABB. Walks the cell span; for each cell, atomicAdd
    on the bucket counter; if within bucketCapacity, write triangleId. Overflow → atomic counter.
    insertTissue flag picks tissueCount vs toolCount.

insertIndexedPointsKernel                                  (Phase 12 NEW)
    Same shape as insertIndexedTrianglesKernel but each thread handles one vertex.
    The AABB is the (point ± contactDistance) cube. Writes to cellToolIds.

generateDenseGridUniqueCandidatePairsKernel
    One block per cell (ALL cells), threads stride over (tissueCount * toolCount) pair products.
    Each pair is encoded as uint64 = (tissueId << 32) | toolId. atomicCAS into pairHashKeys
    table; survivors atomicAdd into candidatePairs[] / candidateCount.
    NOTE: superseded by the active-cell variant below when useToolActiveCellGeneration=true (default).

generateActiveDenseGridUniqueCandidatePairsKernel          (Phase 15, DEFAULT)
    Same per-cell pair logic, but iterates only the tool-built active-cell list
    (device-reads *activeCellCount, grid-strides). Launched <<<1024, 256>>> instead of
    <<<32768, 256>>>. ~38x faster than the all-cells variant on the one-tissue scene.

featureBasedProximityKernel                                (Phase 11; grid-stride since §5.17)
    GRID-STRIDES over all candidate pairs (for idx = tid; idx < pairCount; idx += stride).
    Loads both triangles via indexedTriangleAt. Runs 6 VF + 9 EE tests, keeps the closest
    feature pair. If best distance ≤ contactDistance: atomicAdd into contactCount;
    write DeviceProximityContact; atomicAdd into vfCount/fvCount/eeCount.
    The grid-stride is REQUIRED for correctness — the launch grid is a fixed saturation
    target (1024 blocks), and large scenes produce far more pairs than launched threads.

featureBasedVertexTriangleProximityKernel                 (Phase 12; grid-stride since §5.17)
    GRID-STRIDES over all (triangleId, vertexId) candidate pairs. Loads the triangle and vertex.
    If selfCollisionVertexExclusionStride != 0 and vertexId matches one of the triangle's
    three corner indices: skip (continue). Otherwise runs closestPointOnTriangleBary once,
    emits a VertexFace ProximityContact when distance ≤ contactDistance.

resetProximityCountersKernel
    One-thread kernel that zeros proximityContactCount + proximityOverflowCount +
    vf/fv/ee counters. Trivial cost.
```

### 8.4 Per-path launch sequences

**Exact-contact (legacy default):**

```text
1. resetDenseGridKernel
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel (tissue)
4. insertIndexedTrianglesKernel (tool)
5. generateDenseGridUniqueCandidatePairsKernel
[6. exactTriangleContactKernel — only if computeDeviceContactsWhenContactsStayOnDevice]
```

→ 5 launches + 1 memset for detection-only, 6 launches with exact contacts.

**Tri-tri FBP:**

```text
1-5: same as above (broad cull via computeDenseGridIndexedTriangleContacts with
     readback off so the wrapper does not synchronize)
6. resetProximityCountersKernel
7. featureBasedProximityKernel
```

→ 7 launches + 1 memset.

**V-t self-collision and cross-model:**

```text
1. resetDenseGridKernel
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel (the triangle mesh)
4. insertIndexedPointsKernel    (the vertex set)
5. generateDenseGridUniqueCandidatePairsKernel
6. resetProximityCountersKernel
7. featureBasedVertexTriangleProximityKernel
```

→ Same 7-launch structure, swapping the second `insertIndexedTrianglesKernel`
for `insertIndexedPointsKernel` and the FBP kernel for the v-t variant.

(The benchmark controller reports `avg_kernel_launch_count=6` for v-t paths
because in `computeFeatureBasedVertexTriangleContacts` the `resetProximity` +
`featureBasedVertexTriangle` pair is grouped into 6 explicit launch
increments. Cosmetic accounting — actual SM activity is the same.)

### 8.5 Sync discipline

This is what makes the "fast path" actually fast.

| Where | What | Whose decision |
|---|---|---|
| Broad-cull internal events | `cudaEventRecord(totalStart)` ... `cudaEventRecord(totalEnd)` ... `cudaEventSynchronize(totalEnd)` ... `cudaEventElapsedTime` | Inside `computeDenseGridIndexedTriangleContacts`, gated by always-on event window |
| Broad-cull counter readback | `cudaMemcpy` of rawCandidateCount, candidateCount, overflowCount, stats, contactCount | Gated by `readCountersWhenContactsStayOnDevice`. FBP wrapper sets this to `proximityReadContactCounter`, so it's OFF in production |
| FBP wrapper events | `cudaEventRecord(workspace.fbpStartEvent)` ... `cudaEventRecord(workspace.fbpEndEvent)` ... `cudaEventSynchronize` ... `cudaEventElapsedTime` | Only when `proximityConfig.readContactCounter` is true |
| FBP counter readback | Five `cudaMemcpyAsync` into one pinned host buffer + single `cudaDeviceSynchronize` | Only when `proximityConfig.readContactCounter` is true |
| FBP contact array download | `cudaMemcpy` of N × `sizeof(DeviceProximityContact)` | Only when `proximityConfig.keepContactsOnDevice` is false |

When all readback flags are off (the production detection-only mode), the
narrow phase issues kernels and returns. The next frame's CPU work begins
while the GPU is still finishing the previous frame's FBP kernel. The CPU
narrow_wall measurement therefore captures only the host-side scheduling
cost — it does not block on GPU completion.

This is why one-tissue-one-blade with FBP enabled reports ~0.7 ms narrow_wall
when the FBP kernel itself runs for ~0.5-1 ms on the GPU: the work overlaps
the next frame's host work.

---

## 9. Profiling pipeline

Files:

- `SofaGpuCollision/src/SofaGpuCollision/GpuPipelineProfiling.{h,cpp}`
- `SofaGpuCollision/src/SofaGpuCollision/GpuPipelineBenchmarkController.{h,cpp}`

### 9.1 `StepSnapshot` accumulator

A single thread-local `RuntimeState` holds a `StepSnapshot` (one
`StageSnapshot` for broad phase + one for narrow phase) and a rolling
`AggregateSnapshot` of completed steps. A `std::mutex` guards every write.

The broad/narrow phase write into the current step via `recordBroadPhase`
and `recordNarrowPhase`. The benchmark controller calls `finishStep()` once
per frame, which finalizes the step and folds its stats into the aggregate.

### 9.2 `StageSnapshot` fields (2026-05-25 layout)

```cpp
struct StageSnapshot {
    // wall-time + cuda-event-time
    double wallMilliseconds;                     // chrono around the whole stage
    double gpuKernelMilliseconds;                // sum of cudaEventElapsedTime measurements
    double hostSynchronizationMilliseconds;     // derived: max(0, wall - kernel)

    // host preparation
    double hostPreparationMilliseconds;
    double sofaTriangleExtractionMilliseconds;
    double backendTrianglePackMilliseconds;
    double hostToDeviceMilliseconds;
    double deviceAllocationMilliseconds;

    // per-kernel breakdown (broad cull)
    double denseGridClearMilliseconds;
    double denseGridCounterClearMilliseconds;
    double denseGridTissueAabbMilliseconds;     // legacy split-aabb path
    double denseGridToolAabbMilliseconds;       // legacy split-aabb path
    double denseGridInsertTissueMilliseconds;
    double denseGridInsertToolMilliseconds;
    double denseGridGeneratePairsMilliseconds;
    double denseGridCandidateReadbackMilliseconds;
    double denseGridSortUniqueMilliseconds;
    double denseGridSortUniqueHostMilliseconds;
    double denseGridExactContactMilliseconds;
    double denseGridContactCountReadbackMilliseconds;
    double denseGridContactDownloadMilliseconds;

    // FBP-specific
    double featureBasedProximityKernelMilliseconds;

    // SOFA publication
    double sofaDetectionOutputPublishMilliseconds;

    // byte counters
    std::uint64_t hostToDeviceBytes, deviceToHostBytes, deviceAllocationBytes;

    // launch counters
    std::uint32_t kernelLaunchCount, cudaMemsetCount, workspaceResizeCount;

    // primitive counters
    std::uint32_t inputPrimitiveCount, outputPairCount, outputCandidateCount, outputContactCount;
    std::uint32_t rawCandidateCount, uniqueCandidateCount;
    std::uint32_t gridCellCount, activeMixedCellCount;
    std::uint32_t tissueInsertCount, toolInsertCount;
    std::uint32_t maxTissueCellOccupancy, maxToolCellOccupancy;
    std::uint32_t overflowCount, hashDedupeProbeOverflowCount;

    // FBP per-class breakdown
    std::uint32_t vfContactCount, fvContactCount, eeContactCount;

    bool gpuUsed, cpuFallbackUsed;
};
```

### 9.3 CSV + summary writer

`GpuPipelineBenchmarkController` emits two files per run:

- `<label>_timings.csv` — one row per frame, ~50 columns
- `<label>_summary.txt` — `avg_*` aggregates + scene metadata

The CSV columns map one-to-one to `StageSnapshot` fields (with broad and
narrow values both serialized). The new Phase 11+12 columns are appended at
the end of each row:

```text
narrow_feature_based_proximity_kernel_ms
narrow_host_synchronization_ms
narrow_vf_contact_count
narrow_fv_contact_count
narrow_ee_contact_count
```

The summary file has corresponding `avg_*` lines. The controller writes the
summary on `finalize()` (called from the destructor) so the file appears
after the last frame.

### 9.4 Sampled counter readback

`d_proximityCounterReadbackInterval` controls a sampled mode for long
profiling runs. When `interval > 0` and `m_frameCounter % interval == 0`,
the FBP wrapper takes the readback path (cudaEvents + batched counter
copy) for that frame only. Other frames stay on the no-readback fast path.
This lets you collect VF/FV/EE breakdowns periodically without paying the
sync every frame.

---

## 10. Data ownership map

| Resource | Owner | Lifetime |
|---|---|---|
| Vertex positions on GPU | SOFA `MechanicalObject<CudaVec3f>` | Scene lifetime; we hold a `deviceRead()` pointer per frame |
| Triangle index array on CPU | SOFA `MeshTopology` | Scene lifetime; we cache a flat copy in `m_triangleTopologyCache` |
| Triangle index array on GPU | `DenseGridWorkspace.indexedTissueIndices` / `indexedToolIndices` | Plugin lifetime; uploaded once per topology version |
| Dense grid buffers (cell buckets, pair hash, candidate pairs) | `DenseGridWorkspace` | Plugin lifetime; reallocated only when capacities grow |
| FBP contact buffer | `DenseGridWorkspace.proximityContacts` (cast to `DeviceProximityContact*`) | Plugin lifetime |
| CUDA events | `DenseGridWorkspace.fbpStartEvent` / `fbpEndEvent` | Plugin lifetime (created lazily, destroyed in workspace destructor) |
| Pinned host counter buffer | `DenseGridWorkspace.proximityCountersHostPinned` (5 × uint32) | Plugin lifetime |
| SOFA `DetectionOutput` records | SOFA's `NarrowPhaseDetection` | Per frame; we `clear()` then `push_back` |
| Benchmark CSV/summary files | `GpuPipelineBenchmarkController` | Process lifetime |

Nothing crosses the CPU/GPU boundary on the production fast path. Positions
are GPU-only, indices stay GPU-cached, contacts stay GPU-resident.

---

## 11. Failure modes and fallback semantics

| Failure | Symptom | Fallback |
|---|---|---|
| `cuda_runtime` missing at build time | Plugin builds against `GpuCollisionBackendStub.cpp`; `probe()` returns false | Every backend entry returns false with diagnostic; narrow phase delegates to `BVHNarrowPhase` when `allowCPUFallback=True` |
| No CUDA device at runtime | `probe()` returns false | Same as above |
| Dense-grid bucket overflow | `BackendExecutionStats.overflowCount > 0` | Contact data may be incomplete; the kernel atomically tallies overflow so it's visible. Increase `maxTissueTrianglesPerCell` / `maxToolTrianglesPerCell` or refine the grid resolution |
| Candidate-pair overflow | `BackendExecutionStats.overflowCount > 0` (same counter) and `outputCandidateCount == maxCandidatePairs` | Increase `maxCandidatePairs` |
| Proximity contact overflow | `proximityOverflowCount > 0` | Increase `proximityMaxContacts` |
| Pair hash dedupe probe overflow | `hashDedupeProbeOverflowCount > 0` | Capacity is `nextPowerOfTwo(maxCandidatePairs * 2)`; increase `maxCandidatePairs` to grow the hash table |
| Cross-model v-t with same-template surfaces | (e.g. two `CudaTriangleCollisionModel` instances) | The cross-model branch checks for `dynamic_cast<CudaPointCollisionModel>`; if no point model is present it falls through to the tri-tri branch |
| Indexed extraction fails (e.g. empty topology) | `extractCudaIndexedSurface` returns false | Pair is pushed to `legacyPairs`; the SOFA CPU narrow phase processes it |
| Backend call returns false with diagnostic | `m_reportedFallback` flag emits one `msg_warning` then suppresses | Pair is pushed to legacyPairs |
| Stub backend in use | Every call returns false | CPU fallback covers correctness |

---

## 12. Glossary

**Broad phase** — Filters pairs of collision models that cannot possibly
intersect (by AABB or coarser test) before invoking the narrow phase.

**Narrow phase** — Generates actual contact records (point-pair + normal +
distance) for pairs that survive broad phase.

**Dense grid** — A regular 3D grid of cells. Each cell holds a bucket of
"tissue" primitive ids and a bucket of "tool" primitive ids that overlap the
cell. Candidate pairs are generated cell-by-cell.

**Indexed input** — A surface description where vertex positions and
triangle indices are passed as separate arrays. Avoids the per-triangle
repack of vertices that a "packed" representation (`TrianglePrimitive[]`)
would require.

**Direct device positions** — Using a SOFA CUDA `MechanicalObject`'s
`deviceRead()` pointer directly, skipping any host-side rebuild.

**Topology version** — A frame-stable token that bumps when triangle
indices change. The backend uploads indices only when the workspace's cached
version disagrees with the surface's version.

**SAT** — Separating Axis Theorem. The legacy exact-contact kernel uses 11
candidate separating axes (face normals + edge cross products) to determine
if two triangles intersect.

**FBP** — Feature-Based Proximity. Instead of a boolean SAT intersection
test, FBP computes the closest pair of features (vertex-face or edge-edge)
between two triangles and emits a contact if the distance is below a
threshold. Produces continuous, gradient-smooth contact data suitable for a
constraint solver.

**VF / FV / EE** — Vertex-Face (vertex of A vs face of B), Face-Vertex
(face of A vs vertex of B), Edge-Edge.

**v-t** — Vertex-Triangle. A simplified FBP variant where one side is a
point cloud (no edges, no faces). Each (vertex, triangle) candidate produces
at most one VertexFace contact.

**Self-collision** — A collision-model pair where both sides are the same
mesh (`pair.first == pair.second`). Triggered by SOFA's broad phase when
`selfCollision=True` is set on a `TriangleCollisionModel`.

**Own-corner exclusion** — In the v-t self-collision kernel, the rule that
skips (vertex, triangle) pairs where the vertex is one of the triangle's
own three corners. Prevents false-positive contacts at distance zero.

**Topology cache** — `m_triangleTopologyCache` in `GpuCollisionNarrowPhase`.
Keyed by `CudaTriangleCollisionModel*`. Avoids rebuilding the flat triangle-
index array every frame for static-topology scenes.

**Workspace** — `DenseGridWorkspace` singleton in the CUDA backend that owns
all reusable GPU allocations + persistent CUDA events + pinned counter
buffer. Plugin lifetime; never freed except at unload.

**Detection-only** — A run mode where collision contacts are produced but
not published into SOFA's response pipeline (`copyContactsToHost=false`,
no `ContactManager` in the scene). Used for raw collision-detection
throughput benchmarks.

**Sampled counter readback** — A profiling mode where contact counters are
read back from GPU every Nth frame instead of every frame. Keeps the fast
path clean while still collecting VF/FV/EE breakdowns periodically.

---

## 13. Quick cross-reference

| Concept | File | Line region |
|---|---|---|
| Plugin init | `init.cpp` | whole file |
| Backend probe | `cuda/GpuCollisionBackend.cu` | `BackendStatus probe()` |
| Broad-phase dispatch | `GpuCollisionBroadPhase.cpp` | `endBroadPhase()` |
| Topology cache | `GpuCollisionNarrowPhase.{h,cpp}` | `m_triangleTopologyCache`, `markTriangleTopologyCacheUnused`, `pruneTriangleTopologyCache` |
| Tri-tri FBP dispatch | `GpuCollisionNarrowPhase.cpp` | `endNarrowPhase()` `else if (usingIndexedDenseGrid && d_useFeatureBasedProximity.getValue())` |
| V-t self-collision dispatch | `GpuCollisionNarrowPhase.cpp` | `endNarrowPhase()` `if (routeToVertexTriangle)` |
| V-t cross-model dispatch | `GpuCollisionNarrowPhase.cpp` | `endNarrowPhase()` top-of-loop cross-model branch |
| Closest-point math | `cuda/GpuCollisionBackend.cu` | `closestPointOnTriangleBary`, `closestPointSegmentSegment`, `reconstructFromBary` |
| FBP tri-tri kernel | `cuda/GpuCollisionBackend.cu` | `featureBasedProximityKernel` |
| V-t insertion kernel | `cuda/GpuCollisionBackend.cu` | `insertIndexedPointsKernel` |
| V-t narrow kernel | `cuda/GpuCollisionBackend.cu` | `featureBasedVertexTriangleProximityKernel` |
| Workspace | `cuda/GpuCollisionBackend.cu` | `struct DenseGridWorkspace` |
| FBP public function | `cuda/GpuCollisionBackend.cu` | `computeFeatureBasedProximityContacts` |
| V-t public function | `cuda/GpuCollisionBackend.cu` | `computeFeatureBasedVertexTriangleContacts` |
| CSV/summary writer | `GpuPipelineBenchmarkController.cpp` | `flushRows()`, `writeSummary()` |
| Cross-model publication | `GpuCollisionNarrowPhase.cpp` | `publishCudaPointTriangleContacts` |
| Standalone FBP+v-t bench | `src/tools/DenseGridBackendBench.cpp` | runs all three paths back-to-back, writes 3 CSVs |
| Nsight launcher (new kernels) | `scripts/run_nsight_fbp_profile_wsl.sh` | targets `featureBasedProximityKernel` + `featureBasedVertexTriangleProximityKernel` |

For the latest measured numbers see `guide/plan.md` (current phase verification)
and the run logs under `output/benchmark_logs/`.

---

## 14. Follow-up findings (2026-05-25, post Phase 12 cross-model)

Three follow-up items completed after the initial Phase 12 wiring:

### Cross-model `DetectionOutput` publication

- New helper `publishCudaPointTriangleContacts` uses
  `TDetectionOutputVector<CudaPointCollisionModel, CudaTriangleCollisionModel>`.
- SOFA's `TDetectionOutputVector` is templated on any two `CollisionModel`
  types, so no new specialization was needed.
- The narrow phase now couples `copyContactsToHost` and
  `proximityKeepContactsOnDevice`: when the former is true, the latter is
  forced to false so the proximity contacts get downloaded before
  publication.
- Verified: 332 FPS at 254 contacts/frame with 19 320 D2H bytes
  (vs 1968 FPS detection-only).

### Standalone backend bench (`DenseGridBackendBench`)

- Now runs three phases per invocation: legacy exact-contact, tri-tri FBP,
  v-t cross-model.
- Writes three CSVs: `<label>.csv`, `<label>_fbp.csv`, `<label>_vt.csv`.
- Reference run (12 800 + 224 triangles): exact-contact 4.36 ms wall,
  tri-tri FBP 2.43 ms wall / 488 contacts, v-t cross-model 1.63 ms wall /
  38 contacts.
- Toggle with `SOFA_BACKEND_BENCH_RUN_FBP` / `SOFA_BACKEND_BENCH_RUN_VT`
  env vars (default both on).

### Nsight Compute capture on the new kernels

- `scripts/run_nsight_fbp_profile_wsl.sh` profiles all three scenes
  (tri-tri FBP, v-t self, v-t cross-model).
- Per-kernel metrics in
  `reports/gpu_collision_phase11_12_kernel_profile_20260525.md`.
- Headline: **none of the new kernels are bottlenecks**.
  `featureBasedProximityKernel` is register-bound (68 regs/thread → ~50%
  occupancy cap) but cheap (~17 µs); future tuning options are documented
  but not urgent.

---

## 15. Generation + event optimizations (IMPLEMENTED 2026-05-25)

Two optimizations target the measured bottleneck of the tri-tri FBP fast
path. **Both are now implemented** (Phase 15 opt-in via
`useToolActiveCellGeneration`; Phase 16 always on). Measured result:
one-tissue fast path **221 → 943 FPS (4.3×)**, generation kernel
**300 µs → 7.9 µs (38×)**, contact counts bit-identical. Full numbers in
`guide/plan.md` §5.15-5.16 and `reports/gpu_collision_phase15_16_optimization_20260525.md`.
This section documents the *mechanisms* as built.

### 15.1  Where the time actually goes (the motivation)

Per `reports/end_to_end_verification_20260525.md` §7, the tri-tri FBP path
spends ~380 µs on the GPU, of which **~300 µs (≈80%) is
`generateDenseGridUniqueCandidatePairsKernel`**, plus ~450 µs of CPU/sync
overhead on top, of which **~80 µs is per-frame CUDA event churn** inside
the broad-cull function.

```text
GPU kernel time (≈380 µs):
  generateDenseGridUniqueCandidatePairsKernel   ~300 µs  ← 80%, the target
  insertIndexedTrianglesKernel (blade)           ~30 µs  (launch-bound)
  insertIndexedTrianglesKernel (tissue)          ~29 µs
  featureBasedProximityKernel                    ~17 µs
  resetDenseGridKernel + cudaMemset              ~5 µs

CPU overhead (wall − kernel ≈ 450 µs):
  cudaEventCreate/Destroy ×4 (broad cull)        ~80 µs  ← the target
  7 kernel-launch API calls                      ~40 µs
  C++ setup + extractors                         ~50-100 µs
  SOFA dispatch + NVTX + mutex + stats           ~100+ µs
  inter-kernel stream gaps                       ~150 µs
```

### 15.2  Planned: tool-active-cell candidate generation (§5.15)

**Root cause.** `generateDenseGridUniqueCandidatePairsKernel` launches
`<<<cellCount, 256>>>` = one block per cell for all 32 768 cells. The
blade occupies ~30 cells, so ~99% of blocks read an empty bucket and exit.
The 300 µs is block-scheduling overhead, not pair work (26% SM throughput).

**Mechanism.** Build the **mixed-cell list during the tool insert** — no
separate scan. In `insertTriangleAabbIntoGrid`, for a tool triangle, the
first one to claim slot 0 in a cell that already has tissue appends the
cell id:

```cpp
const uint32_t localIndex = atomicAdd(&grid[cellId].toolCount, 1u);
if (buildToolActiveList && localIndex == 0u && grid[cellId].tissueCount > 0u)
    activeCellIds[atomicAdd(activeCellCount, 1u)] = cellId;   // dedup'd, can't overflow
```

This works because (a) tissue insert runs before tool insert on the
serialized stream so `tissueCount` is final, (b) `localIndex == 0`
deduplicates, (c) `activeCellIds` is already sized to `cellCount`.

Generation then reuses the **existing**
`generateActiveDenseGridUniqueCandidatePairsKernel`, which device-reads
`*activeCellCount` and grid-strides over the active list. Launch a fixed
modest grid (`<<<256, 256>>>`) — no host readback — so the ~30 active cells
run in the first ~30 blocks and the rest exit immediately. **128× fewer
launched blocks.**

**How this differs from `compactActiveDenseGridCellsKernel` (the regressed
`compactActiveCells`).** That kernel scans **all 32 768 cells** in a
separate pass to find mixed cells — the scan reads the whole 256 KB grid,
costing what it saves. The new design **never scans all cells**; it builds
the list as a side effect of the insert it already runs.

**Ordering dependency.** Requires tissue-insert-before-tool-insert, so it
is mutually exclusive with `batchTriangleInsert` (the fused-insert
experiment). The backend must reject the combination.

**New flag.** `useToolActiveCellGeneration` (`DenseGridConfig` + narrow-
phase Data field + `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION`), distinct from
`compactActiveCells`.

Updated launch sequence when enabled:

```text
1. resetDenseGridKernel                       (also zeroes activeCellCount)
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel (tissue)
4. insertIndexedTrianglesKernel (tool, buildToolActiveList=true)  ← builds active list
5. generateActiveDenseGridUniqueCandidatePairsKernel<<<256,256>>> ← ~30 cells, not 32 768
6. resetProximityCountersKernel
7. featureBasedProximityKernel
```

Same 7-launch count; launch #5 swaps the all-cells generator for the
active-cell generator with a 128× smaller grid.

### 15.3  Planned: workspace-cached broad-cull CUDA events (§5.16)

**Root cause.** `computeDenseGridIndexedTriangleContacts` and
`computeDenseGridTriangleContacts` each `cudaEventCreate` / `cudaEventDestroy`
their `stageStart/stageEnd/totalStart/totalEnd` events **every frame**
(~40-80 µs of driver churn).

**Mechanism.** Mirror the Phase 5.14 FBP-event fix: add `broadStageStart`,
`broadStageEnd`, `broadTotalStart`, `broadTotalEnd` + `broadEventsReady` +
`ensureBroadEvents()` to `DenseGridWorkspace`, lazy-create once, reuse
across frames, free in the workspace destructor. `cudaEventRecord`
overwrites the timestamp each frame so reuse is safe.

This extends the workspace's event ownership from just the FBP events to
all timing events, completing the "no per-frame CUDA object creation"
principle the workspace already follows for device buffers.

### 15.4  Combined expectation

| Metric | Current | After §5.15 | After §5.15 + §5.16 |
|---|---:|---:|---:|
| generation kernel | ~300 µs | < 30 µs | < 30 µs |
| host sync | ~285 µs | ~285 µs | ~205 µs |
| narrow wall | ~0.83 ms | ~0.5 ms | ~0.43 ms |
| one-tissue FPS | ~672 | ~900 | ~1000 |

**Measured 2026-05-25 (one-tissue/one-blade, fast path, same cool GPU
back-to-back):**

| Metric | Baseline (flag off) | Active (flag on) |
|---|---:|---:|
| generation kernel grid | 32 768 blocks | **1 024 blocks** |
| generation kernel duration | ~300 µs | **7.9 µs** |
| narrow kernel (all) | 3.83 ms | **0.35 ms** |
| narrow wall | 4.00 ms | **0.56 ms** |
| FPS | 221 | **943** |
| contact count (validation) | 56 (all EE) | 56 (all EE) — identical |
| overflow | 0 | 0 |

### 15.5  Default flip + large-scene validation + grid-stride fix

The large-tissue A/B (79 520 collision elements, subdivided blade) cleared
the default flip and surfaced a separate correctness bug:

- **Large-tissue A/B:** baseline 108 FPS / 4.63 ms, active 116 FPS / 4.09 ms
  (1.08×). Contact output bit-identical (8018 contacts: 5397 VF / 880 FV /
  1741 EE), unique_candidate_count 322 560 identical, overflow 0. Smaller
  win than one-tissue because the subdivided blade touches many cells, so
  the tool/tissue asymmetry is weaker and the FBP kernel itself dominates —
  but still faster, never a regression.
- **Default flipped to ON:** `DenseGridConfig.useToolActiveCellGeneration`,
  the `GpuCollisionNarrowPhase` Data field, and the benchmark-scene env-flag
  defaults are all now `true`. `compactActiveCells` is effectively superseded.
- **Grid-stride correctness fix (§5.17 in plan.md):** the large-scene A/B
  exposed a latent bug — `featureBasedProximityKernel` and
  `featureBasedVertexTriangleProximityKernel` processed one pair per thread
  with no grid-stride, so the fixed 65 536-thread over-launch silently
  dropped ~80% of the 322 560 large-scene pairs. Both kernels now grid-stride
  over all pairs; the launch grid is a pure saturation target (1024 blocks).
  Large-scene contact count corrected from the buggy ~3700 to the true 8018.
  Small scenes (< 65 536 pairs) were unaffected throughout.

The Phase 15 win exceeded the target on small scenes; the event caching
(Phase 16) is always on; the grid-stride fix makes FBP/v-t correct at any
scale.
