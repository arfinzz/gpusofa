# 09 — Vertex-Triangle Paths (Self-Collision & Cross-Model)

Files 06–08 traced the **triangle-triangle** path: two triangle meshes, dense
grid, 15 feature tests per pair. This file covers two *variants* that reuse all
the same machinery but treat one side as a **point cloud** (a set of loose
vertices) instead of a triangle mesh.

Both are real, wired, and tested. They're controlled by the
`useVertexTriangleProximity` Data field.

---

## 9.1 Why a vertex-triangle path at all?

Two surgical use cases the triangle-triangle path doesn't cover well:

1. **Self-collision.** When tissue folds onto itself (think a flap of skin
   doubling over during a cut), the mesh's own vertices come close to its own
   non-adjacent triangles. You want to detect that.

2. **A tool that's a point cloud.** Some instruments are best modeled as a set
   of sample points (the tip of a probe, scattered contact points of a sponge)
   rather than a closed triangle mesh. You want those points tested against the
   tissue surface.

Both reduce to the same question: **for each vertex (a point), what is the
closest tissue triangle, and is it within `contactDistance`?** That's simpler
than triangle-triangle — it's just *one* closest-point-on-triangle test per
(vertex, triangle) pair, not 15.

---

## 9.2 The shared machinery

The vertex-triangle path reuses almost everything from files 06–08:

- The **dense grid** — same cells, same insertion, same candidate generation.
- The **closest-point-on-triangle math** (`closestPointOnTriangleBary`, Ericson
  5.1.5) — the exact same function from file 08.
- The **workspace**, the **over-launch trick**, the **sync bypass**.

What's different:

- One side is inserted as **points** instead of triangles (a new kernel,
  `insertIndexedPointsKernel`).
- The narrow kernel is `featureBasedVertexTriangleProximityKernel` — it runs
  **one** closest-point test per pair, not 15.
- Every contact is, by construction, a **Vertex-Face** contact.

---

## 9.3 Inserting points into the grid — `insertIndexedPointsKernel`

A point has no extent, so its "bounding box" is just a tiny cube of size
`2 × contactDistance` centered on the point:

```cpp
const auto p = positions[tid];     // one vertex
const float r = config.contactDistance;   // 0.03 (or scene-specific)
DeviceAabb aabb;
aabb.minX = p.x - r;  aabb.maxX = p.x + r;
// ... y and z ...
```

Then it's inserted into the cells that box overlaps, exactly like a triangle,
but into the **tool bucket**:

```cpp
const std::uint32_t localIdx = atomicAdd(&grid[cellId].toolCount, 1u);
if (localIdx < bucketCapacity)
    cellToolIds[cellId * bucketCapacity + localIdx] = tid;   // tid = vertex ID
```

So in the vertex-triangle path:

- The **tissue triangles** go into the **tissue buckets** (via the normal
  `insertIndexedTrianglesKernel`).
- The **vertices** go into the **tool buckets** (via the new
  `insertIndexedPointsKernel`).

The candidate-generation kernel then pairs them up exactly as before, producing
`(triangleId, vertexId)` candidate pairs.

---

## 9.4 The narrow kernel — `featureBasedVertexTriangleProximityKernel`

One thread per `(triangle, vertex)` candidate pair:

```cpp
const std::uint64_t pair = candidatePairs[tid];
const std::uint32_t triId   = pair >> 32;          // triangle ID
const std::uint32_t pointId = pair & 0xffffffff;   // vertex ID

// --- self-collision exclusion (see 9.5) ---
if (selfCollisionVertexExclusionStride != 0) {
    const std::uint32_t i0 = triangleIndices[3*triId+0];
    const std::uint32_t i1 = triangleIndices[3*triId+1];
    const std::uint32_t i2 = triangleIndices[3*triId+2];
    if (pointId == i0 || pointId == i1 || pointId == i2) return;   // skip own corners
}

const DeviceTriangle tri = indexedTriangleAt(trianglePositions, triangleIndices, triId);
const float3 p = pointPositions[pointId];

const float3 bary = closestPointOnTriangleBary(p, tri.p0, tri.p1, tri.p2);
const float3 cp   = reconstructFromBary(tri.p0, tri.p1, tri.p2, bary);
const float distSq = dot3(sub3(cp, p), sub3(cp, p));

if (distSq > contactDistance*contactDistance) return;   // too far

// emit a VertexFace contact
const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
contacts[outIdx] = { pointId, triId, /*VertexFace*/, /*bary=(1,0,0)*/, bary, p, cp, normal, dist };
atomicAdd(vfCount, 1u);
```

It's the same closest-point math from file 08, but run *once* (point vs face)
instead of 15 times. That's why the vertex-triangle kernel is about 2× cheaper
than the triangle-triangle kernel and uses only 32 registers per thread vs 68
(measured in `reports/gpu_collision_phase11_12_kernel_profile_20260525.md`).

---

## 9.5 Self-collision and "own-corner exclusion"

Here's a problem unique to self-collision. If you test a mesh against *itself*,
then vertex 500 will be tested against the triangles that *contain* vertex 500.
But a vertex is at distance *zero* from its own triangles — so you'd get a
flood of false "contacts" at distance 0.

The fix is **own-corner exclusion**. The kernel skips any (triangle, vertex)
pair where the vertex is one of that triangle's three corners:

```cpp
if (pointId == i0 || pointId == i1 || pointId == i2) return;
```

This is gated by a runtime flag `selfCollisionVertexExclusionStride`:

- **Self-collision** (same mesh both sides) → exclusion **ON**. A vertex is not
  matched against its own adjacent triangles, only against far-away parts of the
  same mesh.
- **Cross-model** (point cloud vs a *different* triangle mesh) → exclusion
  **OFF**. There's no "own corner" because the vertex and the triangles belong
  to different objects.

### How the kernel knows which mode it's in

The C++ wrapper decides by comparing `surfaceId`s:

```cpp
const bool selfCollision = (pointCloud.surfaceId != 0 &&
                            pointCloud.surfaceId == triangleSurface.surfaceId);
```

Recall from file 05 that `surfaceId` is the collision-model pointer. If both
sides have the *same* surfaceId, they're literally the same mesh → self-
collision → exclusion on. If the IDs differ, they're different objects →
exclusion off. This is why unique surface IDs matter.

---

## 9.6 Self-collision dispatch (SOFA side)

How does the narrow phase decide to run the vertex-triangle self-collision path?
Recall from file 04 that the broad phase emits a `(cm, cm)` pair when a model
has `selfCollision=True`. In `endNarrowPhase`:

```cpp
const bool isSelfCollisionPair = (pair.first == pair.second && pair.first != nullptr);
const bool routeToVertexTriangle =
    usingIndexedDenseGrid &&
    d_useFeatureBasedProximity.getValue() &&
    d_useVertexTriangleProximity.getValue() &&
    isSelfCollisionPair;

if (routeToVertexTriangle) {
    // mirror the cached triangle surface into a point cloud that shares its surfaceId
    backend::PointCloudSurface selfPointCloud;
    selfPointCloud.devicePositions = firstIndexedSurface.devicePositions;  // SAME positions
    selfPointCloud.pointCount      = firstIndexedSurface.vertexCount;
    selfPointCloud.surfaceId       = firstIndexedSurface.surfaceId;        // SAME id → exclusion on
    // ...
    backend::computeFeatureBasedVertexTriangleContacts(selfPointCloud, firstIndexedSurface, ...);
}
```

The trick: the *same* mesh is fed as both the triangle surface AND the point
cloud (its vertices), sharing one surfaceId. The matching ID switches on
own-corner exclusion. Elegant — no extra data, just reusing the surface two ways.

### Verified result

The self-collision smoke test (`test_gpu_self_collision_vertex_triangle_smoke.py`)
is a two-layer slab: 512 vertices, 900 triangles, layers 0.05 apart, contact
distance 0.06 so each top vertex is in range of a bottom triangle. Measured:
**1385 FPS, 2700 contacts, all VertexFace.** 2700 ÷ 512 ≈ 5.3 contacts per
vertex — each vertex's inflated box spans several cells and pairs with several
nearby triangles. Own-corner exclusion keeps it from exploding with
distance-zero self-hits.

---

## 9.7 Cross-model dispatch (point cloud vs different mesh)

The other variant: a separate `CudaPointCollisionModel` (a point cloud
component) against a `CudaTriangleCollisionModel` (the tissue). The broad phase
emits a normal pair `(pointModel, triangleModel)` — two *different* models. The
narrow phase detects this at the top of the per-pair loop:

```cpp
auto* firstAsPoint  = dynamic_cast<CudaPointCollisionModel*>(pair.first->getLast());
auto* secondAsTri   = dynamic_cast<CudaTriangleCollisionModel*>(pair.second->getLast());
// (and the reverse ordering)

if (one side is a point model AND the other is a triangle model) {
    extractCudaPointCloud(pointSideCm, ...);        // grab the point cloud's GPU positions
    extractCudaIndexedSurface(triSideCm, ...);      // grab the triangle surface
    backend::computeFeatureBasedVertexTriangleContacts(crossPointCloud, crossTriangleSurface, ...);
    continue;   // handled this pair, skip the rest
}
```

### Extracting the point cloud — `extractCudaPointCloud`

A `CudaPointCollisionModel` stores its positions in a `MechanicalState`. The
extractor pulls the device pointer directly (zero-copy again):

```cpp
auto* mstate = pointModel->getMechanicalState();
const auto& positions = mstate->read(core::vec_id::read_access::position)->getValue();
outPointCloud.devicePositions = reinterpret_cast<const TriangleVertex*>(positions.deviceRead());
outPointCloud.pointCount = positions.size();
outPointCloud.surfaceId = (uint64) pointModel;   // the point-model pointer
```

Because `surfaceId` is the point-model's pointer, it differs from the triangle
model's pointer → own-corner exclusion stays **off** (correct for cross-model).

### Verified result

The cross-model smoke test (`test_gpu_cross_model_vertex_triangle_smoke.py`) is
a 41×41 tissue grid + an 8×8 tool point cloud hovering 0.04 above it. Measured:
**1968 FPS, 254 contacts, all VertexFace.** Each tool point finds ~4 nearby
tissue triangles.

---

## 9.8 Optional CPU publication

For all the FBP paths, if you set `copyContactsToHost=True`, the contacts are
downloaded and published into SOFA's `DetectionOutput` so a CPU response
pipeline can use them. For cross-model, this uses a special helper:

```cpp
publishCudaPointTriangleContacts(this, pointModel, triangleModel, proximityContacts);
```

which fills a `TDetectionOutputVector<CudaPointCollisionModel, CudaTriangleCollisionModel>`.

The narrow phase couples the flags so this works correctly: turning on
`copyContactsToHost` automatically forces `keepContactsOnDevice` off, so the
contacts actually get downloaded before publication.

Cost: the cross-model scene drops from 1968 FPS (detection-only) to 332 FPS with
publication on, because ~19 KB of contact data crosses PCIe each frame. That's
the price of getting the contacts to the CPU — only pay it if you need them
there.

---

## 9.9 The four paths, side by side

The narrow phase picks exactly one path per pair, based on the flags and the
model types:

| Path | Trigger | Insert kernel | Narrow kernel | Tests/pair |
|---|---|---|---|---|
| Exact-contact (legacy) | default | triangles | `exactDenseGridContactKernel` (SAT) | boolean |
| Tri-tri FBP | `useFeatureBasedProximity` | triangles | `featureBasedProximityKernel` | 15 |
| V-t self-collision | + `useVertexTriangleProximity` + `(cm,cm)` pair | triangles + points (same mesh) | `featureBasedVertexTriangleProximityKernel` | 1 |
| V-t cross-model | + `useVertexTriangleProximity` + (point, triangle) pair | triangles + points (diff meshes) | `featureBasedVertexTriangleProximityKernel` | 1 |

All four share the dense grid, the workspace, and the sync discipline. They
differ only in what's inserted and which narrow kernel runs.

---

## 9.10 Summary

```text
Vertex-triangle = treat one side as loose points, not triangles.
Reuses: dense grid, closest-point-on-triangle math, workspace, over-launch.
New: insertIndexedPointsKernel (points into tool buckets),
     featureBasedVertexTriangleProximityKernel (1 test per pair).
Self-collision: same mesh both sides → own-corner exclusion ON (via matching surfaceId).
Cross-model: point cloud vs different mesh → exclusion OFF.
All v-t contacts are VertexFace. Faster than tri-tri (1 test vs 15).
Measured: self-collision 1385 FPS, cross-model 1968 FPS.
```

Next: why the benchmark is so fast — the synchronization bypass and what
happens at frame end. Go to [10_phase4_sync_and_output.md](10_phase4_sync_and_output.md).
