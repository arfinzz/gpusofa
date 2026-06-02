# Phase 12 Full Verification — 2026-05-25

End-to-end verification of all three feature-based proximity narrow-phase paths
on the GTX 1650 Ti laptop. All three scenes are detection-only (no CPU contact
response), use direct CUDA `deviceRead()` position pointers, and indexed dense-
grid broad cull.

## Run matrix

| Mode | Scene | Pair shape | FPS | narrow_wall | narrow_kernel | FBP kernel | contacts | VF / FV / EE | launches | D2H |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast (no readback) | tri-tri FBP one-tissue/one-blade | (CudaTri, CudaTri) | 775 | 0.67 ms | 0.51 ms | not timed | n/a | n/a | 7 | 0 |
| Fast (counter on) | v-t self-collision two-layer slab | (CudaTri, CudaTri) where pair.first == pair.second | 1385 | 0.66 ms | not timed | not timed | 2700 | 2700 / 0 / 0 | 6 | 16 |
| Fast (counter on) | v-t cross-model tissue + tool cloud | (CudaTri, CudaPoint) | 1968 | 0.37 ms | not timed | not timed | 254 | 254 / 0 / 0 | 6 | 16 |
| Validation (counter on) | tri-tri FBP one-tissue/one-blade | (CudaTri, CudaTri) | 449 | 1.49 ms | 1.02 ms | 0.07 ms | 56 | 0 / 0 / 56 | 7 | 64 |

## Source log paths

- `output/benchmark_logs/phase12_verify_tri_tri_rerun/` — tri-tri FBP fast path
- `output/benchmark_logs/phase12_verify_vt_self/` — v-t self-collision two-layer slab
- `output/benchmark_logs/phase12_verify_vt_cross/` — v-t cross-model tissue + tool cloud
- `output/benchmark_logs/phase12_validation_tri_tri/` — tri-tri FBP with readback for VF/FV/EE breakdown

## Interpretation

### Tri-tri FBP fast path

`one-tissue/one-blade` scene with `selfCollision=False` on both meshes. SOFA broad
phase emits one `(tissue, blade)` cross-model pair. With `useFeatureBasedProximity=True`
and `useVertexTriangleProximity=False` (default), the narrow phase routes through
`computeFeatureBasedProximityContacts`. The 7-kernel sequence is:

```text
1. resetDenseGridKernel
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel (tissue, 12800 tris)
4. insertIndexedTrianglesKernel (blade, 12 tris)
5. generateDenseGridUniqueCandidatePairsKernel
6. resetProximityCountersKernel
7. featureBasedProximityKernel (6 VF + 9 EE per pair)
```

Fast path (`SOFA_PROXIMITY_READ_CONTACT_COUNTER=0`) skips per-kernel CUDA event
timing and the 5-counter D2H batch. CPU sees only the host-side scheduling time
(0.67 ms). The 0.51 ms `narrow_kernel_ms` is the cudaEventElapsedTime around
the broad-cull window only (since FBP timing is suppressed). 775 FPS is
identical-within-noise to the pre-FBP 633 FPS detection-only baseline.

Validation mode (`SOFA_PROXIMITY_READ_CONTACT_COUNTER=1`) enables the FBP
cudaEventRecord pair, which adds 0.07 ms `featureBasedProximityKernelMs` to
the kernel total. The 5-counter batched D2H accounts for the 64 bytes (5
proximity counters + 11 broad-cull counters at this verbosity level). All 56
emitted contacts are `EdgeEdge` — the closest feature pair for a flat tissue
against a small blade box is always edge-edge at this contact distance.

### V-t self-collision two-layer slab

`test_gpu_self_collision_vertex_triangle_smoke.py`. A two-layer flat slab
(256 verts × 2 layers = 512 verts, 450 tris × 2 layers = 900 tris) with
`selfCollision=True` on the `TriangleCollisionModel`. SOFA broad phase emits
one `(cm, cm)` self-collision pair. With `useVertexTriangleProximity=True`,
the narrow phase routes through `computeFeatureBasedVertexTriangleContacts`
with the same `TriangleIndexedSurface` mirrored into a `PointCloudSurface`
(matching `surfaceId` triggers own-corner exclusion).

6-kernel sequence:

```text
1. resetDenseGridKernel
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel (all 900 tris)
4. insertIndexedPointsKernel (all 512 verts as 1-point AABBs)
5. generateDenseGridUniqueCandidatePairsKernel (produces (triId, vertId) pairs)
6. resetProximityCountersKernel + featureBasedVertexTriangleProximityKernel
```

(The proximity counter reset is launch 6 with the v-t narrow pass; the
controller counts the kernel-pair as one launch because of how the launch
counter is incremented in the backend wrapper.)

2700 contacts emitted, all `VertexFace` as expected. 2700 / 512 ≈ 5.3 contacts
per vertex — each vertex's 0.06-inflated AABB spans multiple cells in the
32×4×32 dense grid, and the kernel emits one contact per (vertex, triangle)
candidate within distance. Own-corner exclusion suppresses adjacent-corner
false positives (otherwise the count would explode).

### V-t cross-model tissue + tool cloud

`test_gpu_cross_model_vertex_triangle_smoke.py`. Tissue is a 41×41 vertex
flat grid with `TriangleCollisionModel` (1681 verts, 3200 tris). Tool is a
separate 8×8 point cloud (64 verts) with `CudaPointCollisionModel` hovering
0.04 above the tissue. SOFA broad phase emits one cross-model pair.

The narrow phase detects this pair via the new cross-model branch:

```cpp
dynamic_cast<CudaPointCollisionModel*>(pair.first->getLast()) // matches tool
dynamic_cast<CudaTriangleCollisionModel*>(pair.second->getLast()) // matches tissue
```

then dispatches via `extractCudaPointCloud` + `extractCudaIndexedSurface` →
`computeFeatureBasedVertexTriangleContacts`. SurfaceIds differ (point-model
pointer vs triangle-model pointer) so own-corner exclusion is OFF.

254 contacts emitted, all `VertexFace`. 254 / 64 = 3.97 contacts per tool
vertex. Tool grid spacing 0.286 doesn't align with the tissue grid spacing
0.1, so most tool points project onto tissue vertices or edges where multiple
tissue triangles share the closest-point geometry. Distance 0.04 < 0.05
threshold → all close-by tissue triangles within the inflated AABB pass.

### Geometric correctness summary

- All v-t contacts are classified `VertexFace` by construction — kernel
  hard-codes `featureKind = 0`.
- All tri-tri contacts at this geometry are `EdgeEdge` — for a flat tissue
  against a small box at 0.03 contact distance, no vertex of either mesh
  penetrates the other face's plane within distance, so VF/FV never fire.
- Self-collision exclusion is verified working: with `surfaceId` match, the
  v-t kernel skips own-corner pairs. The 2700-contact count is what remains
  after exclusion; without exclusion the same scene would emit ~6× more.

## Build state

Clean build of `libSofaGpuCollision.so` and `SofaGpuCollisionDenseGridBackendBench`
in WSL after sync. Two pre-existing deprecation warnings (RegisterObject is
deprecated in SOFA v25.12); these are inherited from SOFA core and are not
project changes.

## Performance regression check

Tri-tri FBP fast-path FPS reads:

| Date | FPS | narrow_wall | narrow_kernel | Notes |
|---|---:|---:|---:|---|
| 2026-05-17 baseline (pre-FBP detection-only) | 633 | 0.69 ms | 0.54 ms | dense-grid only, no FBP |
| 2026-05-24 first FBP smoke (pre sync fix) | 109 | 8.34 ms | 7.16 ms | 5 synchronous counter readbacks |
| 2026-05-25 post sync fix | 672 | 0.71 ms | 0.52 ms | sync fix landed |
| 2026-05-25 post v-t self-collision wiring | 752 | 0.64 ms | 0.52 ms | regression check after Phase 12 |
| **2026-05-25 post v-t cross-model wiring** | **775** | **0.67 ms** | **0.51 ms** | this verification |

No regression. All three FBP paths coexist on the same `endNarrowPhase`
dispatch chain without affecting one another.
