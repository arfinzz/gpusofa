# Archived Historical Plan

This file is preserved for context only. It is not the current source of truth.
Use these active documents instead:

```text
guide/setup.md
guide/plan.md
guide/architecture.md
reports/gpu_collision_architecture_calculation_report_20260517.md
reports/gpu_collision_profiling_findings_report_20260517.md
```

---

# GPU Collision Optimization Plan - 2026-05-14

## Goal

Reduce SOFA-integrated dense-grid collision step time by attacking the measured bottlenecks in this order:

1. Stop rebuilding static triangle topology every step.
2. Upload only changing vertex positions, or skip upload entirely when SOFA already has GPU-resident data.
3. Remove blocking D2H counter readbacks from detection-only runs.
4. Reduce launch/memset overhead for small scenes.
5. Improve memory layout/coalescing for AABB and grid insertion.
6. Tune exact contact only after the wrapper path is no longer dominant.

Current clean benchmark baseline:

- GPU dense-grid SOFA scene: 12.032 ms/step.
- Narrow wall time: 9.923 ms.
- GPU kernel time: 1.668 ms.
- Host preparation: 5.335 ms.
- SOFA triangle extraction: 4.186 ms.
- Backend triangle packing: 1.148 ms.
- H2D upload: 0.519 ms.
- Candidate/contact counter readback: about 1.058 ms combined.

## Current Hot Path

The current SOFA path does this every frame and every CUDA triangle pair:

1. `GpuCollisionNarrowPhase::endNarrowPhase()`
2. `tryExtractCudaTriangles(...)`
3. Build fresh `std::vector<backend::TrianglePrimitive>` for both models.
4. `backend::computeDenseGridTriangleContacts(...)`
5. Convert `TrianglePrimitive` into `DeviceTriangle` again.
6. Copy packed triangles H2D.
7. Run dense-grid kernels.
8. Copy several tiny counters D2H even when contacts are not copied back.

Primary files:

- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.h`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h`
- `SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu`
- `SofaGpuCollision/src/SofaGpuCollision/GpuPipelineBenchmarkController.cpp`

## Phase 0 - Guardrails And Baseline

Deliverables:

- Keep the existing dense-grid path working as the default fallback.
- Add feature toggles so each optimization can be tested independently.
- Preserve correctness for collision-response-enabled scenes.
- Preserve detection-only fast path where `copyContactsToHost=false`.

Implementation tasks:

- Add benchmark labels for each new mode:
  - `cacheTopology`
  - `indexedSurfaceBackend`
  - `deferCounterReadback`
  - `activeCellPairGeneration`
  - `fusedAabbInsert`
- Keep current CSV fields and add new fields:
  - `topology_cache_hit_count`
  - `topology_cache_miss_count`
  - `position_upload_bytes`
  - `topology_upload_bytes`
  - `counter_readback_deferred`
  - `active_cell_count`
  - `hash_clear_bytes`

Acceptance:

- Existing validation scenes still run.
- Existing benchmark CSVs still parse.
- Baseline numbers can be reproduced before each phase.

## Phase 1 - Cache SOFA Triangle Topology

Purpose:

Stop reading and re-expanding triangle index topology every frame when topology is unchanged.

Current issue:

`tryExtractCudaTriangles(...)` loops over `outModel->getTriangles()` every frame and creates full packed triangles from positions plus indices. Topology normally does not change, but the code pays the full cost anyway.

Implementation:

1. Add a CUDA-only cache type in `GpuCollisionNarrowPhase.h`.

   Suggested structure:

   ```cpp
   struct CachedTriangleTopology
   {
       CudaTriangleCollisionModel* model { nullptr };
       std::size_t triangleCount { 0 };
       std::size_t vertexCount { 0 };
       std::uint64_t topologyHash { 0 };
       std::vector<std::uint32_t> triangleIndices; // 3 * triangleCount
       std::vector<backend::TrianglePrimitive> packedTrianglesScratch;
       std::uint64_t hits { 0 };
       std::uint64_t misses { 0 };
       bool seenThisFrame { false };
   };
   ```

2. Add a member cache:

   ```cpp
   std::unordered_map<const CudaTriangleCollisionModel*, CachedTriangleTopology> m_triangleTopologyCache;
   ```

3. Replace `tryExtractCudaTriangles(...)` with a cache-aware function:

   - Resolve `CudaTriangleCollisionModel`.
   - Check `vertexCount`, `triangleCount`, and a lightweight topology hash.
   - Rebuild cached indices only on miss.
   - Every frame, update only `packedTrianglesScratch` positions from `getX()`.
   - Use `resize()` and assignment instead of `clear()` plus `push_back()` to avoid repeated growth.

4. Prune cache entries:

   - Mark entries `seenThisFrame=false` in `beginNarrowPhase()`.
   - Mark used models in `endNarrowPhase()`.
   - Erase unseen entries after the phase to avoid dangling model pointers after scene changes.

5. Add toggles:

   - `cacheTriangleTopology=true`
   - `topologyCacheValidation=true` for debug hash checking.

Expected effect:

- Does not eliminate position packing yet.
- Should reduce extraction overhead and allocations.
- Low risk, because the backend API can remain unchanged.

Acceptance:

- Same contact counts as baseline.
- Same raw/unique candidate counts as baseline.
- Topology cache hit rate should become near 100% after warmup.
- `narrow_sofa_triangle_extraction_ms + narrow_backend_triangle_pack_ms` should drop measurably.

## Phase 2 - Split Backend Input Into Topology And Positions

Purpose:

Avoid CPU construction of full `TrianglePrimitive` arrays. Send stable topology once, send changing vertex positions each step.

New backend API:

Add to `GpuCollisionBackend.h`:

```cpp
struct TriangleIndexedSurface
{
    const TriangleVertex* positions { nullptr };
    std::uint32_t vertexCount { 0 };
    const std::uint32_t* triangleIndices { nullptr }; // triplets
    std::uint32_t triangleCount { 0 };
    std::uint64_t topologyVersion { 0 };
    std::uint64_t surfaceId { 0 };
};

bool computeDenseGridIndexedTriangleContacts(
    const TriangleIndexedSurface& tissue,
    const TriangleIndexedSurface& tool,
    const DenseGridConfig& config,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);
```

Backend state:

- Add cached device buffers per `surfaceId`:
  - `DeviceTriangleIndex* triangleIndices`
  - `float4* positions`
  - optional `DeviceTriangle* packedTriangles` during transition
  - `topologyVersion`
  - capacities

Implementation steps:

1. In SOFA wrapper, produce:
   - cached triangle index triplets
   - current position array view/scratch
2. In backend:
   - Upload triangle indices only when `topologyVersion` changes.
   - Upload positions every step.
   - Pack triangles on GPU with a small `packIndexedTrianglesKernel`, or modify existing kernels to read indexed positions directly.
3. First transition path:
   - Keep existing dense-grid kernels.
   - Add `packIndexedTrianglesKernel` to fill `DeviceTriangle`.
   - This removes CPU triangle packing while keeping existing kernel logic.
4. Second path:
   - Teach AABB, insert, and exact contact kernels to read `positions + triangleIndices` directly.
   - Remove the `DeviceTriangle` intermediate for the dense-grid path.

Expected effect:

- Backend triangle packing should approach zero on CPU.
- H2D bytes for small scene should drop from packed triangles to vertex positions plus occasional topology upload.
- For unchanged topology, per-frame topology upload should be zero.

Acceptance:

- `topology_upload_bytes=0` after warmup.
- `position_upload_bytes` only reflects vertex positions.
- Contact/candidate counts match existing backend.
- No regression in large standalone benchmark.

## Phase 3 - GPU-Native SOFA Surface Path

Purpose:

Avoid CPU position extraction when `CudaTriangleCollisionModel` already owns GPU-resident position buffers.

Investigation first:

- Inspect SofaCUDA vector types returned by `CudaTriangleCollisionModel::getX()`.
- Determine whether a stable device pointer can be obtained safely.
- Determine whether topology indices have a GPU-resident representation or must remain cached/uploaded by the plugin.

Implementation options:

1. Best case: direct device view.

   Add:

   ```cpp
   struct DeviceTriangleIndexedSurface
   {
       const float4* devicePositions { nullptr };
       const std::uint32_t* deviceTriangleIndices { nullptr };
       std::uint32_t vertexCount { 0 };
       std::uint32_t triangleCount { 0 };
       std::uint64_t topologyVersion { 0 };
       std::uint64_t surfaceId { 0 };
   };
   ```

   Backend uses SOFA device positions directly and uploads no positions.

2. Middle case: topology cached on device, positions copied from host.

   Use pinned staging plus `cudaMemcpyAsync`.

3. Fallback case: current packed-triangle path.

   Preserve for unsupported models or CPU-only builds.

Acceptance:

- In GPU-native mode, per-frame H2D triangle/position bytes should be zero or near zero.
- SOFA extraction time should drop sharply.
- Fallback remains automatic if a model cannot expose device data.

## Phase 4 - Remove Or Defer D2H Counter Readbacks

Purpose:

Avoid blocking the CPU on tiny D2H copies during detection-only benchmarks.

Current issue:

Even with `copyContactsToHost=false`, the backend reads:

- raw candidate count
- unique candidate count
- overflow count
- dense-grid stats
- contact count
- overflow count again

This costs around 1 ms in the clean integrated benchmark despite copying only a few bytes.

Implementation:

1. Add readback policy to `DenseGridConfig`.

   ```cpp
   enum class CounterReadbackMode : std::uint8_t
   {
       Required,       // collision response or strict correctness
       StatsOnly,      // profiling/benchmark stats
       Deferred,       // async readback, consumed next frame
       Disabled        // detection-only, no host stats needed
   };
   ```

2. Default policy:

   - If `copyContactsToHost=true`: `Required`.
   - If `detailedProfiling=true`: `StatsOnly`.
   - If benchmark wants per-frame stats: `StatsOnly` or `Deferred`.
   - If collision response disabled and stats disabled: `Disabled`.

3. Add pinned counter buffers:

   - double-buffered host counter structs
   - a CUDA stream for async readback
   - CUDA events to check completion next frame

4. Create a single device-side counter struct:

   ```cpp
   struct DeviceDenseGridCounters
   {
       std::uint32_t rawCandidateCount;
       std::uint32_t uniqueCandidateCount;
       std::uint32_t contactCount;
       std::uint32_t overflowCount;
       DeviceDenseGridStats stats;
   };
   ```

5. Copy this struct once when readback is required.

6. In `Disabled` mode:

   - Do not copy counters to host.
   - Return success if kernel launches succeeded.
   - Leave output counts as zero or previous-frame values with a clear CSV flag.

7. In `Deferred` mode:

   - Start `cudaMemcpyAsync` into pinned buffer.
   - Do not synchronize in the current frame.
   - Consume completed data in the next `beginNarrowPhase()` or benchmark controller tick.

Acceptance:

- Detection-only fast path avoids blocking D2H readbacks.
- Collision-response path still reads contact counts and contacts correctly.
- Overflow detection remains strict in `Required` mode.
- CSV clearly marks whether stats are exact, deferred, or unavailable.

## Phase 5 - Reduce Launches And Full-Grid Work

Purpose:

Reduce overhead for small scenes where many launches process tiny or empty workloads.

Current launch pattern:

- clear grid
- 5-6 memset operations
- AABB tissue
- AABB tool
- insert tissue
- insert tool
- generate pairs over every grid cell
- exact contacts
- optional sort/unique path

Recommended changes:

1. Fuse AABB and insert.

   Replace:

   - `triangleAabbKernel`
   - `insertTissueTrianglesKernel`
   - `insertToolTrianglesKernel`

   With:

   - `buildTissueGridKernel`: compute tissue AABB and insert tissue.
   - `buildToolGridKernel`: compute tool AABB and insert tool.

   This cuts two launches and removes global AABB memory traffic in the simple path.

2. Add active-cell pair generation.

   Current `generateDenseGridUniqueCandidatePairsKernel` launches one block per grid cell, even when most cells are empty.

   New strategy:

   - During insert, append cells that become mixed or non-empty into an active-cell list.
   - Generate pairs only over active mixed cells.
   - Keep a fallback full-grid kernel for debugging and validation.

3. Avoid clearing huge hash tables.

   Current GPU hash dedupe clears `pairHashKeys` with `cudaMemset(..., 0xff, pairHashCapacity * sizeof(uint64_t))`.

   Options:

   - Size hash capacity from expected candidate count instead of `maxCandidatePairs`.
   - Use generation-stamped hash slots.
   - Track touched hash slots and clear only those.

4. Fuse counter clear.

   Replace multiple scalar `cudaMemset` calls with one `clearDenseGridCountersKernel`.

   This is less important than hash-table clearing but simpler and cleaner.

Acceptance:

- Kernel launches per dense-grid step decreases.
- Full-grid pair generation no longer launches over tens of thousands of empty cells for small scenes.
- Candidate/contact counts match baseline.
- Nsight Compute no longer reports tiny tool kernels as the dominant inefficiency for the small scene.

## Phase 6 - Revisit Memory Layout For AABB And Insertion

Purpose:

Improve memory coalescing and reduce bytes moved in the memory-dominated kernels.

Current symptoms:

- Tissue AABB is around 72% DRAM utilization in the large backend case.
- Insert kernels are memory-pipeline dominated.
- Exact contact uses packed `DeviceTriangle` AoS data.

Changes:

1. Move toward SoA layout:

   - `float4* positions`
   - `uint3* triangleIndices`
   - separate AABB arrays:
     - `float4* aabbMin`
     - `float4* aabbMax`

2. Reduce intermediate writes:

   - If AABB+insert is fused, avoid writing AABBs unless detailed profiling or debug requires them.
   - For exact contact, load only candidate triangles.

3. Improve grid bucket layout:

   - Separate counts from payload arrays.
   - Keep tissue/tool bucket payloads contiguous.
   - Consider compact active cell records:

     ```cpp
     struct ActiveCell
     {
         std::uint32_t cellId;
         std::uint16_t tissueCount;
         std::uint16_t toolCount;
     };
     ```

4. Validate candidate pair encoding:

   - Current `uint64_t` pair encoding is compact and probably fine.
   - Keep unless profiling shows pair bandwidth dominating.

Acceptance:

- AABB and insert memory throughput improves or total time drops.
- No correctness change.
- Large backend benchmark benefits without hurting small scene latency.

## Phase 7 - Tune Exact Contact Last

Purpose:

Address register pressure only after host/SOFA overhead and launch overhead are reduced.

Current evidence:

- Exact contact uses about 68 registers/thread.
- Achieved occupancy is about 60%.
- It is not the main integrated wall-time bottleneck today.

Possible later work:

- Split helper computations to reduce live ranges.
- Use `__forceinline__` selectively, not everywhere.
- Evaluate 128-thread blocks against 256-thread blocks.
- Store less in `DeviceExactContact` when `copyContactsToHost=false`.
- Add contact-count-only kernel variant for detection-only mode.

Acceptance:

- Exact contact duration improves without reducing numerical correctness.
- Register count decreases or occupancy improves.
- No regression in contact count or contact geometry when contacts are copied to host.

## Suggested Execution Order

1. Phase 1: topology cache with current backend API.
2. Re-benchmark.
3. Phase 4: readback policy and detection-only skip/defer.
4. Re-benchmark.
5. Phase 2: indexed surface backend with topology uploaded once.
6. Re-benchmark.
7. Phase 5: launch reductions, starting with fused AABB+insert.
8. Re-benchmark.
9. Phase 3: GPU-native SOFA device pointer path, if SofaCUDA exposes stable device buffers.
10. Phase 6: SoA/memory layout work.
11. Phase 7: exact contact register tuning.

## Target Milestones

Milestone A: Low-risk wrapper cache

- Topology cache lands.
- Existing backend unchanged.
- Expected clean SOFA step improvement: modest but visible.

Milestone B: Detection-only fast path

- Counter readbacks are skipped or deferred when contacts are not copied to host.
- Expected clean SOFA step improvement: about 0.5-1.0 ms from current measured readback overhead.

Milestone C: Indexed backend

- CPU triangle packing removed.
- Topology upload only on cache miss.
- Expected clean SOFA step improvement: should attack the current 1.148 ms backend pack cost and part of extraction cost.

Milestone D: GPU-native surface feed

- H2D triangle/position upload is eliminated where SofaCUDA exposes device buffers.
- This is the highest-value version but requires confirming SOFA CUDA pointer ownership/lifetime.

Milestone E: Small-scene launch reduction

- Fewer launches and less full-grid work.
- Small one-tissue/one-blade scene should stop looking launch-bound.

## Testing Matrix

Correctness:

- CPU one-tissue/one-blade benchmark.
- GPU one-tissue/one-blade dense-grid benchmark.
- Large tissue/blade dense-grid benchmark.
- Phase 4/5 overlap validation scenes.
- Force topology change mid-run and verify cache invalidation.
- Run with `copyContactsToHost=true` and verify SOFA DetectionOutput is populated.
- Run with `copyContactsToHost=false` and verify no unnecessary contact downloads.

Performance:

- Clean SOFA benchmark, 20+ iterations.
- Standalone backend benchmark, 8+ measured steps.
- Nsight Compute report after each kernel-layout phase.
- Nsight Systems no-capture timeline after each wrapper/readback phase.

Regression gates:

- No candidate/contact count drift unless a feature intentionally changes candidate dedupe strategy.
- No overflow in existing benchmark configurations.
- No new device allocations after warmup.
- No per-frame topology upload after warmup.

## Main Risks

1. SOFA model lifetime.
   - Cache keys are raw model pointers. Mitigate by pruning unseen entries every frame and validating size/hash before reuse.

2. Dynamic topology.
   - Topology may change in some scenes. Mitigate with count/hash checks and explicit cache miss handling.

3. GPU-native pointer lifetime.
   - SofaCUDA device pointers may be transient or hidden. Treat direct pointer use as optional and guarded.

4. Deferred readbacks.
   - Benchmark stats can become one frame late. Mark CSV rows clearly and keep strict mode for correctness.

5. Launch fusion.
   - Fusing kernels can make debugging harder. Keep old kernels behind a debug toggle until performance and correctness are proven.

## First Patch Recommendation

Start with Phase 1 because it is the safest and gives a clean foundation:

- Add `CachedTriangleTopology`.
- Cache triangle indices per `CudaTriangleCollisionModel`.
- Reuse `packedTrianglesScratch` buffers.
- Add cache hit/miss stats.
- Keep `computeDenseGridTriangleContacts(...)` unchanged.

That patch should be small, easy to test, and will make the later indexed-backend work much less risky.
