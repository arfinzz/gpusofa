# GPU Collision Architecture And Calculation Report

Date: 2026-05-17

## Scope

This report explains the current GPU SOFA collision architecture at the implementation level: what data enters the system, how the SOFA wrapper prepares it, which CUDA kernels run, what each calculation does, what data transfers happen, and what output comes back.

The source-of-truth implementation is the direct indexed dense-grid path:

```text
CudaTriangleCollisionModel
  -> cached triangle index topology
  -> direct SOFA CUDA deviceRead() position pointer
  -> TriangleIndexedSurface
  -> cached backend device index buffers
  -> indexed dense-grid CUDA kernels
```

Main files:

```text
SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h
SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu
SofaGpuCollision/src/SofaGpuCollision/GpuPipelineProfiling.h
SofaGpuCollision/src/SofaGpuCollision/GpuPipelineBenchmarkController.cpp
```

## SOFA Integration

The project builds a SOFA plugin named `SofaGpuCollision`. The plugin registers GPU collision components and lets SOFA scenes instantiate them as normal SOFA objects.

Registered components:

```text
GpuCollisionBroadPhase
GpuCollisionNarrowPhase
GpuKinematicRigidController
GpuPipelineBenchmarkController
```

Runtime flow:

```text
SOFA animation step
  -> CollisionPipeline
  -> GpuCollisionBroadPhase
  -> GpuCollisionNarrowPhase
  -> backend::computeDenseGridIndexedTriangleContacts
  -> optional SOFA DetectionOutput publication
  -> GpuPipelineBenchmarkController writes CSV/summary metrics
```

The plugin keeps CPU fallback available. If CUDA is not available, if collision models are not CUDA triangle models, or if the dense-grid path rejects a configuration, the narrow phase can pass pairs back into SOFA's normal CPU path when `allowCPUFallback=true`.

## Input Data

The benchmark scenes provide CUDA triangle collision models. The important input is two triangle surfaces: one tissue surface and one tool/blade surface.

SOFA-side source data:

```text
CudaTriangleCollisionModel::getX()
  CUDA-compatible vertex positions

CudaTriangleCollisionModel::getTriangles()
  triangle topology as three vertex indices per triangle
```

Backend API input:

```cpp
struct TriangleIndexedSurface
{
    const TriangleVertex* positions;
    const TriangleVertex* devicePositions;
    std::uint32_t vertexCount;
    const std::uint32_t* triangleIndices;
    std::uint32_t triangleCount;
    std::uint64_t surfaceId;
    std::uint64_t topologyVersion;
};
```

Meaning:

- `devicePositions` is used in the fast path. It points directly at the SOFA CUDA vector storage.
- `positions` is fallback CPU-visible storage, used only when direct device positions are unavailable.
- `triangleIndices` is a flat CPU topology array: `[i0, i1, i2, i0, i1, i2, ...]`.
- `surfaceId` is derived from the SOFA collision model pointer and is used as a stable cache key.
- `topologyVersion` is a cached topology hash used to detect when device index buffers must be refreshed.

Dense-grid configuration:

```cpp
struct DenseGridConfig
{
    gridMinX/Y/Z, gridMaxX/Y/Z;
    gridResolutionX/Y/Z;
    contactDistance;
    maxTissueTrianglesPerCell;
    maxToolTrianglesPerCell;
    maxCandidatePairs;
    deduplicatePairs;
    copyContactsToHost;
    detailedProfiling;
    usePinnedHostStaging;
    useGpuHashDedupe;
    canonicalPairEmission;
    validateDedupeOnHost;
    readCountersWhenContactsStayOnDevice;
    computeDeviceContactsWhenContactsStayOnDevice;
    compactActiveCells;
    batchTriangleInsert;
};
```

Current benchmark defaults:

```text
useIndexedDenseGridInput=true
useDirectDevicePositions=true
copyContactsToHost=false
readCountersWhenContactsStayOnDevice=false
computeDeviceContactsWhenContactsStayOnDevice=false
useGpuHashDedupe=true
compactActiveCells=false
batchTriangleInsert=false
```

## SOFA Wrapper Preparation

`GpuCollisionNarrowPhase::endNarrowPhase()` loops over pending collision model pairs. For each pair it tries the indexed dense-grid path first when dense grid, topology caching, and indexed input are enabled.

Preparation call:

```text
extractCudaIndexedSurface(pair.first, firstModel, firstIndexedSurface)
extractCudaIndexedSurface(pair.second, secondModel, secondIndexedSurface)
```

The extraction function does this:

1. Casts the model to `CudaTriangleCollisionModel`.
2. Reads `positions = cudaModel->getX()` and `triangles = cudaModel->getTriangles()`.
3. Looks up a topology cache entry keyed by `CudaTriangleCollisionModel*`.
4. Checks whether vertex count, triangle count, or cached flat index size changed.
5. Recomputes a topology hash when size changed or `validateTriangleTopologyCache=true`.
6. If topology changed, rebuilds the flat `triangleIndices` vector and validates every triangle index.
7. If `useDirectDevicePositions=true`, calls `positions.deviceRead()` and reinterprets that CUDA pointer as `TriangleVertex*`.
8. If no direct pointer is available, fills `positionScratch` on the CPU as fallback.
9. Emits `TriangleIndexedSurface`.

Topology hash:

```text
hash starts at 1469598103934665603
for each value:
    hash = (hash xor value) * 1099511628211
values mixed: vertex count, triangle count, and every triangle index
```

Why this matters:

- Static topology scenes do not rebuild full CPU triangle arrays every frame.
- Direct CUDA positions avoid per-frame CPU position extraction and H2D position upload.
- The backend sees indexed data instead of packed `TrianglePrimitive` arrays.

## Backend Workspace

`DenseGridWorkspace` owns reusable CUDA allocations. The workspace grows when a larger scene needs more storage, then stays allocated for steady-state frames.

Main device arrays:

```text
grid
cellTissueIds
cellToolIds
candidatePairs
pairHashKeys
contacts
rawCandidateCount
candidateCount
contactCount
overflowCount
activeCellIds
activeCellCount
denseGridStats
indexedTissuePositions
indexedToolPositions
indexedTissueIndices
indexedToolIndices
```

Pinned fallback staging arrays:

```text
pinnedIndexedTissuePositions
pinnedIndexedToolPositions
pinnedIndexedTissueIndices
pinnedIndexedToolIndices
```

Direct fast-path frames do not use the indexed position staging arrays because position pointers come from SOFA CUDA storage. Device index buffers are cached by:

```text
indexedTissueSurfaceId + indexedTissueTopologyVersion
indexedToolSurfaceId + indexedToolTopologyVersion
```

## Derived Grid Data

The backend first validates the grid. It rejects zero resolution, zero bucket capacity, zero candidate capacity, and invalid min/max bounds.

Core derived values:

```text
cellCount = gridResolutionX * gridResolutionY * gridResolutionZ
tissueBucketCount = cellCount * maxTissueTrianglesPerCell
toolBucketCount = cellCount * maxToolTrianglesPerCell
```

With GPU hash dedupe:

```text
pairHashCapacity = nextPowerOfTwo(maxCandidatePairs * 2)
```

The device config stores inverse cell size:

```text
inverseCellSize.x = gridResolutionX / (gridMaxX - gridMinX)
inverseCellSize.y = gridResolutionY / (gridMaxY - gridMinY)
inverseCellSize.z = gridResolutionZ / (gridMaxZ - gridMinZ)
```

This converts world-space coordinates into dense-grid cell coordinates.

## Data Transfers

### Direct Indexed Fast Path

Measured steady state:

```text
positions H2D: 0 bytes
triangle pack H2D: 0 bytes
counter/contact D2H: 0 bytes
```

Details:

- Vertex positions are already on GPU in SOFA CUDA storage.
- The backend uses the `deviceRead()` pointer directly.
- Triangle indices are uploaded only when `surfaceId` or `topologyVersion` changes. In measured summaries, warmup absorbs the initial topology upload, so steady-state H2D is zero.
- Candidate/contact counters are not copied back when `copyContactsToHost=false`, `readCountersWhenContactsStayOnDevice=false`, and `detailedProfiling=false`.

### Fallback Indexed Path

When direct SOFA device positions are unavailable:

```text
CPU positions -> optional pinned staging -> cudaMemcpyAsync -> indexed position device buffer
CPU triangle indices -> optional pinned staging -> cudaMemcpyAsync -> indexed index device buffer
```

Pinned staging is used only for large fallback uploads. The current threshold is 1 MiB. Small uploads skip pinned staging because the extra host copy can cost more than it saves.

### Validation/Profiling Readback

When counters are requested, the backend reads:

```text
rawCandidateCount
candidateCount
overflowCount
DeviceDenseGridStats
contactCount
overflowCount
```

Current counter-read validation reports 44 D2H bytes per measured frame. The byte count is tiny, but the reads synchronize the CPU with GPU progress, so the wall-time cost can be much larger than the transfer size suggests.

## Kernel Sequence

Current direct indexed detection-only path:

```text
threadCount = 256

1. resetDenseGridKernel<<<ceil(cellCount / 256), 256>>>
2. cudaMemset(pairHashKeys, 0xff)
3. insertIndexedTrianglesKernel<<<ceil(tissueTriangleCount / 256), 256>>>
4. insertIndexedTrianglesKernel<<<ceil(toolTriangleCount / 256), 256>>>
5. generateDenseGridUniqueCandidatePairsKernel<<<cellCount, 256>>>
```

The benchmark reports this as:

```text
avg_kernel_launch_count=5
avg_cuda_memset_count=1
```

Strictly speaking, the 5 counted GPU operations are 4 kernels plus 1 `cudaMemset`; the project counts the memset in the launch count because it is a GPU-side operation on the critical path.

Optional variants:

```text
batchTriangleInsert=true:
  replace the two insert launches with insertIndexedTrianglePairKernel

compactActiveCells=true:
  add compactActiveDenseGridCellsKernel
  run generateActiveDenseGridUniqueCandidatePairsKernel over active cells

computeDeviceContactsWhenContactsStayOnDevice=true or copyContactsToHost=true:
  add exactDenseGridIndexedContactKernel

useGpuHashDedupe=false:
  use candidate generation without GPU hash dedupe, then optional thrust sort/unique
```

The current defaults keep `batchTriangleInsert=false` and `compactActiveCells=false` because profiling showed a regression on the GTX 1650 Ti one-tissue benchmark.

## Calculation Details

### Indexed Triangle Load

`indexedTriangleAt()` converts one triangle id into three vertices:

```text
i0 = triangleIndices[3 * triangleId + 0]
i1 = triangleIndices[3 * triangleId + 1]
i2 = triangleIndices[3 * triangleId + 2]
p0 = positions[i0]
p1 = positions[i1]
p2 = positions[i2]
```

It builds a `DeviceTriangle` with `p0`, `p1`, `p2`, and the triangle id.

### Triangle AABB

For each triangle:

```text
minPoint = min(min(p0, p1), p2)
maxPoint = max(max(p0, p1), p2)

aabb.min = minPoint - contactDistance
aabb.max = maxPoint + contactDistance
```

The inflation allows triangles near each other to be considered candidate contacts before exact testing.

### AABB To Cell Span

The AABB is rejected if it lies fully outside the grid. Otherwise:

```text
cellMin.x = floor((aabb.minX - gridMin.x) * inverseCellSize.x)
cellMin.y = floor((aabb.minY - gridMin.y) * inverseCellSize.y)
cellMin.z = floor((aabb.minZ - gridMin.z) * inverseCellSize.z)

cellMax.x = floor((aabb.maxX - gridMin.x) * inverseCellSize.x)
cellMax.y = floor((aabb.maxY - gridMin.y) * inverseCellSize.y)
cellMax.z = floor((aabb.maxZ - gridMin.z) * inverseCellSize.z)
```

Then every coordinate is clamped to the valid grid range:

```text
x: [0, resolutionX - 1]
y: [0, resolutionY - 1]
z: [0, resolutionZ - 1]
```

### Linear Cell Id

Cell coordinates are flattened as:

```text
cellId = x + y * resolutionX + z * resolutionX * resolutionY
```

### Triangle Insertion

For every cell overlapped by the triangle AABB:

```text
localIndex = atomicAdd(&grid[cellId].tissueCount, 1)
```

or:

```text
localIndex = atomicAdd(&grid[cellId].toolCount, 1)
```

If the local index is within capacity:

```text
cellTissueIds[cellId * maxTissueTrianglesPerCell + localIndex] = tissueTriangleId
cellToolIds[cellId * maxToolTrianglesPerCell + localIndex] = toolTriangleId
```

If capacity is exceeded:

```text
overflowCount++
```

Stats such as insertion count and max cell occupancy are updated only when dense-grid stats are requested. Fast wall-time runs skip those stats atomics.

### Candidate Pair Generation

For each cell:

```text
tissueCount = min(grid[cellId].tissueCount, maxTissueTrianglesPerCell)
toolCount = min(grid[cellId].toolCount, maxToolTrianglesPerCell)
totalPairs = tissueCount * toolCount
```

Threads walk local pairs:

```text
for localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x
```

Mapping from local pair id to triangle ids:

```text
tissueLocal = localPair / toolCount
toolLocal = localPair % toolCount
tissueTriangleId = cellTissueIds[cellId * maxTissueTrianglesPerCell + tissueLocal]
toolTriangleId = cellToolIds[cellId * maxToolTrianglesPerCell + toolLocal]
```

Candidate pair encoding:

```text
candidatePair = (uint64(tissueTriangleId) << 32) | uint64(toolTriangleId)
```

### GPU Hash Dedupe

Duplicate pairs occur because the same triangle pair can overlap more than one grid cell. The default path deduplicates on GPU.

The hash table uses:

```text
empty slot = 0xffffffffffffffff
slot = mixCandidatePairHash(candidatePair) & (pairHashCapacity - 1)
```

Insertion uses linear probing with `atomicCAS`:

```text
if atomicCAS(emptySlot, candidatePair) succeeds:
    outputIndex = atomicAdd(candidateCount, 1)
    candidatePairs[outputIndex] = candidatePair

if slot already contains candidatePair:
    duplicate, skip

if probing exceeds 256 slots:
    hashDedupeProbeOverflowCount++
    overflowCount++
```

The output of this stage is a compact device-side `candidatePairs` array.

### Exact Contact Kernel

The exact contact kernel is skipped in the fastest detection-only benchmark unless:

```text
copyContactsToHost=true
```

or:

```text
computeDeviceContactsWhenContactsStayOnDevice=true
```

When enabled, each candidate pair maps to one CUDA thread:

```text
contactBlocks = ceil(exactCandidateCount / 256)
exactDenseGridIndexedContactKernel<<<contactBlocks, 256>>>
```

The kernel decodes the candidate pair, reloads indexed tissue/tool triangles, calls `exactTriangleIntersection`, and appends a `DeviceExactContact` with `atomicAdd(contactCount, 1)` if an intersection exists.

### CPU Contact Publication

When `copyContactsToHost=true`, the backend reads `contactCount`, downloads `DeviceExactContact` records, converts them to `ExactContact`, and `GpuCollisionNarrowPhase` publishes SOFA `DetectionOutput` objects.

When `copyContactsToHost=false`, the backend returns success without publishing contacts. This is the intended detection-only benchmark mode because collision response is disabled.

## Output Data

Fast detection-only output:

```text
contacts vector: empty
candidate/contact counters: not read back
SOFA DetectionOutput: not published
benchmark metrics: stage timings and byte/launch counters
```

Validation output:

```text
rawCandidateCount
uniqueCandidateCount
overflowCount
activeMixedCellCount
tissue/tool insert counts
max cell occupancies
optional contacts
```

CPU response-enabled output:

```text
ExactContact vector on CPU
SOFA DetectionOutput entries
```

## Profiling Instrumentation

`GpuPipelineBenchmarkController` writes per-step CSV rows and summary files. The narrow-phase metrics come from `BackendExecutionStats` and `profiling::StageSnapshot`.

Important metrics:

```text
avg_step_seconds
avg_fps
avg_narrow_wall_ms
avg_narrow_kernel_ms
avg_narrow_sofa_triangle_extraction_ms
avg_narrow_backend_triangle_pack_ms
avg_narrow_h2d_ms
avg_narrow_candidate_readback_ms
avg_narrow_contact_count_readback_ms
avg_host_to_device_bytes
avg_device_to_host_bytes
avg_kernel_launch_count
avg_cuda_memset_count
avg_narrow_raw_candidate_count
avg_narrow_unique_candidate_count
avg_narrow_overflow_count
```

Nsight artifacts from the latest run:

```text
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_ncu.ncu-rep
output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_nsys_full.nsys-rep
```

Nsight Compute captured these detection-only kernels:

```text
resetDenseGridKernel
insertIndexedTrianglesKernel
insertIndexedTrianglesKernel
generateDenseGridUniqueCandidatePairsKernel
```

The exact contact kernel is absent from the detection-only Nsight run because exact contacts are disabled in the fastest mode.

## Current Design Choices

- Keep CPU fallback for non-CUDA scenes and failure recovery.
- Keep detailed profiling opt-in because CUDA event timing and counter reads add synchronization.
- Keep counter reads off in detection-only performance runs.
- Use direct SOFA CUDA position pointers whenever available.
- Cache topology in the SOFA wrapper and cache index buffers in the backend.
- Keep `cudaMemset` for pair hash reset because it measured faster than a custom hash reset on this GPU.
- Keep active-cell compaction and batched insertion available but disabled by default after measured regression.

## Current Bottlenecks And Risks

Current measured bottleneck candidates:

- Candidate generation and grid insertion dominate large detection-only GPU work.
- Counter readbacks dominate validation-mode wall time even though only 44 bytes are copied.
- Large-scene wall time is noisy in WSL and should be benchmarked with controlled run order and power state before drawing new conclusions.

Deferred work:

- SoA/coalescing redesign for AABB/grid insertion if Nsight confirms memory pressure is the next limiter.
- Exact-contact register-pressure tuning after exact contact becomes a real integrated bottleneck.
- Sampling or async CPU logging of counters if long runs need periodic counts without synchronizing every frame.
