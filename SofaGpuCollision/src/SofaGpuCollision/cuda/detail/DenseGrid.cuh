// DenseGrid.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Dense-grid workspace + kernels (insert/generate/exact, incl. the shared
// cell math: triangleAabb/denseGridCellSpan/denseCellId + pair dedup) and
// the two dense host drivers (packed + indexed).

namespace
{

struct DenseGridWorkspace
{
    DeviceTriangle* tissueTriangles { nullptr };
    DeviceTriangle* toolTriangles { nullptr };
    DeviceAabb* tissueAabbs { nullptr };
    DeviceAabb* toolAabbs { nullptr };
    DeviceCellBucket* grid { nullptr };
    std::uint32_t* activeCellIds { nullptr };
    std::uint32_t* cellTissueIds { nullptr };
    std::uint32_t* cellToolIds { nullptr };
    std::uint64_t* candidatePairs { nullptr };
    unsigned long long* pairHashKeys { nullptr };
    DeviceExactContact* contacts { nullptr };
    std::uint32_t* rawCandidateCount { nullptr };
    std::uint32_t* candidateCount { nullptr };
    std::uint32_t* contactCount { nullptr };
    std::uint32_t* overflowCount { nullptr };
    std::uint32_t* activeCellCount { nullptr };
    DeviceDenseGridStats* denseGridStats { nullptr };
    SofaGpuCollision::backend::TriangleVertex* indexedTissuePositions { nullptr };
    SofaGpuCollision::backend::TriangleVertex* indexedToolPositions { nullptr };
    std::uint32_t* indexedTissueIndices { nullptr };
    std::uint32_t* indexedToolIndices { nullptr };
    DeviceTriangle* pinnedTissueTriangles { nullptr };
    DeviceTriangle* pinnedToolTriangles { nullptr };
    SofaGpuCollision::backend::TriangleVertex* pinnedIndexedTissuePositions { nullptr };
    SofaGpuCollision::backend::TriangleVertex* pinnedIndexedToolPositions { nullptr };
    std::uint32_t* pinnedIndexedTissueIndices { nullptr };
    std::uint32_t* pinnedIndexedToolIndices { nullptr };
    // Feature-based proximity (VF/EE) outputs and counters
    void* proximityContacts { nullptr };  // DeviceProximityContact* (forward-typed)
    std::uint32_t* proximityContactCount { nullptr };
    std::uint32_t* proximityOverflowCount { nullptr };
    std::uint32_t* proximityVfCount { nullptr };
    std::uint32_t* proximityFvCount { nullptr };
    std::uint32_t* proximityEeCount { nullptr };
    std::size_t proximityContactCapacity { 0 };
    // Workspace-owned counter-readback buffer (5 contiguous uint32s). Lets us
    // do one cudaMemcpyAsync per frame instead of five separate copies.
    std::uint32_t* proximityCountersHostPinned { nullptr };  // 5 uint32s, pinned for fast D2H
    // Workspace-owned CUDA events so the FBP path does not pay cudaEventCreate /
    // cudaEventDestroy on every frame. Created lazily on first use.
    cudaEvent_t fbpStartEvent { nullptr };
    cudaEvent_t fbpEndEvent { nullptr };
    bool fbpEventsReady { false };
    // Phase 16: workspace-owned broad-cull timing events. The dense-grid
    // contact functions used to cudaEventCreate/Destroy these 4 events every
    // frame (~40-80 us of driver churn). Lazy-created once, reused, freed in
    // the workspace destructor.
    cudaEvent_t broadStageStart { nullptr };
    cudaEvent_t broadStageEnd { nullptr };
    cudaEvent_t broadTotalStart { nullptr };
    cudaEvent_t broadTotalEnd { nullptr };
    bool broadEventsReady { false };
    // Frame counter for sampled deep-counter readback.
    std::uint64_t frameCounter { 0 };

    std::size_t tissueTriangleCapacity { 0 };
    std::size_t toolTriangleCapacity { 0 };
    std::size_t tissueAabbCapacity { 0 };
    std::size_t toolAabbCapacity { 0 };
    std::size_t gridCellCapacity { 0 };
    std::size_t activeCellCapacity { 0 };
    std::size_t tissueBucketCapacity { 0 };
    std::size_t toolBucketCapacity { 0 };
    std::size_t candidateCapacity { 0 };
    std::size_t pairHashCapacity { 0 };
    std::size_t contactCapacity { 0 };
    std::size_t rawCandidateCounterCapacity { 0 };
    std::size_t candidateCounterCapacity { 0 };
    std::size_t contactCounterCapacity { 0 };
    std::size_t overflowCounterCapacity { 0 };
    std::size_t activeCellCounterCapacity { 0 };
    std::size_t denseGridStatsCapacity { 0 };
    std::size_t indexedTissuePositionCapacity { 0 };
    std::size_t indexedToolPositionCapacity { 0 };
    std::size_t indexedTissueIndexCapacity { 0 };
    std::size_t indexedToolIndexCapacity { 0 };
    std::size_t pinnedTissueTriangleCapacity { 0 };
    std::size_t pinnedToolTriangleCapacity { 0 };
    std::size_t pinnedIndexedTissuePositionCapacity { 0 };
    std::size_t pinnedIndexedToolPositionCapacity { 0 };
    std::size_t pinnedIndexedTissueIndexCapacity { 0 };
    std::size_t pinnedIndexedToolIndexCapacity { 0 };
    std::uint64_t indexedTissueSurfaceId { 0 };
    std::uint64_t indexedToolSurfaceId { 0 };
    std::uint64_t indexedTissueTopologyVersion { 0 };
    std::uint64_t indexedToolTopologyVersion { 0 };

    ~DenseGridWorkspace()
    {
        release();
    }

    void release()
    {
        cudaFree(tissueTriangles);
        cudaFree(toolTriangles);
        cudaFree(tissueAabbs);
        cudaFree(toolAabbs);
        cudaFree(grid);
        cudaFree(activeCellIds);
        cudaFree(cellTissueIds);
        cudaFree(cellToolIds);
        cudaFree(candidatePairs);
        cudaFree(pairHashKeys);
        cudaFree(contacts);
        cudaFree(rawCandidateCount);
        cudaFree(candidateCount);
        cudaFree(contactCount);
        cudaFree(overflowCount);
        cudaFree(activeCellCount);
        cudaFree(denseGridStats);
        cudaFree(indexedTissuePositions);
        cudaFree(indexedToolPositions);
        cudaFree(indexedTissueIndices);
        cudaFree(indexedToolIndices);
        cudaFreeHost(pinnedTissueTriangles);
        cudaFreeHost(pinnedToolTriangles);
        cudaFreeHost(pinnedIndexedTissuePositions);
        cudaFreeHost(pinnedIndexedToolPositions);
        cudaFreeHost(pinnedIndexedTissueIndices);
        cudaFreeHost(pinnedIndexedToolIndices);
        cudaFree(proximityContacts);
        cudaFree(proximityContactCount);
        cudaFree(proximityOverflowCount);
        cudaFree(proximityVfCount);
        cudaFree(proximityFvCount);
        cudaFree(proximityEeCount);
        cudaFreeHost(proximityCountersHostPinned);
        if (fbpEventsReady)
        {
            cudaEventDestroy(fbpStartEvent);
            cudaEventDestroy(fbpEndEvent);
            fbpEventsReady = false;
            fbpStartEvent = nullptr;
            fbpEndEvent = nullptr;
        }
        if (broadEventsReady)
        {
            cudaEventDestroy(broadStageStart);
            cudaEventDestroy(broadStageEnd);
            cudaEventDestroy(broadTotalStart);
            cudaEventDestroy(broadTotalEnd);
            broadEventsReady = false;
            broadStageStart = nullptr;
            broadStageEnd = nullptr;
            broadTotalStart = nullptr;
            broadTotalEnd = nullptr;
        }
        proximityContacts = nullptr;
        proximityContactCount = nullptr;
        proximityOverflowCount = nullptr;
        proximityVfCount = nullptr;
        proximityFvCount = nullptr;
        proximityEeCount = nullptr;
        proximityCountersHostPinned = nullptr;
        proximityContactCapacity = 0;
        tissueTriangles = nullptr;
        toolTriangles = nullptr;
        tissueAabbs = nullptr;
        toolAabbs = nullptr;
        grid = nullptr;
        activeCellIds = nullptr;
        cellTissueIds = nullptr;
        cellToolIds = nullptr;
        candidatePairs = nullptr;
        pairHashKeys = nullptr;
        contacts = nullptr;
        rawCandidateCount = nullptr;
        candidateCount = nullptr;
        contactCount = nullptr;
        overflowCount = nullptr;
        activeCellCount = nullptr;
        denseGridStats = nullptr;
        indexedTissuePositions = nullptr;
        indexedToolPositions = nullptr;
        indexedTissueIndices = nullptr;
        indexedToolIndices = nullptr;
        pinnedTissueTriangles = nullptr;
        pinnedToolTriangles = nullptr;
        pinnedIndexedTissuePositions = nullptr;
        pinnedIndexedToolPositions = nullptr;
        pinnedIndexedTissueIndices = nullptr;
        pinnedIndexedToolIndices = nullptr;
        tissueTriangleCapacity = 0;
        toolTriangleCapacity = 0;
        tissueAabbCapacity = 0;
        toolAabbCapacity = 0;
        gridCellCapacity = 0;
        activeCellCapacity = 0;
        tissueBucketCapacity = 0;
        toolBucketCapacity = 0;
        candidateCapacity = 0;
        pairHashCapacity = 0;
        contactCapacity = 0;
        rawCandidateCounterCapacity = 0;
        candidateCounterCapacity = 0;
        contactCounterCapacity = 0;
        overflowCounterCapacity = 0;
        activeCellCounterCapacity = 0;
        denseGridStatsCapacity = 0;
        indexedTissuePositionCapacity = 0;
        indexedToolPositionCapacity = 0;
        indexedTissueIndexCapacity = 0;
        indexedToolIndexCapacity = 0;
        pinnedTissueTriangleCapacity = 0;
        pinnedToolTriangleCapacity = 0;
        pinnedIndexedTissuePositionCapacity = 0;
        pinnedIndexedToolPositionCapacity = 0;
        pinnedIndexedTissueIndexCapacity = 0;
        pinnedIndexedToolIndexCapacity = 0;
        indexedTissueSurfaceId = 0;
        indexedToolSurfaceId = 0;
        indexedTissueTopologyVersion = 0;
        indexedToolTopologyVersion = 0;
    }

    cudaError_t ensurePinnedHostStaging(
        const std::size_t tissueTriangleCount,
        const std::size_t toolTriangleCount,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensurePinnedHostArray(
            pinnedTissueTriangles,
            pinnedTissueTriangleCapacity,
            tissueTriangleCount,
            newlyAllocatedBytes);
        if (err == cudaSuccess)
        {
            err = ensurePinnedHostArray(
                pinnedToolTriangles,
                pinnedToolTriangleCapacity,
                toolTriangleCount,
                newlyAllocatedBytes);
        }
        return err;
    }

    cudaError_t ensureIndexedPinnedHostStaging(
        const std::size_t tissueVertexCount,
        const std::size_t toolVertexCount,
        const std::size_t tissueIndexCount,
        const std::size_t toolIndexCount,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensurePinnedHostArray(
            pinnedIndexedTissuePositions,
            pinnedIndexedTissuePositionCapacity,
            tissueVertexCount,
            newlyAllocatedBytes);
        if (err == cudaSuccess)
        {
            err = ensurePinnedHostArray(
                pinnedIndexedToolPositions,
                pinnedIndexedToolPositionCapacity,
                toolVertexCount,
                newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensurePinnedHostArray(
                pinnedIndexedTissueIndices,
                pinnedIndexedTissueIndexCapacity,
                tissueIndexCount,
                newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensurePinnedHostArray(
                pinnedIndexedToolIndices,
                pinnedIndexedToolIndexCapacity,
                toolIndexCount,
                newlyAllocatedBytes);
        }
        return err;
    }

    cudaError_t ensureIndexedInput(
        const std::size_t tissueVertexCount,
        const std::size_t toolVertexCount,
        const std::size_t tissueIndexCount,
        const std::size_t toolIndexCount,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensureDeviceArray(
            indexedTissuePositions,
            indexedTissuePositionCapacity,
            tissueVertexCount,
            newlyAllocatedBytes);
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(
                indexedToolPositions,
                indexedToolPositionCapacity,
                toolVertexCount,
                newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(
                indexedTissueIndices,
                indexedTissueIndexCapacity,
                tissueIndexCount,
                newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(
                indexedToolIndices,
                indexedToolIndexCapacity,
                toolIndexCount,
                newlyAllocatedBytes);
        }
        return err;
    }

    cudaError_t ensure(
        const std::size_t tissueTriangleCount,
        const std::size_t toolTriangleCount,
        const std::size_t cellCount,
        const std::size_t tissueBucketCount,
        const std::size_t toolBucketCount,
        const std::size_t candidateCountCapacity,
        const std::size_t pairHashCountCapacity,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensureDeviceArray(tissueTriangles, tissueTriangleCapacity, tissueTriangleCount, newlyAllocatedBytes);
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(toolTriangles, toolTriangleCapacity, toolTriangleCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(tissueAabbs, tissueAabbCapacity, tissueTriangleCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(toolAabbs, toolAabbCapacity, toolTriangleCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(grid, gridCellCapacity, cellCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(activeCellIds, activeCellCapacity, cellCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(cellTissueIds, tissueBucketCapacity, tissueBucketCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(cellToolIds, toolBucketCapacity, toolBucketCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(candidatePairs, candidateCapacity, candidateCountCapacity, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(pairHashKeys, pairHashCapacity, pairHashCountCapacity, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(contacts, contactCapacity, candidateCountCapacity, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(rawCandidateCount, rawCandidateCounterCapacity, 1, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(candidateCount, candidateCounterCapacity, 1, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(contactCount, contactCounterCapacity, 1, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(overflowCount, overflowCounterCapacity, 1, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(activeCellCount, activeCellCounterCapacity, 1, newlyAllocatedBytes);
        }
        if (err == cudaSuccess)
        {
            err = ensureDeviceArray(denseGridStats, denseGridStatsCapacity, 1, newlyAllocatedBytes);
        }
        return err;
    }

    // Allocate device storage for feature-based proximity outputs. The element
    // size argument keeps DeviceProximityContact (declared later in this TU)
    // out of the workspace's type signature.
    cudaError_t ensureProximityStorage(
        const std::size_t contactCount,
        const std::size_t contactElementBytes,
        std::uint64_t& newlyAllocatedBytes)
    {
        const std::size_t requestedBytes = contactCount * contactElementBytes;
        const std::size_t currentBytes = proximityContactCapacity * contactElementBytes;
        cudaError_t err = cudaSuccess;
        if (requestedBytes > currentBytes)
        {
            cudaFree(proximityContacts);
            proximityContacts = nullptr;
            void* devicePtr = nullptr;
            err = cudaMalloc(&devicePtr, requestedBytes);
            if (err == cudaSuccess)
            {
                proximityContacts = devicePtr;
                proximityContactCapacity = contactCount;
                newlyAllocatedBytes += static_cast<std::uint64_t>(requestedBytes);
            }
        }
        if (err == cudaSuccess && proximityContactCount == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&proximityContactCount), sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += sizeof(std::uint32_t);
        }
        if (err == cudaSuccess && proximityOverflowCount == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&proximityOverflowCount), sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += sizeof(std::uint32_t);
        }
        if (err == cudaSuccess && proximityVfCount == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&proximityVfCount), sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += sizeof(std::uint32_t);
        }
        if (err == cudaSuccess && proximityFvCount == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&proximityFvCount), sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += sizeof(std::uint32_t);
        }
        if (err == cudaSuccess && proximityEeCount == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&proximityEeCount), sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += sizeof(std::uint32_t);
        }
        if (err == cudaSuccess && proximityCountersHostPinned == nullptr)
        {
            err = cudaMallocHost(reinterpret_cast<void**>(&proximityCountersHostPinned), 5u * sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += 5u * sizeof(std::uint32_t);
        }
        return err;
    }

    cudaError_t ensureFbpEvents()
    {
        if (fbpEventsReady) return cudaSuccess;
        cudaError_t err = cudaEventCreate(&fbpStartEvent);
        if (err == cudaSuccess) err = cudaEventCreate(&fbpEndEvent);
        fbpEventsReady = (err == cudaSuccess);
        return err;
    }

    cudaError_t ensureBroadEvents()
    {
        if (broadEventsReady) return cudaSuccess;
        cudaError_t err = cudaEventCreate(&broadStageStart);
        if (err == cudaSuccess) err = cudaEventCreate(&broadStageEnd);
        if (err == cudaSuccess) err = cudaEventCreate(&broadTotalStart);
        if (err == cudaSuccess) err = cudaEventCreate(&broadTotalEnd);
        broadEventsReady = (err == cudaSuccess);
        return err;
    }
};

DenseGridWorkspace& denseGridWorkspace()
{
    static DenseGridWorkspace workspace;
    return workspace;
}


__global__ void resetDenseGridKernel(
    DeviceCellBucket* grid,
    const std::uint32_t cellCount,
    unsigned long long* pairHashKeys,
    const std::uint32_t pairHashCapacity,
    const bool resetPairHash,
    std::uint32_t* activeCellCount,
    std::uint32_t* rawCandidateCount,
    std::uint32_t* candidateCount,
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats,
    const bool resetDenseGridStats)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < cellCount)
    {
        grid[id] = DeviceCellBucket { 0u, 0u };
    }
    if (resetPairHash && id < pairHashCapacity)
    {
        pairHashKeys[id] = static_cast<unsigned long long>(kEmptyPairSlot);
    }
    if (id == 0)
    {
        *activeCellCount = 0u;
        *rawCandidateCount = 0u;
        *candidateCount = 0u;
        *contactCount = 0u;
        *overflowCount = 0u;
        if (resetDenseGridStats && denseGridStats != nullptr)
        {
            *denseGridStats = DeviceDenseGridStats {};
        }
    }
}

__global__ void compactActiveDenseGridCellsKernel(
    const DeviceCellBucket* grid,
    const std::uint32_t cellCount,
    const DeviceDenseGridConfig config,
    std::uint32_t* activeCellIds,
    std::uint32_t* activeCellCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= cellCount)
    {
        return;
    }

    const auto bucket = grid[cellId];
    const std::uint32_t tissueCount = min(bucket.tissueCount, config.maxTissueTrianglesPerCell);
    const std::uint32_t toolCount = min(bucket.toolCount, config.maxToolTrianglesPerCell);

    if (denseGridStats != nullptr)
    {
        atomicMax(&denseGridStats->maxTissueCellOccupancy, tissueCount);
        atomicMax(&denseGridStats->maxToolCellOccupancy, toolCount);
    }

    if (tissueCount == 0 || toolCount == 0)
    {
        return;
    }

    const std::uint32_t outputIndex = atomicAdd(activeCellCount, 1u);
    activeCellIds[outputIndex] = cellId;
    if (denseGridStats != nullptr)
    {
        atomicAdd(&denseGridStats->activeMixedCellCount, 1u);
    }
}

__global__ void triangleAabbKernel(
    const DeviceTriangle* triangles,
    const std::uint32_t triangleCount,
    const float contactDistance,
    DeviceAabb* aabbs)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount)
    {
        return;
    }

    const auto triangle = triangles[triangleId];
    const float3 minPoint = min3(min3(triangle.p0, triangle.p1), triangle.p2);
    const float3 maxPoint = max3(max3(triangle.p0, triangle.p1), triangle.p2);

    aabbs[triangleId] = DeviceAabb {
        minPoint.x - contactDistance,
        minPoint.y - contactDistance,
        minPoint.z - contactDistance,
        maxPoint.x + contactDistance,
        maxPoint.y + contactDistance,
        maxPoint.z + contactDistance
    };
}

__device__ DeviceAabb triangleAabb(const DeviceTriangle& triangle, const float contactDistance)
{
    const float3 minPoint = min3(min3(triangle.p0, triangle.p1), triangle.p2);
    const float3 maxPoint = max3(max3(triangle.p0, triangle.p1), triangle.p2);

    return DeviceAabb {
        minPoint.x - contactDistance,
        minPoint.y - contactDistance,
        minPoint.z - contactDistance,
        maxPoint.x + contactDistance,
        maxPoint.y + contactDistance,
        maxPoint.z + contactDistance
    };
}

__device__ DeviceTriangle indexedTriangleAt(
    const BackendTriangleVertex* positions,
    const std::uint32_t* triangleIndices,
    const std::uint32_t triangleId)
{
    const std::uint32_t i0 = triangleIndices[3u * triangleId + 0u];
    const std::uint32_t i1 = triangleIndices[3u * triangleId + 1u];
    const std::uint32_t i2 = triangleIndices[3u * triangleId + 2u];
    const auto p0 = positions[i0];
    const auto p1 = positions[i1];
    const auto p2 = positions[i2];
    return DeviceTriangle {
        make_float3(p0.x, p0.y, p0.z),
        make_float3(p1.x, p1.y, p1.z),
        make_float3(p2.x, p2.y, p2.z),
        triangleId
    };
}

__device__ bool denseGridCellSpan(
    const DeviceAabb& aabb,
    const DeviceDenseGridConfig& config,
    int3& cellMin,
    int3& cellMax)
{
    if (aabb.maxX < config.gridMin.x || aabb.minX > config.gridMax.x ||
        aabb.maxY < config.gridMin.y || aabb.minY > config.gridMax.y ||
        aabb.maxZ < config.gridMin.z || aabb.minZ > config.gridMax.z)
    {
        return false;
    }

    cellMin.x = static_cast<int>(floorf((aabb.minX - config.gridMin.x) * config.inverseCellSize.x));
    cellMin.y = static_cast<int>(floorf((aabb.minY - config.gridMin.y) * config.inverseCellSize.y));
    cellMin.z = static_cast<int>(floorf((aabb.minZ - config.gridMin.z) * config.inverseCellSize.z));
    cellMax.x = static_cast<int>(floorf((aabb.maxX - config.gridMin.x) * config.inverseCellSize.x));
    cellMax.y = static_cast<int>(floorf((aabb.maxY - config.gridMin.y) * config.inverseCellSize.y));
    cellMax.z = static_cast<int>(floorf((aabb.maxZ - config.gridMin.z) * config.inverseCellSize.z));

    cellMin.x = max(0, min(cellMin.x, static_cast<int>(config.resolutionX) - 1));
    cellMin.y = max(0, min(cellMin.y, static_cast<int>(config.resolutionY) - 1));
    cellMin.z = max(0, min(cellMin.z, static_cast<int>(config.resolutionZ) - 1));
    cellMax.x = max(0, min(cellMax.x, static_cast<int>(config.resolutionX) - 1));
    cellMax.y = max(0, min(cellMax.y, static_cast<int>(config.resolutionY) - 1));
    cellMax.z = max(0, min(cellMax.z, static_cast<int>(config.resolutionZ) - 1));
    return true;
}

__device__ std::uint32_t denseCellId(
    const int x,
    const int y,
    const int z,
    const DeviceDenseGridConfig& config)
{
    return static_cast<std::uint32_t>(
        x + y * static_cast<int>(config.resolutionX) +
        z * static_cast<int>(config.resolutionX * config.resolutionY));
}

__device__ void insertTriangleAabbIntoGrid(
    const DeviceAabb& aabb,
    const std::uint32_t triangleId,
    const bool insertTissue,
    const DeviceDenseGridConfig& config,
    DeviceCellBucket* grid,
    std::uint32_t* cellIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats,
    const bool buildToolActiveList = false,
    std::uint32_t* activeCellIds = nullptr,
    std::uint32_t* activeCellCount = nullptr)
{
    int3 cellMin {};
    int3 cellMax {};
    if (!denseGridCellSpan(aabb, config, cellMin, cellMax))
    {
        return;
    }

    const std::uint32_t bucketCapacity =
        insertTissue ? config.maxTissueTrianglesPerCell : config.maxToolTrianglesPerCell;

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        for (int y = cellMin.y; y <= cellMax.y; ++y)
        {
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t cellId = denseCellId(x, y, z, config);
                std::uint32_t* bucketCount =
                    insertTissue ? &grid[cellId].tissueCount : &grid[cellId].toolCount;
                const std::uint32_t localIndex = atomicAdd(bucketCount, 1u);
                if (localIndex < bucketCapacity)
                {
                    cellIds[cellId * bucketCapacity + localIndex] = triangleId;
                    // Phase 15: build the mixed-cell list as a side effect of the
                    // tool insert. The FIRST tool triangle to claim slot 0 in a
                    // cell that ALREADY contains tissue appends the cell id once.
                    // localIndex == 0 gives natural dedupe (each cell at most
                    // once); grid[cellId].tissueCount is final because the tissue
                    // insert kernel completes before the tool insert kernel on the
                    // serialized stream. activeCellIds is sized to cellCount, and
                    // distinct cells <= cellCount, so the append can never overflow.
                    if (buildToolActiveList && !insertTissue && localIndex == 0u &&
                        grid[cellId].tissueCount > 0u)
                    {
                        const std::uint32_t activeIndex = atomicAdd(activeCellCount, 1u);
                        activeCellIds[activeIndex] = cellId;
                    }
                    if (denseGridStats != nullptr)
                    {
                        if (insertTissue)
                        {
                            atomicAdd(&denseGridStats->tissueInsertCount, 1u);
                        }
                        else
                        {
                            atomicAdd(&denseGridStats->toolInsertCount, 1u);
                        }
                    }
                }
                else
                {
                    atomicAdd(overflowCount, 1u);
                }
            }
        }
    }
}

__global__ void insertPackedTrianglesKernel(
    const DeviceTriangle* triangles,
    const std::uint32_t triangleCount,
    const bool insertTissue,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats,
    const bool buildToolActiveList = false,
    std::uint32_t* activeCellIds = nullptr,
    std::uint32_t* activeCellCount = nullptr)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount)
    {
        return;
    }

    insertTriangleAabbIntoGrid(
        triangleAabb(triangles[triangleId], config.contactDistance),
        triangleId,
        insertTissue,
        config,
        grid,
        cellIds,
        overflowCount,
        denseGridStats,
        buildToolActiveList,
        activeCellIds,
        activeCellCount);
}

__global__ void insertIndexedTrianglesKernel(
    const BackendTriangleVertex* positions,
    const std::uint32_t* triangleIndices,
    const std::uint32_t triangleCount,
    const bool insertTissue,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats,
    const bool buildToolActiveList = false,
    std::uint32_t* activeCellIds = nullptr,
    std::uint32_t* activeCellCount = nullptr)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount)
    {
        return;
    }

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    insertTriangleAabbIntoGrid(
        triangleAabb(triangle, config.contactDistance),
        triangleId,
        insertTissue,
        config,
        grid,
        cellIds,
        overflowCount,
        denseGridStats,
        buildToolActiveList,
        activeCellIds,
        activeCellCount);
}

__global__ void insertPackedTrianglePairKernel(
    const DeviceTriangle* tissueTriangles,
    const std::uint32_t tissueTriangleCount,
    const DeviceTriangle* toolTriangles,
    const std::uint32_t toolTriangleCount,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellTissueIds,
    std::uint32_t* cellToolIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t globalTriangleId = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t totalTriangleCount = tissueTriangleCount + toolTriangleCount;
    if (globalTriangleId >= totalTriangleCount)
    {
        return;
    }

    if (globalTriangleId < tissueTriangleCount)
    {
        insertTriangleAabbIntoGrid(
            triangleAabb(tissueTriangles[globalTriangleId], config.contactDistance),
            globalTriangleId,
            true,
            config,
            grid,
            cellTissueIds,
            overflowCount,
            denseGridStats);
        return;
    }

    const std::uint32_t toolTriangleId = globalTriangleId - tissueTriangleCount;
    insertTriangleAabbIntoGrid(
        triangleAabb(toolTriangles[toolTriangleId], config.contactDistance),
        toolTriangleId,
        false,
        config,
        grid,
        cellToolIds,
        overflowCount,
        denseGridStats);
}

__global__ void insertIndexedTrianglePairKernel(
    const BackendTriangleVertex* tissuePositions,
    const std::uint32_t* tissueTriangleIndices,
    const std::uint32_t tissueTriangleCount,
    const BackendTriangleVertex* toolPositions,
    const std::uint32_t* toolTriangleIndices,
    const std::uint32_t toolTriangleCount,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellTissueIds,
    std::uint32_t* cellToolIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t globalTriangleId = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t totalTriangleCount = tissueTriangleCount + toolTriangleCount;
    if (globalTriangleId >= totalTriangleCount)
    {
        return;
    }

    if (globalTriangleId < tissueTriangleCount)
    {
        const DeviceTriangle triangle = indexedTriangleAt(
            tissuePositions,
            tissueTriangleIndices,
            globalTriangleId);
        insertTriangleAabbIntoGrid(
            triangleAabb(triangle, config.contactDistance),
            globalTriangleId,
            true,
            config,
            grid,
            cellTissueIds,
            overflowCount,
            denseGridStats);
        return;
    }

    const std::uint32_t toolTriangleId = globalTriangleId - tissueTriangleCount;
    const DeviceTriangle triangle = indexedTriangleAt(
        toolPositions,
        toolTriangleIndices,
        toolTriangleId);
    insertTriangleAabbIntoGrid(
        triangleAabb(triangle, config.contactDistance),
        toolTriangleId,
        false,
        config,
        grid,
        cellToolIds,
        overflowCount,
        denseGridStats);
}

__global__ void insertTissueTrianglesKernel(
    const DeviceAabb* aabbs,
    const std::uint32_t triangleCount,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellTissueIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount)
    {
        return;
    }

    int3 cellMin {};
    int3 cellMax {};
    if (!denseGridCellSpan(aabbs[triangleId], config, cellMin, cellMax))
    {
        return;
    }

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        for (int y = cellMin.y; y <= cellMax.y; ++y)
        {
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t cellId = denseCellId(x, y, z, config);
                const std::uint32_t localIndex = atomicAdd(&grid[cellId].tissueCount, 1u);
                if (localIndex < config.maxTissueTrianglesPerCell)
                {
                    cellTissueIds[cellId * config.maxTissueTrianglesPerCell + localIndex] = triangleId;
                    if (denseGridStats != nullptr)
                    {
                        atomicAdd(&denseGridStats->tissueInsertCount, 1u);
                    }
                }
                else
                {
                    atomicAdd(overflowCount, 1u);
                }
            }
        }
    }
}

__global__ void insertToolTrianglesKernel(
    const DeviceAabb* aabbs,
    const std::uint32_t triangleCount,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* grid,
    std::uint32_t* cellToolIds,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount)
    {
        return;
    }

    int3 cellMin {};
    int3 cellMax {};
    if (!denseGridCellSpan(aabbs[triangleId], config, cellMin, cellMax))
    {
        return;
    }

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        for (int y = cellMin.y; y <= cellMax.y; ++y)
        {
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t cellId = denseCellId(x, y, z, config);
                const std::uint32_t localIndex = atomicAdd(&grid[cellId].toolCount, 1u);
                if (localIndex < config.maxToolTrianglesPerCell)
                {
                    cellToolIds[cellId * config.maxToolTrianglesPerCell + localIndex] = triangleId;
                    if (denseGridStats != nullptr)
                    {
                        atomicAdd(&denseGridStats->toolInsertCount, 1u);
                    }
                }
                else
                {
                    atomicAdd(overflowCount, 1u);
                }
            }
        }
    }
}

__device__ std::uint64_t encodeCandidatePair(const std::uint32_t tissueTriangleId, const std::uint32_t toolTriangleId)
{
    return (static_cast<std::uint64_t>(tissueTriangleId) << 32u) | static_cast<std::uint64_t>(toolTriangleId);
}

__device__ std::uint64_t mixCandidatePairHash(std::uint64_t value)
{
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33u;
    return value;
}

__device__ bool insertUniqueCandidatePair(
    const std::uint64_t pair,
    unsigned long long* pairHashKeys,
    const std::uint32_t pairHashCapacity,
    std::uint64_t* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats,
    const std::uint32_t maxCandidatePairs)
{
    constexpr std::uint32_t kMaxProbeCount = 256u;
    const std::uint32_t mask = pairHashCapacity - 1u;
    std::uint32_t slot = static_cast<std::uint32_t>(mixCandidatePairHash(pair)) & mask;

    for (std::uint32_t probe = 0; probe < kMaxProbeCount; ++probe)
    {
        const std::uint64_t previous = atomicCAS(
            &pairHashKeys[slot],
            static_cast<unsigned long long>(kEmptyPairSlot),
            static_cast<unsigned long long>(pair));

        if (previous == kEmptyPairSlot)
        {
            const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            if (outputIndex < maxCandidatePairs)
            {
                candidatePairs[outputIndex] = pair;
            }
            else
            {
                atomicAdd(overflowCount, 1u);
            }
            return true;
        }
        if (previous == pair)
        {
            return false;
        }
        slot = (slot + 1u) & mask;
    }

    if (denseGridStats != nullptr)
    {
        atomicAdd(&denseGridStats->hashDedupeProbeOverflowCount, 1u);
    }
    atomicAdd(overflowCount, 1u);
    return false;
}

__device__ bool isCanonicalPairCell(
    const std::uint32_t cellId,
    const DeviceAabb& tissueAabb,
    const DeviceAabb& toolAabb,
    const DeviceDenseGridConfig& config)
{
    int3 tissueCellMin {};
    int3 tissueCellMax {};
    int3 toolCellMin {};
    int3 toolCellMax {};
    if (!denseGridCellSpan(tissueAabb, config, tissueCellMin, tissueCellMax) ||
        !denseGridCellSpan(toolAabb, config, toolCellMin, toolCellMax))
    {
        return false;
    }

    const int canonicalX = max(tissueCellMin.x, toolCellMin.x);
    const int canonicalY = max(tissueCellMin.y, toolCellMin.y);
    const int canonicalZ = max(tissueCellMin.z, toolCellMin.z);
    const int maxX = min(tissueCellMax.x, toolCellMax.x);
    const int maxY = min(tissueCellMax.y, toolCellMax.y);
    const int maxZ = min(tissueCellMax.z, toolCellMax.z);
    if (canonicalX > maxX || canonicalY > maxY || canonicalZ > maxZ)
    {
        return false;
    }

    return cellId == denseCellId(canonicalX, canonicalY, canonicalZ, config);
}

__global__ void generateDenseGridCandidatePairsKernel(
    const DeviceCellBucket* grid,
    const std::uint32_t* cellTissueIds,
    const std::uint32_t* cellToolIds,
    const DeviceAabb* tissueAabbs,
    const DeviceAabb* toolAabbs,
    const DeviceDenseGridConfig config,
    std::uint64_t* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t cellId = blockIdx.x;
    const auto bucket = grid[cellId];
    const std::uint32_t tissueCount = min(bucket.tissueCount, config.maxTissueTrianglesPerCell);
    const std::uint32_t toolCount = min(bucket.toolCount, config.maxToolTrianglesPerCell);
    const std::uint64_t totalPairs =
        static_cast<std::uint64_t>(tissueCount) * static_cast<std::uint64_t>(toolCount);

    if (threadIdx.x == 0 && denseGridStats != nullptr)
    {
        if (tissueCount > 0 && toolCount > 0)
        {
            atomicAdd(&denseGridStats->activeMixedCellCount, 1u);
        }
        atomicMax(&denseGridStats->maxTissueCellOccupancy, tissueCount);
        atomicMax(&denseGridStats->maxToolCellOccupancy, toolCount);
    }

    for (std::uint64_t localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x)
    {
        const std::uint32_t tissueLocal = static_cast<std::uint32_t>(localPair / toolCount);
        const std::uint32_t toolLocal = static_cast<std::uint32_t>(localPair % toolCount);
        const std::uint32_t tissueTriangleId =
            cellTissueIds[cellId * config.maxTissueTrianglesPerCell + tissueLocal];
        const std::uint32_t toolTriangleId =
            cellToolIds[cellId * config.maxToolTrianglesPerCell + toolLocal];

        if (config.canonicalPairEmission &&
            !isCanonicalPairCell(cellId, tissueAabbs[tissueTriangleId], toolAabbs[toolTriangleId], config))
        {
            continue;
        }

        const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
        if (outputIndex < config.maxCandidatePairs)
        {
            candidatePairs[outputIndex] = encodeCandidatePair(tissueTriangleId, toolTriangleId);
        }
        else
        {
            atomicAdd(overflowCount, 1u);
        }
    }
}

__global__ void generateDenseGridUniqueCandidatePairsKernel(
    const DeviceCellBucket* grid,
    const std::uint32_t* cellTissueIds,
    const std::uint32_t* cellToolIds,
    const DeviceDenseGridConfig config,
    std::uint64_t* candidatePairs,
    unsigned long long* pairHashKeys,
    std::uint32_t* rawCandidateCount,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t cellId = blockIdx.x;
    const auto bucket = grid[cellId];
    const std::uint32_t tissueCount = min(bucket.tissueCount, config.maxTissueTrianglesPerCell);
    const std::uint32_t toolCount = min(bucket.toolCount, config.maxToolTrianglesPerCell);
    const std::uint64_t totalPairs =
        static_cast<std::uint64_t>(tissueCount) * static_cast<std::uint64_t>(toolCount);

    if (threadIdx.x == 0 && denseGridStats != nullptr)
    {
        if (tissueCount > 0 && toolCount > 0)
        {
            atomicAdd(&denseGridStats->activeMixedCellCount, 1u);
        }
        atomicMax(&denseGridStats->maxTissueCellOccupancy, tissueCount);
        atomicMax(&denseGridStats->maxToolCellOccupancy, toolCount);
    }

    for (std::uint64_t localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x)
    {
        const std::uint32_t tissueLocal = static_cast<std::uint32_t>(localPair / toolCount);
        const std::uint32_t toolLocal = static_cast<std::uint32_t>(localPair % toolCount);
        const std::uint32_t tissueTriangleId =
            cellTissueIds[cellId * config.maxTissueTrianglesPerCell + tissueLocal];
        const std::uint32_t toolTriangleId =
            cellToolIds[cellId * config.maxToolTrianglesPerCell + toolLocal];

        atomicAdd(rawCandidateCount, 1u);
        insertUniqueCandidatePair(
            encodeCandidatePair(tissueTriangleId, toolTriangleId),
            pairHashKeys,
            config.pairHashCapacity,
            candidatePairs,
            candidateCount,
            overflowCount,
            denseGridStats,
            config.maxCandidatePairs);
    }
}

__global__ void generateActiveDenseGridCandidatePairsKernel(
    const DeviceCellBucket* grid,
    const std::uint32_t* activeCellIds,
    const std::uint32_t* activeCellCount,
    const std::uint32_t* cellTissueIds,
    const std::uint32_t* cellToolIds,
    const DeviceAabb* tissueAabbs,
    const DeviceAabb* toolAabbs,
    const DeviceDenseGridConfig config,
    std::uint64_t* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount)
{
    const std::uint32_t activeCount = *activeCellCount;
    for (std::uint32_t activeIndex = blockIdx.x; activeIndex < activeCount; activeIndex += gridDim.x)
    {
        const std::uint32_t cellId = activeCellIds[activeIndex];
        const auto bucket = grid[cellId];
        const std::uint32_t tissueCount = min(bucket.tissueCount, config.maxTissueTrianglesPerCell);
        const std::uint32_t toolCount = min(bucket.toolCount, config.maxToolTrianglesPerCell);
        const std::uint64_t totalPairs =
            static_cast<std::uint64_t>(tissueCount) * static_cast<std::uint64_t>(toolCount);

        for (std::uint64_t localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x)
        {
            const std::uint32_t tissueLocal = static_cast<std::uint32_t>(localPair / toolCount);
            const std::uint32_t toolLocal = static_cast<std::uint32_t>(localPair % toolCount);
            const std::uint32_t tissueTriangleId =
                cellTissueIds[cellId * config.maxTissueTrianglesPerCell + tissueLocal];
            const std::uint32_t toolTriangleId =
                cellToolIds[cellId * config.maxToolTrianglesPerCell + toolLocal];

            if (config.canonicalPairEmission &&
                !isCanonicalPairCell(cellId, tissueAabbs[tissueTriangleId], toolAabbs[toolTriangleId], config))
            {
                continue;
            }

            const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            if (outputIndex < config.maxCandidatePairs)
            {
                candidatePairs[outputIndex] = encodeCandidatePair(tissueTriangleId, toolTriangleId);
            }
            else
            {
                atomicAdd(overflowCount, 1u);
            }
        }
    }
}

__global__ void generateActiveDenseGridUniqueCandidatePairsKernel(
    const DeviceCellBucket* grid,
    const std::uint32_t* activeCellIds,
    const std::uint32_t* activeCellCount,
    const std::uint32_t* cellTissueIds,
    const std::uint32_t* cellToolIds,
    const DeviceDenseGridConfig config,
    std::uint64_t* candidatePairs,
    unsigned long long* pairHashKeys,
    std::uint32_t* rawCandidateCount,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    DeviceDenseGridStats* denseGridStats)
{
    const std::uint32_t activeCount = *activeCellCount;
    for (std::uint32_t activeIndex = blockIdx.x; activeIndex < activeCount; activeIndex += gridDim.x)
    {
        const std::uint32_t cellId = activeCellIds[activeIndex];
        const auto bucket = grid[cellId];
        const std::uint32_t tissueCount = min(bucket.tissueCount, config.maxTissueTrianglesPerCell);
        const std::uint32_t toolCount = min(bucket.toolCount, config.maxToolTrianglesPerCell);
        const std::uint64_t totalPairs =
            static_cast<std::uint64_t>(tissueCount) * static_cast<std::uint64_t>(toolCount);

        for (std::uint64_t localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x)
        {
            const std::uint32_t tissueLocal = static_cast<std::uint32_t>(localPair / toolCount);
            const std::uint32_t toolLocal = static_cast<std::uint32_t>(localPair % toolCount);
            const std::uint32_t tissueTriangleId =
                cellTissueIds[cellId * config.maxTissueTrianglesPerCell + tissueLocal];
            const std::uint32_t toolTriangleId =
                cellToolIds[cellId * config.maxToolTrianglesPerCell + toolLocal];

            atomicAdd(rawCandidateCount, 1u);
            insertUniqueCandidatePair(
                encodeCandidatePair(tissueTriangleId, toolTriangleId),
                pairHashKeys,
                config.pairHashCapacity,
                candidatePairs,
                candidateCount,
                overflowCount,
                denseGridStats,
                config.maxCandidatePairs);
        }
    }
}

__global__ void exactDenseGridContactKernel(
    const DeviceTriangle* tissueTriangles,
    const DeviceTriangle* toolTriangles,
    const std::uint64_t* candidatePairs,
    const std::uint32_t candidateCount,
    DeviceExactContact* contacts,
    const std::uint32_t maxContactCount,
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount)
{
    const std::uint32_t candidateIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidateIndex >= candidateCount)
    {
        return;
    }

    const std::uint64_t candidatePair = candidatePairs[candidateIndex];
    const std::uint32_t tissueTriangleId = static_cast<std::uint32_t>(candidatePair >> 32u);
    const std::uint32_t toolTriangleId = static_cast<std::uint32_t>(candidatePair & 0xffffffffull);

    DeviceExactContact exactContact {};
    if (exactTriangleIntersection(tissueTriangles[tissueTriangleId], toolTriangles[toolTriangleId], exactContact))
    {
        const std::uint32_t outputIndex = atomicAdd(contactCount, 1u);
        if (outputIndex < maxContactCount)
        {
            contacts[outputIndex] = exactContact;
        }
        else
        {
            atomicAdd(overflowCount, 1u);
        }
    }
}

__global__ void exactDenseGridIndexedContactKernel(
    const BackendTriangleVertex* tissuePositions,
    const std::uint32_t* tissueTriangleIndices,
    const BackendTriangleVertex* toolPositions,
    const std::uint32_t* toolTriangleIndices,
    const std::uint64_t* candidatePairs,
    const std::uint32_t candidateCount,
    DeviceExactContact* contacts,
    const std::uint32_t maxContactCount,
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount)
{
    const std::uint32_t candidateIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidateIndex >= candidateCount)
    {
        return;
    }

    const std::uint64_t candidatePair = candidatePairs[candidateIndex];
    const std::uint32_t tissueTriangleId = static_cast<std::uint32_t>(candidatePair >> 32u);
    const std::uint32_t toolTriangleId = static_cast<std::uint32_t>(candidatePair & 0xffffffffull);
    const DeviceTriangle tissueTriangle = indexedTriangleAt(
        tissuePositions,
        tissueTriangleIndices,
        tissueTriangleId);
    const DeviceTriangle toolTriangle = indexedTriangleAt(
        toolPositions,
        toolTriangleIndices,
        toolTriangleId);

    DeviceExactContact exactContact {};
    if (exactTriangleIntersection(tissueTriangle, toolTriangle, exactContact))
    {
        const std::uint32_t outputIndex = atomicAdd(contactCount, 1u);
        if (outputIndex < maxContactCount)
        {
            contacts[outputIndex] = exactContact;
        }
        else
        {
            atomicAdd(overflowCount, 1u);
        }
    }
}

// ============================================================================
// Feature-based proximity (VF / EE) — Ericson closest-point math + kernel
// ----------------------------------------------------------------------------
// Replaces the SAT-style exactTriangleIntersection narrow phase with a smooth
// closest-feature evaluation. For every candidate triangle pair we run 6
// vertex-face tests (3 verts of A vs face B, 3 verts of B vs face A) and 9
// edge-edge tests (3 edges A x 3 edges B). The closest feature pair within
// contactDistance is emitted as a ProximityContact with barycentric weights,
// suitable for a CUDA constraint solver. No divergent clipping loops; the
// math is dot products and branchless clamping, which suits the GTX 1650 Ti.
// ============================================================================



} // namespace


namespace SofaGpuCollision::backend
{

bool computeDenseGridTriangleContacts(
    const std::vector<TrianglePrimitive>& tissueTriangles,
    const std::vector<TrianglePrimitive>& toolTriangles,
    const DenseGridConfig& config,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    ScopedNvtxRange totalRange("SofaGpuCollision dense-grid narrow phase", config.detailedProfiling);
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount =
            static_cast<std::uint32_t>(tissueTriangles.size() + toolTriangles.size());
    }

    if (tissueTriangles.empty() || toolTriangles.empty())
    {
        diagnostic.clear();
        return true;
    }

    if (config.gridResolutionX == 0 || config.gridResolutionY == 0 || config.gridResolutionZ == 0 ||
        config.maxTissueTrianglesPerCell == 0 || config.maxToolTrianglesPerCell == 0 ||
        config.maxCandidatePairs == 0 ||
        config.gridMaxX <= config.gridMinX ||
        config.gridMaxY <= config.gridMinY ||
        config.gridMaxZ <= config.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration.";
        return false;
    }

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(config.gridResolutionX) *
        static_cast<std::uint64_t>(config.gridResolutionY) *
        static_cast<std::uint64_t>(config.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid cell count exceeds uint32_t indexing capacity.";
        return false;
    }
    const auto cellCount = static_cast<std::uint32_t>(cellCount64);

    const std::uint64_t tissueBucketCount =
        cellCount64 * static_cast<std::uint64_t>(config.maxTissueTrianglesPerCell);
    const std::uint64_t toolBucketCount =
        cellCount64 * static_cast<std::uint64_t>(config.maxToolTrianglesPerCell);
    const std::uint64_t pairHashCount64 = config.useGpuHashDedupe
        ? static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(config.maxCandidatePairs) * 2u))
        : 1ull;
    if (tissueBucketCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) ||
        toolBucketCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)))
    {
        diagnostic = "Dense-grid bucket allocation request is too large.";
        return false;
    }
    if (pairHashCount64 == 0 ||
        pairHashCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid hash dedupe table request exceeds uint32_t indexing capacity.";
        return false;
    }

    auto& workspace = denseGridWorkspace();
    constexpr std::size_t kPinnedStagingMinimumBytes = 1u << 20;
    const std::size_t tissueTriangleBytes = tissueTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t toolTriangleBytes = toolTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t totalTriangleBytes = tissueTriangleBytes + toolTriangleBytes;
    bool usingPinnedHostStaging = false;
    if (config.usePinnedHostStaging && totalTriangleBytes >= kPinnedStagingMinimumBytes)
    {
        std::uint64_t newlyPinnedHostBytes = 0;
        const cudaError_t pinnedErr = workspace.ensurePinnedHostStaging(
            tissueTriangles.size(),
            toolTriangles.size(),
            newlyPinnedHostBytes);
        usingPinnedHostStaging = pinnedErr == cudaSuccess;
    }

    const DeviceTriangle* hostTissueTriangleData = nullptr;
    const DeviceTriangle* hostToolTriangleData = nullptr;

    const auto hostPreparationStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange hostPreparationRange("CPU triangle preparation", config.detailedProfiling);
        if (usingPinnedHostStaging)
        {
            if (tissueTriangleBytes != 0)
            {
                std::memcpy(workspace.pinnedTissueTriangles, tissueTriangles.data(), tissueTriangleBytes);
            }
            if (toolTriangleBytes != 0)
            {
                std::memcpy(workspace.pinnedToolTriangles, toolTriangles.data(), toolTriangleBytes);
            }
            hostTissueTriangleData = workspace.pinnedTissueTriangles;
            hostToolTriangleData = workspace.pinnedToolTriangles;
        }
        else
        {
            hostTissueTriangleData = reinterpret_cast<const DeviceTriangle*>(tissueTriangles.data());
            hostToolTriangleData = reinterpret_cast<const DeviceTriangle*>(toolTriangles.data());
        }
    }

    if (executionStats != nullptr)
    {
        const double packMs = elapsedMillisecondsSince(hostPreparationStart);
        executionStats->hostPreparationMilliseconds += packMs;
        executionStats->backendTrianglePackMilliseconds += packMs;
    }

    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = cellCount;
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(tissueTriangleBytes + toolTriangleBytes);
    }

    std::uint64_t newlyAllocatedBytes = 0;
    const auto deviceAllocationStart = std::chrono::steady_clock::now();
    cudaError_t err = workspace.ensure(
        tissueTriangles.size(),
        toolTriangles.size(),
        static_cast<std::size_t>(cellCount64),
        static_cast<std::size_t>(tissueBucketCount),
        static_cast<std::size_t>(toolBucketCount),
        static_cast<std::size_t>(config.maxCandidatePairs),
        static_cast<std::size_t>(pairHashCount64),
        newlyAllocatedBytes);
    const double deviceAllocationMs = elapsedMillisecondsSince(deviceAllocationStart);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += deviceAllocationMs;
        executionStats->workspaceResizeCount += newlyAllocatedBytes > 0 ? 1u : 0u;
    }

    DeviceTriangle* deviceTissueTriangles = workspace.tissueTriangles;
    DeviceTriangle* deviceToolTriangles = workspace.toolTriangles;
    DeviceAabb* deviceTissueAabbs = workspace.tissueAabbs;
    DeviceAabb* deviceToolAabbs = workspace.toolAabbs;
    DeviceCellBucket* deviceGrid = workspace.grid;
    std::uint32_t* deviceActiveCellIds = workspace.activeCellIds;
    std::uint32_t* deviceCellTissueIds = workspace.cellTissueIds;
    std::uint32_t* deviceCellToolIds = workspace.cellToolIds;
    std::uint64_t* deviceCandidatePairs = workspace.candidatePairs;
    unsigned long long* devicePairHashKeys = workspace.pairHashKeys;
    DeviceExactContact* deviceContacts = workspace.contacts;
    std::uint32_t* deviceRawCandidateCount = workspace.rawCandidateCount;
    std::uint32_t* deviceCandidateCount = workspace.candidateCount;
    std::uint32_t* deviceContactCount = workspace.contactCount;
    std::uint32_t* deviceOverflowCount = workspace.overflowCount;
    std::uint32_t* deviceActiveCellCount = workspace.activeCellCount;
    DeviceDenseGridStats* deviceDenseGridStats = workspace.denseGridStats;
    const bool readFullDenseGridCounters =
        config.copyContactsToHost ||
        config.readCountersWhenContactsStayOnDevice ||
        config.detailedProfiling;
    const bool computeDeviceContacts =
        config.copyContactsToHost ||
        config.computeDeviceContactsWhenContactsStayOnDevice;
    const bool readCandidateCounters =
        readFullDenseGridCounters ||
        computeDeviceContacts;
    DeviceDenseGridStats* activeDeviceDenseGridStats =
        readFullDenseGridCounters ? deviceDenseGridStats : nullptr;
    auto freeAll = []() {};

    const auto hostToDeviceStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange h2dRange("H2D triangle upload", config.detailedProfiling);
        err = cudaMemcpyAsync(deviceTissueTriangles, hostTissueTriangleData, tissueTriangleBytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
        {
            err = cudaMemcpyAsync(deviceToolTriangles, hostToolTriangleData, toolTriangleBytes, cudaMemcpyHostToDevice);
        }
    }
    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceMilliseconds += elapsedMillisecondsSince(hostToDeviceStart);
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    const DeviceDenseGridConfig deviceConfig {
        make_float3(config.gridMinX, config.gridMinY, config.gridMinZ),
        make_float3(config.gridMaxX, config.gridMaxY, config.gridMaxZ),
        make_float3(
            static_cast<float>(config.gridResolutionX) / (config.gridMaxX - config.gridMinX),
            static_cast<float>(config.gridResolutionY) / (config.gridMaxY - config.gridMinY),
            static_cast<float>(config.gridResolutionZ) / (config.gridMaxZ - config.gridMinZ)),
        config.gridResolutionX,
        config.gridResolutionY,
        config.gridResolutionZ,
        config.contactDistance,
        config.maxTissueTrianglesPerCell,
        config.maxToolTrianglesPerCell,
        config.maxCandidatePairs,
        static_cast<std::uint32_t>(pairHashCount64),
        config.useGpuHashDedupe,
        config.canonicalPairEmission
    };

    constexpr std::uint32_t threadCount = 256;
    const auto tissueBlocks =
        static_cast<std::uint32_t>((tissueTriangles.size() + threadCount - 1) / threadCount);
    const auto toolBlocks =
        static_cast<std::uint32_t>((toolTriangles.size() + threadCount - 1) / threadCount);

    // Phase 16: reuse workspace-owned events instead of creating/destroying
    // four CUDA events every frame. The aliases below keep the rest of this
    // function unchanged; destroyStageEvents is now a no-op because the events
    // live for the plugin's lifetime (freed in the workspace destructor).
    err = workspace.ensureBroadEvents();
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }
    cudaEvent_t& stageStart = workspace.broadStageStart;
    cudaEvent_t& stageEnd = workspace.broadStageEnd;
    cudaEvent_t& totalStart = workspace.broadTotalStart;
    cudaEvent_t& totalEnd = workspace.broadTotalEnd;

    std::uint32_t denseGridLaunchCount = 0;
    auto destroyStageEvents = [&]() { /* workspace-owned; freed at plugin unload */ };

    if (!config.detailedProfiling)
    {
        err = cudaEventRecord(totalStart);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }
    }

    double resetGridMs = 0.0;
    err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, resetGridMs, [&]() {
        ScopedNvtxRange range("reset grid/counters", config.detailedProfiling);
        const std::uint32_t resetItems = cellCount;
        const std::uint32_t resetBlocks = (resetItems + threadCount - 1u) / threadCount;
        resetDenseGridKernel<<<resetBlocks, threadCount>>>(
            deviceGrid,
            cellCount,
            devicePairHashKeys,
            static_cast<std::uint32_t>(pairHashCount64),
            false,
            deviceActiveCellCount,
            deviceRawCandidateCount,
            deviceCandidateCount,
            deviceContactCount,
            deviceOverflowCount,
            deviceDenseGridStats,
            readFullDenseGridCounters);
        return cudaGetLastError();
    });
    denseGridLaunchCount += 1;
    if (executionStats != nullptr)
    {
        executionStats->denseGridClearMilliseconds += resetGridMs;
    }
    if (err == cudaSuccess && config.useGpuHashDedupe)
    {
        double hashClearMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, hashClearMs, [&]() {
            ScopedNvtxRange range("clear pair hash", config.detailedProfiling);
            return cudaMemset(
                devicePairHashKeys,
                0xff,
                static_cast<std::size_t>(pairHashCount64) * sizeof(std::uint64_t));
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridCounterClearMilliseconds += hashClearMs;
            executionStats->cudaMemsetCount += 1;
        }
    }
    const bool useFusedAabbInsert = !config.canonicalPairEmission || config.useGpuHashDedupe;
    if (useFusedAabbInsert)
    {
        if (err == cudaSuccess && config.batchTriangleInsert)
        {
            double insertMs = 0.0;
            const auto combinedBlocks = static_cast<std::uint32_t>(
                ((tissueTriangles.size() + toolTriangles.size()) + threadCount - 1) / threadCount);
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertMs, [&]() {
                ScopedNvtxRange range("fused AABB/insert tissue+tool", config.detailedProfiling);
                insertPackedTrianglePairKernel<<<combinedBlocks, threadCount>>>(
                    deviceTissueTriangles,
                    static_cast<std::uint32_t>(tissueTriangles.size()),
                    deviceToolTriangles,
                    static_cast<std::uint32_t>(toolTriangles.size()),
                    deviceConfig,
                    deviceGrid,
                    deviceCellTissueIds,
                    deviceCellToolIds,
                    deviceOverflowCount,
                    activeDeviceDenseGridStats);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridInsertTissueMilliseconds += insertMs;
            }
        }
        if (err == cudaSuccess && !config.batchTriangleInsert)
        {
            double insertTissueMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertTissueMs, [&]() {
                ScopedNvtxRange range("fused AABB/insert tissue", config.detailedProfiling);
                insertPackedTrianglesKernel<<<tissueBlocks, threadCount>>>(
                    deviceTissueTriangles,
                    static_cast<std::uint32_t>(tissueTriangles.size()),
                    true,
                    deviceConfig,
                    deviceGrid,
                    deviceCellTissueIds,
                    deviceOverflowCount,
                    activeDeviceDenseGridStats);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridInsertTissueMilliseconds += insertTissueMs;
            }
        }
        if (err == cudaSuccess && !config.batchTriangleInsert)
        {
            double insertToolMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertToolMs, [&]() {
                ScopedNvtxRange range("fused AABB/insert tool", config.detailedProfiling);
                insertPackedTrianglesKernel<<<toolBlocks, threadCount>>>(
                    deviceToolTriangles,
                    static_cast<std::uint32_t>(toolTriangles.size()),
                    false,
                    deviceConfig,
                    deviceGrid,
                    deviceCellToolIds,
                    deviceOverflowCount,
                    activeDeviceDenseGridStats);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridInsertToolMilliseconds += insertToolMs;
            }
        }
    }
    else
    {
        if (err == cudaSuccess)
        {
            double tissueAabbMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, tissueAabbMs, [&]() {
                ScopedNvtxRange range("AABB tissue", config.detailedProfiling);
                triangleAabbKernel<<<tissueBlocks, threadCount>>>(
                    deviceTissueTriangles,
                    static_cast<std::uint32_t>(tissueTriangles.size()),
                    config.contactDistance,
                    deviceTissueAabbs);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridTissueAabbMilliseconds += tissueAabbMs;
            }
        }
        if (err == cudaSuccess)
        {
            double toolAabbMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, toolAabbMs, [&]() {
                ScopedNvtxRange range("AABB tool", config.detailedProfiling);
                triangleAabbKernel<<<toolBlocks, threadCount>>>(
                    deviceToolTriangles,
                    static_cast<std::uint32_t>(toolTriangles.size()),
                    config.contactDistance,
                    deviceToolAabbs);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridToolAabbMilliseconds += toolAabbMs;
            }
        }
        if (err == cudaSuccess)
        {
            double insertTissueMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertTissueMs, [&]() {
                ScopedNvtxRange range("insert tissue", config.detailedProfiling);
                insertTissueTrianglesKernel<<<tissueBlocks, threadCount>>>(
                    deviceTissueAabbs,
                    static_cast<std::uint32_t>(tissueTriangles.size()),
                    deviceConfig,
                    deviceGrid,
                    deviceCellTissueIds,
                    deviceOverflowCount,
                    activeDeviceDenseGridStats);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridInsertTissueMilliseconds += insertTissueMs;
            }
        }
        if (err == cudaSuccess)
        {
            double insertToolMs = 0.0;
            err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertToolMs, [&]() {
                ScopedNvtxRange range("insert tool", config.detailedProfiling);
                insertToolTrianglesKernel<<<toolBlocks, threadCount>>>(
                    deviceToolAabbs,
                    static_cast<std::uint32_t>(toolTriangles.size()),
                    deviceConfig,
                    deviceGrid,
                    deviceCellToolIds,
                    deviceOverflowCount,
                    activeDeviceDenseGridStats);
                return cudaGetLastError();
            });
            denseGridLaunchCount += 1;
            if (executionStats != nullptr)
            {
                executionStats->denseGridInsertToolMilliseconds += insertToolMs;
            }
        }
    }
    if (err == cudaSuccess && config.compactActiveCells)
    {
        double compactActiveCellsMs = 0.0;
        const std::uint32_t compactBlocks = (cellCount + threadCount - 1u) / threadCount;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, compactActiveCellsMs, [&]() {
            ScopedNvtxRange range("compact active cells", config.detailedProfiling);
            compactActiveDenseGridCellsKernel<<<compactBlocks, threadCount>>>(
                deviceGrid,
                cellCount,
                deviceConfig,
                deviceActiveCellIds,
                deviceActiveCellCount,
                activeDeviceDenseGridStats);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridGeneratePairsMilliseconds += compactActiveCellsMs;
        }
    }
    if (err == cudaSuccess)
    {
        double generatePairsMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, generatePairsMs, [&]() {
            ScopedNvtxRange range("generate pairs", config.detailedProfiling);
            if (config.useGpuHashDedupe)
            {
                if (config.compactActiveCells)
                {
                    const std::uint32_t activeBlocks = std::min(cellCount, 1024u);
                    generateActiveDenseGridUniqueCandidatePairsKernel<<<activeBlocks, threadCount>>>(
                        deviceGrid,
                        deviceActiveCellIds,
                        deviceActiveCellCount,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceConfig,
                        deviceCandidatePairs,
                        devicePairHashKeys,
                        deviceRawCandidateCount,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
                else
                {
                    generateDenseGridUniqueCandidatePairsKernel<<<cellCount, threadCount>>>(
                        deviceGrid,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceConfig,
                        deviceCandidatePairs,
                        devicePairHashKeys,
                        deviceRawCandidateCount,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
            }
            else
            {
                if (config.compactActiveCells)
                {
                    const std::uint32_t activeBlocks = std::min(cellCount, 1024u);
                    generateActiveDenseGridCandidatePairsKernel<<<activeBlocks, threadCount>>>(
                        deviceGrid,
                        deviceActiveCellIds,
                        deviceActiveCellCount,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceTissueAabbs,
                        deviceToolAabbs,
                        deviceConfig,
                        deviceCandidatePairs,
                        deviceCandidateCount,
                        deviceOverflowCount);
                }
                else
                {
                    generateDenseGridCandidatePairsKernel<<<cellCount, threadCount>>>(
                        deviceGrid,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceTissueAabbs,
                        deviceToolAabbs,
                        deviceConfig,
                        deviceCandidatePairs,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
            }
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridGeneratePairsMilliseconds += generatePairsMs;
        }
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        destroyStageEvents();
        freeAll();
        return false;
    }

    std::uint32_t rawCandidateCount = 0;
    std::uint32_t exactCandidateCount = 0;
    std::uint32_t overflowCount = 0;
    DeviceDenseGridStats hostDenseGridStats {};
    const auto candidateReadbackStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange range("D2H candidate counters", config.detailedProfiling);
        if (readFullDenseGridCounters)
        {
            err = cudaMemcpy(
                &rawCandidateCount,
                config.useGpuHashDedupe ? deviceRawCandidateCount : deviceCandidateCount,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost);
            if (err == cudaSuccess && config.useGpuHashDedupe)
            {
                err = cudaMemcpy(&exactCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (err == cudaSuccess && executionStats != nullptr)
            {
                err = cudaMemcpy(&hostDenseGridStats, deviceDenseGridStats, sizeof(DeviceDenseGridStats), cudaMemcpyDeviceToHost);
            }
        }
        else if (readCandidateCounters)
        {
            err = cudaMemcpy(&exactCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (!config.useGpuHashDedupe)
            {
                rawCandidateCount = exactCandidateCount;
            }
        }
    }
    if (executionStats != nullptr)
    {
        if (readCandidateCounters)
        {
            executionStats->denseGridCandidateReadbackMilliseconds += elapsedMillisecondsSince(candidateReadbackStart);
        }
        const auto candidateReadbackBytes = readFullDenseGridCounters
            ? static_cast<std::uint64_t>((config.useGpuHashDedupe ? 3u : 2u) * sizeof(std::uint32_t) + sizeof(DeviceDenseGridStats))
            : (readCandidateCounters ? static_cast<std::uint64_t>(2u * sizeof(std::uint32_t)) : 0u);
        executionStats->deviceToHostBytes += candidateReadbackBytes;
        executionStats->activeMixedCellCount = hostDenseGridStats.activeMixedCellCount;
        executionStats->tissueInsertCount = hostDenseGridStats.tissueInsertCount;
        executionStats->toolInsertCount = hostDenseGridStats.toolInsertCount;
        executionStats->maxTissueCellOccupancy = hostDenseGridStats.maxTissueCellOccupancy;
        executionStats->maxToolCellOccupancy = hostDenseGridStats.maxToolCellOccupancy;
        executionStats->hashDedupeProbeOverflowCount = hostDenseGridStats.hashDedupeProbeOverflowCount;
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        destroyStageEvents();
        freeAll();
        return false;
    }

    const std::uint32_t usedCandidateCount = std::min(rawCandidateCount, config.maxCandidatePairs);
    if (executionStats != nullptr &&
        config.validateDedupeOnHost &&
        !config.useGpuHashDedupe &&
        usedCandidateCount > 0)
    {
        std::vector<std::uint64_t> hostCandidatePairs(usedCandidateCount);
        const auto validationStart = std::chrono::steady_clock::now();
        err = cudaMemcpy(
            hostCandidatePairs.data(),
            deviceCandidatePairs,
            static_cast<std::size_t>(usedCandidateCount) * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }

        std::sort(hostCandidatePairs.begin(), hostCandidatePairs.end());
        const auto hostUniqueEnd = std::unique(hostCandidatePairs.begin(), hostCandidatePairs.end());
        executionStats->hostValidatedUniqueCandidateCount =
            static_cast<std::uint32_t>(hostUniqueEnd - hostCandidatePairs.begin());
        executionStats->denseGridCandidateReadbackMilliseconds += elapsedMillisecondsSince(validationStart);
        executionStats->deviceToHostBytes +=
            static_cast<std::uint64_t>(usedCandidateCount) * sizeof(std::uint64_t);
    }
    if (!config.useGpuHashDedupe)
    {
        exactCandidateCount = usedCandidateCount;
    }
    else
    {
        exactCandidateCount = std::min(exactCandidateCount, config.maxCandidatePairs);
    }
    if (!config.useGpuHashDedupe && config.deduplicatePairs && usedCandidateCount > 1)
    {
        double sortUniqueMs = 0.0;
        const auto sortUniqueHostStart = std::chrono::steady_clock::now();
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, sortUniqueMs, [&]() {
            ScopedNvtxRange range("sort unique", config.detailedProfiling);
            thrust::device_ptr<std::uint64_t> begin(deviceCandidatePairs);
            thrust::device_ptr<std::uint64_t> end = begin + usedCandidateCount;
            thrust::sort(begin, end);
            cudaError_t thrustErr = cudaDeviceSynchronize();
            if (thrustErr != cudaSuccess)
            {
                return thrustErr;
            }
            const auto uniqueEnd = thrust::unique(begin, end);
            thrustErr = cudaDeviceSynchronize();
            if (thrustErr != cudaSuccess)
            {
                return thrustErr;
            }
            exactCandidateCount = static_cast<std::uint32_t>(uniqueEnd - begin);
            return cudaGetLastError();
        });
        const double sortUniqueHostMs = elapsedMillisecondsSince(sortUniqueHostStart);
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridSortUniqueMilliseconds += sortUniqueMs;
            executionStats->denseGridSortUniqueHostMilliseconds += sortUniqueHostMs;
        }
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }
    }

    if (err == cudaSuccess && computeDeviceContacts && exactCandidateCount > 0)
    {
        double exactContactMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, exactContactMs, [&]() {
            ScopedNvtxRange range("exact contacts", config.detailedProfiling);
            const auto contactBlocks = static_cast<std::uint32_t>((exactCandidateCount + threadCount - 1) / threadCount);
            exactDenseGridContactKernel<<<contactBlocks, threadCount>>>(
                deviceTissueTriangles,
                deviceToolTriangles,
                deviceCandidatePairs,
                exactCandidateCount,
                deviceContacts,
                config.maxCandidatePairs,
                deviceContactCount,
                deviceOverflowCount);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridExactContactMilliseconds += exactContactMs;
        }
    }

    if (!config.detailedProfiling && err == cudaSuccess)
    {
        err = cudaEventRecord(totalEnd);
        if (err == cudaSuccess)
        {
            err = cudaEventSynchronize(totalEnd);
        }
    }

    if (executionStats != nullptr)
    {
        if (config.detailedProfiling)
        {
            executionStats->gpuKernelMilliseconds +=
                executionStats->denseGridClearMilliseconds +
                executionStats->denseGridCounterClearMilliseconds +
                executionStats->denseGridTissueAabbMilliseconds +
                executionStats->denseGridToolAabbMilliseconds +
                executionStats->denseGridInsertTissueMilliseconds +
                executionStats->denseGridInsertToolMilliseconds +
                executionStats->denseGridGeneratePairsMilliseconds +
                executionStats->denseGridSortUniqueMilliseconds +
                executionStats->denseGridExactContactMilliseconds;
        }
        else if (err == cudaSuccess)
        {
            float elapsedMs = 0.0f;
            const cudaError_t eventErr = cudaEventElapsedTime(&elapsedMs, totalStart, totalEnd);
            if (eventErr == cudaSuccess)
            {
                executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            }
        }
        executionStats->kernelLaunchCount += denseGridLaunchCount;
        executionStats->rawCandidateCount = rawCandidateCount;
        executionStats->uniqueCandidateCount = exactCandidateCount;
        executionStats->outputPairCount = exactCandidateCount > 0 ? 1u : 0u;
        executionStats->outputCandidateCount = exactCandidateCount;
    }
    destroyStageEvents();

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    std::uint32_t hostContactCount = 0;
    const auto contactCountReadbackStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange range("D2H contact counters", config.detailedProfiling);
        if (readFullDenseGridCounters)
        {
            err = cudaMemcpy(&hostContactCount, deviceContactCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
        }
    }
    if (executionStats != nullptr && readFullDenseGridCounters)
    {
        executionStats->denseGridContactCountReadbackMilliseconds += elapsedMillisecondsSince(contactCountReadbackStart);
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->overflowCount = overflowCount;
        executionStats->outputContactCount = hostContactCount;
        if (readFullDenseGridCounters)
        {
            executionStats->deviceToHostBytes += static_cast<std::uint64_t>(2 * sizeof(std::uint32_t));
        }
    }

    if (overflowCount > 0 ||
        (readFullDenseGridCounters && hostDenseGridStats.hashDedupeProbeOverflowCount > 0) ||
        (!config.useGpuHashDedupe && rawCandidateCount > config.maxCandidatePairs) ||
        (readFullDenseGridCounters && hostContactCount > config.maxCandidatePairs))
    {
        diagnostic = "Dense-grid bucket, candidate, or contact capacity overflowed.";
        freeAll();
        return false;
    }

    if (!config.copyContactsToHost)
    {
        diagnostic.clear();
        return true;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(hostContactCount * sizeof(DeviceExactContact));
    }

    std::vector<DeviceExactContact> hostContacts(hostContactCount);
    if (hostContactCount > 0)
    {
        const auto contactDownloadStart = std::chrono::steady_clock::now();
        {
            ScopedNvtxRange range("D2H contacts", config.detailedProfiling);
            err = cudaMemcpy(
                hostContacts.data(),
                deviceContacts,
                hostContactCount * sizeof(DeviceExactContact),
                cudaMemcpyDeviceToHost);
        }
        if (executionStats != nullptr)
        {
            executionStats->denseGridContactDownloadMilliseconds += elapsedMillisecondsSince(contactDownloadStart);
        }
    }
    freeAll();

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    contacts.reserve(hostContacts.size());
    for (const auto& contact : hostContacts)
    {
        contacts.push_back(ExactContact {
            contact.firstTriangleIndex,
            contact.secondTriangleIndex,
            TriangleVertex { contact.pointOnFirst.x, contact.pointOnFirst.y, contact.pointOnFirst.z },
            TriangleVertex { contact.pointOnSecond.x, contact.pointOnSecond.y, contact.pointOnSecond.z },
            TriangleVertex { contact.normal.x, contact.normal.y, contact.normal.z },
            contact.signedDistance,
        });
    }

    diagnostic.clear();
    return true;
}

bool computeDenseGridIndexedTriangleContacts(
    const TriangleIndexedSurface& tissueSurface,
    const TriangleIndexedSurface& toolSurface,
    const DenseGridConfig& config,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = tissueSurface.triangleCount + toolSurface.triangleCount;
    }

    const auto surfaceIsInvalid = [](const TriangleIndexedSurface& surface) {
        return (surface.positions == nullptr && surface.devicePositions == nullptr) ||
            surface.triangleIndices == nullptr ||
            surface.vertexCount == 0 ||
            surface.triangleCount == 0;
    };
    if (surfaceIsInvalid(tissueSurface) || surfaceIsInvalid(toolSurface))
    {
        diagnostic = "Invalid indexed triangle surface.";
        return false;
    }

    if (config.canonicalPairEmission && !config.useGpuHashDedupe)
    {
        diagnostic = "Indexed dense-grid input does not support canonical pair emission without GPU hash dedupe.";
        return false;
    }

    // Phase 15: tool-active-cell generation builds the active list during the
    // tool insert, which relies on tissue being inserted before tool. Fused
    // insertion (batchTriangleInsert) breaks that ordering.
    if (config.useToolActiveCellGeneration && config.batchTriangleInsert)
    {
        diagnostic = "useToolActiveCellGeneration is incompatible with batchTriangleInsert (the active list relies on tissue-before-tool insert ordering).";
        return false;
    }

    if (config.gridResolutionX == 0 || config.gridResolutionY == 0 || config.gridResolutionZ == 0 ||
        config.maxTissueTrianglesPerCell == 0 || config.maxToolTrianglesPerCell == 0 ||
        config.maxCandidatePairs == 0 ||
        config.gridMaxX <= config.gridMinX ||
        config.gridMaxY <= config.gridMinY ||
        config.gridMaxZ <= config.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration.";
        return false;
    }

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(config.gridResolutionX) *
        static_cast<std::uint64_t>(config.gridResolutionY) *
        static_cast<std::uint64_t>(config.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid cell count exceeds uint32_t indexing capacity.";
        return false;
    }
    const auto cellCount = static_cast<std::uint32_t>(cellCount64);

    const std::uint64_t tissueBucketCount =
        cellCount64 * static_cast<std::uint64_t>(config.maxTissueTrianglesPerCell);
    const std::uint64_t toolBucketCount =
        cellCount64 * static_cast<std::uint64_t>(config.maxToolTrianglesPerCell);
    const std::uint64_t pairHashCount64 = config.useGpuHashDedupe
        ? static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(config.maxCandidatePairs) * 2u))
        : 1ull;
    if (tissueBucketCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) ||
        toolBucketCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)))
    {
        diagnostic = "Dense-grid bucket allocation request is too large.";
        return false;
    }
    if (pairHashCount64 == 0 ||
        pairHashCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid hash dedupe table request exceeds uint32_t indexing capacity.";
        return false;
    }

    auto& workspace = denseGridWorkspace();
    std::uint64_t newlyAllocatedBytes = 0;
    const auto deviceAllocationStart = std::chrono::steady_clock::now();
    cudaError_t err = workspace.ensure(
        0,
        0,
        static_cast<std::size_t>(cellCount64),
        static_cast<std::size_t>(tissueBucketCount),
        static_cast<std::size_t>(toolBucketCount),
        static_cast<std::size_t>(config.maxCandidatePairs),
        static_cast<std::size_t>(pairHashCount64),
        newlyAllocatedBytes);
    if (err == cudaSuccess)
    {
        err = workspace.ensureIndexedInput(
            tissueSurface.devicePositions == nullptr ? tissueSurface.vertexCount : 0u,
            toolSurface.devicePositions == nullptr ? toolSurface.vertexCount : 0u,
            static_cast<std::size_t>(tissueSurface.triangleCount) * 3u,
            static_cast<std::size_t>(toolSurface.triangleCount) * 3u,
            newlyAllocatedBytes);
    }
    const double deviceAllocationMs = elapsedMillisecondsSince(deviceAllocationStart);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = cellCount;
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += deviceAllocationMs;
        executionStats->workspaceResizeCount += newlyAllocatedBytes > 0 ? 1u : 0u;
    }

    const BackendTriangleVertex* deviceTissuePositions =
        tissueSurface.devicePositions == nullptr ? workspace.indexedTissuePositions : tissueSurface.devicePositions;
    const BackendTriangleVertex* deviceToolPositions =
        toolSurface.devicePositions == nullptr ? workspace.indexedToolPositions : toolSurface.devicePositions;
    std::uint32_t* deviceTissueTriangleIndices = workspace.indexedTissueIndices;
    std::uint32_t* deviceToolTriangleIndices = workspace.indexedToolIndices;
    DeviceCellBucket* deviceGrid = workspace.grid;
    std::uint32_t* deviceActiveCellIds = workspace.activeCellIds;
    std::uint32_t* deviceCellTissueIds = workspace.cellTissueIds;
    std::uint32_t* deviceCellToolIds = workspace.cellToolIds;
    std::uint64_t* deviceCandidatePairs = workspace.candidatePairs;
    unsigned long long* devicePairHashKeys = workspace.pairHashKeys;
    DeviceExactContact* deviceContacts = workspace.contacts;
    std::uint32_t* deviceRawCandidateCount = workspace.rawCandidateCount;
    std::uint32_t* deviceCandidateCount = workspace.candidateCount;
    std::uint32_t* deviceContactCount = workspace.contactCount;
    std::uint32_t* deviceOverflowCount = workspace.overflowCount;
    std::uint32_t* deviceActiveCellCount = workspace.activeCellCount;
    DeviceDenseGridStats* deviceDenseGridStats = workspace.denseGridStats;
    const bool readFullDenseGridCounters =
        config.copyContactsToHost ||
        config.readCountersWhenContactsStayOnDevice ||
        config.detailedProfiling;
    const bool computeDeviceContacts =
        config.copyContactsToHost ||
        config.computeDeviceContactsWhenContactsStayOnDevice;
    const bool readCandidateCounters =
        readFullDenseGridCounters ||
        computeDeviceContacts;
    DeviceDenseGridStats* activeDeviceDenseGridStats =
        readFullDenseGridCounters ? deviceDenseGridStats : nullptr;
    auto freeAll = []() {};

    const std::size_t tissuePositionBytes =
        static_cast<std::size_t>(tissueSurface.vertexCount) * sizeof(BackendTriangleVertex);
    const std::size_t toolPositionBytes =
        static_cast<std::size_t>(toolSurface.vertexCount) * sizeof(BackendTriangleVertex);
    const std::size_t tissueIndexBytes =
        static_cast<std::size_t>(tissueSurface.triangleCount) * 3u * sizeof(std::uint32_t);
    const std::size_t toolIndexBytes =
        static_cast<std::size_t>(toolSurface.triangleCount) * 3u * sizeof(std::uint32_t);
    const bool uploadTissueTopology =
        workspace.indexedTissueSurfaceId != tissueSurface.surfaceId ||
        workspace.indexedTissueTopologyVersion != tissueSurface.topologyVersion;
    const bool uploadToolTopology =
        workspace.indexedToolSurfaceId != toolSurface.surfaceId ||
        workspace.indexedToolTopologyVersion != toolSurface.topologyVersion;
    constexpr std::size_t kPinnedIndexedStagingMinimumBytes = 1u << 20;
    const std::size_t indexedUploadBytes =
        (tissueSurface.devicePositions == nullptr ? tissuePositionBytes : 0u) +
        (toolSurface.devicePositions == nullptr ? toolPositionBytes : 0u) +
        (uploadTissueTopology ? tissueIndexBytes : 0u) +
        (uploadToolTopology ? toolIndexBytes : 0u);
    bool usingPinnedIndexedHostStaging = false;
    if (config.usePinnedHostStaging && indexedUploadBytes >= kPinnedIndexedStagingMinimumBytes)
    {
        std::uint64_t newlyPinnedHostBytes = 0;
        const cudaError_t pinnedErr = workspace.ensureIndexedPinnedHostStaging(
            tissueSurface.devicePositions == nullptr ? tissueSurface.vertexCount : 0u,
            toolSurface.devicePositions == nullptr ? toolSurface.vertexCount : 0u,
            uploadTissueTopology ? static_cast<std::size_t>(tissueSurface.triangleCount) * 3u : 0u,
            uploadToolTopology ? static_cast<std::size_t>(toolSurface.triangleCount) * 3u : 0u,
            newlyPinnedHostBytes);
        usingPinnedIndexedHostStaging = pinnedErr == cudaSuccess;
    }

    const auto hostToDeviceStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange h2dRange("H2D indexed surface upload", config.detailedProfiling);
        if (tissueSurface.devicePositions == nullptr)
        {
            err = copyHostArrayToDeviceAsync(
                workspace.indexedTissuePositions,
                tissueSurface.positions,
                workspace.pinnedIndexedTissuePositions,
                static_cast<std::size_t>(tissueSurface.vertexCount),
                usingPinnedIndexedHostStaging);
        }
        if (err == cudaSuccess && toolSurface.devicePositions == nullptr)
        {
            err = copyHostArrayToDeviceAsync(
                workspace.indexedToolPositions,
                toolSurface.positions,
                workspace.pinnedIndexedToolPositions,
                static_cast<std::size_t>(toolSurface.vertexCount),
                usingPinnedIndexedHostStaging);
        }
        if (err == cudaSuccess && uploadTissueTopology)
        {
            err = copyHostArrayToDeviceAsync(
                deviceTissueTriangleIndices,
                tissueSurface.triangleIndices,
                workspace.pinnedIndexedTissueIndices,
                static_cast<std::size_t>(tissueSurface.triangleCount) * 3u,
                usingPinnedIndexedHostStaging);
        }
        if (err == cudaSuccess && uploadToolTopology)
        {
            err = copyHostArrayToDeviceAsync(
                deviceToolTriangleIndices,
                toolSurface.triangleIndices,
                workspace.pinnedIndexedToolIndices,
                static_cast<std::size_t>(toolSurface.triangleCount) * 3u,
                usingPinnedIndexedHostStaging);
        }
    }
    if (err == cudaSuccess && uploadTissueTopology)
    {
        workspace.indexedTissueSurfaceId = tissueSurface.surfaceId;
        workspace.indexedTissueTopologyVersion = tissueSurface.topologyVersion;
    }
    if (err == cudaSuccess && uploadToolTopology)
    {
        workspace.indexedToolSurfaceId = toolSurface.surfaceId;
        workspace.indexedToolTopologyVersion = toolSurface.topologyVersion;
    }
    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceMilliseconds += elapsedMillisecondsSince(hostToDeviceStart);
        if (tissueSurface.devicePositions == nullptr)
        {
            executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(tissuePositionBytes);
        }
        if (toolSurface.devicePositions == nullptr)
        {
            executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(toolPositionBytes);
        }
        if (uploadTissueTopology)
        {
            executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(tissueIndexBytes);
        }
        if (uploadToolTopology)
        {
            executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(toolIndexBytes);
        }
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    const DeviceDenseGridConfig deviceConfig {
        make_float3(config.gridMinX, config.gridMinY, config.gridMinZ),
        make_float3(config.gridMaxX, config.gridMaxY, config.gridMaxZ),
        make_float3(
            static_cast<float>(config.gridResolutionX) / (config.gridMaxX - config.gridMinX),
            static_cast<float>(config.gridResolutionY) / (config.gridMaxY - config.gridMinY),
            static_cast<float>(config.gridResolutionZ) / (config.gridMaxZ - config.gridMinZ)),
        config.gridResolutionX,
        config.gridResolutionY,
        config.gridResolutionZ,
        config.contactDistance,
        config.maxTissueTrianglesPerCell,
        config.maxToolTrianglesPerCell,
        config.maxCandidatePairs,
        static_cast<std::uint32_t>(pairHashCount64),
        config.useGpuHashDedupe,
        config.canonicalPairEmission
    };

    constexpr std::uint32_t threadCount = 256;
    const auto tissueBlocks =
        static_cast<std::uint32_t>((tissueSurface.triangleCount + threadCount - 1) / threadCount);
    const auto toolBlocks =
        static_cast<std::uint32_t>((toolSurface.triangleCount + threadCount - 1) / threadCount);

    // Phase 16: reuse workspace-owned events instead of creating/destroying
    // four CUDA events every frame. The aliases below keep the rest of this
    // function unchanged; destroyStageEvents is now a no-op because the events
    // live for the plugin's lifetime (freed in the workspace destructor).
    err = workspace.ensureBroadEvents();
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }
    cudaEvent_t& stageStart = workspace.broadStageStart;
    cudaEvent_t& stageEnd = workspace.broadStageEnd;
    cudaEvent_t& totalStart = workspace.broadTotalStart;
    cudaEvent_t& totalEnd = workspace.broadTotalEnd;

    std::uint32_t denseGridLaunchCount = 0;
    auto destroyStageEvents = [&]() { /* workspace-owned; freed at plugin unload */ };

    if (!config.detailedProfiling)
    {
        err = cudaEventRecord(totalStart);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }
    }

    double resetGridMs = 0.0;
    err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, resetGridMs, [&]() {
        ScopedNvtxRange range("reset indexed grid/counters", config.detailedProfiling);
        const std::uint32_t resetItems = cellCount;
        const std::uint32_t resetBlocks = (resetItems + threadCount - 1u) / threadCount;
        resetDenseGridKernel<<<resetBlocks, threadCount>>>(
            deviceGrid,
            cellCount,
            devicePairHashKeys,
            static_cast<std::uint32_t>(pairHashCount64),
            false,
            deviceActiveCellCount,
            deviceRawCandidateCount,
            deviceCandidateCount,
            deviceContactCount,
            deviceOverflowCount,
            deviceDenseGridStats,
            readFullDenseGridCounters);
        return cudaGetLastError();
    });
    denseGridLaunchCount += 1;
    if (executionStats != nullptr)
    {
        executionStats->denseGridClearMilliseconds += resetGridMs;
    }
    if (err == cudaSuccess && config.useGpuHashDedupe)
    {
        double hashClearMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, hashClearMs, [&]() {
            ScopedNvtxRange range("clear indexed pair hash", config.detailedProfiling);
            return cudaMemset(
                devicePairHashKeys,
                0xff,
                static_cast<std::size_t>(pairHashCount64) * sizeof(std::uint64_t));
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridCounterClearMilliseconds += hashClearMs;
            executionStats->cudaMemsetCount += 1;
        }
    }

    if (err == cudaSuccess && config.batchTriangleInsert)
    {
        double insertMs = 0.0;
        const auto combinedBlocks = static_cast<std::uint32_t>(
            ((tissueSurface.triangleCount + toolSurface.triangleCount) + threadCount - 1) / threadCount);
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertMs, [&]() {
            ScopedNvtxRange range("indexed AABB/insert tissue+tool", config.detailedProfiling);
            insertIndexedTrianglePairKernel<<<combinedBlocks, threadCount>>>(
                deviceTissuePositions,
                deviceTissueTriangleIndices,
                tissueSurface.triangleCount,
                deviceToolPositions,
                deviceToolTriangleIndices,
                toolSurface.triangleCount,
                deviceConfig,
                deviceGrid,
                deviceCellTissueIds,
                deviceCellToolIds,
                deviceOverflowCount,
                activeDeviceDenseGridStats);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridInsertTissueMilliseconds += insertMs;
        }
    }
    if (err == cudaSuccess && !config.batchTriangleInsert)
    {
        double insertTissueMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertTissueMs, [&]() {
            ScopedNvtxRange range("indexed AABB/insert tissue", config.detailedProfiling);
            insertIndexedTrianglesKernel<<<tissueBlocks, threadCount>>>(
                deviceTissuePositions,
                deviceTissueTriangleIndices,
                tissueSurface.triangleCount,
                true,
                deviceConfig,
                deviceGrid,
                deviceCellTissueIds,
                deviceOverflowCount,
                activeDeviceDenseGridStats);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridInsertTissueMilliseconds += insertTissueMs;
        }
    }
    if (err == cudaSuccess && !config.batchTriangleInsert)
    {
        double insertToolMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, insertToolMs, [&]() {
            ScopedNvtxRange range("indexed AABB/insert tool", config.detailedProfiling);
            insertIndexedTrianglesKernel<<<toolBlocks, threadCount>>>(
                deviceToolPositions,
                deviceToolTriangleIndices,
                toolSurface.triangleCount,
                false,
                deviceConfig,
                deviceGrid,
                deviceCellToolIds,
                deviceOverflowCount,
                activeDeviceDenseGridStats,
                config.useToolActiveCellGeneration,                                  // buildToolActiveList
                config.useToolActiveCellGeneration ? deviceActiveCellIds : nullptr,
                config.useToolActiveCellGeneration ? deviceActiveCellCount : nullptr);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridInsertToolMilliseconds += insertToolMs;
        }
    }
    // Phase 9 compaction: separate full-grid scan (regressed; off by default).
    // Phase 15 (useToolActiveCellGeneration) builds the active list during the
    // tool insert above, so it never runs this scan.
    if (err == cudaSuccess && config.compactActiveCells && !config.useToolActiveCellGeneration)
    {
        double compactActiveCellsMs = 0.0;
        const std::uint32_t compactBlocks = (cellCount + threadCount - 1u) / threadCount;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, compactActiveCellsMs, [&]() {
            ScopedNvtxRange range("indexed compact active cells", config.detailedProfiling);
            compactActiveDenseGridCellsKernel<<<compactBlocks, threadCount>>>(
                deviceGrid,
                cellCount,
                deviceConfig,
                deviceActiveCellIds,
                deviceActiveCellCount,
                activeDeviceDenseGridStats);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridGeneratePairsMilliseconds += compactActiveCellsMs;
        }
    }
    if (err == cudaSuccess)
    {
        // Phase 15: when useToolActiveCellGeneration is set, generate over the
        // tool-built active-cell list (a small fixed grid that device-reads the
        // count and grid-strides) instead of one block per grid cell.
        const bool useActiveGeneration = config.compactActiveCells || config.useToolActiveCellGeneration;
        const std::uint32_t activeBlocks = std::min(cellCount, 1024u);
        double generatePairsMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, generatePairsMs, [&]() {
            ScopedNvtxRange range("indexed generate pairs", config.detailedProfiling);
            if (config.useGpuHashDedupe)
            {
                if (useActiveGeneration)
                {
                    generateActiveDenseGridUniqueCandidatePairsKernel<<<activeBlocks, threadCount>>>(
                        deviceGrid,
                        deviceActiveCellIds,
                        deviceActiveCellCount,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceConfig,
                        deviceCandidatePairs,
                        devicePairHashKeys,
                        deviceRawCandidateCount,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
                else
                {
                    generateDenseGridUniqueCandidatePairsKernel<<<cellCount, threadCount>>>(
                        deviceGrid,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        deviceConfig,
                        deviceCandidatePairs,
                        devicePairHashKeys,
                        deviceRawCandidateCount,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
            }
            else
            {
                if (useActiveGeneration)
                {
                    generateActiveDenseGridCandidatePairsKernel<<<activeBlocks, threadCount>>>(
                        deviceGrid,
                        deviceActiveCellIds,
                        deviceActiveCellCount,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        nullptr,
                        nullptr,
                        deviceConfig,
                        deviceCandidatePairs,
                        deviceCandidateCount,
                        deviceOverflowCount);
                }
                else
                {
                    generateDenseGridCandidatePairsKernel<<<cellCount, threadCount>>>(
                        deviceGrid,
                        deviceCellTissueIds,
                        deviceCellToolIds,
                        nullptr,
                        nullptr,
                        deviceConfig,
                        deviceCandidatePairs,
                        deviceCandidateCount,
                        deviceOverflowCount,
                        activeDeviceDenseGridStats);
                }
            }
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridGeneratePairsMilliseconds += generatePairsMs;
        }
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        destroyStageEvents();
        freeAll();
        return false;
    }

    std::uint32_t rawCandidateCount = 0;
    std::uint32_t exactCandidateCount = 0;
    std::uint32_t overflowCount = 0;
    DeviceDenseGridStats hostDenseGridStats {};
    const auto candidateReadbackStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange range("D2H indexed candidate counters", config.detailedProfiling);
        if (readFullDenseGridCounters)
        {
            err = cudaMemcpy(
                &rawCandidateCount,
                config.useGpuHashDedupe ? deviceRawCandidateCount : deviceCandidateCount,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost);
            if (err == cudaSuccess && config.useGpuHashDedupe)
            {
                err = cudaMemcpy(&exactCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (err == cudaSuccess && executionStats != nullptr)
            {
                err = cudaMemcpy(&hostDenseGridStats, deviceDenseGridStats, sizeof(DeviceDenseGridStats), cudaMemcpyDeviceToHost);
            }
        }
        else if (readCandidateCounters)
        {
            err = cudaMemcpy(&exactCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
            if (!config.useGpuHashDedupe)
            {
                rawCandidateCount = exactCandidateCount;
            }
        }
    }
    if (executionStats != nullptr)
    {
        if (readCandidateCounters)
        {
            executionStats->denseGridCandidateReadbackMilliseconds += elapsedMillisecondsSince(candidateReadbackStart);
        }
        const auto candidateReadbackBytes = readFullDenseGridCounters
            ? static_cast<std::uint64_t>((config.useGpuHashDedupe ? 3u : 2u) * sizeof(std::uint32_t) + sizeof(DeviceDenseGridStats))
            : (readCandidateCounters ? static_cast<std::uint64_t>(2u * sizeof(std::uint32_t)) : 0u);
        executionStats->deviceToHostBytes += candidateReadbackBytes;
        executionStats->activeMixedCellCount = hostDenseGridStats.activeMixedCellCount;
        executionStats->tissueInsertCount = hostDenseGridStats.tissueInsertCount;
        executionStats->toolInsertCount = hostDenseGridStats.toolInsertCount;
        executionStats->maxTissueCellOccupancy = hostDenseGridStats.maxTissueCellOccupancy;
        executionStats->maxToolCellOccupancy = hostDenseGridStats.maxToolCellOccupancy;
        executionStats->hashDedupeProbeOverflowCount = hostDenseGridStats.hashDedupeProbeOverflowCount;
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        destroyStageEvents();
        freeAll();
        return false;
    }

    const std::uint32_t usedCandidateCount = std::min(rawCandidateCount, config.maxCandidatePairs);
    if (executionStats != nullptr &&
        config.validateDedupeOnHost &&
        !config.useGpuHashDedupe &&
        usedCandidateCount > 0)
    {
        std::vector<std::uint64_t> hostCandidatePairs(usedCandidateCount);
        const auto validationStart = std::chrono::steady_clock::now();
        err = cudaMemcpy(
            hostCandidatePairs.data(),
            deviceCandidatePairs,
            static_cast<std::size_t>(usedCandidateCount) * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }

        std::sort(hostCandidatePairs.begin(), hostCandidatePairs.end());
        const auto hostUniqueEnd = std::unique(hostCandidatePairs.begin(), hostCandidatePairs.end());
        executionStats->hostValidatedUniqueCandidateCount =
            static_cast<std::uint32_t>(hostUniqueEnd - hostCandidatePairs.begin());
        executionStats->denseGridCandidateReadbackMilliseconds += elapsedMillisecondsSince(validationStart);
        executionStats->deviceToHostBytes +=
            static_cast<std::uint64_t>(usedCandidateCount) * sizeof(std::uint64_t);
    }
    if (!config.useGpuHashDedupe)
    {
        exactCandidateCount = usedCandidateCount;
    }
    else
    {
        exactCandidateCount = std::min(exactCandidateCount, config.maxCandidatePairs);
    }
    if (!config.useGpuHashDedupe && config.deduplicatePairs && usedCandidateCount > 1)
    {
        double sortUniqueMs = 0.0;
        const auto sortUniqueHostStart = std::chrono::steady_clock::now();
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, sortUniqueMs, [&]() {
            ScopedNvtxRange range("indexed sort unique", config.detailedProfiling);
            thrust::device_ptr<std::uint64_t> begin(deviceCandidatePairs);
            thrust::device_ptr<std::uint64_t> end = begin + usedCandidateCount;
            thrust::sort(begin, end);
            cudaError_t thrustErr = cudaDeviceSynchronize();
            if (thrustErr != cudaSuccess)
            {
                return thrustErr;
            }
            const auto uniqueEnd = thrust::unique(begin, end);
            thrustErr = cudaDeviceSynchronize();
            if (thrustErr != cudaSuccess)
            {
                return thrustErr;
            }
            exactCandidateCount = static_cast<std::uint32_t>(uniqueEnd - begin);
            return cudaGetLastError();
        });
        const double sortUniqueHostMs = elapsedMillisecondsSince(sortUniqueHostStart);
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridSortUniqueMilliseconds += sortUniqueMs;
            executionStats->denseGridSortUniqueHostMilliseconds += sortUniqueHostMs;
        }
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            destroyStageEvents();
            freeAll();
            return false;
        }
    }

    if (err == cudaSuccess && computeDeviceContacts && exactCandidateCount > 0)
    {
        double exactContactMs = 0.0;
        err = runCudaOperation(config.detailedProfiling, stageStart, stageEnd, exactContactMs, [&]() {
            ScopedNvtxRange range("indexed exact contacts", config.detailedProfiling);
            const auto contactBlocks = static_cast<std::uint32_t>((exactCandidateCount + threadCount - 1) / threadCount);
            exactDenseGridIndexedContactKernel<<<contactBlocks, threadCount>>>(
                deviceTissuePositions,
                deviceTissueTriangleIndices,
                deviceToolPositions,
                deviceToolTriangleIndices,
                deviceCandidatePairs,
                exactCandidateCount,
                deviceContacts,
                config.maxCandidatePairs,
                deviceContactCount,
                deviceOverflowCount);
            return cudaGetLastError();
        });
        denseGridLaunchCount += 1;
        if (executionStats != nullptr)
        {
            executionStats->denseGridExactContactMilliseconds += exactContactMs;
        }
    }

    if (!config.detailedProfiling && err == cudaSuccess)
    {
        err = cudaEventRecord(totalEnd);
        if (err == cudaSuccess)
        {
            err = cudaEventSynchronize(totalEnd);
        }
    }

    if (executionStats != nullptr)
    {
        if (config.detailedProfiling)
        {
            executionStats->gpuKernelMilliseconds +=
                executionStats->denseGridClearMilliseconds +
                executionStats->denseGridCounterClearMilliseconds +
                executionStats->denseGridTissueAabbMilliseconds +
                executionStats->denseGridToolAabbMilliseconds +
                executionStats->denseGridInsertTissueMilliseconds +
                executionStats->denseGridInsertToolMilliseconds +
                executionStats->denseGridGeneratePairsMilliseconds +
                executionStats->denseGridSortUniqueMilliseconds +
                executionStats->denseGridExactContactMilliseconds;
        }
        else if (err == cudaSuccess)
        {
            float elapsedMs = 0.0f;
            const cudaError_t eventErr = cudaEventElapsedTime(&elapsedMs, totalStart, totalEnd);
            if (eventErr == cudaSuccess)
            {
                executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            }
        }
        executionStats->kernelLaunchCount += denseGridLaunchCount;
        executionStats->rawCandidateCount = rawCandidateCount;
        executionStats->uniqueCandidateCount = exactCandidateCount;
        executionStats->outputPairCount = exactCandidateCount > 0 ? 1u : 0u;
        executionStats->outputCandidateCount = exactCandidateCount;
    }
    destroyStageEvents();

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    std::uint32_t hostContactCount = 0;
    const auto contactCountReadbackStart = std::chrono::steady_clock::now();
    {
        ScopedNvtxRange range("D2H indexed contact counters", config.detailedProfiling);
        if (readFullDenseGridCounters)
        {
            err = cudaMemcpy(&hostContactCount, deviceContactCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(&overflowCount, deviceOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
            }
        }
    }
    if (executionStats != nullptr && readFullDenseGridCounters)
    {
        executionStats->denseGridContactCountReadbackMilliseconds += elapsedMillisecondsSince(contactCountReadbackStart);
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        freeAll();
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->overflowCount = overflowCount;
        executionStats->outputContactCount = hostContactCount;
        if (readFullDenseGridCounters)
        {
            executionStats->deviceToHostBytes += static_cast<std::uint64_t>(2 * sizeof(std::uint32_t));
        }
    }

    if (overflowCount > 0 ||
        (readFullDenseGridCounters && hostDenseGridStats.hashDedupeProbeOverflowCount > 0) ||
        (!config.useGpuHashDedupe && rawCandidateCount > config.maxCandidatePairs) ||
        (readFullDenseGridCounters && hostContactCount > config.maxCandidatePairs))
    {
        diagnostic = "Dense-grid bucket, candidate, or contact capacity overflowed.";
        freeAll();
        return false;
    }

    if (!config.copyContactsToHost)
    {
        diagnostic.clear();
        return true;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(hostContactCount * sizeof(DeviceExactContact));
    }

    std::vector<DeviceExactContact> hostContacts(hostContactCount);
    if (hostContactCount > 0)
    {
        const auto contactDownloadStart = std::chrono::steady_clock::now();
        {
            ScopedNvtxRange range("D2H indexed contacts", config.detailedProfiling);
            err = cudaMemcpy(
                hostContacts.data(),
                deviceContacts,
                hostContactCount * sizeof(DeviceExactContact),
                cudaMemcpyDeviceToHost);
        }
        if (executionStats != nullptr)
        {
            executionStats->denseGridContactDownloadMilliseconds += elapsedMillisecondsSince(contactDownloadStart);
        }
    }
    freeAll();

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    contacts.reserve(hostContacts.size());
    for (const auto& contact : hostContacts)
    {
        contacts.push_back(ExactContact {
            contact.firstTriangleIndex,
            contact.secondTriangleIndex,
            TriangleVertex { contact.pointOnFirst.x, contact.pointOnFirst.y, contact.pointOnFirst.z },
            TriangleVertex { contact.pointOnSecond.x, contact.pointOnSecond.y, contact.pointOnSecond.z },
            TriangleVertex { contact.normal.x, contact.normal.y, contact.normal.z },
            contact.signedDistance,
        });
    }

    diagnostic.clear();
    return true;
}

// ============================================================================
// computeFeatureBasedProximityContacts
// ----------------------------------------------------------------------------
// Strategy:
//   1. Run the existing indexed dense-grid path with a tweaked config that
//      forces broad-cull-only behavior. This leaves the unique candidate-pair
//      list on the device in workspace.candidatePairs / workspace.candidateCount
//      and skips the SAT-style exact-contact kernel. We pay one D2H readback
//      of the candidate count via readCountersWhenContactsStayOnDevice=true so
//      that we can pick a sensible grid for the FBP kernel below.
//   2. Allocate / reuse a device buffer for ProximityContacts and reset the
//      proximity counters.
//   3. Launch featureBasedProximityKernel over the candidate pairs.
//   4. Optionally read back the proximity contact count and copy contacts to
//      host according to the FeatureBasedProximityConfig flags.
// ============================================================================


} // namespace SofaGpuCollision::backend
