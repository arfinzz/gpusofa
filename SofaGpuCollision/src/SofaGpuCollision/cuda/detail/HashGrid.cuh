// HashGrid.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Optimised spatial-hash broad cull (way 3): workspace (shared with the
// simple hash), mark/compact/fill kernels, tracked pair-dedup inserts,
// mixed-bucket generators, and the hash host driver.

namespace
{

struct HashGridWorkspace
{
    std::uint32_t* cellKeys { nullptr };
    std::uint32_t* slotBucketIds { nullptr };
    std::uint32_t* bucketCellIds { nullptr };
    std::uint32_t* tissueCount { nullptr };
    std::uint32_t* toolCount { nullptr };
    std::uint32_t* tissueIds { nullptr };
    std::uint32_t* toolIds { nullptr };
    std::uint32_t* mixedBucketIds { nullptr };
    std::uint32_t* pairsPerBucket { nullptr };
    std::uint64_t* candidatePairs { nullptr };
    std::uint32_t* compactCandidatePairs { nullptr };
    unsigned long long* pairHashKeys { nullptr };
    std::uint32_t* compactPairHashKeys { nullptr };
    std::uint32_t* touchedPairHashSlots { nullptr };
    std::uint32_t* candidateCount { nullptr };
    std::uint32_t* rawCandidateCount { nullptr };
    std::uint32_t* overflowCount { nullptr };
    std::uint32_t* probeOverflowCount { nullptr };
    std::uint32_t* occupiedBucketCount { nullptr };
    std::uint32_t* mixedBucketCount { nullptr };
    std::uint32_t* touchedPairHashCount { nullptr };
    void* proximityContacts { nullptr };
    std::uint32_t* proximityContactCount { nullptr };
    std::uint32_t* proximityOverflowCount { nullptr };
    std::uint32_t* proximityVfCount { nullptr };
    std::uint32_t* proximityFvCount { nullptr };
    std::uint32_t* proximityEeCount { nullptr };
    std::uint32_t* firstIndices { nullptr };
    std::uint32_t* secondIndices { nullptr };
    BackendTriangleVertex* firstPositions { nullptr };   // used only for host-input fallback
    BackendTriangleVertex* secondPositions { nullptr };
    std::uint32_t* countersHostPinned { nullptr };  // 10 uint32, pinned

    std::size_t cellKeysCapacity { 0 };
    std::size_t slotBucketIdCapacity { 0 };
    std::size_t bucketCellIdCapacity { 0 };
    std::size_t tissueCountCapacity { 0 };
    std::size_t toolCountCapacity { 0 };
    std::size_t mixedBucketIdCapacity { 0 };
    std::size_t pairsPerBucketCapacity { 0 };
    std::size_t tissueIdCapacity { 0 };
    std::size_t toolIdCapacity { 0 };
    std::size_t candidateCapacity { 0 };
    std::size_t compactCandidateCapacity { 0 };
    std::size_t pairHashCapacity { 0 };
    std::size_t compactPairHashCapacity { 0 };
    std::size_t touchedPairHashSlotCapacity { 0 };
    std::size_t contactCapacity { 0 };
    std::size_t firstIndexCapacity { 0 };
    std::size_t secondIndexCapacity { 0 };
    std::size_t firstPositionCapacity { 0 };
    std::size_t secondPositionCapacity { 0 };
    std::uint64_t firstSurfaceId { 0 };
    std::uint64_t secondSurfaceId { 0 };
    std::uint64_t firstTopologyVersion { ~0ull };
    std::uint64_t secondTopologyVersion { ~0ull };
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t endEvent { nullptr };
    bool eventsReady { false };
    bool pairHashKeysInitialized { false };
    bool compactPairHashKeysInitialized { false };
    bool useCompactCandidatePairs { false };
    // CUDA Graph state (Optimization 2026-06-17c). The per-frame kernel sequence
    // is captured once into graphExec and replayed each steady-state frame to cut
    // per-launch CPU overhead (the broad cull is now ~11 sub-10us kernels).
    cudaStream_t captureStream { nullptr };
    cudaGraphExec_t graphExec { nullptr };
    bool graphInstantiated { false };
    // Capture signature: if any of these change, the graph is rebuilt.
    std::uint32_t gSigTableSize { 0 };
    std::uint32_t gSigBucketCap { 0 };
    std::uint32_t gSigFirstTri { ~0u };
    std::uint32_t gSigSecondTri { ~0u };
    std::uint32_t gSigMaxContacts { 0 };
    std::uint32_t gSigPairHash { 0 };
    bool gSigUseCompact { false };
    bool gSigComputeBary { false };
    float gSigContactDist { -1.0f };

    ~HashGridWorkspace() { release(); }

    void release()
    {
        cudaFree(cellKeys); cudaFree(slotBucketIds); cudaFree(bucketCellIds);
        cudaFree(tissueCount); cudaFree(toolCount);
        cudaFree(tissueIds); cudaFree(toolIds); cudaFree(mixedBucketIds);
        cudaFree(pairsPerBucket);
        cudaFree(candidatePairs); cudaFree(compactCandidatePairs);
        cudaFree(pairHashKeys); cudaFree(compactPairHashKeys); cudaFree(touchedPairHashSlots);
        cudaFree(candidateCount); cudaFree(rawCandidateCount);
        cudaFree(overflowCount); cudaFree(probeOverflowCount);
        cudaFree(occupiedBucketCount); cudaFree(mixedBucketCount); cudaFree(touchedPairHashCount);
        cudaFree(proximityContacts);
        cudaFree(proximityContactCount); cudaFree(proximityOverflowCount);
        cudaFree(proximityVfCount); cudaFree(proximityFvCount); cudaFree(proximityEeCount);
        cudaFree(firstIndices); cudaFree(secondIndices);
        cudaFree(firstPositions); cudaFree(secondPositions);
        cudaFreeHost(countersHostPinned);
        if (eventsReady) { cudaEventDestroy(startEvent); cudaEventDestroy(endEvent); }
        if (graphExec != nullptr) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
        if (captureStream != nullptr) { cudaStreamDestroy(captureStream); captureStream = nullptr; }
        graphInstantiated = false;
        cellKeys = nullptr; slotBucketIds = nullptr; bucketCellIds = nullptr;
        tissueCount = nullptr; toolCount = nullptr;
        tissueIds = nullptr; toolIds = nullptr; mixedBucketIds = nullptr;
        pairsPerBucket = nullptr;
        candidatePairs = nullptr; compactCandidatePairs = nullptr;
        pairHashKeys = nullptr; compactPairHashKeys = nullptr; touchedPairHashSlots = nullptr;
        candidateCount = nullptr; rawCandidateCount = nullptr;
        overflowCount = nullptr; probeOverflowCount = nullptr;
        occupiedBucketCount = nullptr; mixedBucketCount = nullptr; touchedPairHashCount = nullptr;
        proximityContacts = nullptr;
        proximityContactCount = nullptr; proximityOverflowCount = nullptr;
        proximityVfCount = nullptr; proximityFvCount = nullptr; proximityEeCount = nullptr;
        firstIndices = nullptr; secondIndices = nullptr;
        firstPositions = nullptr; secondPositions = nullptr;
        countersHostPinned = nullptr;
        startEvent = nullptr; endEvent = nullptr; eventsReady = false;
        tissueIdCapacity = 0; toolIdCapacity = 0;
        candidateCapacity = 0; compactCandidateCapacity = 0;
        pairHashCapacity = 0; compactPairHashCapacity = 0; touchedPairHashSlotCapacity = 0;
        contactCapacity = 0;
        firstIndexCapacity = 0; secondIndexCapacity = 0;
        firstPositionCapacity = 0; secondPositionCapacity = 0;
        cellKeysCapacity = 0; slotBucketIdCapacity = 0; bucketCellIdCapacity = 0;
        tissueCountCapacity = 0; toolCountCapacity = 0;
        mixedBucketIdCapacity = 0; pairsPerBucketCapacity = 0;
        firstSurfaceId = 0; secondSurfaceId = 0;
        firstTopologyVersion = ~0ull; secondTopologyVersion = ~0ull;
        pairHashKeysInitialized = false; compactPairHashKeysInitialized = false;
        useCompactCandidatePairs = false;
    }

    cudaError_t ensureEvents()
    {
        if (eventsReady) return cudaSuccess;
        cudaError_t err = cudaEventCreate(&startEvent);
        if (err == cudaSuccess) err = cudaEventCreate(&endEvent);
        eventsReady = (err == cudaSuccess);
        return err;
    }

    cudaError_t ensure(
        const std::size_t tableSize,
        const std::size_t bucketCapacity,
        const std::size_t maxTissuePerCell,
        const std::size_t maxToolPerCell,
        const std::size_t maxCandidatePairs,
        const std::size_t pairHashCount,
        const std::size_t maxContacts,
        const std::size_t contactElementBytes,
        const std::size_t firstIndexCount,
        const std::size_t secondIndexCount,
        const std::size_t firstVertexCount,    // 0 when first positions are direct device pointers
        const std::size_t secondVertexCount,   // 0 when second positions are direct device pointers
        const bool compactPairs,
        std::uint64_t& newlyAllocatedBytes)
    {
        useCompactCandidatePairs = compactPairs;
        cudaError_t err = ensureDeviceArray(cellKeys, cellKeysCapacity, tableSize, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(slotBucketIds, slotBucketIdCapacity, tableSize, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(bucketCellIds, bucketCellIdCapacity, bucketCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(tissueCount, tissueCountCapacity, bucketCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(toolCount, toolCountCapacity, bucketCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(mixedBucketIds, mixedBucketIdCapacity, bucketCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(pairsPerBucket, pairsPerBucketCapacity, bucketCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(tissueIds, tissueIdCapacity, bucketCapacity * maxTissuePerCell, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(toolIds, toolIdCapacity, bucketCapacity * maxToolPerCell, newlyAllocatedBytes);
        if (err == cudaSuccess && !compactPairs) err = ensureDeviceArray(candidatePairs, candidateCapacity, maxCandidatePairs, newlyAllocatedBytes);
        if (err == cudaSuccess && compactPairs) err = ensureDeviceArray(compactCandidatePairs, compactCandidateCapacity, maxCandidatePairs, newlyAllocatedBytes);
        if (err == cudaSuccess && !compactPairs)
        {
            if (pairHashCount > pairHashCapacity) pairHashKeysInitialized = false;
            err = ensureDeviceArray(pairHashKeys, pairHashCapacity, pairHashCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess && compactPairs)
        {
            if (pairHashCount > compactPairHashCapacity) compactPairHashKeysInitialized = false;
            err = ensureDeviceArray(compactPairHashKeys, compactPairHashCapacity, pairHashCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess) err = ensureDeviceArray(touchedPairHashSlots, touchedPairHashSlotCapacity, pairHashCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(firstIndices, firstIndexCapacity, firstIndexCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(secondIndices, secondIndexCapacity, secondIndexCount, newlyAllocatedBytes);
        if (err == cudaSuccess && firstVertexCount > 0) err = ensureDeviceArray(firstPositions, firstPositionCapacity, firstVertexCount, newlyAllocatedBytes);
        if (err == cudaSuccess && secondVertexCount > 0) err = ensureDeviceArray(secondPositions, secondPositionCapacity, secondVertexCount, newlyAllocatedBytes);
        // (Optimization 2026-06-17b) The per-frame CUB exclusive-scan was dropped:
        // its pairOffsets are unused by the block-per-bucket generator, and the
        // raw-pair total is accumulated into rawCandidateCount by
        // computeCompactHashPairsPerBucketKernel. The scan's pairOffsets/rawTotal
        // buffers, the scanTempStorage temp, and setCompactHashRawTotalKernel were
        // all removed (2026-06-17b cleanup); no CUB dependency remains.
        const std::size_t contactCount = maxContacts;
        if (err == cudaSuccess && (proximityContacts == nullptr || contactCapacity < contactCount))
        {
            cudaFree(proximityContacts); proximityContacts = nullptr;
            void* p = nullptr;
            err = cudaMalloc(&p, contactCount * contactElementBytes);
            if (err == cudaSuccess) { proximityContacts = p; contactCapacity = contactCount; newlyAllocatedBytes += contactCount * contactElementBytes; }
        }
        if (err == cudaSuccess && candidateCount == nullptr)        err = cudaMallocTracked(candidateCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && rawCandidateCount == nullptr)     err = cudaMallocTracked(rawCandidateCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && overflowCount == nullptr)         err = cudaMallocTracked(overflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && probeOverflowCount == nullptr)    err = cudaMallocTracked(probeOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && occupiedBucketCount == nullptr)   err = cudaMallocTracked(occupiedBucketCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && mixedBucketCount == nullptr)      err = cudaMallocTracked(mixedBucketCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && touchedPairHashCount == nullptr)  err = cudaMallocTracked(touchedPairHashCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityContactCount == nullptr) err = cudaMallocTracked(proximityContactCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityOverflowCount == nullptr) err = cudaMallocTracked(proximityOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityVfCount == nullptr)      err = cudaMallocTracked(proximityVfCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityFvCount == nullptr)      err = cudaMallocTracked(proximityFvCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityEeCount == nullptr)      err = cudaMallocTracked(proximityEeCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && countersHostPinned == nullptr)
        {
            err = cudaMallocHost(reinterpret_cast<void**>(&countersHostPinned), 10u * sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += 10u * sizeof(std::uint32_t);
        }
        return err;
    }

private:
    template <class T>
    static cudaError_t cudaMallocTracked(T*& ptr, const std::size_t count, std::uint64_t& bytes)
    {
        void* p = nullptr;
        const cudaError_t err = cudaMalloc(&p, count * sizeof(T));
        if (err == cudaSuccess) { ptr = static_cast<T*>(p); bytes += count * sizeof(T); }
        return err;
    }
};

HashGridWorkspace& hashGridWorkspace()
{
    static HashGridWorkspace workspace;
    return workspace;
}

// Independent workspace instance for the simple direct-bucket hash ("4th way").
// Same struct, but driven with bucketCapacity == tableSize so each hash slot is
// its own bucket (no compaction). Kept separate so a scene can A/B the optimised
// and simple hash paths without thrashing one shared workspace / graph.
HashGridWorkspace& simpleHashGridWorkspace()
{
    static HashGridWorkspace workspace;
    return workspace;
}

// Hash a linear cell id into a table slot (reuses the MurmurHash3 finalizer).
__device__ __forceinline__ std::uint32_t hashCellSlot(const std::uint32_t cellId, const std::uint32_t mask)
{
    return static_cast<std::uint32_t>(mixCandidatePairHash(static_cast<std::uint64_t>(cellId) + 1ull)) & mask;
}

__global__ void resetCompactHashGridKernel(
    std::uint32_t* cellKeys,
    std::uint32_t* slotBucketIds,
    const std::uint32_t tableSize,
    std::uint32_t* tissueCount,
    std::uint32_t* toolCount,
    std::uint32_t* pairsPerBucket,
    const std::uint32_t bucketCapacity,
    std::uint32_t* rawCandidateCount,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount,
    std::uint32_t* occupiedBucketCount,
    std::uint32_t* mixedBucketCount,
    std::uint32_t* touchedPairHashCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < tableSize)
    {
        cellKeys[id] = kEmptyHashCellKey;
        slotBucketIds[id] = kInvalidHashBucket;
    }
    if (id < bucketCapacity)
    {
        tissueCount[id] = 0u;
        toolCount[id] = 0u;
        pairsPerBucket[id] = 0u;
    }
    if (id == 0)
    {
        *rawCandidateCount = 0u;
        *candidateCount = 0u;
        *overflowCount = 0u;
        *probeOverflowCount = 0u;
        *occupiedBucketCount = 0u;
        *mixedBucketCount = 0u;
        *touchedPairHashCount = 0u;
    }
}

// One kernel for both pair-slot widths: SlotT is std::uint32_t (compact pairs)
// or unsigned long long (64-bit pairs). Both empty sentinels are all-ones.
template <class SlotT>
__global__ void clearTouchedPairHashKernel(
    SlotT* pairHashKeys,
    const std::uint32_t* touchedSlots,
    const std::uint32_t* touchedCount)
{
    const std::uint32_t count = *touchedCount;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += stride)
    {
        pairHashKeys[touchedSlots[id]] = static_cast<SlotT>(~0ull);
    }
}

__global__ void markCompactHashGridCellsKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const DeviceDenseGridConfig config,
    const std::uint32_t tableSize,
    const std::uint32_t maxProbe,
    std::uint32_t* cellKeys,
    std::uint32_t* probeOverflowCount)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount) return;

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    int3 cellMin, cellMax;
    if (!denseGridCellSpan(triangleAabb(triangle, config.contactDistance), config, cellMin, cellMax))
        return;

    const std::uint32_t mask = tableSize - 1u;

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    {
        const std::uint32_t cellId = denseCellId(x, y, z, config);
        std::uint32_t slot = hashCellSlot(cellId, mask);
        bool placed = false;
        for (std::uint32_t probe = 0; probe < maxProbe; ++probe)
        {
            const std::uint32_t previous = atomicCAS(&cellKeys[slot], kEmptyHashCellKey, cellId);
            if (previous == kEmptyHashCellKey || previous == cellId)
            {
                placed = true;
                break;
            }
            slot = (slot + 1u) & mask;
        }
        if (!placed) atomicAdd(probeOverflowCount, 1u);
    }
}

__global__ void compactHashGridSlotsKernel(
    const std::uint32_t* cellKeys,
    std::uint32_t* slotBucketIds,
    std::uint32_t* bucketCellIds,
    const std::uint32_t tableSize,
    const std::uint32_t bucketCapacity,
    std::uint32_t* occupiedBucketCount,
    std::uint32_t* overflowCount)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= tableSize) return;
    const std::uint32_t cellId = cellKeys[slot];
    if (cellId == kEmptyHashCellKey) return;

    const std::uint32_t bucketIdx = atomicAdd(occupiedBucketCount, 1u);
    if (bucketIdx < bucketCapacity)
    {
        bucketCellIds[bucketIdx] = cellId;
        slotBucketIds[slot] = bucketIdx;
    }
    else
    {
        slotBucketIds[slot] = kOverflowHashBucket;
        atomicAdd(overflowCount, 1u);
    }
}

__global__ void fillCompactHashGridTrianglesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool insertTissue,
    const DeviceDenseGridConfig config,
    const std::uint32_t tableSize,
    const std::uint32_t bucketCapacity,
    const std::uint32_t maxProbe,
    const std::uint32_t* cellKeys,
    const std::uint32_t* slotBucketIds,
    std::uint32_t* tissueCount,
    std::uint32_t* toolCount,
    std::uint32_t* tissueIds,
    std::uint32_t* toolIds,
    std::uint32_t* mixedBucketIds,
    std::uint32_t* mixedBucketCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount) return;

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    int3 cellMin, cellMax;
    if (!denseGridCellSpan(triangleAabb(triangle, config.contactDistance), config, cellMin, cellMax))
        return;

    const std::uint32_t mask = tableSize - 1u;
    const std::uint32_t bucketCap = insertTissue ? config.maxTissueTrianglesPerCell : config.maxToolTrianglesPerCell;

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    {
        const std::uint32_t cellId = denseCellId(x, y, z, config);
        std::uint32_t slot = hashCellSlot(cellId, mask);
        std::uint32_t bucketIdx = kInvalidHashBucket;
        for (std::uint32_t probe = 0; probe < maxProbe; ++probe)
        {
            const std::uint32_t key = cellKeys[slot];
            if (key == cellId)
            {
                bucketIdx = slotBucketIds[slot];
                break;
            }
            if (key == kEmptyHashCellKey)
            {
                break;
            }
            slot = (slot + 1u) & mask;
        }
        if (bucketIdx == kInvalidHashBucket)
        {
            atomicAdd(probeOverflowCount, 1u);
            continue;
        }
        if (bucketIdx == kOverflowHashBucket || bucketIdx >= bucketCapacity)
        {
            atomicAdd(overflowCount, 1u);
            continue;
        }

        std::uint32_t* countPtr = insertTissue ? &tissueCount[bucketIdx] : &toolCount[bucketIdx];
        const std::uint32_t local = atomicAdd(countPtr, 1u);
        if (local < bucketCap)
        {
            std::uint32_t* ids = insertTissue ? tissueIds : toolIds;
            ids[bucketIdx * bucketCap + local] = triangleId;
            if (!insertTissue && local == 0u && tissueCount[bucketIdx] > 0u)
            {
                const std::uint32_t mixedIndex = atomicAdd(mixedBucketCount, 1u);
                if (mixedIndex < bucketCapacity)
                {
                    mixedBucketIds[mixedIndex] = bucketIdx;
                }
                else
                {
                    atomicAdd(overflowCount, 1u);
                }
            }
        }
        else
        {
            atomicAdd(overflowCount, 1u);
        }
    }
}

__global__ void computeCompactHashPairsPerBucketKernel(
    const std::uint32_t* tissueCount,
    const std::uint32_t* toolCount,
    const std::uint32_t* mixedBucketIds,
    const std::uint32_t* mixedBucketCount,
    const std::uint32_t bucketCapacity,
    const std::uint32_t maxTissuePerCell,
    const std::uint32_t maxToolPerCell,
    std::uint32_t* pairsPerBucket,
    std::uint32_t* rawCandidateCount)
{
    // pairsPerBucket is still written per bucket, but since the CUB scan was
    // dropped (2026-06-17b) nothing reads it downstream; only the accumulated
    // raw-pair total (rawCandidateCount, via atomicAdd below) is consumed.
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= bucketCapacity) return;
    const std::uint32_t mixedCount = min(*mixedBucketCount, bucketCapacity);
    if (id >= mixedCount)
    {
        pairsPerBucket[id] = 0u;
        return;
    }
    const std::uint32_t bucketIdx = mixedBucketIds[id];
    const std::uint32_t t = min(tissueCount[bucketIdx], maxTissuePerCell);
    const std::uint32_t u = min(toolCount[bucketIdx], maxToolPerCell);
    const std::uint32_t pairs = t * u;
    pairsPerBucket[id] = pairs;
    if (pairs > 0u)
    {
        atomicAdd(rawCandidateCount, pairs);
    }
}

__device__ __forceinline__ std::uint32_t encodeCompactCandidatePair(
    const std::uint32_t firstTriangleId,
    const std::uint32_t secondTriangleId)
{
    return (firstTriangleId << 16u) | secondTriangleId;
}

// Pair-type traits for the dedup machinery. PairT is std::uint32_t (16+16
// compact encoding, both meshes <= 65,535 triangles) or std::uint64_t (32+32).
// The atomicCAS slot type differs (unsigned long long for 64-bit pairs); both
// empty sentinels are all-ones, so a single ~0 cast covers both.
template <class PairT> struct PairTraits;
template <> struct PairTraits<std::uint32_t>
{
    using SlotT = std::uint32_t;
    static __device__ __forceinline__ std::uint32_t encode(const std::uint32_t firstId, const std::uint32_t secondId)
    {
        return encodeCompactCandidatePair(firstId, secondId);
    }
};
template <> struct PairTraits<std::uint64_t>
{
    using SlotT = unsigned long long;
    static __device__ __forceinline__ std::uint64_t encode(const std::uint32_t firstId, const std::uint32_t secondId)
    {
        return encodeCandidatePair(firstId, secondId);
    }
};

// Tracked open-addressing dedup insert, shared by the hash / simple-hash /
// sorted-grid (pair-hash mode) generators. Replaces the former 32/64-bit twin
// functions; the instantiations are identical to the old hand-written pair.
template <class PairT>
__device__ bool insertUniqueCandidatePairTracked(
    const PairT pair,
    typename PairTraits<PairT>::SlotT* pairHashKeys,
    std::uint32_t* touchedPairHashSlots,
    std::uint32_t* touchedPairHashCount,
    const std::uint32_t pairHashCapacity,
    PairT* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount,
    const std::uint32_t maxCandidatePairs)
{
    using SlotT = typename PairTraits<PairT>::SlotT;
    constexpr std::uint32_t kMaxProbeCount = 256u;
    constexpr SlotT kEmpty = static_cast<SlotT>(~0ull);
    const std::uint32_t mask = pairHashCapacity - 1u;
    std::uint32_t slot = static_cast<std::uint32_t>(mixCandidatePairHash(pair)) & mask;
    for (std::uint32_t probe = 0; probe < kMaxProbeCount; ++probe)
    {
        const SlotT previous = atomicCAS(&pairHashKeys[slot], kEmpty, static_cast<SlotT>(pair));
        if (previous == kEmpty)
        {
            const std::uint32_t touchedIndex = atomicAdd(touchedPairHashCount, 1u);
            if (touchedIndex < pairHashCapacity)
            {
                touchedPairHashSlots[touchedIndex] = slot;
            }
            else
            {
                atomicAdd(overflowCount, 1u);
            }
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
        if (previous == static_cast<SlotT>(pair))
        {
            return false;
        }
        slot = (slot + 1u) & mask;
    }
    atomicAdd(probeOverflowCount, 1u);
    atomicAdd(overflowCount, 1u);
    return false;
}

// Block-per-mixed-bucket pair generator, shared by the hash and simple-hash
// ways. One template for both pair widths (was a hand-written 32/64 twin pair).
template <class PairT>
__global__ void generateMixedHashCandidatePairsKernel(
    const std::uint32_t* tissueCount,
    const std::uint32_t* toolCount,
    const std::uint32_t* mixedBucketIds,
    const std::uint32_t* mixedBucketCount,
    const std::uint32_t* tissueIds,
    const std::uint32_t* toolIds,
    const std::uint32_t mixedBucketCapacity,
    const std::uint32_t maxTissuePerCell,
    const std::uint32_t maxToolPerCell,
    const std::uint32_t pairHashCapacity,
    const std::uint32_t maxCandidatePairs,
    PairT* candidatePairs,
    typename PairTraits<PairT>::SlotT* pairHashKeys,
    std::uint32_t* touchedPairHashSlots,
    std::uint32_t* touchedPairHashCount,
    std::uint32_t* candidateCount,
    std::uint32_t* probeOverflowCount,
    std::uint32_t* overflowCount)
{
    const std::uint32_t mixedCount = min(*mixedBucketCount, mixedBucketCapacity);
    for (std::uint32_t mixedIndex = blockIdx.x; mixedIndex < mixedCount; mixedIndex += gridDim.x)
    {
        const std::uint32_t bucketIdx = mixedBucketIds[mixedIndex];
        const std::uint32_t t = min(tissueCount[bucketIdx], maxTissuePerCell);
        const std::uint32_t u = min(toolCount[bucketIdx], maxToolPerCell);
        const std::uint64_t totalPairs = static_cast<std::uint64_t>(t) * static_cast<std::uint64_t>(u);
        for (std::uint64_t localPair = threadIdx.x; localPair < totalPairs; localPair += blockDim.x)
        {
            const std::uint32_t tissueLocal = static_cast<std::uint32_t>(localPair / u);
            const std::uint32_t toolLocal = static_cast<std::uint32_t>(localPair % u);
            const std::uint32_t tissueTriId = tissueIds[bucketIdx * maxTissuePerCell + tissueLocal];
            const std::uint32_t toolTriId = toolIds[bucketIdx * maxToolPerCell + toolLocal];
            insertUniqueCandidatePairTracked<PairT>(
                PairTraits<PairT>::encode(tissueTriId, toolTriId),
                pairHashKeys,
                touchedPairHashSlots,
                touchedPairHashCount,
                pairHashCapacity,
                candidatePairs,
                candidateCount,
                overflowCount,
                probeOverflowCount,
                maxCandidatePairs);
        }
    }
}

// ============================================================================
// Simple direct-bucket hash insert ("4th way"). One thread per triangle. For
// each cell the triangle's inflated AABB overlaps, claim the cell's hash slot
// (atomicCAS + linear probe on fmix64) and append the triangle id straight into
// that slot's bucket — a SINGLE pass, no mark/compact/fill. Here the hash slot
// IS the bucket (bucketCap == tableSize), so tissueIds/toolIds are indexed by
// slot exactly as generateMixedHashCandidatePairs{32,64}Kernel expects. Tissue
// must be inserted before tool (separate launches) so the mixed-slot test
// (first tool triangle into a slot that already has tissue) fires once per slot.
// Best-effort: a full bucket or probe exhaustion drops the surplus and bumps the
// overflow / probe-overflow counters.
// ============================================================================


} // namespace


namespace SofaGpuCollision::backend
{

bool computeHashPrefixSumProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const HashPrefixSumConfig& hashConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    HashPrefixSumStats* hashStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = firstSurface.triangleCount + secondSurface.triangleCount;
    }
    if (proximityStats != nullptr) *proximityStats = FeatureBasedProximityStats {};
    if (hashStats != nullptr) *hashStats = HashPrefixSumStats {};

    const auto surfaceInvalid = [](const TriangleIndexedSurface& s) {
        return (s.positions == nullptr && s.devicePositions == nullptr) ||
            s.triangleIndices == nullptr || s.vertexCount == 0 || s.triangleCount == 0;
    };
    if (surfaceInvalid(firstSurface) || surfaceInvalid(secondSurface))
    {
        diagnostic = "Invalid indexed triangle surface (hash path).";
        return false;
    }
    if (gridConfig.gridResolutionX == 0 || gridConfig.gridResolutionY == 0 || gridConfig.gridResolutionZ == 0 ||
        gridConfig.maxTissueTrianglesPerCell == 0 || gridConfig.maxToolTrianglesPerCell == 0 ||
        gridConfig.maxCandidatePairs == 0 ||
        gridConfig.gridMaxX <= gridConfig.gridMinX ||
        gridConfig.gridMaxY <= gridConfig.gridMinY ||
        gridConfig.gridMaxZ <= gridConfig.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration (hash path).";
        return false;
    }

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(gridConfig.gridResolutionX) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionY) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 > std::numeric_limits<std::uint32_t>::max())
    {
        diagnostic = "Hash path grid cell count out of range.";
        return false;
    }
    const std::uint32_t cellCount = static_cast<std::uint32_t>(cellCount64);

    // Derive the hash table size (power of two). Heuristic: ~4 slots per input
    // triangle, so the table comfortably exceeds the number of occupied cells.
    std::uint32_t tableSize = hashConfig.hashTableSize;
    if (tableSize == 0)
    {
        const std::uint64_t guess = static_cast<std::uint64_t>(firstSurface.triangleCount + secondSurface.triangleCount) * 4ull;
        const std::size_t candidate = nextPowerOfTwo(static_cast<std::size_t>(std::max<std::uint64_t>(1024ull, guess)));
        if (candidate > std::numeric_limits<std::uint32_t>::max())
        {
            diagnostic = "Hash path table size out of range.";
            return false;
        }
        tableSize = static_cast<std::uint32_t>(candidate);
    }
    else
    {
        const std::size_t candidate = nextPowerOfTwo(tableSize);
        if (candidate > std::numeric_limits<std::uint32_t>::max())
        {
            diagnostic = "Hash path table size out of range.";
            return false;
        }
        tableSize = static_cast<std::uint32_t>(candidate);
    }
    const std::uint32_t bucketCapacity = std::max(1u, std::min(tableSize, cellCount));
    if (bucketCapacity > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        diagnostic = "Hash path bucket capacity exceeds CUB scan element limit.";
        return false;
    }
    const std::uint64_t pairHashCount64 =
        static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(gridConfig.maxCandidatePairs) * 2u));
    if (pairHashCount64 == 0 || pairHashCount64 > std::numeric_limits<std::uint32_t>::max())
    {
        diagnostic = "Hash path pair-hash table size out of range.";
        return false;
    }
    const bool useCompactCandidatePairs =
        firstSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max() &&
        secondSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max();
    const std::uint32_t maxProbe = std::max(1u, hashConfig.maxProbe);

    auto& ws = hashGridWorkspace();
    std::uint64_t newlyAllocatedBytes = 0;
    const auto allocStart = std::chrono::steady_clock::now();
    cudaError_t err = ws.ensure(
        tableSize,
        bucketCapacity,
        gridConfig.maxTissueTrianglesPerCell,
        gridConfig.maxToolTrianglesPerCell,
        gridConfig.maxCandidatePairs,
        static_cast<std::size_t>(pairHashCount64),
        proximityConfig.maxContacts,
        sizeof(DeviceProximityContact),
        static_cast<std::size_t>(firstSurface.triangleCount) * 3u,
        static_cast<std::size_t>(secondSurface.triangleCount) * 3u,
        firstSurface.devicePositions == nullptr ? firstSurface.vertexCount : 0u,
        secondSurface.devicePositions == nullptr ? secondSurface.vertexCount : 0u,
        useCompactCandidatePairs,
        newlyAllocatedBytes);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Hash workspace alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += elapsedMillisecondsSince(allocStart);
        if (newlyAllocatedBytes > 0) executionStats->workspaceResizeCount += 1;
    }

    // Upload triangle indices on topology change; positions only for host input.
    const bool uploadFirstTopo = ws.firstSurfaceId != firstSurface.surfaceId || ws.firstTopologyVersion != firstSurface.topologyVersion;
    const bool uploadSecondTopo = ws.secondSurfaceId != secondSurface.surfaceId || ws.secondTopologyVersion != secondSurface.topologyVersion;
    const auto h2dStart = std::chrono::steady_clock::now();
    if (uploadFirstTopo)
        err = copyHostArrayToDeviceAsync(ws.firstIndices, firstSurface.triangleIndices, (std::uint32_t*)nullptr, static_cast<std::size_t>(firstSurface.triangleCount) * 3u, false);
    if (err == cudaSuccess && uploadSecondTopo)
        err = copyHostArrayToDeviceAsync(ws.secondIndices, secondSurface.triangleIndices, (std::uint32_t*)nullptr, static_cast<std::size_t>(secondSurface.triangleCount) * 3u, false);
    if (err == cudaSuccess && firstSurface.devicePositions == nullptr)
        err = copyHostArrayToDeviceAsync(ws.firstPositions, firstSurface.positions, (BackendTriangleVertex*)nullptr, static_cast<std::size_t>(firstSurface.vertexCount), false);
    if (err == cudaSuccess && secondSurface.devicePositions == nullptr)
        err = copyHostArrayToDeviceAsync(ws.secondPositions, secondSurface.positions, (BackendTriangleVertex*)nullptr, static_cast<std::size_t>(secondSurface.vertexCount), false);
    if (err == cudaSuccess && uploadFirstTopo) { ws.firstSurfaceId = firstSurface.surfaceId; ws.firstTopologyVersion = firstSurface.topologyVersion; }
    if (err == cudaSuccess && uploadSecondTopo) { ws.secondSurfaceId = secondSurface.surfaceId; ws.secondTopologyVersion = secondSurface.topologyVersion; }
    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceMilliseconds += elapsedMillisecondsSince(h2dStart);
        if (uploadFirstTopo) executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(firstSurface.triangleCount) * 3u * sizeof(std::uint32_t);
        if (uploadSecondTopo) executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(secondSurface.triangleCount) * 3u * sizeof(std::uint32_t);
        if (firstSurface.devicePositions == nullptr) executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(firstSurface.vertexCount) * sizeof(BackendTriangleVertex);
        if (secondSurface.devicePositions == nullptr) executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(secondSurface.vertexCount) * sizeof(BackendTriangleVertex);
    }
    if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }

    const BackendTriangleVertex* deviceFirstPositions = firstSurface.devicePositions != nullptr ? firstSurface.devicePositions : ws.firstPositions;
    const BackendTriangleVertex* deviceSecondPositions = secondSurface.devicePositions != nullptr ? secondSurface.devicePositions : ws.secondPositions;

    const DeviceDenseGridConfig dc = makeDeviceDenseGridConfig(gridConfig, static_cast<std::uint32_t>(pairHashCount64));

    constexpr std::uint32_t threads = 256;
    const std::uint32_t tableBlocks = (tableSize + threads - 1u) / threads;
    const std::uint32_t bucketBlocks = (bucketCapacity + threads - 1u) / threads;
    const std::uint32_t resetItems = std::max(tableSize, bucketCapacity);
    const std::uint32_t resetBlocks = (resetItems + threads - 1u) / threads;
    const std::uint32_t firstBlocks = (firstSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t secondBlocks = (secondSurface.triangleCount + threads - 1u) / threads;
    constexpr std::uint32_t kGenBlocks = 1024;
    const std::uint32_t pairHashCount = static_cast<std::uint32_t>(pairHashCount64);
    const std::uint32_t pairHashClearBlocks =
        std::max(1u, std::min(kGenBlocks, (pairHashCount + threads - 1u) / threads));

    const bool detailedProfiling = gridConfig.detailedProfiling;
    const bool totalTiming = proximityConfig.readContactCounter && !detailedProfiling;
    if (detailedProfiling || totalTiming)
    {
        err = ws.ensureEvents();
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            return false;
        }
    }
    if (totalTiming)
    {
        err = cudaEventRecord(ws.startEvent);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            return false;
        }
    }

    double hashPairClearMs = 0.0;
    double hashResetMs = 0.0;
    double hashInsertTissueMs = 0.0;
    double hashInsertToolMs = 0.0;
    double hashCompactBucketsMs = 0.0;
    double hashPairCountMs = 0.0;
    double hashScanMs = 0.0;
    double hashGeneratePairsMs = 0.0;
    double hashProximityCounterClearMs = 0.0;
    double fbpMs = 0.0;
    std::uint32_t launchCount = 0;
    std::uint32_t memsetCount = 0;

    auto* deviceContacts = reinterpret_cast<DeviceProximityContact*>(ws.proximityContacts);

    // (Optimization 2026-06-17c) The whole per-frame kernel sequence, on stream s.
    // Used for (a) direct execution and (b) CUDA-graph capture. Must stay in sync
    // with the detailed-profiling sequence below (the contact-parity tests catch
    // drift). fullClear=true does a one-time full dedup-table memset (first frame
    // / after a resize); otherwise only the previous frame's touched slots are
    // cleared.
    auto launchAll = [&](cudaStream_t s, bool fullClear)
    {
        if (useCompactCandidatePairs)
        {
            if (fullClear) { cudaMemsetAsync(ws.compactPairHashKeys, 0xff, static_cast<std::size_t>(pairHashCount) * sizeof(std::uint32_t), s); ws.compactPairHashKeysInitialized = true; }
            else clearTouchedPairHashKernel<std::uint32_t><<<pairHashClearBlocks, threads, 0, s>>>(ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        else
        {
            if (fullClear) { cudaMemsetAsync(ws.pairHashKeys, 0xff, static_cast<std::size_t>(pairHashCount) * sizeof(unsigned long long), s); ws.pairHashKeysInitialized = true; }
            else clearTouchedPairHashKernel<unsigned long long><<<pairHashClearBlocks, threads, 0, s>>>(ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        resetCompactHashGridKernel<<<resetBlocks, threads, 0, s>>>(ws.cellKeys, ws.slotBucketIds, tableSize, ws.tissueCount, ws.toolCount, ws.pairsPerBucket, bucketCapacity, ws.rawCandidateCount, ws.candidateCount, ws.overflowCount, ws.probeOverflowCount, ws.occupiedBucketCount, ws.mixedBucketCount, ws.touchedPairHashCount);
        markCompactHashGridCellsKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, dc, tableSize, maxProbe, ws.cellKeys, ws.probeOverflowCount);
        markCompactHashGridCellsKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, dc, tableSize, maxProbe, ws.cellKeys, ws.probeOverflowCount);
        compactHashGridSlotsKernel<<<tableBlocks, threads, 0, s>>>(ws.cellKeys, ws.slotBucketIds, ws.bucketCellIds, tableSize, bucketCapacity, ws.occupiedBucketCount, ws.overflowCount);
        fillCompactHashGridTrianglesKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, true, dc, tableSize, bucketCapacity, maxProbe, ws.cellKeys, ws.slotBucketIds, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, ws.overflowCount, ws.probeOverflowCount);
        fillCompactHashGridTrianglesKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, false, dc, tableSize, bucketCapacity, maxProbe, ws.cellKeys, ws.slotBucketIds, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, ws.overflowCount, ws.probeOverflowCount);
        computeCompactHashPairsPerBucketKernel<<<bucketBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, ws.pairsPerBucket, ws.rawCandidateCount);
        if (useCompactCandidatePairs)
            generateMixedHashCandidatePairsKernel<std::uint32_t><<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.compactCandidatePairs, ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        else
            generateMixedHashCandidatePairsKernel<std::uint64_t><<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.candidatePairs, ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        resetProximityCountersKernel<<<1, 1, 0, s>>>(ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount);
        featureBasedProximityKernel<<<kGenBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, deviceSecondPositions, ws.secondIndices, useCompactCandidatePairs ? nullptr : ws.candidatePairs, useCompactCandidatePairs ? ws.compactCandidatePairs : nullptr, ws.candidateCount, useCompactCandidatePairs, deviceContacts, ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount, proximityConfig.maxContacts, proximityConfig.contactDistance, proximityConfig.computeBarycentrics);
    };

    // CUDA-graph fast path (DEFAULT ON; disable with SOFA_HASH_CUDA_GRAPH=0).
    // Captures the steady-state sequence once and replays it, cutting per-launch
    // overhead (verified 2026-06-17: bit-identical contacts, ~-7% kernel / +10%
    // FPS on the large hash scene). Disabled under detailed profiling (which needs
    // per-stage events). Falls back safely to the direct sequence on any failure.
    static const int kGraphMode = []{ const char* e = std::getenv("SOFA_HASH_CUDA_GRAPH"); return e ? std::atoi(e) : 1; }();
    const bool kGraphsEnabled = (kGraphMode != 0);
    bool ranViaGraph = false;
    if (kGraphsEnabled && !detailedProfiling)
    {
        const bool firstFrame = !(useCompactCandidatePairs ? ws.compactPairHashKeysInitialized : ws.pairHashKeysInitialized);
        const bool sigChanged =
            ws.gSigTableSize != tableSize || ws.gSigBucketCap != bucketCapacity ||
            ws.gSigFirstTri != firstSurface.triangleCount || ws.gSigSecondTri != secondSurface.triangleCount ||
            ws.gSigMaxContacts != proximityConfig.maxContacts || ws.gSigPairHash != pairHashCount ||
            ws.gSigUseCompact != useCompactCandidatePairs || ws.gSigComputeBary != proximityConfig.computeBarycentrics ||
            ws.gSigContactDist != proximityConfig.contactDistance;
        if (newlyAllocatedBytes > 0 || sigChanged) ws.graphInstantiated = false;

        if (firstFrame)
        {
            launchAll(0, /*fullClear=*/true);
            ws.graphInstantiated = false;  // re-capture next (steady) frame
            ranViaGraph = true;
        }
        else
        {
            if (ws.captureStream == nullptr) cudaStreamCreate(&ws.captureStream);
            if (!ws.graphInstantiated && ws.captureStream != nullptr)
            {
                cudaGraph_t graph = nullptr;
                cudaError_t capErr = cudaStreamBeginCapture(ws.captureStream, cudaStreamCaptureModeThreadLocal);
                if (capErr == cudaSuccess)
                {
                    launchAll(ws.captureStream, /*fullClear=*/false);
                    capErr = cudaStreamEndCapture(ws.captureStream, &graph);
                }
                if (capErr == cudaSuccess && graph != nullptr)
                {
                    if (ws.graphExec != nullptr) { cudaGraphExecDestroy(ws.graphExec); ws.graphExec = nullptr; }
#if CUDART_VERSION >= 12000
                    capErr = cudaGraphInstantiate(&ws.graphExec, graph, 0ull);
#else
                    capErr = cudaGraphInstantiate(&ws.graphExec, graph, nullptr, nullptr, 0);
#endif
                    cudaGraphDestroy(graph);
                    if (capErr == cudaSuccess)
                    {
                        ws.graphInstantiated = true;
                        ws.gSigTableSize = tableSize; ws.gSigBucketCap = bucketCapacity;
                        ws.gSigFirstTri = firstSurface.triangleCount; ws.gSigSecondTri = secondSurface.triangleCount;
                        ws.gSigMaxContacts = proximityConfig.maxContacts; ws.gSigPairHash = pairHashCount;
                        ws.gSigUseCompact = useCompactCandidatePairs; ws.gSigComputeBary = proximityConfig.computeBarycentrics;
                        ws.gSigContactDist = proximityConfig.contactDistance;
                    }
                }
                else
                {
                    cudaGetLastError();  // clear any capture error
                }
            }
            if (ws.graphInstantiated)
            {
                err = cudaGraphLaunch(ws.graphExec, 0);
                ranViaGraph = (err == cudaSuccess);
            }
            if (!ranViaGraph)
            {
                launchAll(0, /*fullClear=*/false);  // safe fallback
                ranViaGraph = true;
            }
        }
        launchCount += 11u;
        if (firstFrame) memsetCount += 1u;
        err = cudaGetLastError();
        if (err != cudaSuccess) { diagnostic = std::string("hash graph path: ") + cudaGetErrorString(err); return false; }
    }

    if (!ranViaGraph)
    {
    // Clear only the pair-hash slots touched by the previous frame. The first
    // frame or a capacity growth still needs a full initialization.
    if (useCompactCandidatePairs)
    {
        const bool fullPairHashClear = !ws.compactPairHashKeysInitialized;
        err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashPairClearMs, [&]() {
            ScopedNvtxRange range("hash clear compact touched pair slots", detailedProfiling);
            if (!ws.compactPairHashKeysInitialized)
            {
                const cudaError_t memsetErr = cudaMemset(
                    ws.compactPairHashKeys,
                    0xff,
                    static_cast<std::size_t>(pairHashCount) * sizeof(std::uint32_t));
                if (memsetErr == cudaSuccess) ws.compactPairHashKeysInitialized = true;
                return memsetErr;
            }
            clearTouchedPairHashKernel<std::uint32_t><<<pairHashClearBlocks, threads>>>(
                ws.compactPairHashKeys,
                ws.touchedPairHashSlots,
                ws.touchedPairHashCount);
            return cudaGetLastError();
        });
        if (fullPairHashClear) memsetCount += 1u;
        else launchCount += 1u;
    }
    else
    {
        const bool fullPairHashClear = !ws.pairHashKeysInitialized;
        err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashPairClearMs, [&]() {
            ScopedNvtxRange range("hash clear touched pair slots", detailedProfiling);
            if (!ws.pairHashKeysInitialized)
            {
                const cudaError_t memsetErr = cudaMemset(
                    ws.pairHashKeys,
                    0xff,
                    static_cast<std::size_t>(pairHashCount) * sizeof(unsigned long long));
                if (memsetErr == cudaSuccess) ws.pairHashKeysInitialized = true;
                return memsetErr;
            }
            clearTouchedPairHashKernel<unsigned long long><<<pairHashClearBlocks, threads>>>(
                ws.pairHashKeys,
                ws.touchedPairHashSlots,
                ws.touchedPairHashCount);
            return cudaGetLastError();
        });
        if (fullPairHashClear) memsetCount += 1u;
        else launchCount += 1u;
    }
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash pair-table clear: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashResetMs, [&]() {
        ScopedNvtxRange range("hash reset compact grid", detailedProfiling);
        resetCompactHashGridKernel<<<resetBlocks, threads>>>(
            ws.cellKeys,
            ws.slotBucketIds,
            tableSize,
            ws.tissueCount,
            ws.toolCount,
            ws.pairsPerBucket,
            bucketCapacity,
            ws.rawCandidateCount,
            ws.candidateCount,
            ws.overflowCount,
            ws.probeOverflowCount,
            ws.occupiedBucketCount,
            ws.mixedBucketCount,
            ws.touchedPairHashCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash reset launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashInsertTissueMs, [&]() {
        ScopedNvtxRange range("hash mark tissue cells", detailedProfiling);
        markCompactHashGridCellsKernel<<<firstBlocks, threads>>>(
            deviceFirstPositions,
            ws.firstIndices,
            firstSurface.triangleCount,
            dc,
            tableSize,
            maxProbe,
            ws.cellKeys,
            ws.probeOverflowCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash tissue cell mark launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashInsertToolMs, [&]() {
        ScopedNvtxRange range("hash mark tool cells", detailedProfiling);
        markCompactHashGridCellsKernel<<<secondBlocks, threads>>>(
            deviceSecondPositions,
            ws.secondIndices,
            secondSurface.triangleCount,
            dc,
            tableSize,
            maxProbe,
            ws.cellKeys,
            ws.probeOverflowCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash tool cell mark launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashCompactBucketsMs, [&]() {
        ScopedNvtxRange range("hash compact occupied buckets", detailedProfiling);
        compactHashGridSlotsKernel<<<tableBlocks, threads>>>(
            ws.cellKeys,
            ws.slotBucketIds,
            ws.bucketCellIds,
            tableSize,
            bucketCapacity,
            ws.occupiedBucketCount,
            ws.overflowCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash bucket compaction launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashInsertTissueMs, [&]() {
        ScopedNvtxRange range("hash fill tissue buckets", detailedProfiling);
        fillCompactHashGridTrianglesKernel<<<firstBlocks, threads>>>(
            deviceFirstPositions,
            ws.firstIndices,
            firstSurface.triangleCount,
            true,
            dc,
            tableSize,
            bucketCapacity,
            maxProbe,
            ws.cellKeys,
            ws.slotBucketIds,
            ws.tissueCount,
            ws.toolCount,
            ws.tissueIds,
            ws.toolIds,
            ws.mixedBucketIds,
            ws.mixedBucketCount,
            ws.overflowCount,
            ws.probeOverflowCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash tissue bucket fill launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashInsertToolMs, [&]() {
        ScopedNvtxRange range("hash fill tool buckets", detailedProfiling);
        fillCompactHashGridTrianglesKernel<<<secondBlocks, threads>>>(
            deviceSecondPositions,
            ws.secondIndices,
            secondSurface.triangleCount,
            false,
            dc,
            tableSize,
            bucketCapacity,
            maxProbe,
            ws.cellKeys,
            ws.slotBucketIds,
            ws.tissueCount,
            ws.toolCount,
            ws.tissueIds,
            ws.toolIds,
            ws.mixedBucketIds,
            ws.mixedBucketCount,
            ws.overflowCount,
            ws.probeOverflowCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash tool bucket fill launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashPairCountMs, [&]() {
        ScopedNvtxRange range("hash count mixed-bucket pairs", detailedProfiling);
        computeCompactHashPairsPerBucketKernel<<<bucketBlocks, threads>>>(
            ws.tissueCount,
            ws.toolCount,
            ws.mixedBucketIds,
            ws.mixedBucketCount,
            bucketCapacity,
            gridConfig.maxTissueTrianglesPerCell,
            gridConfig.maxToolTrianglesPerCell,
            ws.pairsPerBucket,
            ws.rawCandidateCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash pair count launch: ") + cudaGetErrorString(err);
        return false;
    }

    // (Optimization 2026-06-17b) The exclusive scan over pairsPerBucket and
    // setCompactHashRawTotalKernel were removed: their pairOffsets/rawTotal
    // outputs are unused now that generateMixedHashCandidatePairs* is
    // block-per-bucket (no global offsets needed). The raw-pair-count statistic
    // is already accumulated into rawCandidateCount by
    // computeCompactHashPairsPerBucketKernel. Saves 2 launches/frame; the
    // candidate set and contacts are bit-identical.
    hashScanMs = 0.0;

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashGeneratePairsMs, [&]() {
        ScopedNvtxRange range("hash generate mixed candidate pairs", detailedProfiling);
        if (useCompactCandidatePairs)
        {
            generateMixedHashCandidatePairsKernel<std::uint32_t><<<kGenBlocks, threads>>>(
                ws.tissueCount,
                ws.toolCount,
                ws.mixedBucketIds,
                ws.mixedBucketCount,
                ws.tissueIds,
                ws.toolIds,
                bucketCapacity,
                gridConfig.maxTissueTrianglesPerCell,
                gridConfig.maxToolTrianglesPerCell,
                pairHashCount,
                gridConfig.maxCandidatePairs,
                ws.compactCandidatePairs,
                ws.compactPairHashKeys,
                ws.touchedPairHashSlots,
                ws.touchedPairHashCount,
                ws.candidateCount,
                ws.probeOverflowCount,
                ws.overflowCount);
        }
        else
        {
            generateMixedHashCandidatePairsKernel<std::uint64_t><<<kGenBlocks, threads>>>(
                ws.tissueCount,
                ws.toolCount,
                ws.mixedBucketIds,
                ws.mixedBucketCount,
                ws.tissueIds,
                ws.toolIds,
                bucketCapacity,
                gridConfig.maxTissueTrianglesPerCell,
                gridConfig.maxToolTrianglesPerCell,
                pairHashCount,
                gridConfig.maxCandidatePairs,
                ws.candidatePairs,
                ws.pairHashKeys,
                ws.touchedPairHashSlots,
                ws.touchedPairHashCount,
                ws.candidateCount,
                ws.probeOverflowCount,
                ws.overflowCount);
        }
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash generate launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, hashProximityCounterClearMs, [&]() {
        ScopedNvtxRange range("hash reset proximity counters", detailedProfiling);
        resetProximityCountersKernel<<<1, 1>>>(
            ws.proximityContactCount,
            ws.proximityOverflowCount,
            ws.proximityVfCount,
            ws.proximityFvCount,
            ws.proximityEeCount);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash proximity counter reset launch: ") + cudaGetErrorString(err);
        return false;
    }

    err = runCudaOperation(detailedProfiling, ws.startEvent, ws.endEvent, fbpMs, [&]() {
        ScopedNvtxRange range("hash FBP narrow phase", detailedProfiling);
        featureBasedProximityKernel<<<kGenBlocks, threads>>>(
            deviceFirstPositions,
            ws.firstIndices,
            deviceSecondPositions,
            ws.secondIndices,
            useCompactCandidatePairs ? nullptr : ws.candidatePairs,
            useCompactCandidatePairs ? ws.compactCandidatePairs : nullptr,
            ws.candidateCount,
            useCompactCandidatePairs,
            deviceContacts,
            ws.proximityContactCount,
            ws.proximityOverflowCount,
            ws.proximityVfCount,
            ws.proximityFvCount,
            ws.proximityEeCount,
            proximityConfig.maxContacts,
            proximityConfig.contactDistance,
            proximityConfig.computeBarycentrics);
        return cudaGetLastError();
    });
    launchCount += 1u;
    if (err != cudaSuccess)
    {
        diagnostic = std::string("hash FBP launch: ") + cudaGetErrorString(err);
        return false;
    }
    }  // end if (!ranViaGraph): direct/detailed sequence

    float kernelMs = 0.0f;
    if (totalTiming)
    {
        err = cudaEventRecord(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventSynchronize(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventElapsedTime(&kernelMs, ws.startEvent, ws.endEvent);
        if (err != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(err);
            return false;
        }
    }

    if (executionStats != nullptr)
    {
        const double detailedKernelMs =
            hashPairClearMs +
            hashResetMs +
            hashInsertTissueMs +
            hashInsertToolMs +
            hashCompactBucketsMs +
            hashPairCountMs +
            hashScanMs +
            hashGeneratePairsMs +
            hashProximityCounterClearMs +
            fbpMs;
        executionStats->kernelLaunchCount += launchCount;
        executionStats->cudaMemsetCount += memsetCount;
        executionStats->gpuKernelMilliseconds += detailedProfiling ? detailedKernelMs : static_cast<double>(kernelMs);
        executionStats->featureBasedProximityKernelMilliseconds += detailedProfiling ? fbpMs : 0.0;
        executionStats->hashGridResetMilliseconds += hashResetMs;
        executionStats->hashGridPairHashClearMilliseconds += hashPairClearMs;
        executionStats->hashGridInsertTissueMilliseconds += hashInsertTissueMs;
        executionStats->hashGridInsertToolMilliseconds += hashInsertToolMs;
        executionStats->hashGridPairCountMilliseconds += hashCompactBucketsMs + hashPairCountMs;
        executionStats->hashGridScanMilliseconds += hashScanMs;
        executionStats->hashGridGeneratePairsMilliseconds += hashGeneratePairsMs;
        executionStats->hashGridProximityCounterClearMilliseconds += hashProximityCounterClearMs;
    }

    // Optional readback (validation/profiling, or when contacts must reach host).
    std::uint32_t hostContactCount = 0, hostUnique = 0, hostRaw = 0, hostOverflow = 0, hostProbe = 0, hostOccupied = 0, hostMixed = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* p = ws.countersHostPinned;
        cudaMemcpyAsync(p + 0, ws.candidateCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 1, ws.rawCandidateCount,     sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 2, ws.overflowCount,         sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 3, ws.probeOverflowCount,    sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 4, ws.occupiedBucketCount,   sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 5, ws.proximityContactCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 6, ws.proximityVfCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 7, ws.proximityFvCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 8, ws.proximityEeCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 9, ws.mixedBucketCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostUnique = p[0]; hostRaw = p[1]; hostOverflow = p[2]; hostProbe = p[3]; hostOccupied = p[4];
        hostContactCount = p[5]; hostVf = p[6]; hostFv = p[7]; hostEe = p[8]; hostMixed = p[9];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 10u * sizeof(std::uint32_t);
            executionStats->rawCandidateCount += hostRaw;
            executionStats->uniqueCandidateCount += hostUnique;
            executionStats->outputCandidateCount += hostUnique;
            executionStats->outputContactCount = hostContactCount;
            executionStats->activeMixedCellCount += std::min(hostMixed, bucketCapacity);
            executionStats->vfContactCount += hostVf;
            executionStats->fvContactCount += hostFv;
            executionStats->eeContactCount += hostEe;
            if (hostOverflow > 0) executionStats->overflowCount += hostOverflow;
            if (hostProbe > 0) executionStats->hashDedupeProbeOverflowCount += hostProbe;
        }
        if (proximityStats != nullptr)
        {
            proximityStats->candidatePairCount = hostUnique;
            proximityStats->emittedContactCount = hostContactCount;
            proximityStats->vfContactCount = hostVf;
            proximityStats->fvContactCount = hostFv;
            proximityStats->eeContactCount = hostEe;
        }
        if (hashStats != nullptr)
        {
            hashStats->hashTableSize = tableSize;
            hashStats->occupiedSlotCount = hostOccupied;
            hashStats->rawPairCount = hostRaw;
            hashStats->uniquePairCount = hostUnique;
            hashStats->hashProbeOverflowCount = hostProbe;
            hashStats->bucketOverflowCount = hostOverflow;
        }
    }

    if (!proximityConfig.keepContactsOnDevice && hostContactCount > 0)
    {
        downloadDeviceProximityContacts(deviceContacts, hostContactCount, proximityConfig.maxContacts, contacts, executionStats);
    }

    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = cellCount;
    }

    diagnostic.clear();
    return true;
}

// ============================================================================
// computeSimpleHashProximityContacts ("4th way" — simple direct-bucket hash)
// ----------------------------------------------------------------------------
// Triangles are stored DIRECTLY into per-cell hash buckets in a single insert
// pass (insertSimpleHashTrianglesKernel), with each table slot acting as its own
// bucket (bucketCapacity == tableSize). No mark, no compact, no pairs-per-bucket
// scan: the 6-kernel steady-state sequence is reset -> insert tissue -> insert
// tool -> generate mixed pairs (deduped) -> reset counters -> FBP. The candidate
// dedup, optional 32-bit packing, touched-slot clear, CUDA graph and FBP narrow
// kernel are all shared with the optimised hash path, so on scenes without
// per-cell overflow the contacts are identical. Best-effort: bucket/probe
// overflow drops the surplus (reported, not fixed up).
// ============================================================================


} // namespace SofaGpuCollision::backend
