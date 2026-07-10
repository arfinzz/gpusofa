// SimpleHash.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Simple direct-bucket hash (way 4): the single-pass insert kernel and the
// simple-hash host driver (reuses the HashGrid kernels + workspace type).

namespace
{

__global__ void insertSimpleHashTrianglesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool insertTissue,
    const DeviceDenseGridConfig config,
    const std::uint32_t tableSize,
    const std::uint32_t maxProbe,
    std::uint32_t* cellKeys,
    std::uint32_t* tissueCount,
    std::uint32_t* toolCount,
    std::uint32_t* tissueIds,
    std::uint32_t* toolIds,
    std::uint32_t* mixedSlotIds,
    std::uint32_t* mixedSlotCount,
    const std::uint32_t mixedCapacity,
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
        bool claimed = false;
        for (std::uint32_t probe = 0; probe < maxProbe; ++probe)
        {
            const std::uint32_t previous = atomicCAS(&cellKeys[slot], kEmptyHashCellKey, cellId);
            if (previous == kEmptyHashCellKey || previous == cellId)
            {
                claimed = true;
                break;
            }
            slot = (slot + 1u) & mask;
        }
        if (!claimed)
        {
            atomicAdd(probeOverflowCount, 1u);
            continue;
        }

        std::uint32_t* countPtr = insertTissue ? &tissueCount[slot] : &toolCount[slot];
        const std::uint32_t local = atomicAdd(countPtr, 1u);
        if (local < bucketCap)
        {
            std::uint32_t* ids = insertTissue ? tissueIds : toolIds;
            ids[slot * bucketCap + local] = triangleId;
            if (!insertTissue && local == 0u && tissueCount[slot] > 0u)
            {
                const std::uint32_t mixedIndex = atomicAdd(mixedSlotCount, 1u);
                if (mixedIndex < mixedCapacity)
                {
                    mixedSlotIds[mixedIndex] = slot;
                }
                else
                {
                    atomicAdd(overflowCount, 1u);
                }
            }
        }
        else
        {
            atomicAdd(overflowCount, 1u);  // best-effort drop
        }
    }
}

// ============================================================================
// Sorted-grid broad cull ("5th way") — tiled binning via sort (Green's method).
// Expand triangles into (cellKey, triId) incidences, sort by key so each cell's
// triangles are CONTIGUOUS, then generate pairs one block per mixed cell with
// the tool run staged in shared memory. Key = (cellId << 1) | isTool, so tissue
// entries sort before tool entries inside each cell and the scanned histogram
// gives every run boundary directly. No per-cell capacity caps -> no
// best-effort triangle drops. Dedup is either home-cell exactly-once emission
// (default; no hash table) or the shared pair-hash machinery (fallback flag).
// ============================================================================



} // namespace


namespace SofaGpuCollision::backend
{

bool computeSimpleHashProximityContacts(
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

    if (indexedSurfaceInvalid(firstSurface) || indexedSurfaceInvalid(secondSurface))
    {
        diagnostic = "Invalid indexed triangle surface (simple-hash path).";
        return false;
    }
    if (gridConfig.gridResolutionX == 0 || gridConfig.gridResolutionY == 0 || gridConfig.gridResolutionZ == 0 ||
        gridConfig.maxTissueTrianglesPerCell == 0 || gridConfig.maxToolTrianglesPerCell == 0 ||
        gridConfig.maxCandidatePairs == 0 ||
        gridConfig.gridMaxX <= gridConfig.gridMinX ||
        gridConfig.gridMaxY <= gridConfig.gridMinY ||
        gridConfig.gridMaxZ <= gridConfig.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration (simple-hash path).";
        return false;
    }

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(gridConfig.gridResolutionX) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionY) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 > std::numeric_limits<std::uint32_t>::max())
    {
        diagnostic = "Simple-hash path grid cell count out of range.";
        return false;
    }
    const std::uint32_t cellCount = static_cast<std::uint32_t>(cellCount64);

    // Table size: ~4 slots per input triangle (power of two). Each slot is a
    // full bucket here, so the low load factor keeps linear probing cheap.
    std::uint32_t tableSize = hashConfig.hashTableSize;
    if (tableSize == 0)
    {
        const std::uint64_t guess = static_cast<std::uint64_t>(firstSurface.triangleCount + secondSurface.triangleCount) * 4ull;
        const std::size_t candidate = nextPowerOfTwo(static_cast<std::size_t>(std::max<std::uint64_t>(1024ull, guess)));
        if (candidate > std::numeric_limits<std::uint32_t>::max()) { diagnostic = "Simple-hash table size out of range."; return false; }
        tableSize = static_cast<std::uint32_t>(candidate);
    }
    else
    {
        const std::size_t candidate = nextPowerOfTwo(tableSize);
        if (candidate > std::numeric_limits<std::uint32_t>::max()) { diagnostic = "Simple-hash table size out of range."; return false; }
        tableSize = static_cast<std::uint32_t>(candidate);
    }
    const std::uint32_t bucketCapacity = tableSize;  // KEY: each hash slot IS a bucket
    const std::uint64_t pairHashCount64 =
        static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(gridConfig.maxCandidatePairs) * 2u));
    if (pairHashCount64 == 0 || pairHashCount64 > std::numeric_limits<std::uint32_t>::max())
    {
        diagnostic = "Simple-hash pair-hash table size out of range.";
        return false;
    }
    const bool useCompactCandidatePairs =
        firstSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max() &&
        secondSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max();
    const std::uint32_t maxProbe = std::max(1u, hashConfig.maxProbe);

    auto& ws = simpleHashGridWorkspace();
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
        diagnostic = std::string("Simple-hash workspace alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += elapsedMillisecondsSince(allocStart);
        if (newlyAllocatedBytes > 0) executionStats->workspaceResizeCount += 1;
    }

    err = uploadSurfacesToWorkspace(ws, firstSurface, secondSurface, executionStats);
    if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }

    const BackendTriangleVertex* deviceFirstPositions = firstSurface.devicePositions != nullptr ? firstSurface.devicePositions : ws.firstPositions;
    const BackendTriangleVertex* deviceSecondPositions = secondSurface.devicePositions != nullptr ? secondSurface.devicePositions : ws.secondPositions;

    const DeviceDenseGridConfig dc = makeDeviceDenseGridConfig(gridConfig, static_cast<std::uint32_t>(pairHashCount64));

    constexpr std::uint32_t threads = 256;
    const std::uint32_t resetBlocks = (tableSize + threads - 1u) / threads;
    const std::uint32_t firstBlocks = (firstSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t secondBlocks = (secondSurface.triangleCount + threads - 1u) / threads;
    constexpr std::uint32_t kGenBlocks = 1024;
    const std::uint32_t pairHashCount = static_cast<std::uint32_t>(pairHashCount64);
    const std::uint32_t pairHashClearBlocks =
        std::max(1u, std::min(kGenBlocks, (pairHashCount + threads - 1u) / threads));

    const bool detailedProfiling = gridConfig.detailedProfiling;
    const bool totalTiming = proximityConfig.readContactCounter && !detailedProfiling;
    const bool measure = detailedProfiling || totalTiming;
    if (measure)
    {
        err = ws.ensureEvents();
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }

    std::uint32_t launchCount = 0;
    std::uint32_t memsetCount = 0;
    auto* deviceContacts = reinterpret_cast<DeviceProximityContact*>(ws.proximityContacts);

    // The whole per-frame kernel sequence on stream s (direct exec + graph
    // capture). fullClear=true does the one-time dedup-table memset.
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
        insertSimpleHashTrianglesKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, true, dc, tableSize, maxProbe, ws.cellKeys, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, ws.overflowCount, ws.probeOverflowCount);
        insertSimpleHashTrianglesKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, false, dc, tableSize, maxProbe, ws.cellKeys, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, ws.overflowCount, ws.probeOverflowCount);
        if (useCompactCandidatePairs)
            generateMixedHashCandidatePairsKernel<std::uint32_t><<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.compactCandidatePairs, ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        else
            generateMixedHashCandidatePairsKernel<std::uint64_t><<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.candidatePairs, ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        resetProximityCountersKernel<<<1, 1, 0, s>>>(ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount);
        featureBasedProximityKernel<<<kGenBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, deviceSecondPositions, ws.secondIndices, useCompactCandidatePairs ? nullptr : ws.candidatePairs, useCompactCandidatePairs ? ws.compactCandidatePairs : nullptr, ws.candidateCount, useCompactCandidatePairs, deviceContacts, ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount, proximityConfig.maxContacts, proximityConfig.contactDistance, proximityConfig.computeBarycentrics);
    };

    if (measure) { err = cudaEventRecord(ws.startEvent); if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; } }

    // CUDA-graph fast path (default ON; disable with SOFA_SIMPLE_HASH_CUDA_GRAPH=0).
    // Same machinery as the optimised hash path; falls back safely on any failure.
    static const int kGraphMode = []{ const char* e = std::getenv("SOFA_SIMPLE_HASH_CUDA_GRAPH"); return e ? std::atoi(e) : 1; }();
    const bool kGraphsEnabled = (kGraphMode != 0);
    bool ranViaGraph = false;
    const bool firstFrame = !(useCompactCandidatePairs ? ws.compactPairHashKeysInitialized : ws.pairHashKeysInitialized);
    if (kGraphsEnabled && !detailedProfiling)
    {
        const std::array<std::uint64_t, 6> graphSignature {
            (static_cast<std::uint64_t>(tableSize) << 32) | bucketCapacity,
            (static_cast<std::uint64_t>(firstSurface.triangleCount) << 32) | secondSurface.triangleCount,
            (static_cast<std::uint64_t>(proximityConfig.maxContacts) << 32) | pairHashCount,
            (useCompactCandidatePairs ? 1ull : 0ull) | (proximityConfig.computeBarycentrics ? 2ull : 0ull),
            floatSignatureBits(proximityConfig.contactDistance),
            0ull };
        ws.graph.invalidateIfChanged(graphSignature, newlyAllocatedBytes > 0);

        if (firstFrame)
        {
            launchAll(0, /*fullClear=*/true);
            ws.graph.instantiated = false;  // re-capture next (steady) frame
            ranViaGraph = true;
        }
        else
        {
            ws.graph.replay(graphSignature, [&](cudaStream_t s) { launchAll(s, /*fullClear=*/false); });
            ranViaGraph = true;
        }
    }
    if (!ranViaGraph)
    {
        launchAll(0, firstFrame);
    }
    launchCount += 7u;            // reset + insert x2 + generate + counters + FBP + touched-clear
    if (firstFrame) { memsetCount += 1u; launchCount -= 1u; }
    err = cudaGetLastError();
    if (err != cudaSuccess) { diagnostic = std::string("simple-hash launch: ") + cudaGetErrorString(err); return false; }

    float kernelMs = 0.0f;
    if (measure)
    {
        err = cudaEventRecord(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventSynchronize(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventElapsedTime(&kernelMs, ws.startEvent, ws.endEvent);
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }
    if (executionStats != nullptr)
    {
        executionStats->kernelLaunchCount += launchCount;
        executionStats->cudaMemsetCount += memsetCount;
        executionStats->gpuKernelMilliseconds += static_cast<double>(kernelMs);
        executionStats->featureBasedProximityKernelMilliseconds += 0.0;
    }

    // Optional readback (validation / contacts to host).
    std::uint32_t hostContactCount = 0, hostUnique = 0, hostRaw = 0, hostOverflow = 0, hostProbe = 0, hostMixed = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* p = ws.countersHostPinned;
        cudaMemcpyAsync(p + 0, ws.candidateCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 1, ws.rawCandidateCount,     sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 2, ws.overflowCount,         sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 3, ws.probeOverflowCount,    sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 5, ws.proximityContactCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 6, ws.proximityVfCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 7, ws.proximityFvCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 8, ws.proximityEeCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 9, ws.mixedBucketCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostUnique = p[0]; hostRaw = p[1]; hostOverflow = p[2]; hostProbe = p[3];
        hostContactCount = p[5]; hostVf = p[6]; hostFv = p[7]; hostEe = p[8]; hostMixed = p[9];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 9u * sizeof(std::uint32_t);
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
            hashStats->occupiedSlotCount = std::min(hostMixed, bucketCapacity);
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
// computeSortedGridProximityContacts ("5th way" — sorted-grid / tiled binning)
// ----------------------------------------------------------------------------
// Green's sorted-particle-grid method adapted to triangles: expand (cellKey,
// triId) incidences (fused per-key histogram + per-triangle AABB store), sort
// them by key so each cell's triangles are contiguous, derive every run
// boundary from the scanned histogram, then generate pairs one block per mixed
// cell with the tool run staged in shared memory. Two toggles:
//   - sort engine: counting sort (default, CUB-free; scatter via bin cursors)
//     or cub::DeviceRadixSort over the 0xff-padded incidence buffer;
//   - dedup: home-cell exactly-once emission (default, no hash table; also
//     pre-culls AABB-disjoint pairs, so uniquePairCount may be SMALLER than the
//     other ways while contacts stay identical) or the shared pair-hash
//     (reproduces the other ways' candidate set exactly).
// No per-cell capacity caps -> no best-effort triangle drops (the incidence
// buffer overflow is counted and never triggers on the test scenes). The whole
// steady-state sequence replays as a CUDA graph (SOFA_SORTED_GRID_CUDA_GRAPH).
// ============================================================================


} // namespace SofaGpuCollision::backend
