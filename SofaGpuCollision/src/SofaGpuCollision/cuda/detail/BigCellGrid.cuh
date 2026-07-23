// BigCellGrid.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking). Added 2026-07-12.
//
// Way 6: big-cell / small-cell FUSED broad cull + narrow phase.
// The small cells are the existing dense-grid cells; a big cell groups
// factor^3 of them (factor <= 4, so a local small-cell id fits in 6 bits).
// Build: a per-big-cell CSR table of (triangleId, localSmallCellId) entries,
// count -> scan -> fill (the "hash" of a big cell is its own id — a perfect,
// collision-free direct index; the scan input is factor^3 smaller than way 5's,
// so the single-block scan is cheap here). Optionally (useHashTableBuild) the
// table is built instead as a literal per-big-cell open-addressing hash
// multi-map, kept toggleable so the two build strategies can be A/B'd.
// Consume: ONE block per mixed big cell stages the tool side into shared
// memory (ids + AABBs + all 9 vertex floats), organizes it into per-small-cell
// runs with an in-shared 64-bin counting sort, then sweeps the tissue entries:
// inflated-AABB overlap -> home-cell ownership at SMALL-cell granularity (the
// same exactly-once rule as way 5, so the surviving pair set is identical) ->
// the identical FBP closest-feature math (fbpComputeClosestFeatureContact)
// inline — no intermediate pair list, contacts go straight to one output array.

namespace
{

// Big-cell geometry derived from the dense-grid config. factor is a power of
// two in [1, 4]; factorShift = log2(factor) so the small->big mapping is shifts.
struct DeviceBigCellGridConfig
{
    std::uint32_t factor;
    std::uint32_t factorShift;
    std::uint32_t bigResX;
    std::uint32_t bigResY;
    std::uint32_t bigResZ;
};

// Opt-in fused-kernel instrumentation. Major-phase cycle fields sum one
// wall-clock sample per participating block/tile; filter cycle fields sum
// participating-thread samples. Neither is a replacement for CUDA-event
// elapsed time because blocks and threads execute concurrently.
struct DeviceBigCellFusedProfile
{
    unsigned long long bigCellIterations;
    unsigned long long tileIterations;
    unsigned long long toolEntriesStaged;
    unsigned long long tissueEntriesVisited;
    unsigned long long smallCellPairVisits;
    unsigned long long inflatedAabbRejects;
    unsigned long long homeCellRejects;
    unsigned long long rawAabbRejects;
    unsigned long long fbpCalls;
    unsigned long long fbpNoContact;
    unsigned long long tileSetupBlockCycles;
    unsigned long long binPrefixBlockCycles;
    unsigned long long toolGatherBlockCycles;
    unsigned long long tissueSweepBlockCycles;
    unsigned long long inflatedAabbThreadCycles;
    unsigned long long homeCellThreadCycles;
    unsigned long long rawAabbThreadCycles;
    unsigned long long fbpThreadCycles;
};

constexpr std::size_t kBigCellDetailedEventCount = 11;

__device__ __forceinline__ std::uint32_t bigCellIdOfSmall(
    const int x, const int y, const int z, const DeviceBigCellGridConfig& bigc)
{
    const std::uint32_t bx = static_cast<std::uint32_t>(x) >> bigc.factorShift;
    const std::uint32_t by = static_cast<std::uint32_t>(y) >> bigc.factorShift;
    const std::uint32_t bz = static_cast<std::uint32_t>(z) >> bigc.factorShift;
    return bx + by * bigc.bigResX + bz * bigc.bigResX * bigc.bigResY;
}

// Local small-cell id inside its big cell, packed into 6 bits (factor <= 4).
__device__ __forceinline__ std::uint32_t bigCellLocalId(
    const int x, const int y, const int z, const DeviceBigCellGridConfig& bigc)
{
    const std::uint32_t mask = bigc.factor - 1u;
    const std::uint32_t lx = static_cast<std::uint32_t>(x) & mask;
    const std::uint32_t ly = static_cast<std::uint32_t>(y) & mask;
    const std::uint32_t lz = static_cast<std::uint32_t>(z) & mask;
    return lx + (ly << bigc.factorShift) + (lz << (2u * bigc.factorShift));
}

struct BigCellWorkspace
{
    std::uint32_t* hist { nullptr };              // binCount = 2*bigCellCount
    std::uint32_t* starts { nullptr };            // binCount + 1 (exclusive scan)
    std::uint32_t* cursors { nullptr };           // binCount (fill write cursors)
    std::uint32_t* entriesPacked { nullptr };     // CSR entries: (triId << 6) | local
    std::uint32_t* hashSlots { nullptr };         // hash-build toggle only
    DeviceAabb* firstAabbs { nullptr };           // per-triangle inflated AABBs
    DeviceAabb* secondAabbs { nullptr };
    std::uint32_t* mixedBigCellIds { nullptr };
    std::uint32_t* entryTotalCount { nullptr };
    std::uint32_t* entryOverflowCount { nullptr };
    std::uint32_t* mixedBigCellCount { nullptr };
    std::uint32_t* pairsTestedCount { nullptr };  // pairs surviving AABB+home-cell
    std::uint32_t* buildOverflowCount { nullptr };// hash-build slot-region overflow
    std::uint32_t* sharedSpillCount { nullptr };// shared-build entries that fell back to the direct global path
    DeviceBigCellFusedProfile* fusedProfile { nullptr };
    void* proximityContacts { nullptr };
    std::uint32_t* proximityContactCount { nullptr };
    std::uint32_t* proximityOverflowCount { nullptr };
    std::uint32_t* proximityVfCount { nullptr };
    std::uint32_t* proximityFvCount { nullptr };
    std::uint32_t* proximityEeCount { nullptr };
    std::uint32_t* firstIndices { nullptr };
    std::uint32_t* secondIndices { nullptr };
    BackendTriangleVertex* firstPositions { nullptr };   // host-input fallback only
    BackendTriangleVertex* secondPositions { nullptr };
    std::uint32_t* countersHostPinned { nullptr };       // 12 uint32, pinned

    std::size_t histCapacity { 0 };
    std::size_t startsCapacity { 0 };
    std::size_t cursorsCapacity { 0 };
    std::size_t entryCapacity { 0 };
    std::size_t hashSlotCapacity { 0 };
    std::size_t firstAabbCapacity { 0 };
    std::size_t secondAabbCapacity { 0 };
    std::size_t mixedBigCellIdCapacity { 0 };
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
    cudaEvent_t detailedEvents[kBigCellDetailedEventCount] {};
    bool detailedEventsReady { false };
    bool firstFrameDone { false };
    CudaGraphReplayer graph;

    ~BigCellWorkspace() { release(); }

    void release()
    {
        cudaFree(hist); cudaFree(starts); cudaFree(cursors);
        cudaFree(entriesPacked); cudaFree(hashSlots);
        cudaFree(firstAabbs); cudaFree(secondAabbs); cudaFree(mixedBigCellIds);
        cudaFree(entryTotalCount); cudaFree(entryOverflowCount); cudaFree(mixedBigCellCount);
        cudaFree(pairsTestedCount); cudaFree(buildOverflowCount); cudaFree(sharedSpillCount);
        cudaFree(fusedProfile);
        cudaFree(proximityContacts);
        cudaFree(proximityContactCount); cudaFree(proximityOverflowCount);
        cudaFree(proximityVfCount); cudaFree(proximityFvCount); cudaFree(proximityEeCount);
        cudaFree(firstIndices); cudaFree(secondIndices);
        cudaFree(firstPositions); cudaFree(secondPositions);
        cudaFreeHost(countersHostPinned);
        if (eventsReady) { cudaEventDestroy(startEvent); cudaEventDestroy(endEvent); }
        if (detailedEventsReady) for (cudaEvent_t event : detailedEvents) cudaEventDestroy(event);
        graph.release();
        hist = nullptr; starts = nullptr; cursors = nullptr;
        entriesPacked = nullptr; hashSlots = nullptr;
        firstAabbs = nullptr; secondAabbs = nullptr; mixedBigCellIds = nullptr;
        entryTotalCount = nullptr; entryOverflowCount = nullptr; mixedBigCellCount = nullptr;
        pairsTestedCount = nullptr; buildOverflowCount = nullptr; sharedSpillCount = nullptr;
        fusedProfile = nullptr;
        proximityContacts = nullptr;
        proximityContactCount = nullptr; proximityOverflowCount = nullptr;
        proximityVfCount = nullptr; proximityFvCount = nullptr; proximityEeCount = nullptr;
        firstIndices = nullptr; secondIndices = nullptr;
        firstPositions = nullptr; secondPositions = nullptr;
        countersHostPinned = nullptr;
        startEvent = nullptr; endEvent = nullptr; eventsReady = false;
        for (cudaEvent_t& event : detailedEvents) event = nullptr; detailedEventsReady = false;
        histCapacity = 0; startsCapacity = 0; cursorsCapacity = 0;
        entryCapacity = 0; hashSlotCapacity = 0;
        firstAabbCapacity = 0; secondAabbCapacity = 0; mixedBigCellIdCapacity = 0;
        contactCapacity = 0;
        firstIndexCapacity = 0; secondIndexCapacity = 0;
        firstPositionCapacity = 0; secondPositionCapacity = 0;
        firstSurfaceId = 0; secondSurfaceId = 0;
        firstTopologyVersion = ~0ull; secondTopologyVersion = ~0ull;
        firstFrameDone = false;
    }

    cudaError_t ensureEvents()
    {
        if (eventsReady) return cudaSuccess;
        cudaError_t err = cudaEventCreate(&startEvent);
        if (err == cudaSuccess) err = cudaEventCreate(&endEvent);
        eventsReady = (err == cudaSuccess);
        return err;
    }

    cudaError_t ensureDetailedEvents()
    {
        if (detailedEventsReady) return cudaSuccess;
        std::size_t created = 0;
        cudaError_t err = cudaSuccess;
        while (created < kBigCellDetailedEventCount)
        {
            err = cudaEventCreate(&detailedEvents[created]);
            if (err != cudaSuccess) break;
            ++created;
        }
        if (err != cudaSuccess)
        {
            for (std::size_t i = 0; i < created; ++i)
            {
                cudaEventDestroy(detailedEvents[i]);
                detailedEvents[i] = nullptr;
            }
            return err;
        }
        detailedEventsReady = true;
        return cudaSuccess;
    }

    cudaError_t ensure(
        const std::size_t binCount,
        const std::size_t bigCellCount,
        const std::size_t entryCapacityRequired,
        const std::size_t hashSlotsRequired,     // 0 on the CSR build
        const std::size_t maxContacts,
        const std::size_t contactElementBytes,
        const std::size_t firstIndexCount,
        const std::size_t secondIndexCount,
        const std::size_t firstVertexCount,
        const std::size_t secondVertexCount,
        const std::size_t firstTriangleCount,
        const std::size_t secondTriangleCount,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensureDeviceArray(hist, histCapacity, binCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(starts, startsCapacity, binCount + 1u, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(cursors, cursorsCapacity, binCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(entriesPacked, entryCapacity, entryCapacityRequired, newlyAllocatedBytes);
        if (err == cudaSuccess && hashSlotsRequired > 0) err = ensureDeviceArray(hashSlots, hashSlotCapacity, hashSlotsRequired, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(firstAabbs, firstAabbCapacity, firstTriangleCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(secondAabbs, secondAabbCapacity, secondTriangleCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(mixedBigCellIds, mixedBigCellIdCapacity, bigCellCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(firstIndices, firstIndexCapacity, firstIndexCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(secondIndices, secondIndexCapacity, secondIndexCount, newlyAllocatedBytes);
        if (err == cudaSuccess && firstVertexCount > 0) err = ensureDeviceArray(firstPositions, firstPositionCapacity, firstVertexCount, newlyAllocatedBytes);
        if (err == cudaSuccess && secondVertexCount > 0) err = ensureDeviceArray(secondPositions, secondPositionCapacity, secondVertexCount, newlyAllocatedBytes);
        if (err == cudaSuccess && (proximityContacts == nullptr || contactCapacity < maxContacts))
        {
            cudaFree(proximityContacts); proximityContacts = nullptr;
            void* p = nullptr;
            err = cudaMalloc(&p, maxContacts * contactElementBytes);
            if (err == cudaSuccess) { proximityContacts = p; contactCapacity = maxContacts; newlyAllocatedBytes += maxContacts * contactElementBytes; }
        }
        if (err == cudaSuccess && entryTotalCount == nullptr)      err = cudaMallocTracked(entryTotalCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && entryOverflowCount == nullptr)   err = cudaMallocTracked(entryOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && mixedBigCellCount == nullptr)    err = cudaMallocTracked(mixedBigCellCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && pairsTestedCount == nullptr)     err = cudaMallocTracked(pairsTestedCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && buildOverflowCount == nullptr)   err = cudaMallocTracked(buildOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && sharedSpillCount == nullptr)  err = cudaMallocTracked(sharedSpillCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && fusedProfile == nullptr)      err = cudaMallocTracked(fusedProfile, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityContactCount == nullptr)  err = cudaMallocTracked(proximityContactCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityOverflowCount == nullptr) err = cudaMallocTracked(proximityOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityVfCount == nullptr)       err = cudaMallocTracked(proximityVfCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityFvCount == nullptr)       err = cudaMallocTracked(proximityFvCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityEeCount == nullptr)       err = cudaMallocTracked(proximityEeCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && countersHostPinned == nullptr)
        {
            err = cudaMallocHost(reinterpret_cast<void**>(&countersHostPinned), 12u * sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += 12u * sizeof(std::uint32_t);
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

BigCellWorkspace& bigCellWorkspace()
{
    static BigCellWorkspace workspace;
    return workspace;
}

// One thread per triangle: store its inflated AABB and bump the per-(bigCell,
// side) histogram once for every overlapped SMALL cell (an entry = one
// (smallCell, triangle) incidence, same granularity as way 5's expansion).
__global__ void countBigCellEntriesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    std::uint32_t* hist,
    DeviceAabb* triangleAabbs,
    std::uint32_t* entryTotalCount)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount) return;

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
    triangleAabbs[triangleId] = aabb;

    int3 cellMin, cellMax;
    if (!denseGridCellSpan(aabb, config, cellMin, cellMax)) return;

    const std::uint32_t span =
        static_cast<std::uint32_t>(cellMax.x - cellMin.x + 1) *
        static_cast<std::uint32_t>(cellMax.y - cellMin.y + 1) *
        static_cast<std::uint32_t>(cellMax.z - cellMin.z + 1);
    atomicAdd(entryTotalCount, span);

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    {
        const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
        atomicAdd(&hist[bin], 1u);
    }
}

// Second walk: scatter each entry to its exact CSR slot (cursor = scanned
// histogram). CSR positions are exact, so the only overflow is total entries
// exceeding the buffer capacity.
__global__ void fillBigCellEntriesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t entryCapacity,
    std::uint32_t* cursors,
    std::uint32_t* entriesPacked,
    std::uint32_t* entryOverflowCount)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount) return;

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);

    int3 cellMin, cellMax;
    if (!denseGridCellSpan(aabb, config, cellMin, cellMax)) return;

    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    {
        const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
        const std::uint32_t pos = atomicAdd(&cursors[bin], 1u);
        if (pos >= entryCapacity)
        {
            atomicAdd(entryOverflowCount, 1u);
            continue;
        }
        entriesPacked[pos] = (triangleId << 6u) | bigCellLocalId(x, y, z, bigc);
    }
}

// ============================================================================
// Shared-memory-privatized CSR build ("populate in shared, merge later").
// Each block owns a FIXED chunk of triangles (spatially blind — the block/
// triangle assignment cannot know spatial grouping before the structure
// exists). The block builds its chunk's contribution in shared memory, then
// merges into the global histogram / CSR entry array with per-bin instead of
// per-entry global atomics. Two organizations, kept as an A/B:
//   mode 1: shared HASH TABLE — insert-or-count keyed by bin; entries staged
//           alongside with their within-bin offset; merge = one global
//           atomicAdd per occupied slot, then scatter into reserved ranges.
//   mode 2: shared SORTED LIST — stage (bin, entry) pairs, in-shared bitonic
//           sort by bin, detect runs, one global reservation per run, the
//           run-start thread writes its run.
// Anything that overflows the shared structures falls back to the direct
// global path (counted in sharedSpillCount) — the entry multiset is identical
// either way, so contacts are unchanged.
// ============================================================================

constexpr std::uint32_t kSharedBinTableSize = 1024;   // pow2; keys+counts = 8 KB
constexpr std::uint32_t kSharedStageCapacity = 2048;  // staged entries per block
constexpr std::uint32_t kSharedBinEmpty = 0xffffffffu;

// Insert-or-find a bin in the block's shared hash table. Returns the slot, or
// -1 when the bounded probe fails (caller falls back to the global path).
__device__ __forceinline__ int bigCellSharedBinSlot(
    std::uint32_t* sKeys, const std::uint32_t bin)
{
    std::uint32_t h = (bin * 2654435761u) & (kSharedBinTableSize - 1u);
    for (std::uint32_t probe = 0; probe < 32u; ++probe)
    {
        const std::uint32_t prev = atomicCAS(&sKeys[h], kSharedBinEmpty, bin);
        if (prev == kSharedBinEmpty || prev == bin) return static_cast<int>(h);
        h = (h + 1u) & (kSharedBinTableSize - 1u);
    }
    return -1;
}

// In-shared bitonic sort over a power-of-two array (the classic i^j network).
// All threads iterate the full index space, so the __syncthreads are uniform.
template <class KeyT>
__device__ void bigCellBitonicSortShared(KeyT* keys, const std::uint32_t n)
{
    for (std::uint32_t k = 2; k <= n; k <<= 1)
    {
        for (std::uint32_t j = k >> 1; j > 0; j >>= 1)
        {
            for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x)
            {
                const std::uint32_t ixj = i ^ j;
                if (ixj > i)
                {
                    const KeyT a = keys[i];
                    const KeyT b = keys[ixj];
                    const bool ascending = ((i & k) == 0u);
                    if ((a > b) == ascending)
                    {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Mode 1 count: shared hash table of per-bin counts, merged with one global
// atomicAdd per occupied slot. Also stores the per-triangle AABBs (this kernel
// replaces countBigCellEntriesKernel when sharedBuildMode==1).
__global__ void countBigCellEntriesSharedHashKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    std::uint32_t* hist,
    DeviceAabb* triangleAabbs,
    std::uint32_t* entryTotalCount,
    std::uint32_t* sharedSpillCount)
{
    __shared__ std::uint32_t sKeys[kSharedBinTableSize];
    __shared__ std::uint32_t sCounts[kSharedBinTableSize];
    for (std::uint32_t i = threadIdx.x; i < kSharedBinTableSize; i += blockDim.x)
    {
        sKeys[i] = kSharedBinEmpty;
        sCounts[i] = 0u;
    }
    __syncthreads();

    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId < triangleCount)
    {
        const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
        const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
        triangleAabbs[triangleId] = aabb;
        int3 cellMin, cellMax;
        if (denseGridCellSpan(aabb, config, cellMin, cellMax))
        {
            const std::uint32_t span =
                static_cast<std::uint32_t>(cellMax.x - cellMin.x + 1) *
                static_cast<std::uint32_t>(cellMax.y - cellMin.y + 1) *
                static_cast<std::uint32_t>(cellMax.z - cellMin.z + 1);
            atomicAdd(entryTotalCount, span);
            for (int z = cellMin.z; z <= cellMax.z; ++z)
            for (int y = cellMin.y; y <= cellMax.y; ++y)
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
                const int slot = bigCellSharedBinSlot(sKeys, bin);
                if (slot >= 0)
                {
                    atomicAdd(&sCounts[slot], 1u);
                }
                else
                {
                    atomicAdd(&hist[bin], 1u);
                    atomicAdd(sharedSpillCount, 1u);
                }
            }
        }
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < kSharedBinTableSize; i += blockDim.x)
    {
        if (sKeys[i] != kSharedBinEmpty && sCounts[i] > 0u)
        {
            atomicAdd(&hist[sKeys[i]], sCounts[i]);
        }
    }
}

// Mode 1 fill: stage entries in shared with their within-slot offsets, reserve
// each occupied bin's range with ONE global atomicAdd, then scatter. Ordering
// invariant: a slot count is incremented ONLY for a successfully staged entry,
// so reserved ranges never contain gaps.
__global__ void fillBigCellEntriesSharedHashKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t entryCapacity,
    std::uint32_t* cursors,
    std::uint32_t* entriesPacked,
    std::uint32_t* entryOverflowCount,
    std::uint32_t* sharedSpillCount)
{
    __shared__ std::uint32_t sKeys[kSharedBinTableSize];
    __shared__ std::uint32_t sCounts[kSharedBinTableSize];
    __shared__ std::uint32_t sBase[kSharedBinTableSize];
    __shared__ std::uint32_t sPacked[kSharedStageCapacity];
    __shared__ std::uint32_t sSlotOff[kSharedStageCapacity];  // (slot << 16) | withinSlotOffset
    __shared__ std::uint32_t sStage;
    for (std::uint32_t i = threadIdx.x; i < kSharedBinTableSize; i += blockDim.x)
    {
        sKeys[i] = kSharedBinEmpty;
        sCounts[i] = 0u;
    }
    if (threadIdx.x == 0) sStage = 0u;
    __syncthreads();

    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId < triangleCount)
    {
        const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
        const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
        int3 cellMin, cellMax;
        if (denseGridCellSpan(aabb, config, cellMin, cellMax))
        {
            for (int z = cellMin.z; z <= cellMax.z; ++z)
            for (int y = cellMin.y; y <= cellMax.y; ++y)
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
                const std::uint32_t packed = (triangleId << 6u) | bigCellLocalId(x, y, z, bigc);
                const std::uint32_t pos = atomicAdd(&sStage, 1u);
                int slot = -1;
                if (pos < kSharedStageCapacity)
                {
                    slot = bigCellSharedBinSlot(sKeys, bin);
                    if (slot >= 0)
                    {
                        const std::uint32_t off = atomicAdd(&sCounts[slot], 1u);
                        sPacked[pos] = packed;
                        sSlotOff[pos] = (static_cast<std::uint32_t>(slot) << 16u) | (off & 0xffffu);
                    }
                    else
                    {
                        sSlotOff[pos] = kSharedBinEmpty;  // hole: staged position unused
                    }
                }
                if (slot < 0)
                {
                    // Direct global fallback (staging full or probe failure).
                    const std::uint32_t gpos = atomicAdd(&cursors[bin], 1u);
                    if (gpos < entryCapacity) entriesPacked[gpos] = packed;
                    else atomicAdd(entryOverflowCount, 1u);
                    atomicAdd(sharedSpillCount, 1u);
                }
            }
        }
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < kSharedBinTableSize; i += blockDim.x)
    {
        if (sKeys[i] != kSharedBinEmpty && sCounts[i] > 0u)
        {
            sBase[i] = atomicAdd(&cursors[sKeys[i]], sCounts[i]);
        }
    }
    __syncthreads();
    const std::uint32_t staged = min(sStage, kSharedStageCapacity);
    for (std::uint32_t i = threadIdx.x; i < staged; i += blockDim.x)
    {
        const std::uint32_t so = sSlotOff[i];
        if (so == kSharedBinEmpty) continue;
        const std::uint32_t pos = sBase[so >> 16u] + (so & 0xffffu);
        if (pos < entryCapacity) entriesPacked[pos] = sPacked[i];
        else atomicAdd(entryOverflowCount, 1u);
    }
}

// Mode 2 count: stage bins, bitonic-sort them in shared, merge one global
// atomicAdd per run (the run-start thread walks its run).
__global__ void countBigCellEntriesSharedSortKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    std::uint32_t* hist,
    DeviceAabb* triangleAabbs,
    std::uint32_t* entryTotalCount,
    std::uint32_t* sharedSpillCount)
{
    __shared__ std::uint32_t sBins[kSharedStageCapacity];
    __shared__ std::uint32_t sStage;
    if (threadIdx.x == 0) sStage = 0u;
    __syncthreads();

    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId < triangleCount)
    {
        const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
        const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
        triangleAabbs[triangleId] = aabb;
        int3 cellMin, cellMax;
        if (denseGridCellSpan(aabb, config, cellMin, cellMax))
        {
            const std::uint32_t span =
                static_cast<std::uint32_t>(cellMax.x - cellMin.x + 1) *
                static_cast<std::uint32_t>(cellMax.y - cellMin.y + 1) *
                static_cast<std::uint32_t>(cellMax.z - cellMin.z + 1);
            atomicAdd(entryTotalCount, span);
            for (int z = cellMin.z; z <= cellMax.z; ++z)
            for (int y = cellMin.y; y <= cellMax.y; ++y)
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
                const std::uint32_t pos = atomicAdd(&sStage, 1u);
                if (pos < kSharedStageCapacity)
                {
                    sBins[pos] = bin;
                }
                else
                {
                    atomicAdd(&hist[bin], 1u);
                    atomicAdd(sharedSpillCount, 1u);
                }
            }
        }
    }
    __syncthreads();
    const std::uint32_t staged = min(sStage, kSharedStageCapacity);
    for (std::uint32_t i = threadIdx.x + staged; i < kSharedStageCapacity; i += blockDim.x)
    {
        sBins[i] = kSharedBinEmpty;  // sorts to the tail (real bins < 2^31)
    }
    __syncthreads();
    bigCellBitonicSortShared(sBins, kSharedStageCapacity);
    for (std::uint32_t i = threadIdx.x; i < staged; i += blockDim.x)
    {
        const std::uint32_t bin = sBins[i];
        if (bin == kSharedBinEmpty) continue;
        if (i > 0 && sBins[i - 1u] == bin) continue;  // not a run start
        std::uint32_t j = i + 1u;
        while (j < staged && sBins[j] == bin) ++j;
        atomicAdd(&hist[bin], j - i);
    }
}

// Mode 2 fill: stage (bin, stagingIndex) 64-bit keys + packed payloads, sort by
// bin, then the run-start thread reserves the run's range with ONE global
// atomicAdd and writes the whole run (consecutive global positions).
__global__ void fillBigCellEntriesSharedSortKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t entryCapacity,
    std::uint32_t* cursors,
    std::uint32_t* entriesPacked,
    std::uint32_t* entryOverflowCount,
    std::uint32_t* sharedSpillCount)
{
    __shared__ unsigned long long sSortKeys[kSharedStageCapacity];  // (bin << 32) | stagingIndex
    __shared__ std::uint32_t sPacked[kSharedStageCapacity];
    __shared__ std::uint32_t sStage;
    if (threadIdx.x == 0) sStage = 0u;
    __syncthreads();

    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId < triangleCount)
    {
        const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
        const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
        int3 cellMin, cellMax;
        if (denseGridCellSpan(aabb, config, cellMin, cellMax))
        {
            for (int z = cellMin.z; z <= cellMax.z; ++z)
            for (int y = cellMin.y; y <= cellMax.y; ++y)
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
                const std::uint32_t packed = (triangleId << 6u) | bigCellLocalId(x, y, z, bigc);
                const std::uint32_t pos = atomicAdd(&sStage, 1u);
                if (pos < kSharedStageCapacity)
                {
                    sPacked[pos] = packed;
                    sSortKeys[pos] = (static_cast<unsigned long long>(bin) << 32) | pos;
                }
                else
                {
                    const std::uint32_t gpos = atomicAdd(&cursors[bin], 1u);
                    if (gpos < entryCapacity) entriesPacked[gpos] = packed;
                    else atomicAdd(entryOverflowCount, 1u);
                    atomicAdd(sharedSpillCount, 1u);
                }
            }
        }
    }
    __syncthreads();
    const std::uint32_t staged = min(sStage, kSharedStageCapacity);
    for (std::uint32_t i = threadIdx.x + staged; i < kSharedStageCapacity; i += blockDim.x)
    {
        sSortKeys[i] = ~0ull;
    }
    __syncthreads();
    bigCellBitonicSortShared(sSortKeys, kSharedStageCapacity);
    for (std::uint32_t i = threadIdx.x; i < staged; i += blockDim.x)
    {
        const std::uint32_t bin = static_cast<std::uint32_t>(sSortKeys[i] >> 32);
        if (bin == kSharedBinEmpty) continue;
        if (i > 0 && static_cast<std::uint32_t>(sSortKeys[i - 1u] >> 32) == bin) continue;
        std::uint32_t j = i + 1u;
        while (j < staged && static_cast<std::uint32_t>(sSortKeys[j] >> 32) == bin) ++j;
        const std::uint32_t runLen = j - i;
        const std::uint32_t base = atomicAdd(&cursors[bin], runLen);
        for (std::uint32_t r = 0; r < runLen; ++r)
        {
            const std::uint32_t pos = base + r;
            const std::uint32_t stagingIndex = static_cast<std::uint32_t>(sSortKeys[i + r] & 0xffffffffull);
            if (pos < entryCapacity) entriesPacked[pos] = sPacked[stagingIndex];
            else atomicAdd(entryOverflowCount, 1u);
        }
    }
}

constexpr std::uint32_t kBigCellEmptySlot = 0xffffffffu;
// (never collides with a real packed entry: triId < 2^26-1 is enforced by the
// driver, so (triId << 6) | local < 0xffffffff)

// Hash-build toggle: reset every slot to empty. A fill kernel, not a
// cudaMemsetAsync — see the WSL2 copy-engine ordering gotcha on
// padSortedGridKeysKernel (SortedGrid.cuh).
__global__ void clearBigCellHashSlotsKernel(
    std::uint32_t* hashSlots,
    const std::uint32_t totalSlots)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalSlots; i += stride)
    {
        hashSlots[i] = kBigCellEmptySlot;
    }
}

// Hash-build toggle: the literal per-big-cell open-addressing hash multi-map.
// Each (big cell, side) owns a fixed slot region; an entry probes linearly from
// hash(localSmallCellId) claiming empty slots with atomicCAS. Also stores the
// per-triangle AABBs and maintains the histogram (mixed-big-cell detection and
// stats), replacing the CSR route's count/scan/fill entirely.
__global__ void insertBigCellHashEntriesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t slotsPerBigCell,   // power of two
    std::uint32_t* hashSlots,
    std::uint32_t* hist,
    DeviceAabb* triangleAabbs,
    std::uint32_t* entryTotalCount,
    std::uint32_t* buildOverflowCount)
{
    const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleId >= triangleCount) return;

    const DeviceTriangle triangle = indexedTriangleAt(positions, triangleIndices, triangleId);
    const DeviceAabb aabb = triangleAabb(triangle, config.contactDistance);
    triangleAabbs[triangleId] = aabb;

    int3 cellMin, cellMax;
    if (!denseGridCellSpan(aabb, config, cellMin, cellMax)) return;

    const std::uint32_t slotMask = slotsPerBigCell - 1u;
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    {
        const std::uint32_t bin = (bigCellIdOfSmall(x, y, z, bigc) << 1) | (isTool ? 1u : 0u);
        const std::uint32_t local = bigCellLocalId(x, y, z, bigc);
        const std::uint32_t packed = (triangleId << 6u) | local;
        const std::uint32_t regionBase = bin * slotsPerBigCell;
        const std::uint32_t h = (local * 2654435761u) & slotMask;
        bool inserted = false;
        for (std::uint32_t probe = 0; probe < slotsPerBigCell; ++probe)
        {
            const std::uint32_t slot = regionBase + ((h + probe) & slotMask);
            const std::uint32_t prev = atomicCAS(&hashSlots[slot], kBigCellEmptySlot, packed);
            if (prev == kBigCellEmptySlot)
            {
                inserted = true;
                break;
            }
        }
        if (inserted)
        {
            atomicAdd(&hist[bin], 1u);
            atomicAdd(entryTotalCount, 1u);
        }
        else
        {
            atomicAdd(buildOverflowCount, 1u);
        }
    }
}

// The way-6 centerpiece: one block per mixed big cell, generation and narrow
// phase fused. Per tool tile: stage packed entries -> in-shared 64-bin counting
// sort by local small cell -> gather tool AABBs + all 3 vertices from global
// ONCE per big cell -> sweep the tissue entries against the shared runs.
// Dedup/pre-cull is way 5's home-cell rule at SMALL-cell granularity, so the
// surviving pair set is provably identical to way 5's; each survivor runs the
// identical FBP math and appends straight to the single contact list.
__device__ __forceinline__ bool bigCellRawAabbGapExceeds(
    const DeviceAabb& a, const DeviceAabb& b, const float thresholdSq)
{
    const float gx = fmaxf(0.0f, fmaxf(a.minX - b.maxX, b.minX - a.maxX));
    const float gy = fmaxf(0.0f, fmaxf(a.minY - b.maxY, b.minY - a.maxY));
    const float gz = fmaxf(0.0f, fmaxf(a.minZ - b.maxZ, b.minZ - a.maxZ));
    return gx * gx + gy * gy + gz * gz > thresholdSq;
}

__global__ void fusedBigCellNarrowKernel(
    const std::uint32_t* __restrict__ starts,
    const std::uint32_t* __restrict__ entriesPacked,
    const std::uint32_t* __restrict__ hashSlots,      // hash-build source (else unused)
    const std::uint32_t slotsPerBigCell,
    const bool useHashBuild,
    const std::uint32_t* __restrict__ mixedBigCellIds,
    const std::uint32_t* mixedBigCellCount,
    const DeviceAabb* __restrict__ tissueAabbs,
    const BackendTriangleVertex* __restrict__ tissuePositions,
    const std::uint32_t* __restrict__ tissueIndices,
    const BackendTriangleVertex* __restrict__ toolPositions,
    const std::uint32_t* __restrict__ toolIndices,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t toolTile,             // <= 256, runtime-chunked
    const std::uint32_t entryCapacity,        // clamps CSR runs when the count pass overflowed the buffer
    DeviceProximityContact* __restrict__ contacts,
    std::uint32_t* pairsTestedCount,
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount,
    std::uint32_t* vfCount,
    std::uint32_t* fvCount,
    std::uint32_t* eeCount,
    const std::uint32_t maxContacts,
    const float contactDistance,
    const bool computeBarycentrics)
{
    __shared__ std::uint32_t sToolIds[256];       // sorted by local small cell
    __shared__ DeviceAabb sToolAabbs[256];
    __shared__ float3 sToolVerts[256 * 3];
    __shared__ std::uint32_t sScratchPacked[256]; // unsorted staging
    __shared__ std::uint32_t sRunStart[65];       // per-local run begin (end = next begin)
    __shared__ std::uint32_t sBinCursor[64];      // histogram, then scatter cursors
    __shared__ std::uint32_t sTileCount;          // staged entries this chunk (hash mode skips empties)

    const float distThreshSq = contactDistance * contactDistance;
    const std::uint32_t mixedCount = *mixedBigCellCount;
    for (std::uint32_t m = blockIdx.x; m < mixedCount; m += gridDim.x)
    {
        const std::uint32_t bigCell = mixedBigCellIds[m];
        const std::uint32_t bx = bigCell % bigc.bigResX;
        const std::uint32_t rem = bigCell / bigc.bigResX;
        const std::uint32_t by = rem % bigc.bigResY;
        const std::uint32_t bz = rem / bigc.bigResY;
        const int baseX = static_cast<int>(bx << bigc.factorShift);
        const int baseY = static_cast<int>(by << bigc.factorShift);
        const int baseZ = static_cast<int>(bz << bigc.factorShift);
        // Entry sources. CSR: contiguous runs from the scanned histogram — the
        // count pass histograms every entry, so when the total exceeded the
        // buffer the scanned starts point past it; clamp every run to the
        // written region (unlike way 5, whose histogram only counts stored
        // entries; dropped entries are tallied in entryOverflowCount).
        // Hash build: each side owns a fixed slot region, swept whole with
        // empty slots skipped — the sparse layout is part of what the A/B
        // measures.
        std::uint32_t tis0, tisSpan, tool0, toolSpan;
        if (useHashBuild)
        {
            const std::uint32_t kTissue = bigCell << 1;
            tis0 = kTissue * slotsPerBigCell;
            tool0 = (kTissue | 1u) * slotsPerBigCell;
            tisSpan = slotsPerBigCell;
            toolSpan = slotsPerBigCell;
        }
        else
        {
            const std::uint32_t kTissue = bigCell << 1;
            tis0 = min(starts[kTissue], entryCapacity);
            tool0 = min(starts[kTissue + 1u], entryCapacity);
            const std::uint32_t tool1 = min(starts[kTissue + 2u], entryCapacity);
            tisSpan = tool0 - tis0;
            toolSpan = tool1 - tool0;
        }

        // Chunk loop: oversized big cells stage the tool side toolTile entries
        // at a time and re-sweep the tissue entries per chunk. Each tool entry
        // lives in exactly one chunk, so no pair is visited twice.
        for (std::uint32_t tBase = 0; tBase < toolSpan; tBase += toolTile)
        {
            const std::uint32_t tile = min(toolTile, toolSpan - tBase);
            __syncthreads();  // previous tile/cell reads are done before rewriting shared
            if (threadIdx.x < 64u) sBinCursor[threadIdx.x] = 0u;
            if (threadIdx.x == 0) sTileCount = 0u;
            __syncthreads();
            if (threadIdx.x < tile)
            {
                const std::uint32_t packed = useHashBuild
                    ? hashSlots[tool0 + tBase + threadIdx.x]
                    : entriesPacked[tool0 + tBase + threadIdx.x];
                if (!useHashBuild || packed != kBigCellEmptySlot)
                {
                    const std::uint32_t pos = atomicAdd(&sTileCount, 1u);
                    sScratchPacked[pos] = packed;
                    atomicAdd(&sBinCursor[packed & 63u], 1u);
                }
            }
            __syncthreads();
            if (sTileCount == 0u) continue;  // uniform: whole chunk empty (hash mode)
            if (threadIdx.x == 0)
            {
                std::uint32_t running = 0;
                for (std::uint32_t i = 0; i < 64u; ++i)
                {
                    sRunStart[i] = running;
                    running += sBinCursor[i];
                    sBinCursor[i] = sRunStart[i];
                }
                sRunStart[64] = running;
            }
            __syncthreads();
            for (std::uint32_t i = threadIdx.x; i < sTileCount; i += blockDim.x)
            {
                const std::uint32_t packed = sScratchPacked[i];
                const std::uint32_t pos = atomicAdd(&sBinCursor[packed & 63u], 1u);
                const std::uint32_t toolTriId = packed >> 6u;
                sToolIds[pos] = toolTriId;
                const DeviceTriangle tt = indexedTriangleAt(toolPositions, toolIndices, toolTriId);
                sToolAabbs[pos] = triangleAabb(tt, 0.0f);
                sToolVerts[3u * pos + 0u] = tt.p0;
                sToolVerts[3u * pos + 1u] = tt.p1;
                sToolVerts[3u * pos + 2u] = tt.p2;
            }
            __syncthreads();

            for (std::uint32_t li = threadIdx.x; li < tisSpan; li += blockDim.x)
            {
                const std::uint32_t packed = useHashBuild
                    ? hashSlots[tis0 + li]
                    : entriesPacked[tis0 + li];
                if (useHashBuild && packed == kBigCellEmptySlot) continue;
                const std::uint32_t local = packed & 63u;
                const std::uint32_t r0 = sRunStart[local];
                const std::uint32_t r1 = sRunStart[local + 1u];
                if (r0 == r1) continue;
                const std::uint32_t tissueTriId = packed >> 6u;
                const DeviceAabb aInflated = tissueAabbs[tissueTriId];
                const std::uint32_t mask = bigc.factor - 1u;
                const int cx = baseX + static_cast<int>(local & mask);
                const int cy = baseY + static_cast<int>((local >> bigc.factorShift) & mask);
                const int cz = baseZ + static_cast<int>(local >> (2u * bigc.factorShift));
                const std::uint32_t smallCellId = denseCellId(cx, cy, cz, config);
                // Tissue vertices load once per (entry, chunk) — not once per pair.
                const DeviceTriangle ta = indexedTriangleAt(tissuePositions, tissueIndices, tissueTriId);
                const float3 aV[3] = { ta.p0, ta.p1, ta.p2 };
                const DeviceAabb aRaw = triangleAabb(ta, 0.0f);
                for (std::uint32_t u = r0; u < r1; ++u)
                {
                    const DeviceAabb bRaw = sToolAabbs[u];
                    const DeviceAabb bInflated {
                        bRaw.minX - contactDistance, bRaw.minY - contactDistance, bRaw.minZ - contactDistance,
                        bRaw.maxX + contactDistance, bRaw.maxY + contactDistance, bRaw.maxZ + contactDistance };
                    const float mx = fmaxf(aInflated.minX, bInflated.minX);
                    const float my = fmaxf(aInflated.minY, bInflated.minY);
                    const float mz = fmaxf(aInflated.minZ, bInflated.minZ);
                    if (mx > fminf(aInflated.maxX, bInflated.maxX) || my > fminf(aInflated.maxY, bInflated.maxY) ||
                        mz > fminf(aInflated.maxZ, bInflated.maxZ))
                    {
                        continue;  // inflated AABBs disjoint -> distance > contactDistance
                    }
                    if (sortedGridClampedCellOfPoint(mx, my, mz, config) != smallCellId)
                    {
                        continue;  // another (shared) small cell owns this pair
                    }
                    atomicAdd(pairsTestedCount, 1u);
                    const float3 bV[3] = {
                        sToolVerts[3u * u + 0u], sToolVerts[3u * u + 1u], sToolVerts[3u * u + 2u] };
                    if (bigCellRawAabbGapExceeds(aRaw, bRaw, distThreshSq)) continue;
                    DeviceProximityContact c;
                    if (!fbpComputeClosestFeatureContact(
                            aV, bV, distThreshSq, computeBarycentrics, c, true))
                    {
                        continue;
                    }
                    c.firstPrimitiveIndex = tissueTriId;
                    c.secondPrimitiveIndex = sToolIds[u];
                    fbpEmitContact(c, contacts, contactCount, overflowCount, vfCount, fvCount, eeCount, maxContacts);
                }
            }
        }
    }
}

__device__ __forceinline__ unsigned long long bigCellWarpReduceSum(unsigned long long value)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void bigCellProfileWarpAtomicAdd(
    unsigned long long* target, unsigned long long value)
{
    value = bigCellWarpReduceSum(value);
    if ((threadIdx.x & (warpSize - 1)) == 0 && value != 0ull)
    {
        atomicAdd(target, value);
    }
}
__global__ void profiledFusedBigCellNarrowKernel(
    const std::uint32_t* __restrict__ starts,
    const std::uint32_t* __restrict__ entriesPacked,
    const std::uint32_t* __restrict__ hashSlots,      // hash-build source (else unused)
    const std::uint32_t slotsPerBigCell,
    const bool useHashBuild,
    const std::uint32_t* __restrict__ mixedBigCellIds,
    const std::uint32_t* mixedBigCellCount,
    const DeviceAabb* __restrict__ tissueAabbs,
    const BackendTriangleVertex* __restrict__ tissuePositions,
    const std::uint32_t* __restrict__ tissueIndices,
    const BackendTriangleVertex* __restrict__ toolPositions,
    const std::uint32_t* __restrict__ toolIndices,
    const DeviceDenseGridConfig config,
    const DeviceBigCellGridConfig bigc,
    const std::uint32_t toolTile,             // <= 256, runtime-chunked
    const std::uint32_t entryCapacity,        // clamps CSR runs when the count pass overflowed the buffer
    DeviceProximityContact* __restrict__ contacts,
    std::uint32_t* pairsTestedCount,
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount,
    std::uint32_t* vfCount,
    std::uint32_t* fvCount,
    std::uint32_t* eeCount,
    const std::uint32_t maxContacts,
    const float contactDistance,
    const bool computeBarycentrics,
    DeviceBigCellFusedProfile* profile)
{
    __shared__ std::uint32_t sToolIds[256];       // sorted by local small cell
    __shared__ DeviceAabb sToolAabbs[256];
    __shared__ float3 sToolVerts[256 * 3];
    __shared__ std::uint32_t sScratchPacked[256]; // unsorted staging
    __shared__ std::uint32_t sRunStart[65];       // per-local run begin (end = next begin)
    __shared__ std::uint32_t sBinCursor[64];      // histogram, then scatter cursors
    __shared__ std::uint32_t sTileCount;          // staged entries this chunk (hash mode skips empties)

    unsigned long long localBigCellIterations = 0ull;
    unsigned long long localTileIterations = 0ull;
    unsigned long long localToolEntriesStaged = 0ull;
    unsigned long long localTissueEntriesVisited = 0ull;
    unsigned long long localSmallCellPairVisits = 0ull;
    unsigned long long localInflatedAabbRejects = 0ull;
    unsigned long long localHomeCellRejects = 0ull;
    unsigned long long localRawAabbRejects = 0ull;
    unsigned long long localFbpCalls = 0ull;
    unsigned long long localFbpNoContact = 0ull;
    unsigned long long localInflatedAabbCycles = 0ull;
    unsigned long long localHomeCellCycles = 0ull;
    unsigned long long localRawAabbCycles = 0ull;
    unsigned long long localFbpCycles = 0ull;
    unsigned long long blockTileSetupCycles = 0ull;
    unsigned long long blockBinPrefixCycles = 0ull;
    unsigned long long blockToolGatherCycles = 0ull;
    unsigned long long blockTissueSweepCycles = 0ull;

    const float distThreshSq = contactDistance * contactDistance;
    const std::uint32_t mixedCount = *mixedBigCellCount;
    for (std::uint32_t m = blockIdx.x; m < mixedCount; m += gridDim.x)
    {
        if (threadIdx.x == 0) ++localBigCellIterations;
        const std::uint32_t bigCell = mixedBigCellIds[m];
        const std::uint32_t bx = bigCell % bigc.bigResX;
        const std::uint32_t rem = bigCell / bigc.bigResX;
        const std::uint32_t by = rem % bigc.bigResY;
        const std::uint32_t bz = rem / bigc.bigResY;
        const int baseX = static_cast<int>(bx << bigc.factorShift);
        const int baseY = static_cast<int>(by << bigc.factorShift);
        const int baseZ = static_cast<int>(bz << bigc.factorShift);
        // Entry sources. CSR: contiguous runs from the scanned histogram — the
        // count pass histograms every entry, so when the total exceeded the
        // buffer the scanned starts point past it; clamp every run to the
        // written region (unlike way 5, whose histogram only counts stored
        // entries; dropped entries are tallied in entryOverflowCount).
        // Hash build: each side owns a fixed slot region, swept whole with
        // empty slots skipped — the sparse layout is part of what the A/B
        // measures.
        std::uint32_t tis0, tisSpan, tool0, toolSpan;
        if (useHashBuild)
        {
            const std::uint32_t kTissue = bigCell << 1;
            tis0 = kTissue * slotsPerBigCell;
            tool0 = (kTissue | 1u) * slotsPerBigCell;
            tisSpan = slotsPerBigCell;
            toolSpan = slotsPerBigCell;
        }
        else
        {
            const std::uint32_t kTissue = bigCell << 1;
            tis0 = min(starts[kTissue], entryCapacity);
            tool0 = min(starts[kTissue + 1u], entryCapacity);
            const std::uint32_t tool1 = min(starts[kTissue + 2u], entryCapacity);
            tisSpan = tool0 - tis0;
            toolSpan = tool1 - tool0;
        }

        // Chunk loop: oversized big cells stage the tool side toolTile entries
        // at a time and re-sweep the tissue entries per chunk. Each tool entry
        // lives in exactly one chunk, so no pair is visited twice.
        for (std::uint32_t tBase = 0; tBase < toolSpan; tBase += toolTile)
        {
            const std::uint32_t tile = min(toolTile, toolSpan - tBase);
            unsigned long long phaseStart = 0ull;
            __syncthreads();  // previous tile/cell reads are done before rewriting shared
            if (threadIdx.x == 0) phaseStart = clock64();
            if (threadIdx.x < 64u) sBinCursor[threadIdx.x] = 0u;
            if (threadIdx.x == 0) sTileCount = 0u;
            __syncthreads();
            if (threadIdx.x < tile)
            {
                const std::uint32_t packed = useHashBuild
                    ? hashSlots[tool0 + tBase + threadIdx.x]
                    : entriesPacked[tool0 + tBase + threadIdx.x];
                if (!useHashBuild || packed != kBigCellEmptySlot)
                {
                    const std::uint32_t pos = atomicAdd(&sTileCount, 1u);
                    sScratchPacked[pos] = packed;
                    atomicAdd(&sBinCursor[packed & 63u], 1u);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                blockTileSetupCycles += clock64() - phaseStart;
                ++localTileIterations;
                localToolEntriesStaged += sTileCount;
                phaseStart = clock64();
            }
            if (sTileCount == 0u) continue;  // uniform: whole chunk empty (hash mode)
            if (threadIdx.x == 0)
            {
                std::uint32_t running = 0;
                for (std::uint32_t i = 0; i < 64u; ++i)
                {
                    sRunStart[i] = running;
                    running += sBinCursor[i];
                    sBinCursor[i] = sRunStart[i];
                }
                sRunStart[64] = running;
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                blockBinPrefixCycles += clock64() - phaseStart;
                phaseStart = clock64();
            }
            for (std::uint32_t i = threadIdx.x; i < sTileCount; i += blockDim.x)
            {
                const std::uint32_t packed = sScratchPacked[i];
                const std::uint32_t pos = atomicAdd(&sBinCursor[packed & 63u], 1u);
                const std::uint32_t toolTriId = packed >> 6u;
                sToolIds[pos] = toolTriId;
                const DeviceTriangle tt = indexedTriangleAt(toolPositions, toolIndices, toolTriId);
                sToolAabbs[pos] = triangleAabb(tt, 0.0f);
                sToolVerts[3u * pos + 0u] = tt.p0;
                sToolVerts[3u * pos + 1u] = tt.p1;
                sToolVerts[3u * pos + 2u] = tt.p2;
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                blockToolGatherCycles += clock64() - phaseStart;
                phaseStart = clock64();
            }

            for (std::uint32_t li = threadIdx.x; li < tisSpan; li += blockDim.x)
            {
                const std::uint32_t packed = useHashBuild
                    ? hashSlots[tis0 + li]
                    : entriesPacked[tis0 + li];
                if (useHashBuild && packed == kBigCellEmptySlot) continue;
                ++localTissueEntriesVisited;
                const std::uint32_t local = packed & 63u;
                const std::uint32_t r0 = sRunStart[local];
                const std::uint32_t r1 = sRunStart[local + 1u];
                if (r0 == r1) continue;
                const std::uint32_t tissueTriId = packed >> 6u;
                const DeviceAabb aInflated = tissueAabbs[tissueTriId];
                const std::uint32_t mask = bigc.factor - 1u;
                const int cx = baseX + static_cast<int>(local & mask);
                const int cy = baseY + static_cast<int>((local >> bigc.factorShift) & mask);
                const int cz = baseZ + static_cast<int>(local >> (2u * bigc.factorShift));
                const std::uint32_t smallCellId = denseCellId(cx, cy, cz, config);
                // Tissue vertices load once per (entry, chunk) — not once per pair.
                const DeviceTriangle ta = indexedTriangleAt(tissuePositions, tissueIndices, tissueTriId);
                const float3 aV[3] = { ta.p0, ta.p1, ta.p2 };
                const DeviceAabb aRaw = triangleAabb(ta, 0.0f);
                    for (std::uint32_t u = r0; u < r1; ++u)
                    {
                        ++localSmallCellPairVisits;
                        const DeviceAabb bRaw = sToolAabbs[u];
                        unsigned long long filterStart = clock64();
                        const DeviceAabb bInflated {
                            bRaw.minX - contactDistance, bRaw.minY - contactDistance, bRaw.minZ - contactDistance,
                            bRaw.maxX + contactDistance, bRaw.maxY + contactDistance, bRaw.maxZ + contactDistance };
                        const float mx = fmaxf(aInflated.minX, bInflated.minX);
                        const float my = fmaxf(aInflated.minY, bInflated.minY);
                        const float mz = fmaxf(aInflated.minZ, bInflated.minZ);
                        const bool inflatedRejected =
                            mx > fminf(aInflated.maxX, bInflated.maxX) ||
                            my > fminf(aInflated.maxY, bInflated.maxY) ||
                            mz > fminf(aInflated.maxZ, bInflated.maxZ);
                        localInflatedAabbCycles += clock64() - filterStart;
                        if (inflatedRejected)
                        {
                            ++localInflatedAabbRejects;
                            continue;
                        }
                        filterStart = clock64();
                        const bool homeCellRejected =
                            sortedGridClampedCellOfPoint(mx, my, mz, config) != smallCellId;
                        localHomeCellCycles += clock64() - filterStart;
                        if (homeCellRejected)
                        {
                            ++localHomeCellRejects;
                            continue;
                        }
                        atomicAdd(pairsTestedCount, 1u);
                        const float3 bV[3] = {
                            sToolVerts[3u * u + 0u], sToolVerts[3u * u + 1u], sToolVerts[3u * u + 2u] };
                        filterStart = clock64();
                        const bool rawAabbRejected = bigCellRawAabbGapExceeds(aRaw, bRaw, distThreshSq);
                        localRawAabbCycles += clock64() - filterStart;
                        if (rawAabbRejected)
                        {
                            ++localRawAabbRejects;
                            continue;
                        }
                        ++localFbpCalls;
                        DeviceProximityContact c;
                        filterStart = clock64();
                        const bool contactFound = fbpComputeClosestFeatureContact(
                            aV, bV, distThreshSq, computeBarycentrics, c, true);
                        localFbpCycles += clock64() - filterStart;
                        if (!contactFound)
                        {
                            ++localFbpNoContact;
                            continue;
                        }
                        c.firstPrimitiveIndex = tissueTriId;
                        c.secondPrimitiveIndex = sToolIds[u];
                        fbpEmitContact(c, contacts, contactCount, overflowCount, vfCount, fvCount, eeCount, maxContacts);
                    }
            }
            __syncthreads();
            if (threadIdx.x == 0) blockTissueSweepCycles += clock64() - phaseStart;
        }
    }

    bigCellProfileWarpAtomicAdd(&profile->bigCellIterations, localBigCellIterations);
        bigCellProfileWarpAtomicAdd(&profile->tileIterations, localTileIterations);
        bigCellProfileWarpAtomicAdd(&profile->toolEntriesStaged, localToolEntriesStaged);
        bigCellProfileWarpAtomicAdd(&profile->tissueEntriesVisited, localTissueEntriesVisited);
        bigCellProfileWarpAtomicAdd(&profile->smallCellPairVisits, localSmallCellPairVisits);
        bigCellProfileWarpAtomicAdd(&profile->inflatedAabbRejects, localInflatedAabbRejects);
        bigCellProfileWarpAtomicAdd(&profile->homeCellRejects, localHomeCellRejects);
        bigCellProfileWarpAtomicAdd(&profile->rawAabbRejects, localRawAabbRejects);
        bigCellProfileWarpAtomicAdd(&profile->fbpCalls, localFbpCalls);
        bigCellProfileWarpAtomicAdd(&profile->fbpNoContact, localFbpNoContact);
        bigCellProfileWarpAtomicAdd(&profile->tileSetupBlockCycles, blockTileSetupCycles);
        bigCellProfileWarpAtomicAdd(&profile->binPrefixBlockCycles, blockBinPrefixCycles);
        bigCellProfileWarpAtomicAdd(&profile->toolGatherBlockCycles, blockToolGatherCycles);
        bigCellProfileWarpAtomicAdd(&profile->tissueSweepBlockCycles, blockTissueSweepCycles);
        bigCellProfileWarpAtomicAdd(&profile->inflatedAabbThreadCycles, localInflatedAabbCycles);
        bigCellProfileWarpAtomicAdd(&profile->homeCellThreadCycles, localHomeCellCycles);
        bigCellProfileWarpAtomicAdd(&profile->rawAabbThreadCycles, localRawAabbCycles);
    bigCellProfileWarpAtomicAdd(&profile->fbpThreadCycles, localFbpCycles);
}

} // namespace


namespace SofaGpuCollision::backend
{

bool computeBigCellFusedProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const BigCellConfig& bigConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    BigCellStats* bigStats,
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
    if (bigStats != nullptr) *bigStats = BigCellStats {};

    if (indexedSurfaceInvalid(firstSurface) || indexedSurfaceInvalid(secondSurface))
    {
        diagnostic = "Invalid indexed triangle surface (big-cell path).";
        return false;
    }
    if (gridConfig.gridResolutionX == 0 || gridConfig.gridResolutionY == 0 || gridConfig.gridResolutionZ == 0 ||
        gridConfig.gridMaxX <= gridConfig.gridMinX ||
        gridConfig.gridMaxY <= gridConfig.gridMinY ||
        gridConfig.gridMaxZ <= gridConfig.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration (big-cell path).";
        return false;
    }
    const std::uint32_t factor = bigConfig.bigCellFactor;
    if (factor != 1u && factor != 2u && factor != 4u)
    {
        diagnostic = "Big-cell factor must be 1, 2 or 4 (local ids are packed into 6 bits).";
        return false;
    }
    // Entries pack the triangle id into the upper 26 bits.
    if (firstSurface.triangleCount >= (1u << 26) || secondSurface.triangleCount >= (1u << 26))
    {
        diagnostic = "Big-cell path supports at most 2^26-1 triangles per surface.";
        return false;
    }
    const std::uint32_t toolTile = std::max(1u, std::min(256u, bigConfig.toolTileCapacity));

    std::uint32_t factorShift = 0;
    while ((1u << factorShift) < factor) ++factorShift;
    const std::uint32_t bigResX = (gridConfig.gridResolutionX + factor - 1u) / factor;
    const std::uint32_t bigResY = (gridConfig.gridResolutionY + factor - 1u) / factor;
    const std::uint32_t bigResZ = (gridConfig.gridResolutionZ + factor - 1u) / factor;
    const std::uint64_t bigCellCount64 =
        static_cast<std::uint64_t>(bigResX) * static_cast<std::uint64_t>(bigResY) * static_cast<std::uint64_t>(bigResZ);
    if (bigCellCount64 == 0 || (bigCellCount64 << 1) >= (1ull << 31))
    {
        diagnostic = "Big-cell count out of range.";
        return false;
    }
    const std::uint32_t bigCellCount = static_cast<std::uint32_t>(bigCellCount64);
    const std::uint32_t binCount = bigCellCount * 2u;

    const std::uint32_t totalTriangles = firstSurface.triangleCount + secondSurface.triangleCount;
    const std::uint32_t entryCapacity = std::max(
        4096u,
        std::max(1u, bigConfig.entryCapacityScale) * totalTriangles);
    const bool useHashBuild = bigConfig.useHashTableBuild;
    std::uint32_t slotsPerBigCell = 0;
    if (useHashBuild)
    {
        slotsPerBigCell = static_cast<std::uint32_t>(nextPowerOfTwo(
            std::min(4096u, std::max(64u, bigConfig.hashSlotsPerBigCell))));
        const std::uint64_t totalSlots64 = static_cast<std::uint64_t>(binCount) * slotsPerBigCell;
        if (totalSlots64 > (1ull << 27))
        {
            diagnostic = "Big-cell hash build: slot table too large (lower hashSlotsPerBigCell or raise bigCellFactor).";
            return false;
        }
    }
    const std::uint32_t sharedBuildMode = bigConfig.sharedBuildMode;
    if (sharedBuildMode > 2u)
    {
        diagnostic = "Big-cell sharedBuildMode must be 0 (off), 1 (shared hash) or 2 (shared sorted list).";
        return false;
    }
    if (sharedBuildMode != 0u && useHashBuild)
    {
        diagnostic = "Big-cell sharedBuildMode privatizes the CSR build; disable useHashTableBuild to use it.";
        return false;
    }

    auto& ws = bigCellWorkspace();
    std::uint64_t newlyAllocatedBytes = 0;
    const auto allocStart = std::chrono::steady_clock::now();
    cudaError_t err = ws.ensure(
        binCount,
        bigCellCount,
        entryCapacity,
        useHashBuild ? static_cast<std::size_t>(binCount) * slotsPerBigCell : 0u,
        proximityConfig.maxContacts,
        sizeof(DeviceProximityContact),
        static_cast<std::size_t>(firstSurface.triangleCount) * 3u,
        static_cast<std::size_t>(secondSurface.triangleCount) * 3u,
        firstSurface.devicePositions == nullptr ? firstSurface.vertexCount : 0u,
        secondSurface.devicePositions == nullptr ? secondSurface.vertexCount : 0u,
        firstSurface.triangleCount,
        secondSurface.triangleCount,
        newlyAllocatedBytes);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Big-cell workspace alloc failed: ") + cudaGetErrorString(err);
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

    const DeviceDenseGridConfig dc = makeDeviceDenseGridConfig(gridConfig, 0u);
    const DeviceBigCellGridConfig bigc { factor, factorShift, bigResX, bigResY, bigResZ };

    constexpr std::uint32_t threads = 256;
    const std::uint32_t resetBlocks = std::max(1u, std::min(1024u, (binCount + threads - 1u) / threads));
    const std::uint32_t firstBlocks = (firstSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t secondBlocks = (secondSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t mixedBlocks = (bigCellCount + threads - 1u) / threads;
    constexpr std::uint32_t kFusedBlocks = 1024;

    const bool profileInternals = bigConfig.profileFusedInternals;
    const bool detailedProfiling = gridConfig.detailedProfiling || profileInternals;
    const bool totalTiming = proximityConfig.readContactCounter && !detailedProfiling;
    if (detailedProfiling)
    {
        err = ws.ensureDetailedEvents();
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }
    else if (totalTiming)
    {
        err = ws.ensureEvents();
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }

    if (bigStats != nullptr)
    {
        bigStats->bigCellCount = bigCellCount;
        bigStats->fusedGridBlocks = kFusedBlocks;
        bigStats->fusedBlockThreads = threads;
    }

    // Runtime occupancy is a launch-limit calculation for the uninstrumented
    // winner specialization. It complements, but does not replace, achieved
    // occupancy from Nsight Compute.
    if (detailedProfiling && bigStats != nullptr)
    {
        cudaFuncAttributes attrs {};
        int activeBlocks = 0;
        int device = 0;
        cudaDeviceProp props {};
        err = cudaFuncGetAttributes(&attrs, fusedBigCellNarrowKernel);
        if (err == cudaSuccess) err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, fusedBigCellNarrowKernel, threads, 0);
        if (err == cudaSuccess) err = cudaGetDevice(&device);
        if (err == cudaSuccess) err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Big-cell occupancy query failed: ") + cudaGetErrorString(err);
            return false;
        }
        bigStats->fusedRegistersPerThread = static_cast<std::uint32_t>(attrs.numRegs);
        bigStats->fusedStaticSharedBytes = static_cast<std::uint32_t>(attrs.sharedSizeBytes);
        bigStats->fusedLocalBytesPerThread = static_cast<std::uint32_t>(attrs.localSizeBytes);
        bigStats->fusedMaxThreadsPerBlock = static_cast<std::uint32_t>(attrs.maxThreadsPerBlock);
        bigStats->fusedActiveBlocksPerSm = static_cast<std::uint32_t>(activeBlocks);
        bigStats->deviceMultiprocessorCount = static_cast<std::uint32_t>(props.multiProcessorCount);
        bigStats->deviceMaxThreadsPerSm = static_cast<std::uint32_t>(props.maxThreadsPerMultiProcessor);
        bigStats->deviceWarpSize = static_cast<std::uint32_t>(props.warpSize);
        bigStats->deviceClockRateKHz = static_cast<std::uint32_t>(props.clockRate);
        bigStats->fusedTheoreticalOccupancyPercent = props.maxThreadsPerMultiProcessor > 0
            ? 100.0 * static_cast<double>(activeBlocks * static_cast<int>(threads)) /
                static_cast<double>(props.maxThreadsPerMultiProcessor)
            : 0.0;
        if (profileInternals)
        {
            cudaFuncAttributes profiledAttrs {};
            int profiledActiveBlocks = 0;
            err = cudaFuncGetAttributes(&profiledAttrs, profiledFusedBigCellNarrowKernel);
            if (err == cudaSuccess) err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &profiledActiveBlocks, profiledFusedBigCellNarrowKernel, threads, 0);
            if (err != cudaSuccess)
            {
                diagnostic = std::string("Big-cell instrumented occupancy query failed: ") + cudaGetErrorString(err);
                return false;
            }
            bigStats->profiledRegistersPerThread = static_cast<std::uint32_t>(profiledAttrs.numRegs);
            bigStats->profiledStaticSharedBytes = static_cast<std::uint32_t>(profiledAttrs.sharedSizeBytes);
            bigStats->profiledLocalBytesPerThread = static_cast<std::uint32_t>(profiledAttrs.localSizeBytes);
            bigStats->profiledActiveBlocksPerSm = static_cast<std::uint32_t>(profiledActiveBlocks);
            bigStats->profiledTheoreticalOccupancyPercent = props.maxThreadsPerMultiProcessor > 0
                ? 100.0 * static_cast<double>(profiledActiveBlocks * static_cast<int>(threads)) /
                    static_cast<double>(props.maxThreadsPerMultiProcessor)
                : 0.0;
        }
    }

    auto* deviceContacts = reinterpret_cast<DeviceProximityContact*>(ws.proximityContacts);
    cudaError_t detailedEventError = cudaSuccess;
    cudaError_t profileResetError = cudaSuccess;
    auto recordDetailedEvent = [&](const std::size_t index, cudaStream_t s)
    {
        if (detailedProfiling && detailedEventError == cudaSuccess)
        {
            detailedEventError = cudaEventRecord(ws.detailedEvents[index], s);
        }
    };

    auto launchAll = [&](cudaStream_t s)
    {
        resetSortedGridKernel<<<resetBlocks, threads, 0, s>>>(
            ws.hist, binCount, ws.entryTotalCount, ws.entryOverflowCount, ws.mixedBigCellCount,
            ws.pairsTestedCount, ws.sharedSpillCount, ws.buildOverflowCount, nullptr);
        recordDetailedEvent(1, s);
        if (useHashBuild)
        {
            const std::uint32_t totalSlots = binCount * slotsPerBigCell;
            const std::uint32_t clearBlocks = std::max(1u, std::min(1024u, (totalSlots + threads - 1u) / threads));
            clearBigCellHashSlotsKernel<<<clearBlocks, threads, 0, s>>>(ws.hashSlots, totalSlots);
            recordDetailedEvent(2, s);
            insertBigCellHashEntriesKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                slotsPerBigCell, ws.hashSlots, ws.hist, ws.firstAabbs, ws.entryTotalCount, ws.buildOverflowCount);
            recordDetailedEvent(3, s);
            insertBigCellHashEntriesKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                slotsPerBigCell, ws.hashSlots, ws.hist, ws.secondAabbs, ws.entryTotalCount, ws.buildOverflowCount);
            recordDetailedEvent(4, s);
            recordDetailedEvent(5, s);
            recordDetailedEvent(6, s);
            recordDetailedEvent(7, s);
        }
        else if (sharedBuildMode == 1u)
        {
            recordDetailedEvent(2, s);
            countBigCellEntriesSharedHashKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                ws.hist, ws.firstAabbs, ws.entryTotalCount, ws.sharedSpillCount);
            recordDetailedEvent(3, s);
            countBigCellEntriesSharedHashKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                ws.hist, ws.secondAabbs, ws.entryTotalCount, ws.sharedSpillCount);
            recordDetailedEvent(4, s);
            exclusiveScanSortedGridBinsKernel<<<1, 1024, 0, s>>>(ws.hist, ws.starts, ws.cursors, binCount);
            recordDetailedEvent(5, s);
            fillBigCellEntriesSharedHashKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount, ws.sharedSpillCount);
            recordDetailedEvent(6, s);
            fillBigCellEntriesSharedHashKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount, ws.sharedSpillCount);
            recordDetailedEvent(7, s);
        }
        else if (sharedBuildMode == 2u)
        {
            recordDetailedEvent(2, s);
            countBigCellEntriesSharedSortKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                ws.hist, ws.firstAabbs, ws.entryTotalCount, ws.sharedSpillCount);
            recordDetailedEvent(3, s);
            countBigCellEntriesSharedSortKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                ws.hist, ws.secondAabbs, ws.entryTotalCount, ws.sharedSpillCount);
            recordDetailedEvent(4, s);
            exclusiveScanSortedGridBinsKernel<<<1, 1024, 0, s>>>(ws.hist, ws.starts, ws.cursors, binCount);
            recordDetailedEvent(5, s);
            fillBigCellEntriesSharedSortKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount, ws.sharedSpillCount);
            recordDetailedEvent(6, s);
            fillBigCellEntriesSharedSortKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount, ws.sharedSpillCount);
            recordDetailedEvent(7, s);
        }
        else
        {
            recordDetailedEvent(2, s);
            countBigCellEntriesKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                ws.hist, ws.firstAabbs, ws.entryTotalCount);
            recordDetailedEvent(3, s);
            countBigCellEntriesKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                ws.hist, ws.secondAabbs, ws.entryTotalCount);
            recordDetailedEvent(4, s);
            exclusiveScanSortedGridBinsKernel<<<1, 1024, 0, s>>>(ws.hist, ws.starts, ws.cursors, binCount);
            recordDetailedEvent(5, s);
            fillBigCellEntriesKernel<<<firstBlocks, threads, 0, s>>>(
                deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount);
            recordDetailedEvent(6, s);
            fillBigCellEntriesKernel<<<secondBlocks, threads, 0, s>>>(
                deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, bigc,
                entryCapacity, ws.cursors, ws.entriesPacked, ws.entryOverflowCount);
            recordDetailedEvent(7, s);
        }
        buildSortedGridMixedCellsKernel<<<mixedBlocks, threads, 0, s>>>(
            ws.hist, bigCellCount, ws.mixedBigCellIds, ws.mixedBigCellCount);
        recordDetailedEvent(8, s);
        if (profileInternals)
        {
            profileResetError = cudaMemsetAsync(ws.fusedProfile, 0, sizeof(DeviceBigCellFusedProfile), s);
        }
        resetProximityCountersKernel<<<1, 1, 0, s>>>(
            ws.proximityContactCount, ws.proximityOverflowCount,
            ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount);
        recordDetailedEvent(9, s);
        if (profileInternals)
        {
            profiledFusedBigCellNarrowKernel<<<kFusedBlocks, threads, 0, s>>>(
                ws.starts, ws.entriesPacked, ws.hashSlots, slotsPerBigCell, useHashBuild,
                ws.mixedBigCellIds, ws.mixedBigCellCount,
                ws.firstAabbs,
                deviceFirstPositions, ws.firstIndices, deviceSecondPositions, ws.secondIndices,
                dc, bigc, toolTile, entryCapacity, deviceContacts,
                ws.pairsTestedCount, ws.proximityContactCount, ws.proximityOverflowCount,
                ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount,
                proximityConfig.maxContacts, proximityConfig.contactDistance,
                proximityConfig.computeBarycentrics, ws.fusedProfile);
        }
        else
        {
            fusedBigCellNarrowKernel<<<kFusedBlocks, threads, 0, s>>>(
                ws.starts, ws.entriesPacked, ws.hashSlots, slotsPerBigCell, useHashBuild,
                ws.mixedBigCellIds, ws.mixedBigCellCount,
                ws.firstAabbs,
                deviceFirstPositions, ws.firstIndices, deviceSecondPositions, ws.secondIndices,
                dc, bigc, toolTile, entryCapacity, deviceContacts,
                ws.pairsTestedCount, ws.proximityContactCount, ws.proximityOverflowCount,
                ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount,
                proximityConfig.maxContacts, proximityConfig.contactDistance,
                proximityConfig.computeBarycentrics);
        }
        recordDetailedEvent(10, s);
    };

    if (detailedProfiling)
    {
        recordDetailedEvent(0, 0);
        if (detailedEventError != cudaSuccess)
        {
            diagnostic = cudaGetErrorString(detailedEventError);
            return false;
        }
    }
    else if (totalTiming)
    {
        err = cudaEventRecord(ws.startEvent);
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }

    static const int kGraphMode = []{ const char* e = std::getenv("SOFA_BIGCELL_CUDA_GRAPH"); return e ? std::atoi(e) : 1; }();
    const bool kGraphsEnabled = (kGraphMode != 0);
    bool ranViaGraph = false;
    if (kGraphsEnabled && !detailedProfiling)
    {
        const std::array<std::uint64_t, 6> graphSignature {
            (static_cast<std::uint64_t>(binCount) << 32) | entryCapacity,
            (static_cast<std::uint64_t>(firstSurface.triangleCount) << 32) | secondSurface.triangleCount,
            (static_cast<std::uint64_t>(proximityConfig.maxContacts) << 32) | (sharedBuildMode << 24) | (factor << 16) | toolTile,
            (proximityConfig.computeBarycentrics ? 1ull : 0ull) | (useHashBuild ? 2ull : 0ull),
            floatSignatureBits(proximityConfig.contactDistance),
            1ull | (static_cast<std::uint64_t>(slotsPerBigCell) << 8) };  // way tag + hash sizing
        ws.graph.invalidateIfChanged(graphSignature, newlyAllocatedBytes > 0);
        if (!ws.firstFrameDone)
        {
            launchAll(0);
            ws.graph.instantiated = false;
            ranViaGraph = true;
        }
        else
        {
            ws.graph.replay(graphSignature, [&](cudaStream_t s) { launchAll(s); });
            ranViaGraph = true;
        }
    }
    if (!ranViaGraph)
    {
        launchAll(0);
    }
    ws.firstFrameDone = true;
    // CSR: reset + count x2 + scan + fill x2 + mixed + counters + fused = 9
    // hash: reset + clear + insert x2 + mixed + counters + fused = 7
    std::uint32_t launchCount = useHashBuild ? 7u : 9u;
    if (profileResetError != cudaSuccess)
    {
        diagnostic = std::string("big-cell profile reset: ") + cudaGetErrorString(profileResetError);
        return false;
    }
    if (detailedEventError != cudaSuccess)
    {
        diagnostic = std::string("big-cell detailed event: ") + cudaGetErrorString(detailedEventError);
        return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { diagnostic = std::string("big-cell launch: ") + cudaGetErrorString(err); return false; }

    float kernelMs = 0.0f;
    float resetMs = 0.0f, buildClearMs = 0.0f, eventMarkerGapMs = 0.0f;
    float firstCountMs = 0.0f, secondCountMs = 0.0f, scanMs = 0.0f;
    float firstFillMs = 0.0f, secondFillMs = 0.0f, mixedMs = 0.0f;
    float proximityResetMs = 0.0f, fusedMs = 0.0f;
    if (detailedProfiling)
    {
        err = cudaEventSynchronize(ws.detailedEvents[10]);
        auto elapsed = [&](float& value, const std::size_t from, const std::size_t to)
        {
            if (err == cudaSuccess)
            {
                err = cudaEventElapsedTime(&value, ws.detailedEvents[from], ws.detailedEvents[to]);
            }
        };
        elapsed(kernelMs, 0, 10);
        elapsed(resetMs, 0, 1);
        elapsed(buildClearMs, 1, 2);
        elapsed(firstCountMs, 2, 3);
        elapsed(secondCountMs, 3, 4);
        elapsed(scanMs, 4, 5);
        elapsed(firstFillMs, 5, 6);
        elapsed(secondFillMs, 6, 7);
        elapsed(mixedMs, 7, 8);
        elapsed(proximityResetMs, 8, 9);
        elapsed(fusedMs, 9, 10);
        if (!useHashBuild)
        {
            eventMarkerGapMs = buildClearMs;
            buildClearMs = 0.0f;
        }
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }
    else if (totalTiming)
    {
        err = cudaEventRecord(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventSynchronize(ws.endEvent);
        if (err == cudaSuccess) err = cudaEventElapsedTime(&kernelMs, ws.startEvent, ws.endEvent);
        if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; }
    }
    if (bigStats != nullptr && detailedProfiling)
    {
        bigStats->resetMilliseconds = resetMs;
        bigStats->buildClearMilliseconds = buildClearMs;
        bigStats->eventMarkerGapMilliseconds = eventMarkerGapMs;
        bigStats->firstCountOrInsertMilliseconds = firstCountMs;
        bigStats->secondCountOrInsertMilliseconds = secondCountMs;
        bigStats->scanMilliseconds = scanMs;
        bigStats->firstFillMilliseconds = firstFillMs;
        bigStats->secondFillMilliseconds = secondFillMs;
        bigStats->mixedCellBuildMilliseconds = mixedMs;
        bigStats->proximityResetMilliseconds = proximityResetMs;
        bigStats->fusedKernelMilliseconds = fusedMs;
        bigStats->totalPipelineMilliseconds = kernelMs;
    }
    if (executionStats != nullptr)
    {
        executionStats->kernelLaunchCount += launchCount;
        executionStats->cudaMemsetCount += profileInternals ? 1u : 0u;
        executionStats->gpuKernelMilliseconds += static_cast<double>(kernelMs);
        executionStats->featureBasedProximityKernelMilliseconds += static_cast<double>(fusedMs);
    }

    if (profileInternals)
    {
        DeviceBigCellFusedProfile hostProfile {};
        err = cudaMemcpy(&hostProfile, ws.fusedProfile, sizeof(hostProfile), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("big-cell internal profile readback: ") + cudaGetErrorString(err);
            return false;
        }
        if (executionStats != nullptr) executionStats->deviceToHostBytes += sizeof(hostProfile);
        if (bigStats != nullptr)
        {
            bigStats->profiledBigCellIterations = hostProfile.bigCellIterations;
            bigStats->profiledTileIterations = hostProfile.tileIterations;
            bigStats->profiledToolEntriesStaged = hostProfile.toolEntriesStaged;
            bigStats->profiledTissueEntriesVisited = hostProfile.tissueEntriesVisited;
            bigStats->profiledSmallCellPairVisits = hostProfile.smallCellPairVisits;
            bigStats->profiledInflatedAabbRejects = hostProfile.inflatedAabbRejects;
            bigStats->profiledHomeCellRejects = hostProfile.homeCellRejects;
            bigStats->profiledRawAabbRejects = hostProfile.rawAabbRejects;
            bigStats->profiledFbpCalls = hostProfile.fbpCalls;
            bigStats->profiledFbpNoContact = hostProfile.fbpNoContact;
            bigStats->profiledTileSetupBlockCycles = hostProfile.tileSetupBlockCycles;
            bigStats->profiledBinPrefixBlockCycles = hostProfile.binPrefixBlockCycles;
            bigStats->profiledToolGatherBlockCycles = hostProfile.toolGatherBlockCycles;
            bigStats->profiledTissueSweepBlockCycles = hostProfile.tissueSweepBlockCycles;
            bigStats->profiledInflatedAabbThreadCycles = hostProfile.inflatedAabbThreadCycles;
            bigStats->profiledHomeCellThreadCycles = hostProfile.homeCellThreadCycles;
            bigStats->profiledRawAabbThreadCycles = hostProfile.rawAabbThreadCycles;
            bigStats->profiledFbpThreadCycles = hostProfile.fbpThreadCycles;
        }
    }

    std::uint32_t hostPairsTested = 0, hostEntries = 0, hostEntryOverflow = 0, hostMixed = 0;
    std::uint32_t hostBuildOverflow = 0, hostContactCount = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* p = ws.countersHostPinned;
        cudaMemcpyAsync(p + 0, ws.pairsTestedCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 1, ws.entryTotalCount,         sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 2, ws.entryOverflowCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 3, ws.mixedBigCellCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 4, ws.buildOverflowCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 5, ws.proximityContactCount,   sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 6, ws.proximityOverflowCount,  sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 7, ws.proximityVfCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 8, ws.proximityFvCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 9, ws.proximityEeCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 10, ws.sharedSpillCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostPairsTested = p[0]; hostEntries = p[1]; hostEntryOverflow = p[2]; hostMixed = p[3];
        hostBuildOverflow = p[4]; hostContactCount = p[5];
        hostVf = p[7]; hostFv = p[8]; hostEe = p[9];
        const std::uint32_t hostSharedSpill = p[10];
        const std::uint32_t hostProxOverflow = p[6];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 11u * sizeof(std::uint32_t);
            executionStats->uniqueCandidateCount += hostPairsTested;
            executionStats->outputCandidateCount += hostPairsTested;
            executionStats->rawCandidateCount += hostEntries;
            executionStats->outputContactCount = hostContactCount;
            executionStats->activeMixedCellCount += hostMixed;
            executionStats->vfContactCount += hostVf;
            executionStats->fvContactCount += hostFv;
            executionStats->eeContactCount += hostEe;
            const std::uint32_t totalOverflow = hostEntryOverflow + hostBuildOverflow + hostProxOverflow;
            if (totalOverflow > 0) executionStats->overflowCount += totalOverflow;
        }
        if (proximityStats != nullptr)
        {
            proximityStats->candidatePairCount = hostPairsTested;
            proximityStats->emittedContactCount = hostContactCount;
            proximityStats->vfContactCount = hostVf;
            proximityStats->fvContactCount = hostFv;
            proximityStats->eeContactCount = hostEe;
        }
        if (bigStats != nullptr)
        {
            bigStats->bigCellCount = bigCellCount;
            bigStats->entryCount = hostEntries;
            bigStats->mixedBigCellCount = hostMixed;
            bigStats->pairsTestedCount = hostPairsTested;
            bigStats->entryOverflowCount = hostEntryOverflow;
            bigStats->buildOverflowCount = hostBuildOverflow;
            bigStats->sharedSpillCount = hostSharedSpill;
        }
    }

    if (!proximityConfig.keepContactsOnDevice && hostContactCount > 0)
    {
        downloadDeviceProximityContacts(deviceContacts, hostContactCount, proximityConfig.maxContacts, contacts, executionStats);
    }

    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = bigCellCount;
    }

    diagnostic.clear();
    return true;
}


} // namespace SofaGpuCollision::backend
