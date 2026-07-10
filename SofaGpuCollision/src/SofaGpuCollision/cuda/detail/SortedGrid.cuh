// SortedGrid.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Sorted-grid / tiled-binning broad cull (way 5): workspace, expansion +
// counting-sort + generation kernels (home-cell dedup), CUB toggle, and the
// sorted-grid host driver.

namespace
{

struct SortedGridWorkspace
{
    std::uint32_t* hist { nullptr };            // binCount = 2*cellCount
    std::uint32_t* starts { nullptr };          // binCount + 1 (exclusive scan)
    std::uint32_t* cursors { nullptr };         // binCount (scatter write cursors)
    std::uint32_t* keys { nullptr };            // incidence keys (unsorted)
    std::uint32_t* vals { nullptr };            // incidence triangle ids (unsorted)
    std::uint32_t* keysOut { nullptr };         // CUB route sorted keys (unused otherwise)
    std::uint32_t* valsSorted { nullptr };      // sorted triangle ids (both routes)
    DeviceAabb* firstAabbs { nullptr };         // per-triangle inflated AABBs (home-cell)
    DeviceAabb* secondAabbs { nullptr };
    std::uint32_t* mixedCellIds { nullptr };    // cellCount capacity
    std::uint64_t* candidatePairs { nullptr };
    std::uint32_t* compactCandidatePairs { nullptr };
    unsigned long long* pairHashKeys { nullptr };        // pair-hash dedup fallback only
    std::uint32_t* compactPairHashKeys { nullptr };
    std::uint32_t* touchedPairHashSlots { nullptr };
    std::uint32_t* incidenceCount { nullptr };
    std::uint32_t* incidenceOverflowCount { nullptr };
    std::uint32_t* mixedCellCount { nullptr };
    std::uint32_t* candidateCount { nullptr };
    std::uint32_t* overflowCount { nullptr };
    std::uint32_t* probeOverflowCount { nullptr };
    std::uint32_t* touchedPairHashCount { nullptr };
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
    void* cubTemp { nullptr };                           // CUB route temp storage
    std::size_t cubTempBytes { 0 };

    std::size_t histCapacity { 0 };
    std::size_t startsCapacity { 0 };
    std::size_t cursorsCapacity { 0 };
    std::size_t keysCapacity { 0 };
    std::size_t valsCapacity { 0 };
    std::size_t keysOutCapacity { 0 };
    std::size_t valsSortedCapacity { 0 };
    std::size_t firstAabbCapacity { 0 };
    std::size_t secondAabbCapacity { 0 };
    std::size_t mixedCellIdCapacity { 0 };
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
    bool firstFrameDone { false };
    CudaGraphReplayer graph;
    // CUB radix-sort health: 0 = unprobed, 1 = verified good, -1 = produced an
    // invalid ordering on this system -> process-wide fallback to the counting
    // sort. (Observed on WSL2/CUDA 12.0: DeviceRadixSort intermittently returns
    // cudaSuccess with garbage output, decided per process; frame-0 probe
    // catches it because the failure mode is process-constant.)
    int cubSortHealth { 0 };

    ~SortedGridWorkspace() { release(); }

    void release()
    {
        cudaFree(hist); cudaFree(starts); cudaFree(cursors);
        cudaFree(keys); cudaFree(vals); cudaFree(keysOut); cudaFree(valsSorted);
        cudaFree(firstAabbs); cudaFree(secondAabbs); cudaFree(mixedCellIds);
        cudaFree(candidatePairs); cudaFree(compactCandidatePairs);
        cudaFree(pairHashKeys); cudaFree(compactPairHashKeys); cudaFree(touchedPairHashSlots);
        cudaFree(incidenceCount); cudaFree(incidenceOverflowCount); cudaFree(mixedCellCount);
        cudaFree(candidateCount); cudaFree(overflowCount); cudaFree(probeOverflowCount);
        cudaFree(touchedPairHashCount);
        cudaFree(proximityContacts);
        cudaFree(proximityContactCount); cudaFree(proximityOverflowCount);
        cudaFree(proximityVfCount); cudaFree(proximityFvCount); cudaFree(proximityEeCount);
        cudaFree(firstIndices); cudaFree(secondIndices);
        cudaFree(firstPositions); cudaFree(secondPositions);
        cudaFree(cubTemp);
        cudaFreeHost(countersHostPinned);
        if (eventsReady) { cudaEventDestroy(startEvent); cudaEventDestroy(endEvent); }
        graph.release();
        hist = nullptr; starts = nullptr; cursors = nullptr;
        keys = nullptr; vals = nullptr; keysOut = nullptr; valsSorted = nullptr;
        firstAabbs = nullptr; secondAabbs = nullptr; mixedCellIds = nullptr;
        candidatePairs = nullptr; compactCandidatePairs = nullptr;
        pairHashKeys = nullptr; compactPairHashKeys = nullptr; touchedPairHashSlots = nullptr;
        incidenceCount = nullptr; incidenceOverflowCount = nullptr; mixedCellCount = nullptr;
        candidateCount = nullptr; overflowCount = nullptr; probeOverflowCount = nullptr;
        touchedPairHashCount = nullptr;
        proximityContacts = nullptr;
        proximityContactCount = nullptr; proximityOverflowCount = nullptr;
        proximityVfCount = nullptr; proximityFvCount = nullptr; proximityEeCount = nullptr;
        firstIndices = nullptr; secondIndices = nullptr;
        firstPositions = nullptr; secondPositions = nullptr;
        cubTemp = nullptr; cubTempBytes = 0;
        countersHostPinned = nullptr;
        startEvent = nullptr; endEvent = nullptr; eventsReady = false;
        histCapacity = 0; startsCapacity = 0; cursorsCapacity = 0;
        keysCapacity = 0; valsCapacity = 0; keysOutCapacity = 0; valsSortedCapacity = 0;
        firstAabbCapacity = 0; secondAabbCapacity = 0; mixedCellIdCapacity = 0;
        candidateCapacity = 0; compactCandidateCapacity = 0;
        pairHashCapacity = 0; compactPairHashCapacity = 0; touchedPairHashSlotCapacity = 0;
        contactCapacity = 0;
        firstIndexCapacity = 0; secondIndexCapacity = 0;
        firstPositionCapacity = 0; secondPositionCapacity = 0;
        firstSurfaceId = 0; secondSurfaceId = 0;
        firstTopologyVersion = ~0ull; secondTopologyVersion = ~0ull;
        pairHashKeysInitialized = false; compactPairHashKeysInitialized = false;
        firstFrameDone = false;
        cubSortHealth = 0;
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
        const std::size_t binCount,
        const std::size_t cellCount,
        const std::size_t incidenceCapacity,
        const std::size_t maxCandidatePairs,
        const std::size_t pairHashCount,       // 0 when home-cell dedup (no table)
        const std::size_t maxContacts,
        const std::size_t contactElementBytes,
        const std::size_t firstIndexCount,
        const std::size_t secondIndexCount,
        const std::size_t firstVertexCount,
        const std::size_t secondVertexCount,
        const std::size_t firstTriangleCount,
        const std::size_t secondTriangleCount,
        const bool compactPairs,
        const bool useCub,
        const int cubEndBit,
        std::uint64_t& newlyAllocatedBytes)
    {
        cudaError_t err = ensureDeviceArray(hist, histCapacity, binCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(starts, startsCapacity, binCount + 1u, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(cursors, cursorsCapacity, binCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(keys, keysCapacity, incidenceCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(vals, valsCapacity, incidenceCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess && useCub) err = ensureDeviceArray(keysOut, keysOutCapacity, incidenceCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(valsSorted, valsSortedCapacity, incidenceCapacity, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(firstAabbs, firstAabbCapacity, firstTriangleCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(secondAabbs, secondAabbCapacity, secondTriangleCount, newlyAllocatedBytes);
        if (err == cudaSuccess) err = ensureDeviceArray(mixedCellIds, mixedCellIdCapacity, cellCount, newlyAllocatedBytes);
        if (err == cudaSuccess && !compactPairs) err = ensureDeviceArray(candidatePairs, candidateCapacity, maxCandidatePairs, newlyAllocatedBytes);
        if (err == cudaSuccess && compactPairs) err = ensureDeviceArray(compactCandidatePairs, compactCandidateCapacity, maxCandidatePairs, newlyAllocatedBytes);
        if (err == cudaSuccess && pairHashCount > 0 && !compactPairs)
        {
            if (pairHashCount > pairHashCapacity) pairHashKeysInitialized = false;
            err = ensureDeviceArray(pairHashKeys, pairHashCapacity, pairHashCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess && pairHashCount > 0 && compactPairs)
        {
            if (pairHashCount > compactPairHashCapacity) compactPairHashKeysInitialized = false;
            err = ensureDeviceArray(compactPairHashKeys, compactPairHashCapacity, pairHashCount, newlyAllocatedBytes);
        }
        if (err == cudaSuccess && pairHashCount > 0) err = ensureDeviceArray(touchedPairHashSlots, touchedPairHashSlotCapacity, pairHashCount, newlyAllocatedBytes);
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
        if (err == cudaSuccess && incidenceCount == nullptr)          err = cudaMallocTracked(incidenceCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && incidenceOverflowCount == nullptr)  err = cudaMallocTracked(incidenceOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && mixedCellCount == nullptr)          err = cudaMallocTracked(mixedCellCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && candidateCount == nullptr)          err = cudaMallocTracked(candidateCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && overflowCount == nullptr)           err = cudaMallocTracked(overflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && probeOverflowCount == nullptr)      err = cudaMallocTracked(probeOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && touchedPairHashCount == nullptr)    err = cudaMallocTracked(touchedPairHashCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityContactCount == nullptr)   err = cudaMallocTracked(proximityContactCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityOverflowCount == nullptr)  err = cudaMallocTracked(proximityOverflowCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityVfCount == nullptr)        err = cudaMallocTracked(proximityVfCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityFvCount == nullptr)        err = cudaMallocTracked(proximityFvCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && proximityEeCount == nullptr)        err = cudaMallocTracked(proximityEeCount, 1, newlyAllocatedBytes);
        if (err == cudaSuccess && countersHostPinned == nullptr)
        {
            err = cudaMallocHost(reinterpret_cast<void**>(&countersHostPinned), 12u * sizeof(std::uint32_t));
            if (err == cudaSuccess) newlyAllocatedBytes += 12u * sizeof(std::uint32_t);
        }
        // CUB temp storage sized for the padded incidence sort (queried once,
        // preallocated so the sort stays CUDA-graph capturable).
        if (err == cudaSuccess && useCub)
        {
            std::size_t requiredBytes = 0;
            err = cub::DeviceRadixSort::SortPairs(
                nullptr, requiredBytes,
                keys, keysOut, vals, valsSorted,
                static_cast<int>(incidenceCapacity), 0, cubEndBit);
            if (err == cudaSuccess && requiredBytes > cubTempBytes)
            {
                cudaFree(cubTemp); cubTemp = nullptr; cubTempBytes = 0;
                err = cudaMalloc(&cubTemp, requiredBytes);
                if (err == cudaSuccess) { cubTempBytes = requiredBytes; newlyAllocatedBytes += requiredBytes; }
            }
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

SortedGridWorkspace& sortedGridWorkspace()
{
    static SortedGridWorkspace workspace;
    return workspace;
}

__global__ void resetSortedGridKernel(
    std::uint32_t* hist,
    const std::uint32_t binCount,
    std::uint32_t* incidenceCount,
    std::uint32_t* incidenceOverflowCount,
    std::uint32_t* mixedCellCount,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount,
    std::uint32_t* touchedPairHashCount)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < binCount; i += stride)
    {
        hist[i] = 0u;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *incidenceCount = 0u;
        *incidenceOverflowCount = 0u;
        *mixedCellCount = 0u;
        *candidateCount = 0u;
        *overflowCount = 0u;
        *probeOverflowCount = 0u;
        if (touchedPairHashCount != nullptr) *touchedPairHashCount = 0u;
    }
}

// One thread per triangle: store its inflated AABB, reserve a contiguous slice
// of the incidence buffer with a single atomicAdd (order does not matter — the
// sort restores it), write one (key, triId) entry per overlapped cell, and bump
// the per-key histogram in the same pass (no separate histogram kernel).
__global__ void expandSortedGridIncidencesKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t* __restrict__ triangleIndices,
    const std::uint32_t triangleCount,
    const bool isTool,
    const DeviceDenseGridConfig config,
    const std::uint32_t incidenceCapacity,
    std::uint32_t* hist,
    std::uint32_t* keys,
    std::uint32_t* vals,
    DeviceAabb* triangleAabbs,
    std::uint32_t* incidenceCount,
    std::uint32_t* incidenceOverflowCount)
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
    const std::uint32_t base = atomicAdd(incidenceCount, span);

    std::uint32_t offset = 0;
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int x = cellMin.x; x <= cellMax.x; ++x, ++offset)
    {
        const std::uint32_t slot = base + offset;
        if (slot >= incidenceCapacity)
        {
            atomicAdd(incidenceOverflowCount, 1u);
            continue;
        }
        const std::uint32_t key = (denseCellId(x, y, z, config) << 1) | (isTool ? 1u : 0u);
        keys[slot] = key;
        vals[slot] = triangleId;
        atomicAdd(&hist[key], 1u);
    }
}

// Single-block chained exclusive scan over the per-key histogram. binCount is
// at most a few hundred thousand, so one 1024-thread block sweeping it in
// chunks is microseconds — and the result doubles as (a) the cell-run starts
// used by generation and (b) the scatter write cursors. starts[binCount] gets
// the grand total.
__global__ void exclusiveScanSortedGridBinsKernel(
    const std::uint32_t* __restrict__ hist,
    std::uint32_t* starts,
    std::uint32_t* cursors,
    const std::uint32_t binCount)
{
    __shared__ std::uint32_t temp[2][1024];
    __shared__ std::uint32_t chunkTotal;
    const std::uint32_t tid = threadIdx.x;
    std::uint32_t running = 0;
    for (std::uint32_t base = 0; base < binCount; base += blockDim.x)
    {
        const std::uint32_t i = base + tid;
        const std::uint32_t v = (i < binCount) ? hist[i] : 0u;
        std::uint32_t buf = 0;
        temp[0][tid] = v;
        __syncthreads();
        for (std::uint32_t off = 1; off < blockDim.x; off <<= 1)
        {
            const std::uint32_t dst = 1u - buf;
            std::uint32_t x = temp[buf][tid];
            if (tid >= off) x += temp[buf][tid - off];
            temp[dst][tid] = x;
            buf = dst;
            __syncthreads();
        }
        const std::uint32_t inclusive = temp[buf][tid];
        if (i < binCount)
        {
            const std::uint32_t s = running + (inclusive - v);
            starts[i] = s;
            cursors[i] = s;
        }
        if (tid == blockDim.x - 1) chunkTotal = inclusive;
        __syncthreads();
        running += chunkTotal;
        __syncthreads();
    }
    if (tid == 0) starts[binCount] = running;
}

// Counting-sort scatter: place each incidence's triangle id at its bin cursor.
// Order inside a bin is nondeterministic (atomic), which is fine — pairs are
// consumed independently by the FBP kernel.
__global__ void scatterSortedGridIncidencesKernel(
    const std::uint32_t* __restrict__ keys,
    const std::uint32_t* __restrict__ vals,
    const std::uint32_t* incidenceCount,
    const std::uint32_t incidenceCapacity,
    std::uint32_t* cursors,
    std::uint32_t* valsSorted)
{
    const std::uint32_t count = min(*incidenceCount, incidenceCapacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const std::uint32_t pos = atomicAdd(&cursors[keys[i]], 1u);
        valsSorted[pos] = vals[i];
    }
}

// Pad the unwritten tail of the key buffer so it sorts after every real key
// (CUB route only). This deliberately runs as a compute kernel AFTER the
// expansion kernels instead of a whole-buffer cudaMemsetAsync BEFORE them: an
// async memset can execute on a copy/DMA engine, and on WSL2 the engine
// assignment is decided per process — losing that ordering race clobbered every
// real key to 0xffffffff and silently emptied the sorted grid (observed as a
// per-process all-frames-broken coin flip, 2026-07-03). A fill kernel on the
// same stream is unambiguous.
__global__ void padSortedGridKeysKernel(
    std::uint32_t* keys,
    const std::uint32_t* incidenceCount,
    const std::uint32_t incidenceCapacity)
{
    const std::uint32_t total = min(*incidenceCount, incidenceCapacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = total + blockIdx.x * blockDim.x + threadIdx.x; i < incidenceCapacity; i += stride)
    {
        keys[i] = 0xffffffffu;
    }
}

// Diagnostic (SOFA_SORTED_GRID_VERIFY=1): validate the sorted key stream —
// ascending order, no padding keys inside the real range, and every entry
// sitting inside its bin's [starts[k], starts[k+1]) run (the exact property the
// generation kernel relies on). Violations land in probeOverflowCount.
__global__ void verifySortedGridRunsKernel(
    const std::uint32_t* __restrict__ keysOut,
    const std::uint32_t* __restrict__ starts,
    const std::uint32_t* incidenceCount,
    const std::uint32_t incidenceCapacity,
    const std::uint32_t binCount,
    std::uint32_t* violations)
{
    const std::uint32_t total = min(*incidenceCount, incidenceCapacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const std::uint32_t k = keysOut[i];
        bool bad = (k >= binCount);
        if (!bad && (i < starts[k] || i >= starts[k + 1u])) bad = true;
        if (i > 0 && keysOut[i - 1u] > k) bad = true;
        if (bad) atomicAdd(violations, 1u);
    }
}

__global__ void buildSortedGridMixedCellsKernel(
    const std::uint32_t* __restrict__ hist,
    const std::uint32_t cellCount,
    std::uint32_t* mixedCellIds,
    std::uint32_t* mixedCellCount)
{
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cellCount) return;
    if (hist[2u * c] > 0u && hist[2u * c + 1u] > 0u)
    {
        mixedCellIds[atomicAdd(mixedCellCount, 1u)] = c;
    }
}

// Clamped cell of a point, matching denseGridCellSpan's clamping exactly so
// the home-cell owner is always one of the cells both triangles were binned in.
__device__ __forceinline__ std::uint32_t sortedGridClampedCellOfPoint(
    const float px, const float py, const float pz, const DeviceDenseGridConfig& config)
{
    int x = static_cast<int>(floorf((px - config.gridMin.x) * config.inverseCellSize.x));
    int y = static_cast<int>(floorf((py - config.gridMin.y) * config.inverseCellSize.y));
    int z = static_cast<int>(floorf((pz - config.gridMin.z) * config.inverseCellSize.z));
    x = max(0, min(x, static_cast<int>(config.resolutionX) - 1));
    y = max(0, min(y, static_cast<int>(config.resolutionY) - 1));
    z = max(0, min(z, static_cast<int>(config.resolutionZ) - 1));
    return denseCellId(x, y, z, config);
}

// One block per mixed cell. The tool run (typically the short side) is staged
// through shared memory in 256-entry tiles (ids + AABBs) — the shared-memory
// privatization step — and the block's threads split the tissue x toolTile
// cross product by divide/modulo. Dedup is home-cell exactly-once by default
// (emit only from the cell holding the min corner of the two AABBs' overlap;
// disjoint AABBs are skipped outright, which is safe because both are inflated
// by contactDistance) or the shared pair-hash when usePairHashDedup is set
// (that mode reproduces the other ways' candidate set exactly).
// One template for both pair widths (was a hand-written 32/64 twin pair);
// PairTraits comes from HashGrid.cuh, included before this file.
template <class PairT>
__global__ void generateSortedGridCandidatePairsKernel(
    const std::uint32_t* __restrict__ starts,
    const std::uint32_t* __restrict__ valsSorted,
    const std::uint32_t* __restrict__ mixedCellIds,
    const std::uint32_t* mixedCellCount,
    const DeviceAabb* __restrict__ tissueAabbs,
    const DeviceAabb* __restrict__ toolAabbs,
    const DeviceDenseGridConfig config,
    const bool usePairHashDedup,
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
    __shared__ std::uint32_t sToolIds[256];
    __shared__ DeviceAabb sToolAabbs[256];
    const std::uint32_t mixedCount = *mixedCellCount;
    for (std::uint32_t m = blockIdx.x; m < mixedCount; m += gridDim.x)
    {
        const std::uint32_t cell = mixedCellIds[m];
        const std::uint32_t kT = cell << 1;
        const std::uint32_t t0 = starts[kT];
        const std::uint32_t u0 = starts[kT + 1u];
        const std::uint32_t u1 = starts[kT + 2u];
        const std::uint32_t tCount = u0 - t0;
        const std::uint32_t uCount = u1 - u0;
        for (std::uint32_t uBase = 0; uBase < uCount; uBase += 256u)
        {
            const std::uint32_t uTile = min(256u, uCount - uBase);
            __syncthreads();
            if (threadIdx.x < uTile)
            {
                const std::uint32_t ui = valsSorted[u0 + uBase + threadIdx.x];
                sToolIds[threadIdx.x] = ui;
                sToolAabbs[threadIdx.x] = toolAabbs[ui];
            }
            __syncthreads();
            const std::uint32_t pairsInTile = tCount * uTile;
            for (std::uint32_t lp = threadIdx.x; lp < pairsInTile; lp += blockDim.x)
            {
                const std::uint32_t tissueTriId = valsSorted[t0 + lp / uTile];
                const std::uint32_t uLocal = lp % uTile;
                const std::uint32_t toolTriId = sToolIds[uLocal];
                if (usePairHashDedup)
                {
                    insertUniqueCandidatePairTracked<PairT>(
                        PairTraits<PairT>::encode(tissueTriId, toolTriId),
                        pairHashKeys, touchedPairHashSlots, touchedPairHashCount,
                        pairHashCapacity, candidatePairs, candidateCount,
                        overflowCount, probeOverflowCount, maxCandidatePairs);
                    continue;
                }
                const DeviceAabb a = tissueAabbs[tissueTriId];
                const DeviceAabb b = sToolAabbs[uLocal];
                const float mx = fmaxf(a.minX, b.minX);
                const float my = fmaxf(a.minY, b.minY);
                const float mz = fmaxf(a.minZ, b.minZ);
                if (mx > fminf(a.maxX, b.maxX) || my > fminf(a.maxY, b.maxY) || mz > fminf(a.maxZ, b.maxZ))
                {
                    continue;  // inflated AABBs disjoint -> distance > contactDistance
                }
                if (sortedGridClampedCellOfPoint(mx, my, mz, config) != cell)
                {
                    continue;  // another (shared) cell owns this pair
                }
                const std::uint32_t idx = atomicAdd(candidateCount, 1u);
                if (idx < maxCandidatePairs) candidatePairs[idx] = PairTraits<PairT>::encode(tissueTriId, toolTriId);
                else atomicAdd(overflowCount, 1u);
            }
        }
        __syncthreads();
    }
}


} // namespace


namespace SofaGpuCollision::backend
{

bool computeSortedGridProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const SortedGridConfig& sortedConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    SortedGridStats* sortedStats,
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
    if (sortedStats != nullptr) *sortedStats = SortedGridStats {};

    if (indexedSurfaceInvalid(firstSurface) || indexedSurfaceInvalid(secondSurface))
    {
        diagnostic = "Invalid indexed triangle surface (sorted-grid path).";
        return false;
    }
    if (gridConfig.gridResolutionX == 0 || gridConfig.gridResolutionY == 0 || gridConfig.gridResolutionZ == 0 ||
        gridConfig.maxCandidatePairs == 0 ||
        gridConfig.gridMaxX <= gridConfig.gridMinX ||
        gridConfig.gridMaxY <= gridConfig.gridMinY ||
        gridConfig.gridMaxZ <= gridConfig.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration (sorted-grid path).";
        return false;
    }

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(gridConfig.gridResolutionX) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionY) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 * 2ull + 2ull > std::numeric_limits<std::uint32_t>::max() / 2u ||
        (cellCount64 << 1) >= (1ull << 31))
    {
        diagnostic = "Sorted-grid cell count out of range.";
        return false;
    }
    const std::uint32_t cellCount = static_cast<std::uint32_t>(cellCount64);
    const std::uint32_t binCount = cellCount * 2u;

    // CUB sort width: smallest bit count covering all real keys, plus one so
    // the 0xffffffff padding entries sort strictly after every real key.
    int cubEndBit = 1;
    while ((1u << cubEndBit) < binCount) ++cubEndBit;
    ++cubEndBit;

    const std::uint32_t totalTriangles = firstSurface.triangleCount + secondSurface.triangleCount;
    const std::uint32_t incidenceCapacity = std::max(
        4096u,
        std::max(1u, sortedConfig.incidenceCapacityScale) * totalTriangles);
    const bool useCompactCandidatePairs =
        firstSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max() &&
        secondSurface.triangleCount <= std::numeric_limits<std::uint16_t>::max();
    const bool usePairHashDedup = sortedConfig.usePairHashDedup;
    std::uint64_t pairHashCount64 = 0;
    if (usePairHashDedup)
    {
        pairHashCount64 = static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(gridConfig.maxCandidatePairs) * 2u));
        if (pairHashCount64 == 0 || pairHashCount64 > std::numeric_limits<std::uint32_t>::max())
        {
            diagnostic = "Sorted-grid pair-hash table size out of range.";
            return false;
        }
    }
    const std::uint32_t pairHashCount = static_cast<std::uint32_t>(pairHashCount64);

    auto& ws = sortedGridWorkspace();
    // Engine actually used this frame: the CUB request is honoured only while
    // the frame-0 health probe hasn't condemned it (see cubSortHealth).
    bool useCubEffective = sortedConfig.useCubRadixSort && ws.cubSortHealth >= 0;
    std::uint64_t newlyAllocatedBytes = 0;
    const auto allocStart = std::chrono::steady_clock::now();
    cudaError_t err = ws.ensure(
        binCount,
        cellCount,
        incidenceCapacity,
        gridConfig.maxCandidatePairs,
        static_cast<std::size_t>(pairHashCount64),
        proximityConfig.maxContacts,
        sizeof(DeviceProximityContact),
        static_cast<std::size_t>(firstSurface.triangleCount) * 3u,
        static_cast<std::size_t>(secondSurface.triangleCount) * 3u,
        firstSurface.devicePositions == nullptr ? firstSurface.vertexCount : 0u,
        secondSurface.devicePositions == nullptr ? secondSurface.vertexCount : 0u,
        firstSurface.triangleCount,
        secondSurface.triangleCount,
        useCompactCandidatePairs,
        useCubEffective,
        cubEndBit,
        newlyAllocatedBytes);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Sorted-grid workspace alloc failed: ") + cudaGetErrorString(err);
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

    const DeviceDenseGridConfig dc = makeDeviceDenseGridConfig(gridConfig, pairHashCount);

    constexpr std::uint32_t threads = 256;
    const std::uint32_t resetBlocks = std::max(1u, std::min(1024u, (binCount + threads - 1u) / threads));
    const std::uint32_t firstBlocks = (firstSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t secondBlocks = (secondSurface.triangleCount + threads - 1u) / threads;
    const std::uint32_t mixedBlocks = (cellCount + threads - 1u) / threads;
    constexpr std::uint32_t kGenBlocks = 1024;
    const std::uint32_t pairHashClearBlocks = usePairHashDedup
        ? std::max(1u, std::min(kGenBlocks, (pairHashCount + threads - 1u) / threads))
        : 1u;

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
    cudaError_t cubSortError = cudaSuccess;  // captured from the CUB call inside launchAll
    auto* deviceContacts = reinterpret_cast<DeviceProximityContact*>(ws.proximityContacts);

    // The whole per-frame kernel sequence on stream s (direct exec + graph
    // capture). fullClear only matters in pair-hash dedup mode (one-time table
    // memset); home-cell mode has no persistent tables to initialize.
    auto launchAll = [&](cudaStream_t s, bool fullClear)
    {
        if (usePairHashDedup)
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
        }
        resetSortedGridKernel<<<resetBlocks, threads, 0, s>>>(ws.hist, binCount, ws.incidenceCount, ws.incidenceOverflowCount, ws.mixedCellCount, ws.candidateCount, ws.overflowCount, ws.probeOverflowCount, usePairHashDedup ? ws.touchedPairHashCount : nullptr);
        expandSortedGridIncidencesKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, false, dc, incidenceCapacity, ws.hist, ws.keys, ws.vals, ws.firstAabbs, ws.incidenceCount, ws.incidenceOverflowCount);
        expandSortedGridIncidencesKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, true, dc, incidenceCapacity, ws.hist, ws.keys, ws.vals, ws.secondAabbs, ws.incidenceCount, ws.incidenceOverflowCount);
        if (useCubEffective)
        {
            // Pad the unwritten key tail AFTER expansion (see kernel comment —
            // never a whole-buffer async memset before it).
            padSortedGridKeysKernel<<<kGenBlocks, threads, 0, s>>>(ws.keys, ws.incidenceCount, incidenceCapacity);
        }
        exclusiveScanSortedGridBinsKernel<<<1, 1024, 0, s>>>(ws.hist, ws.starts, ws.cursors, binCount);
        if (useCubEffective)
        {
            std::size_t tempBytes = ws.cubTempBytes;
            const cudaError_t sortErr = cub::DeviceRadixSort::SortPairs(
                ws.cubTemp, tempBytes,
                ws.keys, ws.keysOut, ws.vals, ws.valsSorted,
                static_cast<int>(incidenceCapacity), 0, cubEndBit, s);
            if (sortErr != cudaSuccess && cubSortError == cudaSuccess) cubSortError = sortErr;
            static const bool kVerifySort = []{ const char* e = std::getenv("SOFA_SORTED_GRID_VERIFY"); return e && std::atoi(e) != 0; }();
            if (kVerifySort)
            {
                verifySortedGridRunsKernel<<<kGenBlocks, threads, 0, s>>>(ws.keysOut, ws.starts, ws.incidenceCount, incidenceCapacity, binCount, ws.probeOverflowCount);
            }
        }
        else
        {
            scatterSortedGridIncidencesKernel<<<kGenBlocks, threads, 0, s>>>(ws.keys, ws.vals, ws.incidenceCount, incidenceCapacity, ws.cursors, ws.valsSorted);
        }
        buildSortedGridMixedCellsKernel<<<mixedBlocks, threads, 0, s>>>(ws.hist, cellCount, ws.mixedCellIds, ws.mixedCellCount);
        if (useCompactCandidatePairs)
            generateSortedGridCandidatePairsKernel<std::uint32_t><<<kGenBlocks, threads, 0, s>>>(ws.starts, ws.valsSorted, ws.mixedCellIds, ws.mixedCellCount, ws.firstAabbs, ws.secondAabbs, dc, usePairHashDedup, pairHashCount, gridConfig.maxCandidatePairs, ws.compactCandidatePairs, ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        else
            generateSortedGridCandidatePairsKernel<std::uint64_t><<<kGenBlocks, threads, 0, s>>>(ws.starts, ws.valsSorted, ws.mixedCellIds, ws.mixedCellCount, ws.firstAabbs, ws.secondAabbs, dc, usePairHashDedup, pairHashCount, gridConfig.maxCandidatePairs, ws.candidatePairs, ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        resetProximityCountersKernel<<<1, 1, 0, s>>>(ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount);
        featureBasedProximityKernel<<<kGenBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, deviceSecondPositions, ws.secondIndices, useCompactCandidatePairs ? nullptr : ws.candidatePairs, useCompactCandidatePairs ? ws.compactCandidatePairs : nullptr, ws.candidateCount, useCompactCandidatePairs, deviceContacts, ws.proximityContactCount, ws.proximityOverflowCount, ws.proximityVfCount, ws.proximityFvCount, ws.proximityEeCount, proximityConfig.maxContacts, proximityConfig.contactDistance, proximityConfig.computeBarycentrics);
    };

    // Frame-0 CUB health probe: verify the sorted stream once per process
    // (one small kernel + a 4-byte sync on the first frame only). If the sort
    // came back invalid — the WSL2 failure documented on the workspace field —
    // flip this process to the counting engine and redo the frame.
    auto probeCubSortHealth = [&]() -> cudaError_t
    {
        if (!useCubEffective || ws.cubSortHealth != 0) return cudaSuccess;
        verifySortedGridRunsKernel<<<kGenBlocks, threads>>>(ws.keysOut, ws.starts, ws.incidenceCount, incidenceCapacity, binCount, ws.probeOverflowCount);
        std::uint32_t violations = 0;
        const cudaError_t copyErr = cudaMemcpy(&violations, ws.probeOverflowCount, sizeof(violations), cudaMemcpyDeviceToHost);
        if (copyErr != cudaSuccess) return copyErr;
        if (violations == 0)
        {
            ws.cubSortHealth = 1;
            return cudaSuccess;
        }
        ws.cubSortHealth = -1;
        useCubEffective = false;
        std::fprintf(stderr,
            "[SofaGpuCollision] sorted-grid: CUB radix sort returned an invalid ordering on this system "
            "(%u violations) — falling back to the counting sort for this process.\n",
            violations);
        launchAll(0, /*fullClear=*/false);  // redo the frame on the counting engine
        return cudaGetLastError();
    };

    if (measure) { err = cudaEventRecord(ws.startEvent); if (err != cudaSuccess) { diagnostic = cudaGetErrorString(err); return false; } }

    // CUDA-graph fast path (default ON; disable with SOFA_SORTED_GRID_CUDA_GRAPH=0).
    static const int kGraphMode = []{ const char* e = std::getenv("SOFA_SORTED_GRID_CUDA_GRAPH"); return e ? std::atoi(e) : 1; }();
    const bool kGraphsEnabled = (kGraphMode != 0);
    bool ranViaGraph = false;
    const bool pairHashNeedsInit = usePairHashDedup &&
        !(useCompactCandidatePairs ? ws.compactPairHashKeysInitialized : ws.pairHashKeysInitialized);
    const bool firstFrame = !ws.firstFrameDone || pairHashNeedsInit;
    if (kGraphsEnabled && !detailedProfiling)
    {
        const std::array<std::uint64_t, 6> graphSignature {
            (static_cast<std::uint64_t>(binCount) << 32) | incidenceCapacity,
            (static_cast<std::uint64_t>(firstSurface.triangleCount) << 32) | secondSurface.triangleCount,
            (static_cast<std::uint64_t>(proximityConfig.maxContacts) << 32) | pairHashCount,
            (useCompactCandidatePairs ? 1ull : 0ull) | (proximityConfig.computeBarycentrics ? 2ull : 0ull) |
                (useCubEffective ? 4ull : 0ull) | (usePairHashDedup ? 8ull : 0ull),
            floatSignatureBits(proximityConfig.contactDistance),
            0ull };
        ws.graph.invalidateIfChanged(graphSignature, newlyAllocatedBytes > 0);

        if (firstFrame)
        {
            launchAll(0, /*fullClear=*/usePairHashDedup);
            err = probeCubSortHealth();
            if (err != cudaSuccess) { diagnostic = std::string("sorted-grid cub probe: ") + cudaGetErrorString(err); return false; }
            ws.graph.instantiated = false;
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
        launchAll(0, firstFrame && usePairHashDedup);
        err = probeCubSortHealth();
        if (err != cudaSuccess) { diagnostic = std::string("sorted-grid cub probe: ") + cudaGetErrorString(err); return false; }
    }
    ws.firstFrameDone = true;
    // reset + expand x2 + scan + (scatter|sort) + mixed + generate + counters + FBP
    launchCount += 9u;
    if (usePairHashDedup) { if (firstFrame) memsetCount += 1u; else launchCount += 1u; }
    if (useCubEffective) launchCount += 1u;  // key tail-pad kernel
    if (cubSortError != cudaSuccess)
    {
        diagnostic = std::string("sorted-grid CUB sort: ") + cudaGetErrorString(cubSortError);
        return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { diagnostic = std::string("sorted-grid launch: ") + cudaGetErrorString(err); return false; }

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
    }

    // Optional readback (validation / contacts to host).
    std::uint32_t hostUnique = 0, hostIncidences = 0, hostIncidenceOverflow = 0, hostMixed = 0;
    std::uint32_t hostPairOverflow = 0, hostProbe = 0, hostContactCount = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* p = ws.countersHostPinned;
        cudaMemcpyAsync(p + 0, ws.candidateCount,          sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 1, ws.incidenceCount,          sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 2, ws.incidenceOverflowCount,  sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 3, ws.mixedCellCount,          sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 4, ws.overflowCount,           sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 5, ws.probeOverflowCount,      sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 6, ws.proximityContactCount,   sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 7, ws.proximityOverflowCount,  sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 8, ws.proximityVfCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 9, ws.proximityFvCount,        sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(p + 10, ws.proximityEeCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostUnique = p[0]; hostIncidences = p[1]; hostIncidenceOverflow = p[2]; hostMixed = p[3];
        hostPairOverflow = p[4]; hostProbe = p[5]; hostContactCount = p[6];
        hostVf = p[8]; hostFv = p[9]; hostEe = p[10];
        const std::uint32_t hostProxOverflow = p[7];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 11u * sizeof(std::uint32_t);
            executionStats->uniqueCandidateCount += hostUnique;
            executionStats->outputCandidateCount += hostUnique;
            executionStats->rawCandidateCount += hostIncidences;   // incidences stand in for "raw" work
            executionStats->outputContactCount = hostContactCount;
            executionStats->activeMixedCellCount += hostMixed;
            executionStats->vfContactCount += hostVf;
            executionStats->fvContactCount += hostFv;
            executionStats->eeContactCount += hostEe;
            const std::uint32_t totalOverflow = hostPairOverflow + hostIncidenceOverflow + hostProxOverflow;
            if (totalOverflow > 0) executionStats->overflowCount += totalOverflow;
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
        if (sortedStats != nullptr)
        {
            sortedStats->binCount = binCount;
            sortedStats->incidenceCount = hostIncidences;
            sortedStats->mixedCellCount = hostMixed;
            sortedStats->uniquePairCount = hostUnique;
            sortedStats->incidenceOverflowCount = hostIncidenceOverflow;
            sortedStats->pairOverflowCount = hostPairOverflow;
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


} // namespace SofaGpuCollision::backend
