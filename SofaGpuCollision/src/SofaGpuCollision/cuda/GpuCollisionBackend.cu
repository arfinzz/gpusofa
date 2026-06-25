#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <cuda_runtime.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SOFAGPUCOLLISION_HAS_NVTX 1
#else
#define SOFAGPUCOLLISION_HAS_NVTX 0
#endif
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{

struct DeviceAabb
{
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
};

struct DeviceTreePairRange
{
    std::uint32_t firstOffset;
    std::uint32_t firstCount;
    std::uint32_t secondOffset;
    std::uint32_t secondCount;
    std::uint32_t firstBaseIndex;
    std::uint32_t secondBaseIndex;
};

struct DeviceIndexPair
{
    std::uint32_t first;
    std::uint32_t second;
};

struct DeviceContactCandidate
{
    std::uint32_t pairIndex;
    std::uint32_t firstLeafIndex;
    std::uint32_t secondLeafIndex;
};

struct DeviceTriangle
{
    float3 p0;
    float3 p1;
    float3 p2;
    std::uint32_t triangleIndex;
};

static_assert(std::is_trivially_copyable_v<SofaGpuCollision::backend::TriangleVertex>);
static_assert(std::is_trivially_copyable_v<SofaGpuCollision::backend::TrianglePrimitive>);
static_assert(sizeof(SofaGpuCollision::backend::TriangleVertex) == sizeof(float3));
static_assert(alignof(SofaGpuCollision::backend::TriangleVertex) == alignof(float3));
static_assert(sizeof(SofaGpuCollision::backend::TrianglePrimitive) == sizeof(DeviceTriangle));
static_assert(alignof(SofaGpuCollision::backend::TrianglePrimitive) == alignof(DeviceTriangle));

using BackendTriangleVertex = SofaGpuCollision::backend::TriangleVertex;

struct DeviceExactContact
{
    std::uint32_t firstTriangleIndex;
    std::uint32_t secondTriangleIndex;
    float3 pointOnFirst;
    float3 pointOnSecond;
    float3 normal;
    float signedDistance;
};

struct DeviceDenseGridConfig
{
    float3 gridMin;
    float3 gridMax;
    float3 inverseCellSize;
    std::uint32_t resolutionX;
    std::uint32_t resolutionY;
    std::uint32_t resolutionZ;
    float contactDistance;
    std::uint32_t maxTissueTrianglesPerCell;
    std::uint32_t maxToolTrianglesPerCell;
    std::uint32_t maxCandidatePairs;
    std::uint32_t pairHashCapacity;
    bool useGpuHashDedupe;
    bool canonicalPairEmission;
};

struct DeviceCellBucket
{
    std::uint32_t tissueCount;
    std::uint32_t toolCount;
};

struct DeviceDenseGridStats
{
    std::uint32_t activeMixedCellCount;
    std::uint32_t tissueInsertCount;
    std::uint32_t toolInsertCount;
    std::uint32_t maxTissueCellOccupancy;
    std::uint32_t maxToolCellOccupancy;
    std::uint32_t hashDedupeProbeOverflowCount;
};

constexpr std::uint64_t kEmptyPairSlot = 0xffffffffffffffffull;

struct ScopedNvtxRange
{
    bool enabled { false };

    ScopedNvtxRange(const char* name, const bool active)
        : enabled(active)
    {
#if SOFAGPUCOLLISION_HAS_NVTX
        if (enabled)
        {
            nvtxRangePushA(name);
        }
#else
        (void)name;
        (void)active;
#endif
    }

    ~ScopedNvtxRange()
    {
#if SOFAGPUCOLLISION_HAS_NVTX
        if (enabled)
        {
            nvtxRangePop();
        }
#endif
    }
};

template<class T>
cudaError_t ensureDeviceArray(T*& pointer, std::size_t& capacity, const std::size_t required, std::uint64_t& newlyAllocatedBytes)
{
    if (required <= capacity)
    {
        return cudaSuccess;
    }

    cudaFree(pointer);
    pointer = nullptr;
    capacity = 0;

    const cudaError_t err = cudaMalloc(&pointer, required * sizeof(T));
    if (err == cudaSuccess)
    {
        capacity = required;
        newlyAllocatedBytes += static_cast<std::uint64_t>(required * sizeof(T));
    }
    return err;
}

template<class T>
cudaError_t ensurePinnedHostArray(T*& pointer, std::size_t& capacity, const std::size_t required, std::uint64_t& newlyAllocatedBytes)
{
    if (required <= capacity)
    {
        return cudaSuccess;
    }

    cudaFreeHost(pointer);
    pointer = nullptr;
    capacity = 0;

    const cudaError_t err = cudaHostAlloc(&pointer, required * sizeof(T), cudaHostAllocDefault);
    if (err == cudaSuccess)
    {
        capacity = required;
        newlyAllocatedBytes += static_cast<std::uint64_t>(required * sizeof(T));
    }
    return err;
}

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

std::size_t nextPowerOfTwo(std::size_t value)
{
    if (value <= 1)
    {
        return 1;
    }
    --value;
    for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1)
    {
        value |= value >> shift;
    }
    return value + 1;
}

double elapsedMillisecondsSince(const std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

template<class Callable>
cudaError_t timeCudaOperation(
    cudaEvent_t startEvent,
    cudaEvent_t endEvent,
    double& targetMilliseconds,
    Callable callable)
{
    cudaError_t err = cudaEventRecord(startEvent);
    if (err != cudaSuccess)
    {
        return err;
    }

    err = callable();
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaEventRecord(endEvent);
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaEventSynchronize(endEvent);
    if (err != cudaSuccess)
    {
        return err;
    }

    float elapsedMs = 0.0f;
    err = cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
    if (err == cudaSuccess)
    {
        targetMilliseconds += static_cast<double>(elapsedMs);
    }
    return err;
}

template<class Callable>
cudaError_t runCudaOperation(
    const bool detailedProfiling,
    cudaEvent_t startEvent,
    cudaEvent_t endEvent,
    double& targetMilliseconds,
    Callable callable)
{
    if (detailedProfiling)
    {
        return timeCudaOperation(startEvent, endEvent, targetMilliseconds, callable);
    }

    return callable();
}

template<class T>
cudaError_t copyHostArrayToDeviceAsync(
    T* deviceDestination,
    const T* hostSource,
    T* pinnedStaging,
    const std::size_t elementCount,
    const bool usePinnedStaging)
{
    if (elementCount == 0)
    {
        return cudaSuccess;
    }

    const T* uploadSource = hostSource;
    if (usePinnedStaging)
    {
        std::memcpy(pinnedStaging, hostSource, elementCount * sizeof(T));
        uploadSource = pinnedStaging;
    }

    return cudaMemcpyAsync(
        deviceDestination,
        uploadSource,
        elementCount * sizeof(T),
        cudaMemcpyHostToDevice);
}

__device__ bool overlapsOnAxis(const float aMin, const float aMax, const float bMin, const float bMax)
{
    return aMin <= bMax && bMin <= aMax;
}

__device__ float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 mul3(const float3 a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 min3(const float3 a, const float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ float3 max3(const float3 a, const float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ float lengthSquared3(const float3 v)
{
    return dot3(v, v);
}

__device__ float3 normalizeOrZero(const float3 v)
{
    const float lenSq = lengthSquared3(v);
    if (lenSq <= 1.0e-12f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float invLen = rsqrtf(lenSq);
    return mul3(v, invLen);
}

__device__ void projectTriangle(const DeviceTriangle& triangle, const float3 axis, float& minProjection, float& maxProjection)
{
    const float d0 = dot3(triangle.p0, axis);
    const float d1 = dot3(triangle.p1, axis);
    const float d2 = dot3(triangle.p2, axis);

    minProjection = fminf(d0, fminf(d1, d2));
    maxProjection = fmaxf(d0, fmaxf(d1, d2));
}

__device__ bool testSeparatingAxis(
    const DeviceTriangle& first,
    const DeviceTriangle& second,
    const float3 axis,
    float& bestOverlap,
    float3& bestAxis)
{
    const float axisLenSq = lengthSquared3(axis);
    if (axisLenSq <= 1.0e-12f)
    {
        return true;
    }

    const float3 normalizedAxis = normalizeOrZero(axis);

    float firstMin = 0.0f;
    float firstMax = 0.0f;
    float secondMin = 0.0f;
    float secondMax = 0.0f;
    projectTriangle(first, normalizedAxis, firstMin, firstMax);
    projectTriangle(second, normalizedAxis, secondMin, secondMax);

    if (!overlapsOnAxis(firstMin, firstMax, secondMin, secondMax))
    {
        return false;
    }

    const float overlap = fminf(firstMax, secondMax) - fmaxf(firstMin, secondMin);
    if (overlap < bestOverlap)
    {
        bestOverlap = overlap;
        bestAxis = normalizedAxis;
    }

    return true;
}

__device__ bool exactTriangleIntersection(
    const DeviceTriangle& first,
    const DeviceTriangle& second,
    DeviceExactContact& outContact)
{
    const float3 firstEdges[3] = {
        sub3(first.p1, first.p0),
        sub3(first.p2, first.p1),
        sub3(first.p0, first.p2),
    };
    const float3 secondEdges[3] = {
        sub3(second.p1, second.p0),
        sub3(second.p2, second.p1),
        sub3(second.p0, second.p2),
    };

    const float3 firstNormal = cross3(firstEdges[0], sub3(first.p2, first.p0));
    const float3 secondNormal = cross3(secondEdges[0], sub3(second.p2, second.p0));

    float bestOverlap = 1.0e30f;
    float3 bestAxis = make_float3(0.0f, 0.0f, 0.0f);

    if (!testSeparatingAxis(first, second, firstNormal, bestOverlap, bestAxis))
    {
        return false;
    }
    if (!testSeparatingAxis(first, second, secondNormal, bestOverlap, bestAxis))
    {
        return false;
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (!testSeparatingAxis(first, second, cross3(firstEdges[i], secondEdges[j]), bestOverlap, bestAxis))
            {
                return false;
            }
        }
    }

    const float3 firstCentroid = mul3(add3(add3(first.p0, first.p1), first.p2), 1.0f / 3.0f);
    const float3 secondCentroid = mul3(add3(add3(second.p0, second.p1), second.p2), 1.0f / 3.0f);
    float3 contactNormal = bestAxis;
    if (lengthSquared3(contactNormal) <= 1.0e-12f)
    {
        contactNormal = normalizeOrZero(firstNormal);
        if (lengthSquared3(contactNormal) <= 1.0e-12f)
        {
            contactNormal = normalizeOrZero(secondNormal);
        }
    }

    if (dot3(contactNormal, sub3(secondCentroid, firstCentroid)) < 0.0f)
    {
        contactNormal = mul3(contactNormal, -1.0f);
    }

    const float3 sharedMidpoint = mul3(add3(firstCentroid, secondCentroid), 0.5f);
    outContact.firstTriangleIndex = first.triangleIndex;
    outContact.secondTriangleIndex = second.triangleIndex;
    outContact.pointOnFirst = sharedMidpoint;
    outContact.pointOnSecond = sharedMidpoint;
    outContact.normal = contactNormal;
    outContact.signedDistance = -bestOverlap;
    return true;
}

__global__ void broadPhaseCompactKernel(
    const DeviceAabb* aabbs,
    const std::uint32_t count,
    DeviceIndexPair* pairs,
    std::uint32_t* pairCount)
{
    const std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count || j >= count || i >= j)
    {
        return;
    }

    const auto a = aabbs[i];
    const auto b = aabbs[j];

    const bool overlaps =
        overlapsOnAxis(a.minX, a.maxX, b.minX, b.maxX) &&
        overlapsOnAxis(a.minY, a.maxY, b.minY, b.maxY) &&
        overlapsOnAxis(a.minZ, a.maxZ, b.minZ, b.maxZ);

    if (overlaps)
    {
        const std::uint32_t outputIndex = atomicAdd(pairCount, 1u);
        pairs[outputIndex] = DeviceIndexPair { i, j };
    }
}

__global__ void treePairOverlapKernel(
    const DeviceAabb* firstTrees,
    const DeviceAabb* secondTrees,
    const DeviceTreePairRange* pairRanges,
    const std::uint32_t pairCount,
    DeviceContactCandidate* contactCandidates,
    std::uint32_t* candidateCount)
{
    const std::uint32_t pairIndex = blockIdx.x;
    if (pairIndex >= pairCount)
    {
        return;
    }

    const auto range = pairRanges[pairIndex];
    const std::uint64_t totalTests =
        static_cast<std::uint64_t>(range.firstCount) * static_cast<std::uint64_t>(range.secondCount);

    for (std::uint64_t linearIndex = threadIdx.x; linearIndex < totalTests; linearIndex += blockDim.x)
    {
        const std::uint32_t i = static_cast<std::uint32_t>(linearIndex / range.secondCount);
        const std::uint32_t j = static_cast<std::uint32_t>(linearIndex % range.secondCount);

        const auto a = firstTrees[range.firstOffset + i];
        const auto b = secondTrees[range.secondOffset + j];

        const bool overlaps =
            overlapsOnAxis(a.minX, a.maxX, b.minX, b.maxX) &&
            overlapsOnAxis(a.minY, a.maxY, b.minY, b.maxY) &&
            overlapsOnAxis(a.minZ, a.maxZ, b.minZ, b.maxZ);

        if (overlaps)
        {
            const std::uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            contactCandidates[outputIndex] = DeviceContactCandidate {
                pairIndex,
                range.firstBaseIndex + i,
                range.secondBaseIndex + j,
            };
        }
    }
}

__global__ void exactTriangleContactKernel(
    const DeviceTriangle* firstTriangles,
    const std::uint32_t firstTriangleCount,
    const DeviceTriangle* secondTriangles,
    const std::uint32_t secondTriangleCount,
    DeviceExactContact* contacts,
    std::uint32_t* contactCount)
{
    const std::uint32_t firstIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t secondIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (firstIndex >= firstTriangleCount || secondIndex >= secondTriangleCount)
    {
        return;
    }

    DeviceExactContact exactContact {};
    if (exactTriangleIntersection(firstTriangles[firstIndex], secondTriangles[secondIndex], exactContact))
    {
        const std::uint32_t outputIndex = atomicAdd(contactCount, 1u);
        contacts[outputIndex] = exactContact;
    }
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

struct DeviceProximityContact
{
    std::uint32_t firstPrimitiveIndex;
    std::uint32_t secondPrimitiveIndex;
    std::uint8_t  featureKind;             // 0=VF, 1=FV, 2=EE
    std::uint8_t  firstFeatureLocalIndex;  // 0..2 (vertex or edge start)
    std::uint8_t  secondFeatureLocalIndex; // 0..2
    std::uint8_t  reserved;
    float         firstBary[3];
    float         secondBary[3];
    float3        pointOnFirst;
    float3        pointOnSecond;
    float3        normal;
    float         signedDistance;
};

// Ericson 5.1.5 — closest point on triangle (a,b,c) to point p, in barycentrics.
// Returns make_float3(u,v,w) with u+v+w==1 such that closest = u*a + v*b + w*c.
__device__ __forceinline__ float3 closestPointOnTriangleBary(
    const float3 p, const float3 a, const float3 b, const float3 c)
{
    const float3 ab = sub3(b, a);
    const float3 ac = sub3(c, a);
    const float3 ap = sub3(p, a);
    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        return make_float3(1.0f, 0.0f, 0.0f); // vertex A region
    }
    const float3 bp = sub3(p, b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        return make_float3(0.0f, 1.0f, 0.0f); // vertex B region
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float denom = (d1 - d3);
        const float v = denom != 0.0f ? d1 / denom : 0.0f;
        return make_float3(1.0f - v, v, 0.0f); // edge AB region
    }
    const float3 cp = sub3(p, c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        return make_float3(0.0f, 0.0f, 1.0f); // vertex C region
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float denom = (d2 - d6);
        const float w = denom != 0.0f ? d2 / denom : 0.0f;
        return make_float3(1.0f - w, 0.0f, w); // edge AC region
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float denom = ((d4 - d3) + (d5 - d6));
        const float w = denom != 0.0f ? (d4 - d3) / denom : 0.0f;
        return make_float3(0.0f, 1.0f - w, w); // edge BC region
    }
    const float denom = va + vb + vc;
    const float invDenom = denom != 0.0f ? 1.0f / denom : 0.0f;
    const float v = vb * invDenom;
    const float w = vc * invDenom;
    return make_float3(1.0f - v - w, v, w); // interior
}

// Reconstruct world-space point from triangle vertices + barycentrics.
__device__ __forceinline__ float3 reconstructFromBary(
    const float3 a, const float3 b, const float3 c, const float3 bary)
{
    return add3(add3(mul3(a, bary.x), mul3(b, bary.y)), mul3(c, bary.z));
}

// Ericson 5.1.9 — closest points of two line segments p1q1 and p2q2.
// Returns s, t and the closest points c1, c2.
__device__ __forceinline__ void closestPointSegmentSegment(
    const float3 p1, const float3 q1,
    const float3 p2, const float3 q2,
    float& s, float& t,
    float3& c1, float3& c2)
{
    const float3 d1 = sub3(q1, p1);
    const float3 d2 = sub3(q2, p2);
    const float3 r  = sub3(p1, p2);
    const float a = dot3(d1, d1);
    const float e = dot3(d2, d2);
    const float f = dot3(d2, r);
    constexpr float kEps = 1.0e-20f;

    if (a <= kEps && e <= kEps)
    {
        s = 0.0f; t = 0.0f;
        c1 = p1; c2 = p2;
        return;
    }
    if (a <= kEps)
    {
        s = 0.0f;
        t = fminf(1.0f, fmaxf(0.0f, f / e));
    }
    else
    {
        const float c = dot3(d1, r);
        if (e <= kEps)
        {
            t = 0.0f;
            s = fminf(1.0f, fmaxf(0.0f, -c / a));
        }
        else
        {
            const float b = dot3(d1, d2);
            const float denom = a * e - b * b;
            s = denom != 0.0f ? fminf(1.0f, fmaxf(0.0f, (b * f - c * e) / denom)) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = fminf(1.0f, fmaxf(0.0f, -c / a));
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = fminf(1.0f, fmaxf(0.0f, (b - c) / a));
            }
        }
    }
    c1 = add3(p1, mul3(d1, s));
    c2 = add3(p2, mul3(d2, t));
}

// One thread per candidate pair. Each thread runs 6 VF + 9 EE and keeps the
// closest feature pair under the contact-distance threshold.
__global__ void featureBasedProximityKernel(
    const BackendTriangleVertex* __restrict__ firstPositions,
    const std::uint32_t* __restrict__ firstIndices,
    const BackendTriangleVertex* __restrict__ secondPositions,
    const std::uint32_t* __restrict__ secondIndices,
    const std::uint64_t* __restrict__ candidatePairs64,
    const std::uint32_t* __restrict__ candidatePairs32,
    const std::uint32_t* __restrict__ candidatePairCount,
    const bool useCompactCandidatePairs,
    DeviceProximityContact* __restrict__ contacts,
    std::uint32_t* __restrict__ contactCount,
    std::uint32_t* __restrict__ overflowCount,
    std::uint32_t* __restrict__ vfCount,
    std::uint32_t* __restrict__ fvCount,
    std::uint32_t* __restrict__ eeCount,
    const std::uint32_t maxContacts,
    const float contactDistance,
    const bool computeBarycentrics)
{
    // Grid-stride over ALL candidate pairs. The launch grid is a fixed modest
    // size (over-launch), so the stride loop is what guarantees every pair is
    // processed even when pairCount exceeds the launched thread count (large
    // scenes produce far more than 65 536 candidate pairs).
    const std::uint32_t pairCount = *candidatePairCount;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    const float distThreshSq = contactDistance * contactDistance;
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pairCount; idx += stride)
    {
    std::uint32_t aIdx = 0u;
    std::uint32_t bIdx = 0u;
    if (useCompactCandidatePairs)
    {
        const std::uint32_t pair = candidatePairs32[idx];
        aIdx = pair >> 16u;
        bIdx = pair & 0xffffu;
    }
    else
    {
        const std::uint64_t pair = candidatePairs64[idx];
        aIdx = static_cast<std::uint32_t>(pair >> 32);
        bIdx = static_cast<std::uint32_t>(pair & 0xffffffffu);
    }

    const DeviceTriangle ta = indexedTriangleAt(firstPositions, firstIndices, aIdx);
    const DeviceTriangle tb = indexedTriangleAt(secondPositions, secondIndices, bIdx);
    const float3 aV[3] = { ta.p0, ta.p1, ta.p2 };
    const float3 bV[3] = { tb.p0, tb.p1, tb.p2 };

    // Cheap conservative AABB pre-reject (2026-06-17): if the two triangles'
    // axis-aligned boxes are separated by more than contactDistance, their
    // closest features cannot be within contactDistance, so skip the 15
    // closest-feature tests. The squared box gap is exact and never drops a real
    // contact (the triangles are contained in their boxes), so output is
    // bit-identical — this only avoids wasted math on far same-cell pairs.
    {
        float3 aMin = aV[0], aMax = aV[0];
        float3 bMin = bV[0], bMax = bV[0];
        #pragma unroll
        for (int k = 1; k < 3; ++k)
        {
            aMin = make_float3(fminf(aMin.x, aV[k].x), fminf(aMin.y, aV[k].y), fminf(aMin.z, aV[k].z));
            aMax = make_float3(fmaxf(aMax.x, aV[k].x), fmaxf(aMax.y, aV[k].y), fmaxf(aMax.z, aV[k].z));
            bMin = make_float3(fminf(bMin.x, bV[k].x), fminf(bMin.y, bV[k].y), fminf(bMin.z, bV[k].z));
            bMax = make_float3(fmaxf(bMax.x, bV[k].x), fmaxf(bMax.y, bV[k].y), fmaxf(bMax.z, bV[k].z));
        }
        const float gx = fmaxf(0.0f, fmaxf(aMin.x - bMax.x, bMin.x - aMax.x));
        const float gy = fmaxf(0.0f, fmaxf(aMin.y - bMax.y, bMin.y - aMax.y));
        const float gz = fmaxf(0.0f, fmaxf(aMin.z - bMax.z, bMin.z - aMax.z));
        if (gx * gx + gy * gy + gz * gz > distThreshSq)
        {
            continue;
        }
    }

    float bestDistSq = INFINITY;
    int bestKind = 0;
    int bestI = 0;
    int bestJ = 0;
    float3 bestP1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 bestP2 = make_float3(0.0f, 0.0f, 0.0f);
    float3 bestBary1 = make_float3(1.0f, 0.0f, 0.0f);
    float3 bestBary2 = make_float3(1.0f, 0.0f, 0.0f);

    // 3 VF: vertex of A vs face B
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        const float3 bary = closestPointOnTriangleBary(aV[i], bV[0], bV[1], bV[2]);
        const float3 cp = reconstructFromBary(bV[0], bV[1], bV[2], bary);
        const float3 diff = sub3(aV[i], cp);
        const float d2v = dot3(diff, diff);
        if (d2v < bestDistSq)
        {
            bestDistSq = d2v;
            bestKind = 0; bestI = i; bestJ = 0;
            bestP1 = aV[i]; bestP2 = cp;
            bestBary1 = make_float3(1.0f, 0.0f, 0.0f);
            bestBary2 = bary;
        }
    }

    // 3 FV: vertex of B vs face A
    #pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        const float3 bary = closestPointOnTriangleBary(bV[j], aV[0], aV[1], aV[2]);
        const float3 cp = reconstructFromBary(aV[0], aV[1], aV[2], bary);
        const float3 diff = sub3(cp, bV[j]);
        const float d2v = dot3(diff, diff);
        if (d2v < bestDistSq)
        {
            bestDistSq = d2v;
            bestKind = 1; bestI = 0; bestJ = j;
            bestP1 = cp; bestP2 = bV[j];
            bestBary1 = bary;
            bestBary2 = make_float3(1.0f, 0.0f, 0.0f);
        }
    }

    // 9 EE: each edge of A vs each edge of B
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        const float3 p1 = aV[i];
        const float3 q1 = aV[(i + 1) % 3];
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            const float3 p2 = bV[j];
            const float3 q2 = bV[(j + 1) % 3];
            float s = 0.0f, t = 0.0f;
            float3 c1, c2;
            closestPointSegmentSegment(p1, q1, p2, q2, s, t, c1, c2);
            const float3 diff = sub3(c1, c2);
            const float d2v = dot3(diff, diff);
            if (d2v < bestDistSq)
            {
                bestDistSq = d2v;
                bestKind = 2; bestI = i; bestJ = j;
                bestP1 = c1; bestP2 = c2;
                bestBary1 = make_float3(1.0f - s, s, 0.0f);
                bestBary2 = make_float3(1.0f - t, t, 0.0f);
            }
        }
    }

    if (bestDistSq > distThreshSq)
    {
        continue;
    }

    const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
    if (outIdx >= maxContacts)
    {
        atomicAdd(overflowCount, 1u);
        continue;
    }

    DeviceProximityContact c;
    c.firstPrimitiveIndex = aIdx;
    c.secondPrimitiveIndex = bIdx;
    c.featureKind = static_cast<std::uint8_t>(bestKind);
    c.firstFeatureLocalIndex = static_cast<std::uint8_t>(bestI);
    c.secondFeatureLocalIndex = static_cast<std::uint8_t>(bestJ);
    c.reserved = 0;
    if (computeBarycentrics)
    {
        c.firstBary[0] = bestBary1.x;  c.firstBary[1] = bestBary1.y;  c.firstBary[2] = bestBary1.z;
        c.secondBary[0] = bestBary2.x; c.secondBary[1] = bestBary2.y; c.secondBary[2] = bestBary2.z;
    }
    else
    {
        c.firstBary[0] = 1.0f;  c.firstBary[1] = 0.0f;  c.firstBary[2] = 0.0f;
        c.secondBary[0] = 1.0f; c.secondBary[1] = 0.0f; c.secondBary[2] = 0.0f;
    }
    c.pointOnFirst  = bestP1;
    c.pointOnSecond = bestP2;
    const float3 sep = sub3(bestP2, bestP1);
    const float dist = sqrtf(bestDistSq);
    c.signedDistance = dist;
    if (dist > 1.0e-12f)
    {
        const float invLen = 1.0f / dist;
        c.normal = make_float3(sep.x * invLen, sep.y * invLen, sep.z * invLen);
    }
    else
    {
        c.normal = make_float3(0.0f, 1.0f, 0.0f);
    }
    contacts[outIdx] = c;

    if (bestKind == 0)      atomicAdd(vfCount, 1u);
    else if (bestKind == 1) atomicAdd(fvCount, 1u);
    else                    atomicAdd(eeCount, 1u);
    } // grid-stride loop
}

// Resets the proximity contact counters and per-class tallies.
__global__ void resetProximityCountersKernel(
    std::uint32_t* contactCount,
    std::uint32_t* overflowCount,
    std::uint32_t* vfCount,
    std::uint32_t* fvCount,
    std::uint32_t* eeCount)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *contactCount  = 0u;
        *overflowCount = 0u;
        *vfCount = 0u;
        *fvCount = 0u;
        *eeCount = 0u;
    }
}

// ============================================================================
// Vertex-Triangle Proximity (Phase 12)
// ----------------------------------------------------------------------------
// Treats each vertex of a point cloud as a primitive inserted into the dense
// grid alongside triangles. The candidate-pair generator then emits
// (triangleId << 32) | vertexId pairs naturally, and a focused narrow-pass
// kernel runs closestPointOnTriangleBary per pair. Use cases:
//   (a) self-collision: pass the same mesh's vertex set + triangle surface
//   (b) cross-model: a separate PointCollisionModel against a triangle mesh
// ============================================================================

// Insert each vertex of the cloud into the tool-side cell buckets. The
// "vertex AABB" is a tiny inflated box around the point; we reuse the same
// denseGridCellSpan helper as triangles for consistency.
__global__ void insertIndexedPointsKernel(
    const BackendTriangleVertex* __restrict__ positions,
    const std::uint32_t pointCount,
    const DeviceDenseGridConfig config,
    DeviceCellBucket* __restrict__ grid,
    std::uint32_t* __restrict__ cellToolIds,
    std::uint32_t* __restrict__ overflowCount,
    DeviceDenseGridStats* __restrict__ stats)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pointCount) return;

    const auto p = positions[tid];
    const float r = config.contactDistance;

    DeviceAabb aabb;
    aabb.minX = p.x - r; aabb.minY = p.y - r; aabb.minZ = p.z - r;
    aabb.maxX = p.x + r; aabb.maxY = p.y + r; aabb.maxZ = p.z + r;

    int3 cellMin, cellMax;
    if (!denseGridCellSpan(aabb, config, cellMin, cellMax)) return;

    const std::uint32_t bucketCapacity = config.maxToolTrianglesPerCell;
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        for (int y = cellMin.y; y <= cellMax.y; ++y)
        {
            for (int x = cellMin.x; x <= cellMax.x; ++x)
            {
                const std::uint32_t cellId =
                    static_cast<std::uint32_t>(x) +
                    static_cast<std::uint32_t>(y) * config.resolutionX +
                    static_cast<std::uint32_t>(z) * config.resolutionX * config.resolutionY;
                const std::uint32_t localIdx = atomicAdd(&grid[cellId].toolCount, 1u);
                if (localIdx < bucketCapacity)
                {
                    cellToolIds[cellId * bucketCapacity + localIdx] = tid;
                    if (stats != nullptr)
                    {
                        atomicAdd(&stats->toolInsertCount, 1u);
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

// One thread per (triangleId, vertexId) candidate pair produced by the
// existing dense-grid candidate generator. Runs closestPointOnTriangleBary.
__global__ void featureBasedVertexTriangleProximityKernel(
    const BackendTriangleVertex* __restrict__ trianglePositions,
    const std::uint32_t* __restrict__ triangleIndices,
    const BackendTriangleVertex* __restrict__ pointPositions,
    const std::uint64_t* __restrict__ candidatePairs,
    const std::uint32_t* __restrict__ candidatePairCount,
    DeviceProximityContact* __restrict__ contacts,
    std::uint32_t* __restrict__ contactCount,
    std::uint32_t* __restrict__ overflowCount,
    std::uint32_t* __restrict__ vfCount,
    const std::uint32_t maxContacts,
    const float contactDistance,
    const bool computeBarycentrics,
    const std::uint32_t selfCollisionVertexExclusionStride)  // 0 to disable; otherwise skip (triId, vertId) where vertId is one of the triangle's 3 vertices
{
    // Grid-stride over ALL (triangle, vertex) candidate pairs so the fixed
    // modest launch grid still processes every pair when pairCount exceeds the
    // launched thread count.
    const std::uint32_t pairCount = *candidatePairCount;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pairCount; idx += stride)
    {
    const std::uint64_t pair = candidatePairs[idx];
    const std::uint32_t triId   = static_cast<std::uint32_t>(pair >> 32);
    const std::uint32_t pointId = static_cast<std::uint32_t>(pair & 0xffffffffu);

    // Self-collision exclusion: skip pairs where the candidate vertex is one
    // of the triangle's own corner vertices. Without this every vertex would
    // report distance 0 to its own adjacent triangles. The stride argument
    // gates this for cross-model cases (stride=0 means "do not exclude").
    if (selfCollisionVertexExclusionStride != 0)
    {
        const std::uint32_t i0 = triangleIndices[3u * triId + 0u];
        const std::uint32_t i1 = triangleIndices[3u * triId + 1u];
        const std::uint32_t i2 = triangleIndices[3u * triId + 2u];
        if (pointId == i0 || pointId == i1 || pointId == i2)
        {
            continue;
        }
    }

    const DeviceTriangle tri = indexedTriangleAt(trianglePositions, triangleIndices, triId);
    const auto vp = pointPositions[pointId];
    const float3 p = make_float3(vp.x, vp.y, vp.z);

    const float3 bary = closestPointOnTriangleBary(p, tri.p0, tri.p1, tri.p2);
    const float3 cp = reconstructFromBary(tri.p0, tri.p1, tri.p2, bary);
    const float3 diff = sub3(cp, p);
    const float distSq = dot3(diff, diff);

    if (distSq > contactDistance * contactDistance) continue;

    const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
    if (outIdx >= maxContacts)
    {
        atomicAdd(overflowCount, 1u);
        continue;
    }

    DeviceProximityContact c;
    c.firstPrimitiveIndex  = pointId;  // "first" side is the vertex
    c.secondPrimitiveIndex = triId;    // "second" side is the triangle
    c.featureKind = static_cast<std::uint8_t>(0);  // VertexFace
    c.firstFeatureLocalIndex  = 0;
    c.secondFeatureLocalIndex = 0;
    c.reserved = 0;
    if (computeBarycentrics)
    {
        c.firstBary[0] = 1.0f; c.firstBary[1] = 0.0f; c.firstBary[2] = 0.0f;
        c.secondBary[0] = bary.x; c.secondBary[1] = bary.y; c.secondBary[2] = bary.z;
    }
    else
    {
        c.firstBary[0] = 1.0f; c.firstBary[1] = 0.0f; c.firstBary[2] = 0.0f;
        c.secondBary[0] = 1.0f; c.secondBary[1] = 0.0f; c.secondBary[2] = 0.0f;
    }
    c.pointOnFirst  = p;
    c.pointOnSecond = cp;
    const float dist = sqrtf(distSq);
    if (dist > 1.0e-12f)
    {
        const float inv = 1.0f / dist;
        c.normal = make_float3(diff.x * inv, diff.y * inv, diff.z * inv);
    }
    else
    {
        c.normal = make_float3(0.0f, 1.0f, 0.0f);
    }
    c.signedDistance = dist;
    contacts[outIdx] = c;
    atomicAdd(vfCount, 1u);
    } // grid-stride loop
}

// ============================================================================
// EXPERIMENTAL (experiment/hash-prefixsum-broadphase): spatial-hash + prefix-sum
// broad cull. Self-contained — does not touch DenseGridWorkspace or the dense
// kernels. Reuses the device math helpers (indexedTriangleAt, triangleAabb,
// denseGridCellSpan, denseCellId, mixCandidatePairHash, insertUniqueCandidatePair)
// and the FBP narrow kernel (featureBasedProximityKernel).
// ============================================================================

constexpr std::uint32_t kEmptyHashCellKey = 0xffffffffu;

constexpr std::uint32_t kInvalidHashBucket = 0xffffffffu;
constexpr std::uint32_t kOverflowHashBucket = 0xfffffffeu;
constexpr std::uint32_t kEmptyCompactPairSlot = 0xffffffffu;

// Open-addressing spatial hash of occupied cells. Table slots store only
// (cellKey -> compact bucket id); per-cell counts and primitive ids live in
// compact bucket arrays sized by the maximum possible occupied cells, not by the
// hash-table slot count. The pair hash keeps a touched-slot list so steady-state
// frames clear only slots written by the previous frame.
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

__global__ void clearTouchedPairHash64Kernel(
    unsigned long long* pairHashKeys,
    const std::uint32_t* touchedSlots,
    const std::uint32_t* touchedCount)
{
    const std::uint32_t count = *touchedCount;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += stride)
    {
        pairHashKeys[touchedSlots[id]] = static_cast<unsigned long long>(kEmptyPairSlot);
    }
}

__global__ void clearTouchedPairHash32Kernel(
    std::uint32_t* pairHashKeys,
    const std::uint32_t* touchedSlots,
    const std::uint32_t* touchedCount)
{
    const std::uint32_t count = *touchedCount;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += stride)
    {
        pairHashKeys[touchedSlots[id]] = kEmptyCompactPairSlot;
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

__device__ bool insertUniqueCandidatePair64Tracked(
    const std::uint64_t pair,
    unsigned long long* pairHashKeys,
    std::uint32_t* touchedPairHashSlots,
    std::uint32_t* touchedPairHashCount,
    const std::uint32_t pairHashCapacity,
    std::uint64_t* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount,
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
        if (previous == pair)
        {
            return false;
        }
        slot = (slot + 1u) & mask;
    }
    atomicAdd(probeOverflowCount, 1u);
    atomicAdd(overflowCount, 1u);
    return false;
}

__device__ bool insertUniqueCandidatePair32Tracked(
    const std::uint32_t pair,
    std::uint32_t* pairHashKeys,
    std::uint32_t* touchedPairHashSlots,
    std::uint32_t* touchedPairHashCount,
    const std::uint32_t pairHashCapacity,
    std::uint32_t* candidatePairs,
    std::uint32_t* candidateCount,
    std::uint32_t* overflowCount,
    std::uint32_t* probeOverflowCount,
    const std::uint32_t maxCandidatePairs)
{
    constexpr std::uint32_t kMaxProbeCount = 256u;
    const std::uint32_t mask = pairHashCapacity - 1u;
    std::uint32_t slot = static_cast<std::uint32_t>(mixCandidatePairHash(pair)) & mask;
    for (std::uint32_t probe = 0; probe < kMaxProbeCount; ++probe)
    {
        const std::uint32_t previous = atomicCAS(&pairHashKeys[slot], kEmptyCompactPairSlot, pair);
        if (previous == kEmptyCompactPairSlot)
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
        if (previous == pair)
        {
            return false;
        }
        slot = (slot + 1u) & mask;
    }
    atomicAdd(probeOverflowCount, 1u);
    atomicAdd(overflowCount, 1u);
    return false;
}

__global__ void generateMixedHashCandidatePairs64Kernel(
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
    std::uint64_t* candidatePairs,
    unsigned long long* pairHashKeys,
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
            insertUniqueCandidatePair64Tracked(
                encodeCandidatePair(tissueTriId, toolTriId),
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

__global__ void generateMixedHashCandidatePairs32Kernel(
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
    std::uint32_t* candidatePairs,
    std::uint32_t* pairHashKeys,
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
            insertUniqueCandidatePair32Tracked(
                encodeCompactCandidatePair(tissueTriId, toolTriId),
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

} // namespace

namespace SofaGpuCollision::backend
{

BackendStatus probe()
{
    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        return { false, cudaGetErrorString(err) };
    }

    if (deviceCount <= 0)
    {
        return { false, "CUDA runtime was found, but no GPU devices were reported." };
    }

    return {
        true,
        "CUDA runtime detected. Kernel integration points are compiled and ready for implementation."
    };
}

bool computeBroadPhasePairs(
    const std::vector<AxisAlignedBoundingBox>& boxes,
    std::vector<BroadPhaseIndexPair>& pairs,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    pairs.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
    }

    if (boxes.size() < 2)
    {
        diagnostic.clear();
        return true;
    }
    if (executionStats != nullptr)
    {
        executionStats->inputPrimitiveCount = static_cast<std::uint32_t>(boxes.size());
    }

    std::vector<DeviceAabb> hostBoxes;
    hostBoxes.reserve(boxes.size());
    for (const auto& box : boxes)
    {
        hostBoxes.push_back(DeviceAabb {
            box.minX, box.minY, box.minZ,
            box.maxX, box.maxY, box.maxZ
        });
    }

    DeviceAabb* deviceBoxes = nullptr;
    DeviceIndexPair* devicePairs = nullptr;
    std::uint32_t* devicePairCount = nullptr;

    const std::size_t boxBytes = hostBoxes.size() * sizeof(DeviceAabb);
    const std::size_t maxPairCount = (hostBoxes.size() * (hostBoxes.size() - 1)) / 2;
    const std::size_t pairBytes = maxPairCount * sizeof(DeviceIndexPair);
    const std::size_t pairCountBytes = sizeof(std::uint32_t);
    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(boxBytes);
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(boxBytes + pairBytes + pairCountBytes);
    }

    cudaError_t err = cudaMalloc(&deviceBoxes, boxBytes);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMalloc(&devicePairs, pairBytes);
    if (err != cudaSuccess)
    {
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMalloc(&devicePairCount, pairCountBytes);
    if (err != cudaSuccess)
    {
        cudaFree(devicePairs);
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(deviceBoxes, hostBoxes.data(), boxBytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemset(devicePairCount, 0, pairCountBytes);
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t tileSize = 16;
        const dim3 blockSize(tileSize, tileSize, 1);
        const dim3 gridSize(
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            (static_cast<std::uint32_t>(hostBoxes.size()) + tileSize - 1) / tileSize,
            1);
        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        broadPhaseCompactKernel<<<gridSize, blockSize>>>(
            deviceBoxes,
            static_cast<std::uint32_t>(hostBoxes.size()),
            devicePairs,
            devicePairCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostPairCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostPairCount, devicePairCount, pairCountBytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(devicePairCount);
        cudaFree(devicePairs);
        cudaFree(deviceBoxes);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            pairCountBytes + hostPairCount * sizeof(DeviceIndexPair));
    }

    std::vector<DeviceIndexPair> hostPairs(hostPairCount);
    if (hostPairCount > 0)
    {
        err = cudaMemcpy(hostPairs.data(), devicePairs, hostPairCount * sizeof(DeviceIndexPair), cudaMemcpyDeviceToHost);
    }

    cudaFree(devicePairCount);
    cudaFree(devicePairs);
    cudaFree(deviceBoxes);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = hostPairCount;
    }

    pairs.reserve(hostPairs.size());
    for (const auto& pair : hostPairs)
    {
        pairs.emplace_back(pair.first, pair.second);
    }

    diagnostic.clear();
    return true;
}

bool prefilterNarrowPhasePairs(
    const std::vector<NarrowPhaseTreePair>& inputTrees,
    std::vector<std::uint32_t>& survivingPairIndices,
    std::vector<NarrowPhaseContactCandidate>* contactCandidates,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    survivingPairIndices.clear();
    if (contactCandidates != nullptr)
    {
        contactCandidates->clear();
    }
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = static_cast<std::uint32_t>(inputTrees.size());
    }

    if (inputTrees.empty())
    {
        diagnostic.clear();
        return true;
    }

    std::vector<DeviceAabb> flatFirstTrees;
    std::vector<DeviceAabb> flatSecondTrees;
    std::vector<DeviceTreePairRange> pairRanges;
    std::vector<std::uint32_t> originalPairIndices;

    flatFirstTrees.reserve(1024);
    flatSecondTrees.reserve(1024);
    pairRanges.reserve(inputTrees.size());
    originalPairIndices.reserve(inputTrees.size());

    for (std::uint32_t pairIndex = 0; pairIndex < inputTrees.size(); ++pairIndex)
    {
        const auto& treePair = inputTrees[pairIndex];

        if (treePair.firstTree.empty() || treePair.secondTree.empty())
        {
            survivingPairIndices.push_back(pairIndex);
            continue;
        }

        const auto firstBegin = treePair.firstTree.size() > 1 ? 1u : 0u;
        const auto secondBegin = treePair.secondTree.size() > 1 ? 1u : 0u;
        const auto firstOffset = static_cast<std::uint32_t>(flatFirstTrees.size());
        const auto secondOffset = static_cast<std::uint32_t>(flatSecondTrees.size());

        for (std::size_t i = firstBegin; i < treePair.firstTree.size(); ++i)
        {
            const auto& box = treePair.firstTree[i];
            flatFirstTrees.push_back(DeviceAabb { box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ });
        }

        for (std::size_t i = secondBegin; i < treePair.secondTree.size(); ++i)
        {
            const auto& box = treePair.secondTree[i];
            flatSecondTrees.push_back(DeviceAabb { box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ });
        }

        const auto firstCount = static_cast<std::uint32_t>(flatFirstTrees.size()) - firstOffset;
        const auto secondCount = static_cast<std::uint32_t>(flatSecondTrees.size()) - secondOffset;

        if (firstCount == 0 || secondCount == 0)
        {
            survivingPairIndices.push_back(pairIndex);
            continue;
        }

        pairRanges.push_back(DeviceTreePairRange {
            firstOffset,
            firstCount,
            secondOffset,
            secondCount,
            static_cast<std::uint32_t>(firstBegin),
            static_cast<std::uint32_t>(secondBegin),
        });
        originalPairIndices.push_back(pairIndex);
    }

    if (pairRanges.empty())
    {
        diagnostic.clear();
        return true;
    }

    DeviceAabb* deviceFirstTrees = nullptr;
    DeviceAabb* deviceSecondTrees = nullptr;
    DeviceTreePairRange* devicePairRanges = nullptr;
    DeviceContactCandidate* deviceContactCandidates = nullptr;
    std::uint32_t* deviceCandidateCount = nullptr;

    std::uint64_t maxCandidateCount = 0;
    for (const auto& range : pairRanges)
    {
        maxCandidateCount += static_cast<std::uint64_t>(range.firstCount) * static_cast<std::uint64_t>(range.secondCount);
    }

    cudaError_t err = cudaMalloc(&deviceFirstTrees, flatFirstTrees.size() * sizeof(DeviceAabb));
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceSecondTrees, flatSecondTrees.size() * sizeof(DeviceAabb));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&devicePairRanges, pairRanges.size() * sizeof(DeviceTreePairRange));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceContactCandidates, maxCandidateCount * sizeof(DeviceContactCandidate));
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceCandidateCount, sizeof(std::uint32_t));
    }
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        cudaFree(deviceFirstTrees);
        cudaFree(deviceSecondTrees);
        cudaFree(devicePairRanges);
        cudaFree(deviceContactCandidates);
        cudaFree(deviceCandidateCount);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(
            flatFirstTrees.size() * sizeof(DeviceAabb) +
            flatSecondTrees.size() * sizeof(DeviceAabb) +
            pairRanges.size() * sizeof(DeviceTreePairRange));
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(
            flatFirstTrees.size() * sizeof(DeviceAabb) +
            flatSecondTrees.size() * sizeof(DeviceAabb) +
            pairRanges.size() * sizeof(DeviceTreePairRange) +
            maxCandidateCount * sizeof(DeviceContactCandidate) +
            sizeof(std::uint32_t));
    }

    err = cudaMemcpy(
        deviceFirstTrees, flatFirstTrees.data(), flatFirstTrees.size() * sizeof(DeviceAabb), cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(
            deviceSecondTrees, flatSecondTrees.data(), flatSecondTrees.size() * sizeof(DeviceAabb), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(
            devicePairRanges, pairRanges.data(), pairRanges.size() * sizeof(DeviceTreePairRange), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(deviceCandidateCount, 0, sizeof(std::uint32_t));
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t threadCount = 256;
        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        treePairOverlapKernel<<<static_cast<std::uint32_t>(pairRanges.size()), threadCount>>>(
            deviceFirstTrees,
            deviceSecondTrees,
            devicePairRanges,
            static_cast<std::uint32_t>(pairRanges.size()),
            deviceContactCandidates,
            deviceCandidateCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostCandidateCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostCandidateCount, deviceCandidateCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTrees);
        cudaFree(deviceSecondTrees);
        cudaFree(devicePairRanges);
        cudaFree(deviceContactCandidates);
        cudaFree(deviceCandidateCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            sizeof(std::uint32_t) + hostCandidateCount * sizeof(DeviceContactCandidate));
    }

    std::vector<DeviceContactCandidate> hostCandidates(hostCandidateCount);
    if (hostCandidateCount > 0)
    {
        err = cudaMemcpy(
            hostCandidates.data(),
            deviceContactCandidates,
            hostCandidateCount * sizeof(DeviceContactCandidate),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFirstTrees);
    cudaFree(deviceSecondTrees);
    cudaFree(devicePairRanges);
    cudaFree(deviceContactCandidates);
    cudaFree(deviceCandidateCount);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputCandidateCount = hostCandidateCount;
    }

    std::vector<std::uint8_t> keepPair(pairRanges.size(), 0);
    for (const auto& candidate : hostCandidates)
    {
        if (candidate.pairIndex < keepPair.size())
        {
            keepPair[candidate.pairIndex] = 1;
            if (contactCandidates != nullptr)
            {
                contactCandidates->push_back(NarrowPhaseContactCandidate {
                    originalPairIndices[candidate.pairIndex],
                    candidate.firstLeafIndex,
                    candidate.secondLeafIndex,
                });
            }
        }
    }

    for (std::size_t i = 0; i < keepPair.size(); ++i)
    {
        if (keepPair[i] != 0)
        {
            survivingPairIndices.push_back(originalPairIndices[i]);
        }
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = static_cast<std::uint32_t>(survivingPairIndices.size());
    }

    diagnostic.clear();
    return true;
}

bool computeExactTriangleContacts(
    const std::vector<TrianglePrimitive>& firstTriangles,
    const std::vector<TrianglePrimitive>& secondTriangles,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount =
            static_cast<std::uint32_t>(firstTriangles.size() + secondTriangles.size());
    }

    if (firstTriangles.empty() || secondTriangles.empty())
    {
        diagnostic.clear();
        return true;
    }

    std::vector<DeviceTriangle> hostFirstTriangles;
    std::vector<DeviceTriangle> hostSecondTriangles;
    hostFirstTriangles.reserve(firstTriangles.size());
    hostSecondTriangles.reserve(secondTriangles.size());

    for (const auto& triangle : firstTriangles)
    {
        hostFirstTriangles.push_back(DeviceTriangle {
            make_float3(triangle.p0.x, triangle.p0.y, triangle.p0.z),
            make_float3(triangle.p1.x, triangle.p1.y, triangle.p1.z),
            make_float3(triangle.p2.x, triangle.p2.y, triangle.p2.z),
            triangle.triangleIndex,
        });
    }

    for (const auto& triangle : secondTriangles)
    {
        hostSecondTriangles.push_back(DeviceTriangle {
            make_float3(triangle.p0.x, triangle.p0.y, triangle.p0.z),
            make_float3(triangle.p1.x, triangle.p1.y, triangle.p1.z),
            make_float3(triangle.p2.x, triangle.p2.y, triangle.p2.z),
            triangle.triangleIndex,
        });
    }

    DeviceTriangle* deviceFirstTriangles = nullptr;
    DeviceTriangle* deviceSecondTriangles = nullptr;
    DeviceExactContact* deviceContacts = nullptr;
    std::uint32_t* deviceContactCount = nullptr;

    const std::size_t firstBytes = hostFirstTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t secondBytes = hostSecondTriangles.size() * sizeof(DeviceTriangle);
    const std::size_t maxContactCount = hostFirstTriangles.size() * hostSecondTriangles.size();
    const std::size_t contactBytes = maxContactCount * sizeof(DeviceExactContact);
    const std::size_t contactCountBytes = sizeof(std::uint32_t);

    if (executionStats != nullptr)
    {
        executionStats->hostToDeviceBytes += static_cast<std::uint64_t>(firstBytes + secondBytes);
        executionStats->deviceAllocationBytes += static_cast<std::uint64_t>(firstBytes + secondBytes + contactBytes + contactCountBytes);
    }

    cudaError_t err = cudaMalloc(&deviceFirstTriangles, firstBytes);
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceSecondTriangles, secondBytes);
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceContacts, contactBytes);
    }
    if (err == cudaSuccess)
    {
        err = cudaMalloc(&deviceContactCount, contactCountBytes);
    }

    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTriangles);
        cudaFree(deviceSecondTriangles);
        cudaFree(deviceContacts);
        cudaFree(deviceContactCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(deviceFirstTriangles, hostFirstTriangles.data(), firstBytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(deviceSecondTriangles, hostSecondTriangles.data(), secondBytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(deviceContactCount, 0, contactCountBytes);
    }

    if (err == cudaSuccess)
    {
        constexpr std::uint32_t tileSize = 16;
        const dim3 blockSize(tileSize, tileSize, 1);
        const dim3 gridSize(
            (static_cast<std::uint32_t>(hostSecondTriangles.size()) + tileSize - 1) / tileSize,
            (static_cast<std::uint32_t>(hostFirstTriangles.size()) + tileSize - 1) / tileSize,
            1);

        cudaEvent_t kernelStart {};
        cudaEvent_t kernelEnd {};
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventRecord(kernelStart);
        exactTriangleContactKernel<<<gridSize, blockSize>>>(
            deviceFirstTriangles,
            static_cast<std::uint32_t>(hostFirstTriangles.size()),
            deviceSecondTriangles,
            static_cast<std::uint32_t>(hostSecondTriangles.size()),
            deviceContacts,
            deviceContactCount);
        cudaEventRecord(kernelEnd);
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess && executionStats != nullptr)
        {
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, kernelStart, kernelEnd);
            executionStats->gpuKernelMilliseconds += static_cast<double>(elapsedMs);
            executionStats->kernelLaunchCount += 1;
        }
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
    }

    std::uint32_t hostContactCount = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy(&hostContactCount, deviceContactCount, contactCountBytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstTriangles);
        cudaFree(deviceSecondTriangles);
        cudaFree(deviceContacts);
        cudaFree(deviceContactCount);
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->deviceToHostBytes += static_cast<std::uint64_t>(
            contactCountBytes + hostContactCount * sizeof(DeviceExactContact));
    }

    std::vector<DeviceExactContact> hostContacts(hostContactCount);
    if (hostContactCount > 0)
    {
        err = cudaMemcpy(
            hostContacts.data(),
            deviceContacts,
            hostContactCount * sizeof(DeviceExactContact),
            cudaMemcpyDeviceToHost);
    }

    cudaFree(deviceFirstTriangles);
    cudaFree(deviceSecondTriangles);
    cudaFree(deviceContacts);
    cudaFree(deviceContactCount);

    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    if (executionStats != nullptr)
    {
        executionStats->outputPairCount = hostContactCount > 0 ? 1u : 0u;
        executionStats->outputCandidateCount = hostContactCount;
        executionStats->outputContactCount = hostContactCount;
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
bool computeFeatureBasedProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (proximityStats != nullptr)
    {
        *proximityStats = FeatureBasedProximityStats {};
    }
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = firstSurface.triangleCount + secondSurface.triangleCount;
    }

    // Step 1: broad cull. Force the existing path into a clean
    // "candidates-only" mode so it does the dense-grid work but skips the
    // SAT-style exact-contact kernel. We deliberately leave counter readback
    // OFF — the FBP kernel reads *candidatePairCount from device memory on
    // first thread of each block (L2-cached after first warp), and over-launch
    // covers any unknown pair count with cheap per-thread early-exit.
    DenseGridConfig broadConfig = gridConfig;
    broadConfig.copyContactsToHost = false;
    broadConfig.computeDeviceContactsWhenContactsStayOnDevice = false;
    broadConfig.readCountersWhenContactsStayOnDevice = proximityConfig.readContactCounter;
    broadConfig.detailedProfiling = false;

    std::vector<ExactContact> ignoredExactContacts;
    BackendExecutionStats broadStats;
    std::string broadDiagnostic;
    const bool broadOk = computeDenseGridIndexedTriangleContacts(
        firstSurface,
        secondSurface,
        broadConfig,
        ignoredExactContacts,
        broadDiagnostic,
        &broadStats);
    if (!broadOk)
    {
        diagnostic = std::string("Feature-based proximity broad cull failed: ") + broadDiagnostic;
        return false;
    }
    if (executionStats != nullptr)
    {
        // Forward the broad-cull stats so the wrapper sees the full picture.
        executionStats->gpuKernelMilliseconds += broadStats.gpuKernelMilliseconds;
        executionStats->hostPreparationMilliseconds += broadStats.hostPreparationMilliseconds;
        executionStats->hostToDeviceMilliseconds += broadStats.hostToDeviceMilliseconds;
        executionStats->deviceAllocationMilliseconds += broadStats.deviceAllocationMilliseconds;
        executionStats->denseGridClearMilliseconds += broadStats.denseGridClearMilliseconds;
        executionStats->denseGridInsertTissueMilliseconds += broadStats.denseGridInsertTissueMilliseconds;
        executionStats->denseGridInsertToolMilliseconds += broadStats.denseGridInsertToolMilliseconds;
        executionStats->denseGridGeneratePairsMilliseconds += broadStats.denseGridGeneratePairsMilliseconds;
        executionStats->hostToDeviceBytes += broadStats.hostToDeviceBytes;
        executionStats->deviceToHostBytes += broadStats.deviceToHostBytes;
        executionStats->deviceAllocationBytes += broadStats.deviceAllocationBytes;
        executionStats->kernelLaunchCount += broadStats.kernelLaunchCount;
        executionStats->cudaMemsetCount += broadStats.cudaMemsetCount;
        executionStats->workspaceResizeCount += broadStats.workspaceResizeCount;
        executionStats->rawCandidateCount += broadStats.rawCandidateCount;
        executionStats->uniqueCandidateCount += broadStats.uniqueCandidateCount;
        executionStats->gridCellCount = broadStats.gridCellCount;
        executionStats->overflowCount += broadStats.overflowCount;
    }
    if (proximityStats != nullptr)
    {
        proximityStats->candidatePairCount = broadStats.uniqueCandidateCount;
    }

    // Step 2: allocate proximity output and counters
    auto& workspace = denseGridWorkspace();
    std::uint64_t newlyAllocatedBytes = 0;
    const auto allocStart = std::chrono::steady_clock::now();
    cudaError_t err = workspace.ensureProximityStorage(
        proximityConfig.maxContacts,
        sizeof(DeviceProximityContact),
        newlyAllocatedBytes);
    const double allocMs = elapsedMillisecondsSince(allocStart);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Proximity storage alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += allocMs;
        if (newlyAllocatedBytes > 0) executionStats->workspaceResizeCount += 1;
    }

    auto* deviceProximityContacts = reinterpret_cast<DeviceProximityContact*>(workspace.proximityContacts);

    // Step 3: reset counters + launch FBP kernel.
    // We only take CUDA events when we actually need fbpMs (i.e. readContactCounter).
    // In the production detection-only path, the kernel runs fully async and
    // the next frame's stage_start cuts in before any sync happens here.
    const bool needFbpTiming = proximityConfig.readContactCounter;
    if (needFbpTiming)
    {
        err = workspace.ensureFbpEvents();
        if (err != cudaSuccess)
        {
            diagnostic = std::string("FBP event create failed: ") + cudaGetErrorString(err);
            return false;
        }
        cudaEventRecord(workspace.fbpStartEvent);
    }

    resetProximityCountersKernel<<<1, 1>>>(
        workspace.proximityContactCount,
        workspace.proximityOverflowCount,
        workspace.proximityVfCount,
        workspace.proximityFvCount,
        workspace.proximityEeCount);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    const BackendTriangleVertex* deviceFirstPositions =
        firstSurface.devicePositions == nullptr ? workspace.indexedTissuePositions : firstSurface.devicePositions;
    const BackendTriangleVertex* deviceSecondPositions =
        secondSurface.devicePositions == nullptr ? workspace.indexedToolPositions : secondSurface.devicePositions;

    // Over-launch with a sensible cap. The kernel does `if (tid >= *count) return`
    // so empty threads exit in ~1 instruction. Cap at 65536 threads = 256 blocks
    // to avoid wasting time launching 7813 empty blocks when maxCandidatePairs is huge.
    // For surgical-scene workloads candidate counts rarely exceed ~10k so 256 blocks
    // = 65 536 threads is a comfortable upper bound and runs as one scheduler wave
    // on the 16 SMs of a GTX 1650 Ti.
    // The kernel grid-strides over all candidate pairs, so this is a
    // saturation target, not a correctness bound: 1024 blocks fills the 16-SM
    // GTX 1650 Ti and the stride loop covers any pairCount (large scenes can
    // produce hundreds of thousands of pairs).
    constexpr std::uint32_t threadCount = 256;
    constexpr std::uint32_t kFbpMaxBlocks = 1024;
    const std::uint32_t fbpUpperPairs = std::min(gridConfig.maxCandidatePairs, kFbpMaxBlocks * threadCount);
    const std::uint32_t fbpBlocks =
        std::max(1u, (fbpUpperPairs + threadCount - 1u) / threadCount);

    featureBasedProximityKernel<<<fbpBlocks, threadCount>>>(
        deviceFirstPositions,
        workspace.indexedTissueIndices,
        deviceSecondPositions,
        workspace.indexedToolIndices,
        workspace.candidatePairs,
        nullptr,
        workspace.candidateCount,
        false,
        deviceProximityContacts,
        workspace.proximityContactCount,
        workspace.proximityOverflowCount,
        workspace.proximityVfCount,
        workspace.proximityFvCount,
        workspace.proximityEeCount,
        proximityConfig.maxContacts,
        proximityConfig.contactDistance,
        proximityConfig.computeBarycentrics);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    float fbpMs = 0.0f;
    if (needFbpTiming)
    {
        cudaEventRecord(workspace.fbpEndEvent);
        cudaEventSynchronize(workspace.fbpEndEvent);
        cudaEventElapsedTime(&fbpMs, workspace.fbpStartEvent, workspace.fbpEndEvent);
    }
    if (executionStats != nullptr)
    {
        executionStats->gpuKernelMilliseconds += fbpMs;
        executionStats->featureBasedProximityKernelMilliseconds += fbpMs;
    }

    // Step 4: optional batched readback.
    // Old path was 5 separate synchronous cudaMemcpy calls (~250 us total on
    // the GTX 1650 Ti). New path issues 5 async copies into a pinned host
    // buffer + one single cudaDeviceSynchronize, which collapses to ~50 us.
    std::uint32_t hostContactCount = 0;
    std::uint32_t hostOverflowCount = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* dst = workspace.proximityCountersHostPinned;
        cudaMemcpyAsync(dst + 0, workspace.proximityContactCount,  sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 1, workspace.proximityOverflowCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 2, workspace.proximityVfCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 3, workspace.proximityFvCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 4, workspace.proximityEeCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostContactCount  = dst[0];
        hostOverflowCount = dst[1];
        hostVf            = dst[2];
        hostFv            = dst[3];
        hostEe            = dst[4];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 5u * sizeof(std::uint32_t);
        }
        if (proximityStats != nullptr)
        {
            proximityStats->emittedContactCount = hostContactCount;
            proximityStats->vfContactCount = hostVf;
            proximityStats->fvContactCount = hostFv;
            proximityStats->eeContactCount = hostEe;
        }
        if (executionStats != nullptr)
        {
            executionStats->vfContactCount += hostVf;
            executionStats->fvContactCount += hostFv;
            executionStats->eeContactCount += hostEe;
        }
    }

    if (!proximityConfig.keepContactsOnDevice && hostContactCount > 0)
    {
        const std::uint32_t copyCount = std::min(hostContactCount, proximityConfig.maxContacts);
        std::vector<DeviceProximityContact> hostBuffer(copyCount);
        cudaMemcpy(hostBuffer.data(), deviceProximityContacts,
            copyCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += static_cast<std::uint64_t>(copyCount) * sizeof(DeviceProximityContact);
        }
        contacts.reserve(copyCount);
        for (const auto& c : hostBuffer)
        {
            ProximityContact out {};
            out.firstPrimitiveIndex = c.firstPrimitiveIndex;
            out.secondPrimitiveIndex = c.secondPrimitiveIndex;
            out.featureKind = static_cast<ProximityFeatureKind>(c.featureKind);
            out.firstFeatureLocalIndex = c.firstFeatureLocalIndex;
            out.secondFeatureLocalIndex = c.secondFeatureLocalIndex;
            out.reserved = 0;
            out.firstBarycentrics[0] = c.firstBary[0];
            out.firstBarycentrics[1] = c.firstBary[1];
            out.firstBarycentrics[2] = c.firstBary[2];
            out.secondBarycentrics[0] = c.secondBary[0];
            out.secondBarycentrics[1] = c.secondBary[1];
            out.secondBarycentrics[2] = c.secondBary[2];
            out.pointOnFirst  = TriangleVertex { c.pointOnFirst.x,  c.pointOnFirst.y,  c.pointOnFirst.z };
            out.pointOnSecond = TriangleVertex { c.pointOnSecond.x, c.pointOnSecond.y, c.pointOnSecond.z };
            out.normal        = TriangleVertex { c.normal.x,        c.normal.y,        c.normal.z };
            out.signedDistance = c.signedDistance;
            contacts.push_back(out);
        }
    }

    if (executionStats != nullptr)
    {
        executionStats->outputContactCount = hostContactCount;
        if (hostOverflowCount > 0) executionStats->overflowCount += hostOverflowCount;
    }

    diagnostic.clear();
    return true;
}

// ============================================================================
// computeFeatureBasedVertexTriangleContacts
// ----------------------------------------------------------------------------
// Vertex-triangle feature-based proximity. Inserts the triangle mesh into the
// tissue side of the dense grid (using the existing insertIndexedTrianglesKernel)
// and the vertex cloud into the tool side (using the new insertIndexedPointsKernel),
// then runs the existing candidate-pair generator to produce (triangleId, vertexId)
// pairs and finally featureBasedVertexTriangleProximityKernel for the narrow pass.
//
// Self-collision is supported: pass the same vertex set + triangle surface and
// set `selfCollisionExclusion = true` in the proximityConfig (encoded by re-
// purposing `emitOnePerPair == false`). When enabled, the narrow kernel skips
// pairs where the candidate vertex is one of the triangle's own corner vertices.
// ============================================================================
bool computeFeatureBasedVertexTriangleContacts(
    const PointCloudSurface& pointCloud,
    const TriangleIndexedSurface& triangleSurface,
    const DenseGridConfig& gridConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats)
{
    contacts.clear();
    if (executionStats != nullptr)
    {
        *executionStats = BackendExecutionStats {};
        executionStats->inputPrimitiveCount = triangleSurface.triangleCount + pointCloud.pointCount;
    }
    if (proximityStats != nullptr)
    {
        *proximityStats = FeatureBasedProximityStats {};
    }

    if ((pointCloud.positions == nullptr && pointCloud.devicePositions == nullptr) ||
        pointCloud.pointCount == 0 ||
        (triangleSurface.positions == nullptr && triangleSurface.devicePositions == nullptr) ||
        triangleSurface.triangleIndices == nullptr ||
        triangleSurface.triangleCount == 0)
    {
        diagnostic = "Vertex-triangle proximity: invalid point cloud or triangle surface.";
        return false;
    }

    if (gridConfig.gridResolutionX == 0 || gridConfig.gridResolutionY == 0 || gridConfig.gridResolutionZ == 0 ||
        gridConfig.maxTissueTrianglesPerCell == 0 || gridConfig.maxToolTrianglesPerCell == 0 ||
        gridConfig.maxCandidatePairs == 0 ||
        gridConfig.gridMaxX <= gridConfig.gridMinX ||
        gridConfig.gridMaxY <= gridConfig.gridMinY ||
        gridConfig.gridMaxZ <= gridConfig.gridMinZ)
    {
        diagnostic = "Invalid dense-grid configuration.";
        return false;
    }

    // The self-collision exclusion is gated by detecting same-surface input:
    // if pointCloud.surfaceId == triangleSurface.surfaceId we are testing a
    // mesh against itself and must skip own-corner vertices.
    const bool selfCollision = (pointCloud.surfaceId != 0 &&
                                pointCloud.surfaceId == triangleSurface.surfaceId);

    const std::uint64_t cellCount64 =
        static_cast<std::uint64_t>(gridConfig.gridResolutionX) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionY) *
        static_cast<std::uint64_t>(gridConfig.gridResolutionZ);
    if (cellCount64 == 0 || cellCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid cell count exceeds uint32_t indexing capacity.";
        return false;
    }
    const auto cellCount = static_cast<std::uint32_t>(cellCount64);

    const std::uint64_t tissueBucketCount = cellCount64 * static_cast<std::uint64_t>(gridConfig.maxTissueTrianglesPerCell);
    const std::uint64_t toolBucketCount   = cellCount64 * static_cast<std::uint64_t>(gridConfig.maxToolTrianglesPerCell);
    const std::uint64_t pairHashCount64   = gridConfig.useGpuHashDedupe
        ? static_cast<std::uint64_t>(nextPowerOfTwo(static_cast<std::size_t>(gridConfig.maxCandidatePairs) * 2u))
        : 1ull;
    if (pairHashCount64 == 0 || pairHashCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        diagnostic = "Dense-grid hash dedupe table size out of range.";
        return false;
    }

    auto& workspace = denseGridWorkspace();
    std::uint64_t newlyAllocatedBytes = 0;
    const auto allocStart = std::chrono::steady_clock::now();
    cudaError_t err = workspace.ensure(
        0,
        0,
        static_cast<std::size_t>(cellCount64),
        static_cast<std::size_t>(tissueBucketCount),
        static_cast<std::size_t>(toolBucketCount),
        static_cast<std::size_t>(gridConfig.maxCandidatePairs),
        static_cast<std::size_t>(pairHashCount64),
        newlyAllocatedBytes);
    if (err == cudaSuccess)
    {
        err = workspace.ensureIndexedInput(
            triangleSurface.devicePositions == nullptr ? triangleSurface.vertexCount : 0u,
            pointCloud.devicePositions == nullptr ? pointCloud.pointCount : 0u,
            static_cast<std::size_t>(triangleSurface.triangleCount) * 3u,
            0u,
            newlyAllocatedBytes);
    }
    if (err == cudaSuccess)
    {
        err = workspace.ensureProximityStorage(
            proximityConfig.maxContacts,
            sizeof(DeviceProximityContact),
            newlyAllocatedBytes);
    }
    const double allocMs = elapsedMillisecondsSince(allocStart);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Vertex-triangle workspace alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = cellCount;
        executionStats->deviceAllocationBytes += newlyAllocatedBytes;
        executionStats->deviceAllocationMilliseconds += allocMs;
        if (newlyAllocatedBytes > 0) executionStats->workspaceResizeCount += 1;
    }

    // Upload triangle positions and indices (if host-side input).
    const std::size_t triPosBytes = static_cast<std::size_t>(triangleSurface.vertexCount) * sizeof(BackendTriangleVertex);
    const std::size_t triIdxBytes = static_cast<std::size_t>(triangleSurface.triangleCount) * 3u * sizeof(std::uint32_t);
    const std::size_t pointPosBytes = static_cast<std::size_t>(pointCloud.pointCount) * sizeof(BackendTriangleVertex);
    const bool uploadTriangleTopology =
        workspace.indexedTissueSurfaceId != triangleSurface.surfaceId ||
        workspace.indexedTissueTopologyVersion != triangleSurface.topologyVersion;

    const auto h2dStart = std::chrono::steady_clock::now();
    if (triangleSurface.devicePositions == nullptr)
    {
        err = copyHostArrayToDeviceAsync(
            workspace.indexedTissuePositions,
            triangleSurface.positions,
            workspace.pinnedIndexedTissuePositions,
            static_cast<std::size_t>(triangleSurface.vertexCount),
            false);
        if (executionStats != nullptr) executionStats->hostToDeviceBytes += triPosBytes;
    }
    if (err == cudaSuccess && uploadTriangleTopology)
    {
        err = copyHostArrayToDeviceAsync(
            workspace.indexedTissueIndices,
            triangleSurface.triangleIndices,
            workspace.pinnedIndexedTissueIndices,
            static_cast<std::size_t>(triangleSurface.triangleCount) * 3u,
            false);
        if (executionStats != nullptr) executionStats->hostToDeviceBytes += triIdxBytes;
    }
    if (err == cudaSuccess && pointCloud.devicePositions == nullptr)
    {
        err = copyHostArrayToDeviceAsync(
            workspace.indexedToolPositions,
            pointCloud.positions,
            workspace.pinnedIndexedToolPositions,
            static_cast<std::size_t>(pointCloud.pointCount),
            false);
        if (executionStats != nullptr) executionStats->hostToDeviceBytes += pointPosBytes;
    }
    if (err == cudaSuccess && uploadTriangleTopology)
    {
        workspace.indexedTissueSurfaceId = triangleSurface.surfaceId;
        workspace.indexedTissueTopologyVersion = triangleSurface.topologyVersion;
    }
    if (executionStats != nullptr) executionStats->hostToDeviceMilliseconds += elapsedMillisecondsSince(h2dStart);
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    const BackendTriangleVertex* deviceTrianglePositions =
        triangleSurface.devicePositions == nullptr ? workspace.indexedTissuePositions : triangleSurface.devicePositions;
    const BackendTriangleVertex* devicePointPositions =
        pointCloud.devicePositions == nullptr ? workspace.indexedToolPositions : pointCloud.devicePositions;

    const DeviceDenseGridConfig deviceConfig {
        make_float3(gridConfig.gridMinX, gridConfig.gridMinY, gridConfig.gridMinZ),
        make_float3(gridConfig.gridMaxX, gridConfig.gridMaxY, gridConfig.gridMaxZ),
        make_float3(
            static_cast<float>(gridConfig.gridResolutionX) / (gridConfig.gridMaxX - gridConfig.gridMinX),
            static_cast<float>(gridConfig.gridResolutionY) / (gridConfig.gridMaxY - gridConfig.gridMinY),
            static_cast<float>(gridConfig.gridResolutionZ) / (gridConfig.gridMaxZ - gridConfig.gridMinZ)),
        gridConfig.gridResolutionX,
        gridConfig.gridResolutionY,
        gridConfig.gridResolutionZ,
        gridConfig.contactDistance,
        gridConfig.maxTissueTrianglesPerCell,
        gridConfig.maxToolTrianglesPerCell,
        gridConfig.maxCandidatePairs,
        static_cast<std::uint32_t>(pairHashCount64),
        gridConfig.useGpuHashDedupe,
        false
    };

    constexpr std::uint32_t threadCount = 256;
    const std::uint32_t resetBlocks = (cellCount + threadCount - 1u) / threadCount;
    const std::uint32_t triBlocks   = (triangleSurface.triangleCount + threadCount - 1u) / threadCount;
    const std::uint32_t pointBlocks = (pointCloud.pointCount + threadCount - 1u) / threadCount;

    // 1) Reset grid + counters
    resetDenseGridKernel<<<resetBlocks, threadCount>>>(
        workspace.grid,
        cellCount,
        workspace.pairHashKeys,
        static_cast<std::uint32_t>(pairHashCount64),
        false,
        workspace.activeCellCount,
        workspace.rawCandidateCount,
        workspace.candidateCount,
        workspace.contactCount,
        workspace.overflowCount,
        workspace.denseGridStats,
        false);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    // 2) Clear pair hash via cudaMemset (matches existing dense-grid path)
    if (gridConfig.useGpuHashDedupe)
    {
        cudaMemsetAsync(workspace.pairHashKeys, 0xff, pairHashCount64 * sizeof(unsigned long long));
        if (executionStats != nullptr) executionStats->cudaMemsetCount += 1;
    }

    // 3) Insert triangles into tissue buckets
    insertIndexedTrianglesKernel<<<triBlocks, threadCount>>>(
        deviceTrianglePositions,
        workspace.indexedTissueIndices,
        triangleSurface.triangleCount,
        true,  // insertTissue
        deviceConfig,
        workspace.grid,
        workspace.cellTissueIds,
        workspace.overflowCount,
        nullptr);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    // 4) Insert points into tool buckets
    insertIndexedPointsKernel<<<pointBlocks, threadCount>>>(
        devicePointPositions,
        pointCloud.pointCount,
        deviceConfig,
        workspace.grid,
        workspace.cellToolIds,
        workspace.overflowCount,
        nullptr);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    // 5) Generate candidate pairs
    generateDenseGridUniqueCandidatePairsKernel<<<cellCount, threadCount>>>(
        workspace.grid,
        workspace.cellTissueIds,
        workspace.cellToolIds,
        deviceConfig,
        workspace.candidatePairs,
        workspace.pairHashKeys,
        workspace.rawCandidateCount,
        workspace.candidateCount,
        workspace.overflowCount,
        nullptr);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    // 6) Reset proximity counters
    auto* deviceProximityContacts = reinterpret_cast<DeviceProximityContact*>(workspace.proximityContacts);
    resetProximityCountersKernel<<<1, 1>>>(
        workspace.proximityContactCount,
        workspace.proximityOverflowCount,
        workspace.proximityVfCount,
        workspace.proximityFvCount,
        workspace.proximityEeCount);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    // 7) Narrow phase: feature-based vertex-triangle proximity.
    // Grid-strides over all pairs; 1024 blocks is a GPU-saturation target.
    constexpr std::uint32_t kFbpMaxBlocks = 1024;
    const std::uint32_t fbpUpperPairs = std::min(gridConfig.maxCandidatePairs, kFbpMaxBlocks * threadCount);
    const std::uint32_t fbpBlocks = std::max(1u, (fbpUpperPairs + threadCount - 1u) / threadCount);
    featureBasedVertexTriangleProximityKernel<<<fbpBlocks, threadCount>>>(
        deviceTrianglePositions,
        workspace.indexedTissueIndices,
        devicePointPositions,
        workspace.candidatePairs,
        workspace.candidateCount,
        deviceProximityContacts,
        workspace.proximityContactCount,
        workspace.proximityOverflowCount,
        workspace.proximityVfCount,
        proximityConfig.maxContacts,
        proximityConfig.contactDistance,
        proximityConfig.computeBarycentrics,
        selfCollision ? 1u : 0u);
    if (executionStats != nullptr) executionStats->kernelLaunchCount += 1;

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = cudaGetErrorString(err);
        return false;
    }

    // 8) Optional readback (batched via pinned host buffer)
    std::uint32_t hostContactCount = 0;
    std::uint32_t hostVf = 0, hostFv = 0, hostEe = 0;
    if (proximityConfig.readContactCounter || !proximityConfig.keepContactsOnDevice)
    {
        std::uint32_t* dst = workspace.proximityCountersHostPinned;
        cudaMemcpyAsync(dst + 0, workspace.proximityContactCount,  sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 2, workspace.proximityVfCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 3, workspace.proximityFvCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(dst + 4, workspace.proximityEeCount,       sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        hostContactCount = dst[0];
        hostVf = dst[2];
        hostFv = dst[3];
        hostEe = dst[4];
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += 4u * sizeof(std::uint32_t);
            executionStats->vfContactCount += hostVf;
            executionStats->fvContactCount += hostFv;
            executionStats->eeContactCount += hostEe;
        }
        if (proximityStats != nullptr)
        {
            proximityStats->emittedContactCount = hostContactCount;
            proximityStats->vfContactCount = hostVf;
            proximityStats->fvContactCount = hostFv;
            proximityStats->eeContactCount = hostEe;
        }
    }

    if (!proximityConfig.keepContactsOnDevice && hostContactCount > 0)
    {
        const std::uint32_t copyCount = std::min(hostContactCount, proximityConfig.maxContacts);
        std::vector<DeviceProximityContact> hostBuffer(copyCount);
        cudaMemcpy(hostBuffer.data(), deviceProximityContacts,
            copyCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
        if (executionStats != nullptr)
        {
            executionStats->deviceToHostBytes += static_cast<std::uint64_t>(copyCount) * sizeof(DeviceProximityContact);
        }
        contacts.reserve(copyCount);
        for (const auto& c : hostBuffer)
        {
            ProximityContact out {};
            out.firstPrimitiveIndex = c.firstPrimitiveIndex;
            out.secondPrimitiveIndex = c.secondPrimitiveIndex;
            out.featureKind = static_cast<ProximityFeatureKind>(c.featureKind);
            out.firstFeatureLocalIndex = c.firstFeatureLocalIndex;
            out.secondFeatureLocalIndex = c.secondFeatureLocalIndex;
            out.firstBarycentrics[0] = c.firstBary[0];
            out.firstBarycentrics[1] = c.firstBary[1];
            out.firstBarycentrics[2] = c.firstBary[2];
            out.secondBarycentrics[0] = c.secondBary[0];
            out.secondBarycentrics[1] = c.secondBary[1];
            out.secondBarycentrics[2] = c.secondBary[2];
            out.pointOnFirst  = TriangleVertex { c.pointOnFirst.x,  c.pointOnFirst.y,  c.pointOnFirst.z };
            out.pointOnSecond = TriangleVertex { c.pointOnSecond.x, c.pointOnSecond.y, c.pointOnSecond.z };
            out.normal        = TriangleVertex { c.normal.x,        c.normal.y,        c.normal.z };
            out.signedDistance = c.signedDistance;
            contacts.push_back(out);
        }
    }

    if (executionStats != nullptr)
    {
        executionStats->outputContactCount = hostContactCount;
    }

    diagnostic.clear();
    return true;
}

// ============================================================================
// computeHashPrefixSumProximityContacts (EXPERIMENTAL)
// ----------------------------------------------------------------------------
// Spatial-hash broad phase using compact occupied buckets. The steady-state path
// clears only touched pair-hash slots from the previous frame, inserts topology
// into compact buckets, scans bucket pair counts with persistent CUB storage, and
// generates only mixed occupied buckets before reusing the FBP narrow kernel.
// ============================================================================
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

    const DeviceDenseGridConfig dc {
        make_float3(gridConfig.gridMinX, gridConfig.gridMinY, gridConfig.gridMinZ),
        make_float3(gridConfig.gridMaxX, gridConfig.gridMaxY, gridConfig.gridMaxZ),
        make_float3(
            static_cast<float>(gridConfig.gridResolutionX) / (gridConfig.gridMaxX - gridConfig.gridMinX),
            static_cast<float>(gridConfig.gridResolutionY) / (gridConfig.gridMaxY - gridConfig.gridMinY),
            static_cast<float>(gridConfig.gridResolutionZ) / (gridConfig.gridMaxZ - gridConfig.gridMinZ)),
        gridConfig.gridResolutionX, gridConfig.gridResolutionY, gridConfig.gridResolutionZ,
        gridConfig.contactDistance,
        gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell,
        gridConfig.maxCandidatePairs, static_cast<std::uint32_t>(pairHashCount64),
        true, false
    };

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
            else clearTouchedPairHash32Kernel<<<pairHashClearBlocks, threads, 0, s>>>(ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        else
        {
            if (fullClear) { cudaMemsetAsync(ws.pairHashKeys, 0xff, static_cast<std::size_t>(pairHashCount) * sizeof(unsigned long long), s); ws.pairHashKeysInitialized = true; }
            else clearTouchedPairHash64Kernel<<<pairHashClearBlocks, threads, 0, s>>>(ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        resetCompactHashGridKernel<<<resetBlocks, threads, 0, s>>>(ws.cellKeys, ws.slotBucketIds, tableSize, ws.tissueCount, ws.toolCount, ws.pairsPerBucket, bucketCapacity, ws.rawCandidateCount, ws.candidateCount, ws.overflowCount, ws.probeOverflowCount, ws.occupiedBucketCount, ws.mixedBucketCount, ws.touchedPairHashCount);
        markCompactHashGridCellsKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, dc, tableSize, maxProbe, ws.cellKeys, ws.probeOverflowCount);
        markCompactHashGridCellsKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, dc, tableSize, maxProbe, ws.cellKeys, ws.probeOverflowCount);
        compactHashGridSlotsKernel<<<tableBlocks, threads, 0, s>>>(ws.cellKeys, ws.slotBucketIds, ws.bucketCellIds, tableSize, bucketCapacity, ws.occupiedBucketCount, ws.overflowCount);
        fillCompactHashGridTrianglesKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, true, dc, tableSize, bucketCapacity, maxProbe, ws.cellKeys, ws.slotBucketIds, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, ws.overflowCount, ws.probeOverflowCount);
        fillCompactHashGridTrianglesKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, false, dc, tableSize, bucketCapacity, maxProbe, ws.cellKeys, ws.slotBucketIds, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, ws.overflowCount, ws.probeOverflowCount);
        computeCompactHashPairsPerBucketKernel<<<bucketBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, ws.pairsPerBucket, ws.rawCandidateCount);
        if (useCompactCandidatePairs)
            generateMixedHashCandidatePairs32Kernel<<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.compactCandidatePairs, ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        else
            generateMixedHashCandidatePairs64Kernel<<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.candidatePairs, ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
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
            clearTouchedPairHash32Kernel<<<pairHashClearBlocks, threads>>>(
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
            clearTouchedPairHash64Kernel<<<pairHashClearBlocks, threads>>>(
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
            generateMixedHashCandidatePairs32Kernel<<<kGenBlocks, threads>>>(
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
            generateMixedHashCandidatePairs64Kernel<<<kGenBlocks, threads>>>(
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
        const std::uint32_t copyCount = std::min(hostContactCount, proximityConfig.maxContacts);
        std::vector<DeviceProximityContact> hostBuffer(copyCount);
        cudaMemcpy(hostBuffer.data(), deviceContacts, copyCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
        if (executionStats != nullptr) executionStats->deviceToHostBytes += static_cast<std::uint64_t>(copyCount) * sizeof(DeviceProximityContact);
        contacts.reserve(copyCount);
        for (const auto& c : hostBuffer)
        {
            ProximityContact out {};
            out.firstPrimitiveIndex = c.firstPrimitiveIndex;
            out.secondPrimitiveIndex = c.secondPrimitiveIndex;
            out.featureKind = static_cast<ProximityFeatureKind>(c.featureKind);
            out.firstFeatureLocalIndex = c.firstFeatureLocalIndex;
            out.secondFeatureLocalIndex = c.secondFeatureLocalIndex;
            out.firstBarycentrics[0] = c.firstBary[0]; out.firstBarycentrics[1] = c.firstBary[1]; out.firstBarycentrics[2] = c.firstBary[2];
            out.secondBarycentrics[0] = c.secondBary[0]; out.secondBarycentrics[1] = c.secondBary[1]; out.secondBarycentrics[2] = c.secondBary[2];
            out.pointOnFirst = TriangleVertex { c.pointOnFirst.x, c.pointOnFirst.y, c.pointOnFirst.z };
            out.pointOnSecond = TriangleVertex { c.pointOnSecond.x, c.pointOnSecond.y, c.pointOnSecond.z };
            out.normal = TriangleVertex { c.normal.x, c.normal.y, c.normal.z };
            out.signedDistance = c.signedDistance;
            contacts.push_back(out);
        }
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

    const auto surfaceInvalid = [](const TriangleIndexedSurface& s) {
        return (s.positions == nullptr && s.devicePositions == nullptr) ||
            s.triangleIndices == nullptr || s.vertexCount == 0 || s.triangleCount == 0;
    };
    if (surfaceInvalid(firstSurface) || surfaceInvalid(secondSurface))
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

    const DeviceDenseGridConfig dc {
        make_float3(gridConfig.gridMinX, gridConfig.gridMinY, gridConfig.gridMinZ),
        make_float3(gridConfig.gridMaxX, gridConfig.gridMaxY, gridConfig.gridMaxZ),
        make_float3(
            static_cast<float>(gridConfig.gridResolutionX) / (gridConfig.gridMaxX - gridConfig.gridMinX),
            static_cast<float>(gridConfig.gridResolutionY) / (gridConfig.gridMaxY - gridConfig.gridMinY),
            static_cast<float>(gridConfig.gridResolutionZ) / (gridConfig.gridMaxZ - gridConfig.gridMinZ)),
        gridConfig.gridResolutionX, gridConfig.gridResolutionY, gridConfig.gridResolutionZ,
        gridConfig.contactDistance,
        gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell,
        gridConfig.maxCandidatePairs, static_cast<std::uint32_t>(pairHashCount64),
        true, false
    };

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
            else clearTouchedPairHash32Kernel<<<pairHashClearBlocks, threads, 0, s>>>(ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        else
        {
            if (fullClear) { cudaMemsetAsync(ws.pairHashKeys, 0xff, static_cast<std::size_t>(pairHashCount) * sizeof(unsigned long long), s); ws.pairHashKeysInitialized = true; }
            else clearTouchedPairHash64Kernel<<<pairHashClearBlocks, threads, 0, s>>>(ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount);
        }
        resetCompactHashGridKernel<<<resetBlocks, threads, 0, s>>>(ws.cellKeys, ws.slotBucketIds, tableSize, ws.tissueCount, ws.toolCount, ws.pairsPerBucket, bucketCapacity, ws.rawCandidateCount, ws.candidateCount, ws.overflowCount, ws.probeOverflowCount, ws.occupiedBucketCount, ws.mixedBucketCount, ws.touchedPairHashCount);
        insertSimpleHashTrianglesKernel<<<firstBlocks, threads, 0, s>>>(deviceFirstPositions, ws.firstIndices, firstSurface.triangleCount, true, dc, tableSize, maxProbe, ws.cellKeys, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, ws.overflowCount, ws.probeOverflowCount);
        insertSimpleHashTrianglesKernel<<<secondBlocks, threads, 0, s>>>(deviceSecondPositions, ws.secondIndices, secondSurface.triangleCount, false, dc, tableSize, maxProbe, ws.cellKeys, ws.tissueCount, ws.toolCount, ws.tissueIds, ws.toolIds, ws.mixedBucketIds, ws.mixedBucketCount, bucketCapacity, ws.overflowCount, ws.probeOverflowCount);
        if (useCompactCandidatePairs)
            generateMixedHashCandidatePairs32Kernel<<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.compactCandidatePairs, ws.compactPairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
        else
            generateMixedHashCandidatePairs64Kernel<<<kGenBlocks, threads, 0, s>>>(ws.tissueCount, ws.toolCount, ws.mixedBucketIds, ws.mixedBucketCount, ws.tissueIds, ws.toolIds, bucketCapacity, gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell, pairHashCount, gridConfig.maxCandidatePairs, ws.candidatePairs, ws.pairHashKeys, ws.touchedPairHashSlots, ws.touchedPairHashCount, ws.candidateCount, ws.probeOverflowCount, ws.overflowCount);
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
            ws.graphInstantiated = false;
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
                    cudaGetLastError();
                }
            }
            if (ws.graphInstantiated)
            {
                err = cudaGraphLaunch(ws.graphExec, 0);
                ranViaGraph = (err == cudaSuccess);
            }
            if (!ranViaGraph)
            {
                launchAll(0, /*fullClear=*/false);
                ranViaGraph = true;
            }
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
        const std::uint32_t copyCount = std::min(hostContactCount, proximityConfig.maxContacts);
        std::vector<DeviceProximityContact> hostBuffer(copyCount);
        cudaMemcpy(hostBuffer.data(), deviceContacts, copyCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
        if (executionStats != nullptr) executionStats->deviceToHostBytes += static_cast<std::uint64_t>(copyCount) * sizeof(DeviceProximityContact);
        contacts.reserve(copyCount);
        for (const auto& c : hostBuffer)
        {
            ProximityContact out {};
            out.firstPrimitiveIndex = c.firstPrimitiveIndex;
            out.secondPrimitiveIndex = c.secondPrimitiveIndex;
            out.featureKind = static_cast<ProximityFeatureKind>(c.featureKind);
            out.firstFeatureLocalIndex = c.firstFeatureLocalIndex;
            out.secondFeatureLocalIndex = c.secondFeatureLocalIndex;
            out.firstBarycentrics[0] = c.firstBary[0]; out.firstBarycentrics[1] = c.firstBary[1]; out.firstBarycentrics[2] = c.firstBary[2];
            out.secondBarycentrics[0] = c.secondBary[0]; out.secondBarycentrics[1] = c.secondBary[1]; out.secondBarycentrics[2] = c.secondBary[2];
            out.pointOnFirst = TriangleVertex { c.pointOnFirst.x, c.pointOnFirst.y, c.pointOnFirst.z };
            out.pointOnSecond = TriangleVertex { c.pointOnSecond.x, c.pointOnSecond.y, c.pointOnSecond.z };
            out.normal = TriangleVertex { c.normal.x, c.normal.y, c.normal.z };
            out.signedDistance = c.signedDistance;
            contacts.push_back(out);
        }
    }

    if (executionStats != nullptr)
    {
        executionStats->gridCellCount = cellCount;
    }

    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
