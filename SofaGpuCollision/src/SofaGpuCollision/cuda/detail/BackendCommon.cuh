// BackendCommon.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Shared device structs/constants, NVTX + allocation/copy/timing helpers,
// and the float3/AABB device math used by every broad cull.

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




// Validity check for an indexed triangle surface, shared by the hash /
// simple-hash / sorted-grid drivers (was three identical lambdas).
bool indexedSurfaceInvalid(const SofaGpuCollision::backend::TriangleIndexedSurface& s)
{
    return (s.positions == nullptr && s.devicePositions == nullptr) ||
        s.triangleIndices == nullptr || s.vertexCount == 0 || s.triangleCount == 0;
}

// Bit-exact float packing for graph signatures (a raw compare, not an
// arithmetic one, so NaN/-0 quirks cannot alias two different configs).
std::uint64_t floatSignatureBits(const float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// CUDA-graph capture/replay machinery shared by the hash / simple-hash /
// sorted-grid drivers (was three copies). The driver owns first-frame
// semantics (direct run, table init, any probing); this class owns the
// steady-state capture -> instantiate -> replay -> safe-direct-fallback
// sequence. The signature is an opaque field pack built by the driver: any
// change (or a workspace resize) invalidates the instantiated graph, exactly
// like the per-field comparisons it replaces.
struct CudaGraphReplayer
{
    cudaStream_t captureStream { nullptr };
    cudaGraphExec_t graphExec { nullptr };
    bool instantiated { false };
    std::array<std::uint64_t, 6> signature {};

    void release()
    {
        if (graphExec != nullptr) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
        if (captureStream != nullptr) { cudaStreamDestroy(captureStream); captureStream = nullptr; }
        instantiated = false;
        signature = {};
    }

    void invalidateIfChanged(const std::array<std::uint64_t, 6>& sig, const bool forceInvalidate)
    {
        if (forceInvalidate || sig != signature) instantiated = false;
    }

    // Steady-state frames only. launchAll(stream) must record the whole
    // per-frame kernel sequence on the given stream.
    template <class LaunchAll>
    void replay(const std::array<std::uint64_t, 6>& sig, LaunchAll&& launchAll)
    {
        if (captureStream == nullptr) cudaStreamCreate(&captureStream);
        if (!instantiated && captureStream != nullptr)
        {
            cudaGraph_t graph = nullptr;
            cudaError_t capErr = cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal);
            if (capErr == cudaSuccess)
            {
                launchAll(captureStream);
                capErr = cudaStreamEndCapture(captureStream, &graph);
            }
            if (capErr == cudaSuccess && graph != nullptr)
            {
                if (graphExec != nullptr) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
#if CUDART_VERSION >= 12000
                capErr = cudaGraphInstantiate(&graphExec, graph, 0ull);
#else
                capErr = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
#endif
                cudaGraphDestroy(graph);
                if (capErr == cudaSuccess)
                {
                    instantiated = true;
                    signature = sig;
                }
            }
            else
            {
                cudaGetLastError();  // clear any capture error
            }
        }
        bool ranViaGraph = false;
        if (instantiated)
        {
            ranViaGraph = (cudaGraphLaunch(graphExec, 0) == cudaSuccess);
        }
        if (!ranViaGraph)
        {
            launchAll(static_cast<cudaStream_t>(0));  // safe direct fallback
        }
    }
};

// Topology/position upload for the hash / simple-hash / sorted-grid workspaces
// (identical field layout; was three copy-pasted blocks). Indices re-upload
// only on topology change; positions only when the caller has host input.
template <class Workspace>
cudaError_t uploadSurfacesToWorkspace(
    Workspace& ws,
    const SofaGpuCollision::backend::TriangleIndexedSurface& firstSurface,
    const SofaGpuCollision::backend::TriangleIndexedSurface& secondSurface,
    SofaGpuCollision::backend::BackendExecutionStats* executionStats)
{
    const bool uploadFirstTopo = ws.firstSurfaceId != firstSurface.surfaceId || ws.firstTopologyVersion != firstSurface.topologyVersion;
    const bool uploadSecondTopo = ws.secondSurfaceId != secondSurface.surfaceId || ws.secondTopologyVersion != secondSurface.topologyVersion;
    const auto h2dStart = std::chrono::steady_clock::now();
    cudaError_t err = cudaSuccess;
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
    return err;
}

// Build the device-side grid config from the public DenseGridConfig. Shared by
// the hash / simple-hash / sorted-grid drivers (was three copy-pasted blocks;
// the two dense drivers predate it and keep their own construction).
DeviceDenseGridConfig makeDeviceDenseGridConfig(
    const SofaGpuCollision::backend::DenseGridConfig& gridConfig,
    const std::uint32_t pairHashCapacity)
{
    return DeviceDenseGridConfig {
        make_float3(gridConfig.gridMinX, gridConfig.gridMinY, gridConfig.gridMinZ),
        make_float3(gridConfig.gridMaxX, gridConfig.gridMaxY, gridConfig.gridMaxZ),
        make_float3(
            static_cast<float>(gridConfig.gridResolutionX) / (gridConfig.gridMaxX - gridConfig.gridMinX),
            static_cast<float>(gridConfig.gridResolutionY) / (gridConfig.gridMaxY - gridConfig.gridMinY),
            static_cast<float>(gridConfig.gridResolutionZ) / (gridConfig.gridMaxZ - gridConfig.gridMinZ)),
        gridConfig.gridResolutionX, gridConfig.gridResolutionY, gridConfig.gridResolutionZ,
        gridConfig.contactDistance,
        gridConfig.maxTissueTrianglesPerCell, gridConfig.maxToolTrianglesPerCell,
        gridConfig.maxCandidatePairs, pairHashCapacity,
        true, false
    };
}

} // namespace
