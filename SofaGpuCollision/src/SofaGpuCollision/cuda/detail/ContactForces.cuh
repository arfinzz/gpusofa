// ContactForces.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included last (needs DeviceProximityContact + every proximity driver's
// recordContactHandle call). Added 2026-07-15 as Tier 1: the CONSUMER side of
// the collision pipeline.
//
// Until now the contact buffer was a dead end: either it stayed on the device
// and nothing read it, or it was copied into SOFA's host DetectionOutput. This
// module closes the loop — one kernel reads the device contacts and scatter-adds
// forces directly into SOFA's CudaVec3f force vectors, so a frame can run
// collision + response without the state ever touching the host.
//
// The scatter is deliberately factored into contactVertexWeights(): turning a
// contact into per-vertex weights is exactly what the FUTURE constraint path
// needs to build its Jacobian rows, so that decoding does not get buried inside
// the penalty law. (Same discipline as extracting fbpComputeClosestFeatureContact.)

namespace
{

// ----------------------------------------------------------------------------
// Contact -> parent-vertex weights.
//
// A contact is found between two TRIANGLES, but forces must land on the
// VERTICES that own them. The contact's closest feature can be a vertex, a
// face, or an edge, and the barycentric convention differs per case
// (see ProximityContact in GpuCollisionBackend.h):
//
//   VF (kind 0): first side is a VERTEX  -> firstBary  = (1,0,0), index = which vertex
//                second side is a FACE   -> secondBary = face barycentrics
//   FV (kind 1): first side is a FACE    -> firstBary  = face barycentrics
//                second side is a VERTEX -> secondBary = (1,0,0), index = which vertex
//   EE (kind 2): both sides are EDGES    -> bary = (1-s, s, 0) along the edge whose
//                                           start vertex is the local index
//
// This collapses all three into "weights on the triangle's 3 vertices", which is
// the only form the force scatter (and later, a constraint Jacobian) cares about.
// Weights always sum to 1, so the scattered force is conserved.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void contactVertexWeights(
    const std::uint8_t featureKind,     // 0 = VF, 1 = FV, 2 = EE
    const std::uint8_t localIndex,      // vertex id (VF/FV) or edge-start id (EE)
    const float* __restrict__ bary,     // the side's barycentrics
    const bool isFirstSide,
    float outWeights[3])
{
    outWeights[0] = 0.0f;
    outWeights[1] = 0.0f;
    outWeights[2] = 0.0f;

    const bool sideIsVertex =
        (featureKind == 0u && isFirstSide) ||   // VF: vertex on the first side
        (featureKind == 1u && !isFirstSide);    // FV: vertex on the second side
    const bool sideIsEdge = (featureKind == 2u);

    if (sideIsVertex)
    {
        outWeights[localIndex % 3u] = 1.0f;
    }
    else if (sideIsEdge)
    {
        const std::uint32_t a = localIndex % 3u;
        const std::uint32_t b = (a + 1u) % 3u;
        outWeights[a] = bary[0];
        outWeights[b] = bary[1];
    }
    else
    {
        // Face side: the barycentrics are already per triangle vertex, in order.
        outWeights[0] = bary[0];
        outWeights[1] = bary[1];
        outWeights[2] = bary[2];
    }
}

// Scatter a force onto a triangle's 3 vertices, weighted. One atomicAdd per
// component; contacts sharing a vertex are summed correctly.
__device__ __forceinline__ void scatterVertexForce(
    float* __restrict__ forces,                  // Vec3f array (3 floats per vertex)
    const std::uint32_t* __restrict__ indices,   // 3 per triangle
    const std::uint32_t triangleId,
    const float weights[3],
    const float3 force)
{
    #pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        const float w = weights[k];
        if (w == 0.0f) continue;
        const std::uint32_t vertexId = indices[3u * triangleId + static_cast<std::uint32_t>(k)];
        float* dst = forces + 3u * vertexId;
        atomicAdd(dst + 0, force.x * w);
        atomicAdd(dst + 1, force.y * w);
        atomicAdd(dst + 2, force.z * w);
    }
}

// Gather the interpolated value (position/velocity) at a contact point from the
// triangle's 3 vertices — the transpose of scatterVertexForce.
__device__ __forceinline__ float3 gatherVertexValue(
    const float* __restrict__ values,
    const std::uint32_t* __restrict__ indices,
    const std::uint32_t triangleId,
    const float weights[3])
{
    float3 out = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        const float w = weights[k];
        if (w == 0.0f) continue;
        const std::uint32_t vertexId = indices[3u * triangleId + static_cast<std::uint32_t>(k)];
        const float* src = values + 3u * vertexId;
        out.x += src[0] * w;
        out.y += src[1] * w;
        out.z += src[2] * w;
    }
    return out;
}

// ----------------------------------------------------------------------------
// Penalty response. One thread per contact.
//
//   depth = contactDistance - distance          (> 0 when closer than the margin)
//   vn    = (v_second - v_first) . normal       (< 0 while approaching)
//   F     = max(0, stiffness*depth - damping*vn)
//
// The normal points from the first surface's contact point toward the second's,
// so the first side is pushed along -n and the second along +n: equal and
// opposite, which Gate 2 checks by reduction.
//
// Note the contacts carry an UNSIGNED distance (FbpKernels writes sqrt of the
// squared closest-feature distance), so this is a proximity penalty that turns
// on at contactDistance rather than at interpenetration. That matches how the
// contacts were generated — they only exist within contactDistance at all.
// ----------------------------------------------------------------------------
__global__ void accumulateContactPenaltyForcesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    float* __restrict__ firstForces,
    float* __restrict__ secondForces,
    const float* __restrict__ firstVelocities,   // may be null
    const float* __restrict__ secondVelocities,  // may be null
    const float stiffness,
    const float damping,
    const float contactDistance,
    std::uint32_t* __restrict__ activeCount)
{
    const std::uint32_t total = min(*contactCount, capacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const DeviceProximityContact c = contacts[i];

        const float depth = contactDistance - c.signedDistance;
        if (depth <= 0.0f) continue;

        float wFirst[3];
        float wSecond[3];
        contactVertexWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wFirst);
        contactVertexWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wSecond);

        float magnitude = stiffness * depth;
        if (damping > 0.0f && firstVelocities != nullptr && secondVelocities != nullptr)
        {
            const float3 v1 = gatherVertexValue(firstVelocities, firstIndices, c.firstPrimitiveIndex, wFirst);
            const float3 v2 = gatherVertexValue(secondVelocities, secondIndices, c.secondPrimitiveIndex, wSecond);
            const float3 vrel = sub3(v2, v1);
            magnitude -= damping * dot3(vrel, c.normal);
        }
        if (magnitude <= 0.0f) continue;  // separating faster than the spring pulls

        const float3 force = make_float3(
            c.normal.x * magnitude, c.normal.y * magnitude, c.normal.z * magnitude);

        // First side pushed away from the second, second pushed away from the first.
        const float3 negForce = make_float3(-force.x, -force.y, -force.z);
        scatterVertexForce(firstForces, firstIndices, c.firstPrimitiveIndex, wFirst, negForce);
        scatterVertexForce(secondForces, secondIndices, c.secondPrimitiveIndex, wSecond, force);

        if (activeCount != nullptr) atomicAdd(activeCount, 1u);
    }
}

// Stiffness-times-dx for implicit integration. For an active contact the
// penalty force derivative is K = stiffness * (n outer n) acting on the relative
// displacement, so df = -kFactor * stiffness * (n . dRel) * n on the first side
// and +the same on the second. Damping is deliberately excluded (SOFA passes
// its own factors for the damping term; treating it here would double count).
__global__ void accumulateContactPenaltyDForcesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    float* __restrict__ firstDForces,
    float* __restrict__ secondDForces,
    const float* __restrict__ firstDx,
    const float* __restrict__ secondDx,
    const float stiffness,
    const float contactDistance,
    const float kFactor)
{
    const std::uint32_t total = min(*contactCount, capacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const DeviceProximityContact c = contacts[i];
        if (contactDistance - c.signedDistance <= 0.0f) continue;

        float wFirst[3];
        float wSecond[3];
        contactVertexWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wFirst);
        contactVertexWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wSecond);

        const float3 dx1 = gatherVertexValue(firstDx, firstIndices, c.firstPrimitiveIndex, wFirst);
        const float3 dx2 = gatherVertexValue(secondDx, secondIndices, c.secondPrimitiveIndex, wSecond);
        const float dn = dot3(sub3(dx2, dx1), c.normal);
        const float scale = kFactor * stiffness * dn;

        const float3 df = make_float3(
            c.normal.x * scale, c.normal.y * scale, c.normal.z * scale);
        const float3 negDf = make_float3(-df.x, -df.y, -df.z);
        scatterVertexForce(firstDForces, firstIndices, c.firstPrimitiveIndex, wFirst, negDf);
        scatterVertexForce(secondDForces, secondIndices, c.secondPrimitiveIndex, wSecond, df);
    }
}

// Scratch counter for the "how many contacts actually produced force" stat.
struct ContactForceWorkspace
{
    std::uint32_t* activeCount { nullptr };
    std::uint32_t* activeCountHostPinned { nullptr };

    ~ContactForceWorkspace()
    {
        cudaFree(activeCount);
        cudaFreeHost(activeCountHostPinned);
        activeCount = nullptr;
        activeCountHostPinned = nullptr;
    }

    cudaError_t ensure()
    {
        cudaError_t err = cudaSuccess;
        if (activeCount == nullptr)
        {
            void* p = nullptr;
            err = cudaMalloc(&p, sizeof(std::uint32_t));
            if (err == cudaSuccess) activeCount = static_cast<std::uint32_t*>(p);
        }
        if (err == cudaSuccess && activeCountHostPinned == nullptr)
        {
            err = cudaMallocHost(reinterpret_cast<void**>(&activeCountHostPinned), sizeof(std::uint32_t));
        }
        return err;
    }
};

ContactForceWorkspace& contactForceWorkspace()
{
    static ContactForceWorkspace workspace;
    return workspace;
}

// Shared validation for both public entry points.
//
// The caller names its two surfaces in scene order, but the narrow phase
// receives collision-model pairs in whatever order SOFA's broad phase emitted
// them — which is frequently the reverse. Rather than make every caller guess,
// accept both orders and report which one matched: when `outSwapped` is true
// the caller's "first" surface is the handle's SECOND, so the force vectors
// must be exchanged before the kernel sees them.
bool resolveContactHandle(
    const std::uint64_t firstSurfaceId,
    const std::uint64_t secondSurfaceId,
    const RecordedContactHandle*& out,
    bool& outSwapped,
    std::string& diagnostic)
{
    out = findContactHandle(firstSurfaceId, secondSurfaceId, outSwapped);
    if (out != nullptr)
    {
        return true;
    }

    const ContactHandleRegistry& registry = contactHandleRegistry();
    std::string recorded;
    std::size_t liveCount = 0;
    for (const auto& handle : registry.slots)
    {
        if (!handle.valid) continue;
        ++liveCount;
        if (!recorded.empty()) recorded += ", ";
        recorded += std::to_string(handle.firstSurfaceId) + "/" + std::to_string(handle.secondSurfaceId);
    }
    diagnostic = liveCount == 0
        ? "No device contact handle recorded — run a proximity computation first."
        : "No contact handle for surface pair " + std::to_string(firstSurfaceId) + "/" +
          std::to_string(secondSurfaceId) + " (recorded pairs: " + recorded + ")." +
          (registry.evictions > 0
              ? " NOTE: " + std::to_string(registry.evictions) +
                " handle eviction(s) so far — more live collision pairs than registry slots."
              : std::string());
    return false;
}

} // namespace


namespace SofaGpuCollision::backend
{

bool accumulateContactPenaltyForces(
    const ContactPenaltyConfig& config,
    const std::uint64_t firstSurfaceId,
    const std::uint64_t secondSurfaceId,
    void* deviceFirstForces,
    void* deviceSecondForces,
    const void* deviceFirstVelocities,
    const void* deviceSecondVelocities,
    ContactPenaltyStats* stats,
    std::string& diagnostic)
{
    if (stats != nullptr) *stats = ContactPenaltyStats {};
    if (deviceFirstForces == nullptr || deviceSecondForces == nullptr)
    {
        diagnostic = "Null device force vector.";
        return false;
    }

    const RecordedContactHandle* handle = nullptr;
    bool swapped = false;
    if (!resolveContactHandle(firstSurfaceId, secondSurfaceId, handle, swapped, diagnostic)) return false;

    // Bind the caller's vectors to the HANDLE's surface order.
    if (swapped)
    {
        std::swap(deviceFirstForces, deviceSecondForces);
        std::swap(deviceFirstVelocities, deviceSecondVelocities);
    }

    auto& ws = contactForceWorkspace();
    cudaError_t err = ws.ensure();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Contact-force workspace alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    cudaMemsetAsync(ws.activeCount, 0, sizeof(std::uint32_t));

    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = 256;
    accumulateContactPenaltyForcesKernel<<<blocks, threads>>>(
        handle->contacts, handle->countDevice, handle->capacity,
        handle->firstIndices, handle->secondIndices,
        static_cast<float*>(deviceFirstForces),
        static_cast<float*>(deviceSecondForces),
        static_cast<const float*>(deviceFirstVelocities),
        static_cast<const float*>(deviceSecondVelocities),
        config.stiffness, config.damping, config.contactDistance,
        ws.activeCount);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Contact penalty force launch: ") + cudaGetErrorString(err);
        return false;
    }

    // Stats are opt-in: reading them costs a sync, so only do it when asked.
    if (stats != nullptr)
    {
        std::uint32_t hostContactCount = 0;
        cudaMemcpy(&hostContactCount, handle->countDevice, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(ws.activeCountHostPinned, ws.activeCount, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        stats->contactCount = std::min(hostContactCount, handle->capacity);
        stats->activeContactCount = *ws.activeCountHostPinned;
    }

    diagnostic.clear();
    return true;
}

bool accumulateContactPenaltyDForces(
    const ContactPenaltyConfig& config,
    const std::uint64_t firstSurfaceId,
    const std::uint64_t secondSurfaceId,
    const float kFactor,
    void* deviceFirstDForces,
    void* deviceSecondDForces,
    const void* deviceFirstDx,
    const void* deviceSecondDx,
    std::string& diagnostic)
{
    if (deviceFirstDForces == nullptr || deviceSecondDForces == nullptr ||
        deviceFirstDx == nullptr || deviceSecondDx == nullptr)
    {
        diagnostic = "Null device dforce/dx vector.";
        return false;
    }

    const RecordedContactHandle* handle = nullptr;
    bool swapped = false;
    if (!resolveContactHandle(firstSurfaceId, secondSurfaceId, handle, swapped, diagnostic)) return false;

    if (swapped)
    {
        std::swap(deviceFirstDForces, deviceSecondDForces);
        std::swap(deviceFirstDx, deviceSecondDx);
    }

    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = 256;
    accumulateContactPenaltyDForcesKernel<<<blocks, threads>>>(
        handle->contacts, handle->countDevice, handle->capacity,
        handle->firstIndices, handle->secondIndices,
        static_cast<float*>(deviceFirstDForces),
        static_cast<float*>(deviceSecondDForces),
        static_cast<const float*>(deviceFirstDx),
        static_cast<const float*>(deviceSecondDx),
        config.stiffness, config.contactDistance, kFactor);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Contact penalty dforce launch: ") + cudaGetErrorString(err);
        return false;
    }

    diagnostic.clear();
    return true;
}

// ----------------------------------------------------------------------------
// Gate 1 + Gate 2 — self-validation.
//
// Gate 1 asks: does the GPU kernel compute the same forces a straightforward
// host implementation would, from the identical contacts? The host reference
// below is written independently of the kernel (plain loops, no shared code
// beyond the weight convention it is testing) so agreement is evidence, not
// tautology.
//
// Gate 2 asks: is every force matched by an equal and opposite one? Summing all
// force vectors over BOTH bodies must give ~0 regardless of the penalty law —
// it tests only the scatter, which is where sign and index errors live.
// ----------------------------------------------------------------------------
bool validateContactPenaltyForces(
    const ContactPenaltyConfig& config,
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    ContactForceValidation* validation,
    std::string& diagnostic)
{
    if (validation != nullptr) *validation = ContactForceValidation {};

    const RecordedContactHandle* handle = nullptr;
    bool swapped = false;
    if (!resolveContactHandle(firstSurface.surfaceId, secondSurface.surfaceId, handle, swapped, diagnostic))
    {
        return false;
    }
    if (swapped)
    {
        diagnostic = "validateContactPenaltyForces expects the surfaces in the recorded order.";
        return false;
    }

    const std::size_t firstVertexCount = firstSurface.vertexCount;
    const std::size_t secondVertexCount = secondSurface.vertexCount;

    float* deviceFirstForces = nullptr;
    float* deviceSecondForces = nullptr;
    std::uint32_t* deviceActive = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&deviceFirstForces), firstVertexCount * 3u * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&deviceSecondForces), secondVertexCount * 3u * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&deviceActive), sizeof(std::uint32_t));
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstForces); cudaFree(deviceSecondForces); cudaFree(deviceActive);
        diagnostic = std::string("validation alloc failed: ") + cudaGetErrorString(err);
        return false;
    }
    cudaMemset(deviceFirstForces, 0, firstVertexCount * 3u * sizeof(float));
    cudaMemset(deviceSecondForces, 0, secondVertexCount * 3u * sizeof(float));
    cudaMemset(deviceActive, 0, sizeof(std::uint32_t));

    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = 256;
    accumulateContactPenaltyForcesKernel<<<blocks, threads>>>(
        handle->contacts, handle->countDevice, handle->capacity,
        handle->firstIndices, handle->secondIndices,
        deviceFirstForces, deviceSecondForces,
        nullptr, nullptr,                       // damping off: velocities are not part of this gate
        config.stiffness, 0.0f, config.contactDistance,
        deviceActive);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(deviceFirstForces); cudaFree(deviceSecondForces); cudaFree(deviceActive);
        diagnostic = std::string("validation kernel: ") + cudaGetErrorString(err);
        return false;
    }

    // Pull back what the GPU produced, plus the contacts it produced it from.
    std::uint32_t hostContactCount = 0;
    std::uint32_t hostActiveCount = 0;
    cudaMemcpy(&hostContactCount, handle->countDevice, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostActiveCount, deviceActive, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    hostContactCount = std::min(hostContactCount, handle->capacity);

    std::vector<float> gpuFirst(firstVertexCount * 3u, 0.0f);
    std::vector<float> gpuSecond(secondVertexCount * 3u, 0.0f);
    cudaMemcpy(gpuFirst.data(), deviceFirstForces, gpuFirst.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuSecond.data(), deviceSecondForces, gpuSecond.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<DeviceProximityContact> hostContacts(hostContactCount);
    if (hostContactCount > 0)
    {
        cudaMemcpy(hostContacts.data(), handle->contacts,
                   hostContactCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
    }
    std::vector<std::uint32_t> hostFirstIndices(firstSurface.triangleCount * 3u);
    std::vector<std::uint32_t> hostSecondIndices(secondSurface.triangleCount * 3u);
    cudaMemcpy(hostFirstIndices.data(), handle->firstIndices, hostFirstIndices.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSecondIndices.data(), handle->secondIndices, hostSecondIndices.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceFirstForces);
    cudaFree(deviceSecondForces);
    cudaFree(deviceActive);

    // ---- independent host reference ----
    auto hostWeights = [](const std::uint8_t kind, const std::uint8_t localIndex,
                          const float* bary, const bool isFirstSide, float w[3]) {
        w[0] = w[1] = w[2] = 0.0f;
        const bool sideIsVertex = (kind == 0u && isFirstSide) || (kind == 1u && !isFirstSide);
        if (sideIsVertex)
        {
            w[localIndex % 3u] = 1.0f;
        }
        else if (kind == 2u)
        {
            const unsigned a = localIndex % 3u;
            w[a] = bary[0];
            w[(a + 1u) % 3u] = bary[1];
        }
        else
        {
            w[0] = bary[0]; w[1] = bary[1]; w[2] = bary[2];
        }
    };

    std::vector<float> refFirst(firstVertexCount * 3u, 0.0f);
    std::vector<float> refSecond(secondVertexCount * 3u, 0.0f);
    for (const auto& c : hostContacts)
    {
        const float depth = config.contactDistance - c.signedDistance;
        if (depth <= 0.0f) continue;
        const float magnitude = config.stiffness * depth;
        if (magnitude <= 0.0f) continue;

        float wa[3], wb[3];
        hostWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wa);
        hostWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wb);
        const float fx = c.normal.x * magnitude;
        const float fy = c.normal.y * magnitude;
        const float fz = c.normal.z * magnitude;

        for (int k = 0; k < 3; ++k)
        {
            if (wa[k] != 0.0f)
            {
                const std::uint32_t v = hostFirstIndices[3u * c.firstPrimitiveIndex + k];
                refFirst[3u * v + 0u] -= fx * wa[k];
                refFirst[3u * v + 1u] -= fy * wa[k];
                refFirst[3u * v + 2u] -= fz * wa[k];
            }
            if (wb[k] != 0.0f)
            {
                const std::uint32_t v = hostSecondIndices[3u * c.secondPrimitiveIndex + k];
                refSecond[3u * v + 0u] += fx * wb[k];
                refSecond[3u * v + 1u] += fy * wb[k];
                refSecond[3u * v + 2u] += fz * wb[k];
            }
        }
    }

    // ---- Gate 1b: reconstruct each contact point from the decoded weights ----
    // Independent of the force math entirely: if the weights address the right
    // vertices with the right coefficients, then sum(w_i * vertexPosition_i)
    // must reproduce the contact point that the collision kernel computed via
    // closest-feature math. A wrong convention (wrong vertex, swapped edge
    // endpoints, face-vs-vertex confusion) breaks this immediately, whereas the
    // force comparison alone would not.
    double maxPointError = 0.0;
    double maxWeightSumError = 0.0;
    bool pointCheckRan = false;
    if (firstSurface.positions != nullptr && secondSurface.positions != nullptr)
    {
        pointCheckRan = true;
        const auto reconstruct = [](const BackendTriangleVertex* positions,
                                    const std::vector<std::uint32_t>& indices,
                                    const std::uint32_t triangleId,
                                    const float w[3], float out[3]) {
            out[0] = out[1] = out[2] = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                if (w[k] == 0.0f) continue;
                const std::uint32_t v = indices[3u * triangleId + k];
                out[0] += positions[v].x * w[k];
                out[1] += positions[v].y * w[k];
                out[2] += positions[v].z * w[k];
            }
        };
        for (const auto& c : hostContacts)
        {
            float wa[3], wb[3];
            hostWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wa);
            hostWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wb);

            maxWeightSumError = std::max(maxWeightSumError,
                static_cast<double>(std::fabs((wa[0] + wa[1] + wa[2]) - 1.0f)));
            maxWeightSumError = std::max(maxWeightSumError,
                static_cast<double>(std::fabs((wb[0] + wb[1] + wb[2]) - 1.0f)));

            float pa[3], pb[3];
            reconstruct(firstSurface.positions, hostFirstIndices, c.firstPrimitiveIndex, wa, pa);
            reconstruct(secondSurface.positions, hostSecondIndices, c.secondPrimitiveIndex, wb, pb);

            maxPointError = std::max(maxPointError, static_cast<double>(std::sqrt(
                (pa[0] - c.pointOnFirst.x) * (pa[0] - c.pointOnFirst.x) +
                (pa[1] - c.pointOnFirst.y) * (pa[1] - c.pointOnFirst.y) +
                (pa[2] - c.pointOnFirst.z) * (pa[2] - c.pointOnFirst.z))));
            maxPointError = std::max(maxPointError, static_cast<double>(std::sqrt(
                (pb[0] - c.pointOnSecond.x) * (pb[0] - c.pointOnSecond.x) +
                (pb[1] - c.pointOnSecond.y) * (pb[1] - c.pointOnSecond.y) +
                (pb[2] - c.pointOnSecond.z) * (pb[2] - c.pointOnSecond.z))));
        }
    }

    double maxErr = 0.0;
    double maxRef = 0.0;
    double netX = 0.0, netY = 0.0, netZ = 0.0;
    double totalMag = 0.0;
    const auto accumulate = [&](const std::vector<float>& gpu, const std::vector<float>& ref) {
        for (std::size_t i = 0; i + 2 < gpu.size(); i += 3)
        {
            for (int k = 0; k < 3; ++k)
            {
                maxErr = std::max(maxErr, static_cast<double>(std::fabs(gpu[i + k] - ref[i + k])));
                maxRef = std::max(maxRef, static_cast<double>(std::fabs(ref[i + k])));
            }
            netX += gpu[i + 0]; netY += gpu[i + 1]; netZ += gpu[i + 2];
            totalMag += std::sqrt(
                static_cast<double>(gpu[i + 0]) * gpu[i + 0] +
                static_cast<double>(gpu[i + 1]) * gpu[i + 1] +
                static_cast<double>(gpu[i + 2]) * gpu[i + 2]);
        }
    };
    accumulate(gpuFirst, refFirst);
    accumulate(gpuSecond, refSecond);

    if (validation != nullptr)
    {
        validation->contactCount = hostContactCount;
        validation->activeContactCount = hostActiveCount;
        validation->maxAbsErrorVsReference = maxErr;
        validation->maxReferenceMagnitude = maxRef;
        validation->netForceMagnitude = std::sqrt(netX * netX + netY * netY + netZ * netZ);
        validation->totalForceMagnitude = totalMag;
        validation->maxContactPointError = maxPointError;
        validation->maxWeightSumError = maxWeightSumError;
        validation->contactPointCheckRan = pointCheckRan;
    }

    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
