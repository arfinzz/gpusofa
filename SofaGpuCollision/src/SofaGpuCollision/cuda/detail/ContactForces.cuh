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
// Which side is which.
//
// The collision kernel reports an UNSIGNED distance, and a normal from the
// first surface's contact point to the second's. Once a point crosses the other
// surface that normal flips, so a plain penalty pushes the point FURTHER
// through. Each triangle's outward normal (from its winding) is used as a side
// reference instead:
//   * contact normal agrees with the reference -> the surfaces are on their
//     correct sides; direction and distance are used unchanged;
//   * it disagrees -> they overlap; the direction is flipped and the distance
//     becomes a negative separation, so the force grows with depth;
//   * no usable contact normal (touching / crossing) -> the reference itself.
// The reference is the second face's normal for VF (reversed: it points from the
// first surface toward the second), the first face's normal for FV, and their
// difference for EE. Surfaces must be wound with normals pointing outward.
//
// Plain unit-normalisation here, not normalizeOrZero: its 1e-6 length cut-off
// would zero the normal of any triangle a few millimetres across in SI units.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float3 unitOrZero(const float3 v)
{
    const float lenSq = lengthSquared3(v);
    return lenSq > 1.0e-30f ? mul3(v, rsqrtf(lenSq)) : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float3 loadVertex(const float* __restrict__ positions, const std::uint32_t v)
{
    return make_float3(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

__device__ __forceinline__ float3 outwardNormal(
    const float* __restrict__ positions,
    const std::uint32_t* __restrict__ indices,
    const std::uint32_t triangleId)
{
    const float3 p0 = loadVertex(positions, indices[3u * triangleId]);
    const float3 p1 = loadVertex(positions, indices[3u * triangleId + 1u]);
    const float3 p2 = loadVertex(positions, indices[3u * triangleId + 2u]);
    return unitOrZero(cross3(sub3(p1, p0), sub3(p2, p0)));
}

// Direction from the first surface toward the second (the second is pushed
// along it, the first against it) and the separation measured along it.
// Null positions = the old unsigned law.
__device__ __forceinline__ void contactSeparation(
    const DeviceProximityContact& c,
    const float* __restrict__ firstPositions,
    const std::uint32_t* __restrict__ firstIndices,
    const float* __restrict__ secondPositions,
    const std::uint32_t* __restrict__ secondIndices,
    const float contactDistance,
    float3& direction,
    float& separation)
{
    direction = c.normal;
    separation = c.signedDistance;
    if (firstPositions == nullptr || secondPositions == nullptr) return;

    float3 reference;
    if (c.featureKind == 0u)
    {
        reference = mul3(outwardNormal(secondPositions, secondIndices, c.secondPrimitiveIndex), -1.0f);
    }
    else if (c.featureKind == 1u)
    {
        reference = outwardNormal(firstPositions, firstIndices, c.firstPrimitiveIndex);
    }
    else
    {
        reference = unitOrZero(sub3(
            outwardNormal(firstPositions, firstIndices, c.firstPrimitiveIndex),
            outwardNormal(secondPositions, secondIndices, c.secondPrimitiveIndex)));
    }
    if (lengthSquared3(reference) == 0.0f) return;  // degenerate triangles: keep the old law

    const bool normalKnown = lengthSquared3(c.normal) > 0.0f &&
                             c.signedDistance > 1.0e-4f * contactDistance;
    if (!normalKnown)
    {
        direction = reference;
        separation = 0.0f;
    }
    else if (dot3(c.normal, reference) < 0.0f)
    {
        direction = mul3(c.normal, -1.0f);
        separation = -c.signedDistance;
    }
}

// ----------------------------------------------------------------------------
// Penalty response. One thread per contact.
//
//   depth = contactDistance - separation        (> 0 when closer than the margin)
//   vn    = (v_second - v_first) . direction    (< 0 while approaching)
//   F     = max(0, stiffness*depth - damping*vn)
//
// The first side is pushed along -direction and the second along +direction:
// equal and opposite, which Gate 2 checks by reduction. With side awareness
// off (null positions) direction/separation are the contact's own normal and
// unsigned distance, i.e. a proximity penalty that turns on at contactDistance.
// ----------------------------------------------------------------------------
__global__ void accumulateContactPenaltyForcesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const float* __restrict__ firstPositions,    // null = unsigned law
    const float* __restrict__ secondPositions,
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

        float3 n;
        float separation;
        contactSeparation(c, firstPositions, firstIndices, secondPositions, secondIndices,
                          contactDistance, n, separation);
        const float depth = contactDistance - separation;
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
            magnitude -= damping * dot3(sub3(v2, v1), n);
        }
        if (magnitude <= 0.0f) continue;  // separating faster than the spring pulls

        const float3 force = mul3(n, magnitude);

        // First side pushed away from the second, second pushed away from the first.
        scatterVertexForce(firstForces, firstIndices, c.firstPrimitiveIndex, wFirst, mul3(force, -1.0f));
        scatterVertexForce(secondForces, secondIndices, c.secondPrimitiveIndex, wSecond, force);

        if (activeCount != nullptr) atomicAdd(activeCount, 1u);
    }
}

// Stiffness-times-dx for implicit integration: df += kFactor * (df/dx) * dx.
// The force on the second side is n * stiffness * (contactDistance - separation)
// and separation grows with n . (x_second - x_first), so
//   df/dx_second = -stiffness * (n outer n),   df/dx_first = +stiffness * (n outer n).
// Hence df_second = -kFactor * stiffness * (n . dRel) * n and df_first = -df_second,
// the same convention as SOFA's PenalityContactForceField (with the implicit
// solver's negative kFactor this ADDS stiffness to the system). Uses the same
// direction and active set as the force pass. Damping is left to SOFA's own
// b-factor terms; treating it here would double count.
__global__ void accumulateContactPenaltyDForcesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const float* __restrict__ firstPositions,    // null = unsigned law
    const float* __restrict__ secondPositions,
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

        float3 n;
        float separation;
        contactSeparation(c, firstPositions, firstIndices, secondPositions, secondIndices,
                          contactDistance, n, separation);
        if (contactDistance - separation <= 0.0f) continue;

        float wFirst[3];
        float wSecond[3];
        contactVertexWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wFirst);
        contactVertexWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wSecond);

        const float3 dx1 = gatherVertexValue(firstDx, firstIndices, c.firstPrimitiveIndex, wFirst);
        const float3 dx2 = gatherVertexValue(secondDx, secondIndices, c.secondPrimitiveIndex, wSecond);
        const float dn = dot3(sub3(dx2, dx1), n);

        const float3 dfSecond = mul3(n, -kFactor * stiffness * dn);
        scatterVertexForce(firstDForces, firstIndices, c.firstPrimitiveIndex, wFirst, mul3(dfSecond, -1.0f));
        scatterVertexForce(secondDForces, secondIndices, c.secondPrimitiveIndex, wSecond, dfSecond);
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

// False when the narrow phase did not compute this pair in the current
// collision pass: the pair has no contacts now, and the handle's buffer still
// holds the last ones it had. Applying those would keep pushing two bodies that
// have already separated.
bool computedThisPass(const RecordedContactHandle& handle)
{
    return handle.collisionPass == contactHandleRegistry().collisionPass;
}

// For the force entry points: finds this pair's contacts from the current pass.
// Returns false only on a real fault (the registry had to evict handles because
// more pairs are live than it has slots). A pair with no recorded contacts - its
// bodies have not come within reach yet, or not in this pass - is the normal
// no-contact state: `out` stays null, and the caller applies nothing.
bool currentContactsFor(
    const std::uint64_t firstSurfaceId,
    const std::uint64_t secondSurfaceId,
    const RecordedContactHandle*& out,
    bool& outSwapped,
    std::string& diagnostic)
{
    out = nullptr;
    const RecordedContactHandle* handle = nullptr;
    if (!resolveContactHandle(firstSurfaceId, secondSurfaceId, handle, outSwapped, diagnostic))
    {
        if (contactHandleRegistry().evictions > 0) return false;
        diagnostic.clear();
        return true;
    }
    if (computedThisPass(*handle)) out = handle;
    diagnostic.clear();
    return true;
}

} // namespace


namespace SofaGpuCollision::backend
{

void beginContactFrame()
{
    ++contactHandleRegistry().collisionPass;
}

bool accumulateContactPenaltyForces(
    const ContactPenaltyConfig& config,
    const std::uint64_t firstSurfaceId,
    const std::uint64_t secondSurfaceId,
    void* deviceFirstForces,
    void* deviceSecondForces,
    const void* deviceFirstVelocities,
    const void* deviceSecondVelocities,
    const void* deviceFirstPositions,
    const void* deviceSecondPositions,
    ContactPenaltyStats* stats,
    std::string& diagnostic)
{
    if (stats != nullptr) *stats = ContactPenaltyStats {};
    if (deviceFirstForces == nullptr || deviceSecondForces == nullptr)
    {
        diagnostic = "Null device force vector.";
        return false;
    }
    if (config.useSurfaceNormals && (deviceFirstPositions == nullptr || deviceSecondPositions == nullptr))
    {
        diagnostic = "useSurfaceNormals needs both surfaces' device positions.";
        return false;
    }
    if (!config.useSurfaceNormals)
    {
        deviceFirstPositions = nullptr;
        deviceSecondPositions = nullptr;
    }

    const RecordedContactHandle* handle = nullptr;
    bool swapped = false;
    if (!currentContactsFor(firstSurfaceId, secondSurfaceId, handle, swapped, diagnostic)) return false;
    if (handle == nullptr) return true;   // no contacts this pass: no force (stats stay 0)

    // Bind the caller's vectors to the HANDLE's surface order.
    if (swapped)
    {
        std::swap(deviceFirstForces, deviceSecondForces);
        std::swap(deviceFirstVelocities, deviceSecondVelocities);
        std::swap(deviceFirstPositions, deviceSecondPositions);
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
        static_cast<const float*>(deviceFirstPositions),
        static_cast<const float*>(deviceSecondPositions),
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
    const void* deviceFirstPositions,
    const void* deviceSecondPositions,
    std::string& diagnostic)
{
    if (deviceFirstDForces == nullptr || deviceSecondDForces == nullptr ||
        deviceFirstDx == nullptr || deviceSecondDx == nullptr)
    {
        diagnostic = "Null device dforce/dx vector.";
        return false;
    }
    if (config.useSurfaceNormals && (deviceFirstPositions == nullptr || deviceSecondPositions == nullptr))
    {
        diagnostic = "useSurfaceNormals needs both surfaces' device positions.";
        return false;
    }
    if (!config.useSurfaceNormals)
    {
        deviceFirstPositions = nullptr;
        deviceSecondPositions = nullptr;
    }

    const RecordedContactHandle* handle = nullptr;
    bool swapped = false;
    if (!currentContactsFor(firstSurfaceId, secondSurfaceId, handle, swapped, diagnostic)) return false;
    if (handle == nullptr) return true;   // no contacts this pass: no stiffness

    if (swapped)
    {
        std::swap(deviceFirstDForces, deviceSecondDForces);
        std::swap(deviceFirstDx, deviceSecondDx);
        std::swap(deviceFirstPositions, deviceSecondPositions);
    }

    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = 256;
    accumulateContactPenaltyDForcesKernel<<<blocks, threads>>>(
        handle->contacts, handle->countDevice, handle->capacity,
        handle->firstIndices, handle->secondIndices,
        static_cast<const float*>(deviceFirstPositions),
        static_cast<const float*>(deviceSecondPositions),
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
void clearRecordedContactHandles()
{
    ContactHandleRegistry& registry = contactHandleRegistry();
    for (auto& handle : registry.slots)
    {
        handle = RecordedContactHandle {};
    }
    registry.nextSlot = 0;
    registry.evictions = 0;
}

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

    // Side awareness reads positions. The reference below needs them on the host
    // anyway, so upload those rather than trusting any device copy.
    const bool sideAware = config.useSurfaceNormals;
    if (sideAware && (firstSurface.positions == nullptr || secondSurface.positions == nullptr))
    {
        diagnostic = "validateContactPenaltyForces with useSurfaceNormals needs host positions.";
        return false;
    }

    float* deviceFirstForces = nullptr;
    float* deviceSecondForces = nullptr;
    float* deviceFirstPositions = nullptr;
    float* deviceSecondPositions = nullptr;
    std::uint32_t* deviceActive = nullptr;
    const auto freeAll = [&]() {
        cudaFree(deviceFirstForces); cudaFree(deviceSecondForces);
        cudaFree(deviceFirstPositions); cudaFree(deviceSecondPositions);
        cudaFree(deviceActive);
    };
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&deviceFirstForces), firstVertexCount * 3u * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&deviceSecondForces), secondVertexCount * 3u * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&deviceActive), sizeof(std::uint32_t));
    if (err == cudaSuccess && sideAware)
    {
        err = cudaMalloc(reinterpret_cast<void**>(&deviceFirstPositions), firstVertexCount * 3u * sizeof(float));
        if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&deviceSecondPositions), secondVertexCount * 3u * sizeof(float));
        if (err == cudaSuccess) err = cudaMemcpy(deviceFirstPositions, firstSurface.positions,
                                                 firstVertexCount * 3u * sizeof(float), cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy(deviceSecondPositions, secondSurface.positions,
                                                 secondVertexCount * 3u * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess)
    {
        freeAll();
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
        deviceFirstPositions, deviceSecondPositions,
        deviceFirstForces, deviceSecondForces,
        nullptr, nullptr,                       // damping off: velocities are not part of this gate
        config.stiffness, 0.0f, config.contactDistance,
        deviceActive);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        freeAll();
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

    freeAll();

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

    // Side rule, written separately from the device version and in double.
    const auto hostFaceNormal = [](const BackendTriangleVertex* positions,
                                   const std::vector<std::uint32_t>& indices,
                                   const std::uint32_t triangleId, double n[3]) {
        const BackendTriangleVertex& a = positions[indices[3u * triangleId]];
        const BackendTriangleVertex& b = positions[indices[3u * triangleId + 1u]];
        const BackendTriangleVertex& d = positions[indices[3u * triangleId + 2u]];
        const double e1[3] = { double(b.x) - a.x, double(b.y) - a.y, double(b.z) - a.z };
        const double e2[3] = { double(d.x) - a.x, double(d.y) - a.y, double(d.z) - a.z };
        n[0] = e1[1] * e2[2] - e1[2] * e2[1];
        n[1] = e1[2] * e2[0] - e1[0] * e2[2];
        n[2] = e1[0] * e2[1] - e1[1] * e2[0];
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] = len > 1e-15 ? n[k] / len : 0.0;
    };

    std::vector<float> refFirst(firstVertexCount * 3u, 0.0f);
    std::vector<float> refSecond(secondVertexCount * 3u, 0.0f);
    for (const auto& c : hostContacts)
    {
        double dir[3] = { c.normal.x, c.normal.y, c.normal.z };
        double separation = c.signedDistance;
        if (sideAware)
        {
            double ref[3];
            if (c.featureKind == 0u)
            {
                hostFaceNormal(secondSurface.positions, hostSecondIndices, c.secondPrimitiveIndex, ref);
                for (double& r : ref) r = -r;
            }
            else if (c.featureKind == 1u)
            {
                hostFaceNormal(firstSurface.positions, hostFirstIndices, c.firstPrimitiveIndex, ref);
            }
            else
            {
                double na[3], nb[3];
                hostFaceNormal(firstSurface.positions, hostFirstIndices, c.firstPrimitiveIndex, na);
                hostFaceNormal(secondSurface.positions, hostSecondIndices, c.secondPrimitiveIndex, nb);
                for (int k = 0; k < 3; ++k) ref[k] = na[k] - nb[k];
                const double len = std::sqrt(ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2]);
                for (double& r : ref) r = len > 1e-15 ? r / len : 0.0;
            }
            const double refLen = ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2];
            const double normalLen = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            if (refLen > 0.0)
            {
                if (normalLen == 0.0 || c.signedDistance <= 1.0e-4f * config.contactDistance)
                {
                    for (int k = 0; k < 3; ++k) dir[k] = ref[k];
                    separation = 0.0;
                }
                else if (dir[0] * ref[0] + dir[1] * ref[1] + dir[2] * ref[2] < 0.0)
                {
                    for (double& v : dir) v = -v;
                    separation = -separation;
                }
            }
        }

        const double depth = config.contactDistance - separation;
        if (depth <= 0.0) continue;
        const double magnitude = config.stiffness * depth;

        float wa[3], wb[3];
        hostWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, wa);
        hostWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, wb);
        const float fx = static_cast<float>(dir[0] * magnitude);
        const float fy = static_cast<float>(dir[1] * magnitude);
        const float fz = static_cast<float>(dir[2] * magnitude);

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

// ----------------------------------------------------------------------------
// Gates 2b + 2c — side awareness and stiffness sign on hand-built cases.
//
// A large face in the plane y = 0, wound so its normal is +y (its outside), and
// a small triangle whose vertex 0 sits a distance h from it, the rest far away,
// so collision detection finds exactly one contact on that vertex. Run through
// the real dense-grid detection and the real kernels. Whatever the side, the
// vertex must be pushed toward +y; inside, harder than outside.
// ----------------------------------------------------------------------------
bool validateContactSideAwareness(ContactSideValidation* validation, std::string& diagnostic)
{
    ContactSideValidation result;
    constexpr float stiffness = 1000.0f;
    constexpr float contactDistance = 0.05f;
    constexpr float h = 0.02f;       // vertex distance from the face
    constexpr float delta = 0.004f;  // outward nudge for the stiffness check
    result.expectedOutside = static_cast<double>(stiffness) * (contactDistance - h);
    result.expectedInside = static_cast<double>(stiffness) * (contactDistance + h);
    result.expectedStiffnessDf = -static_cast<double>(stiffness) * delta;

    const std::vector<BackendTriangleVertex> bigFace = {
        { -1.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, -1.0f } };  // normal +y
    const auto smallTriangle = [](const float tipY, const float restY) {
        return std::vector<BackendTriangleVertex> {
            { 0.0f, tipY, 0.0f }, { -0.2f, restY, 0.1f }, { 0.2f, restY, 0.1f } };
    };
    const std::vector<std::uint32_t> oneTriangle = { 0u, 1u, 2u };

    DenseGridConfig grid;
    grid.gridMinX = -1.5f; grid.gridMinY = -1.0f; grid.gridMinZ = -1.5f;
    grid.gridMaxX = 1.5f;  grid.gridMaxY = 1.0f;  grid.gridMaxZ = 1.5f;
    grid.gridResolutionX = 12; grid.gridResolutionY = 8; grid.gridResolutionZ = 12;
    grid.contactDistance = contactDistance;
    grid.maxCandidatePairs = 1024;
    grid.copyContactsToHost = false;
    FeatureBasedProximityConfig proximity;
    proximity.contactDistance = contactDistance;
    proximity.keepContactsOnDevice = true;
    proximity.readContactCounter = true;
    proximity.maxContacts = 64;

    // Detect one contact between (first, second), then evaluate the force kernel,
    // or - with dxFirstVertex0 set - the stiffness kernel, and return the result
    // on vertex 0 of the requested side.
    std::uint64_t nextSurfaceId = 0x51DE0000ull;
    const auto runCase = [&](const std::vector<BackendTriangleVertex>& firstPositions,
                             const std::vector<BackendTriangleVertex>& secondPositions,
                             const bool sideAware,
                             const bool readSecondSide,
                             const float* dxFirstVertex0,
                             float out[3]) -> bool {
        TriangleIndexedSurface first;
        first.positions = firstPositions.data();
        first.vertexCount = 3;
        first.triangleIndices = oneTriangle.data();
        first.triangleCount = 1;
        first.surfaceId = ++nextSurfaceId;
        TriangleIndexedSurface second = first;
        second.positions = secondPositions.data();
        second.surfaceId = ++nextSurfaceId;

        std::vector<ProximityContact> unused;
        FeatureBasedProximityStats stats;
        if (!computeFeatureBasedProximityContacts(first, second, grid, proximity, unused, &stats, diagnostic))
        {
            return false;
        }
        if (stats.emittedContactCount != 1u)
        {
            diagnostic = "side check expected exactly 1 contact, got " + std::to_string(stats.emittedContactCount);
            return false;
        }
        bool swapped = false;
        const RecordedContactHandle* handle = findContactHandle(first.surfaceId, second.surfaceId, swapped);
        if (handle == nullptr || swapped)
        {
            diagnostic = "side check: no contact handle recorded for the case";
            return false;
        }

        float* buffers[6] = {};  // positions x2, outputs x2, dx x2
        const std::size_t bytes = 9u * sizeof(float);
        cudaError_t err = cudaSuccess;
        for (float*& b : buffers)
        {
            if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&b), bytes);
            if (err == cudaSuccess) err = cudaMemset(b, 0, bytes);
        }
        if (err == cudaSuccess) err = cudaMemcpy(buffers[0], firstPositions.data(), bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy(buffers[1], secondPositions.data(), bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess && dxFirstVertex0 != nullptr)
        {
            err = cudaMemcpy(buffers[4], dxFirstVertex0, 3u * sizeof(float), cudaMemcpyHostToDevice);
        }
        const float* p1 = sideAware ? buffers[0] : nullptr;
        const float* p2 = sideAware ? buffers[1] : nullptr;
        if (err == cudaSuccess)
        {
            if (dxFirstVertex0 == nullptr)
            {
                accumulateContactPenaltyForcesKernel<<<1, 32>>>(
                    handle->contacts, handle->countDevice, handle->capacity,
                    handle->firstIndices, handle->secondIndices, p1, p2,
                    buffers[2], buffers[3], nullptr, nullptr,
                    stiffness, 0.0f, contactDistance, nullptr);
            }
            else
            {
                accumulateContactPenaltyDForcesKernel<<<1, 32>>>(
                    handle->contacts, handle->countDevice, handle->capacity,
                    handle->firstIndices, handle->secondIndices, p1, p2,
                    buffers[2], buffers[3], buffers[4], buffers[5],
                    stiffness, contactDistance, 1.0f);
            }
            err = cudaDeviceSynchronize();
        }
        if (err == cudaSuccess)
        {
            err = cudaMemcpy(out, readSecondSide ? buffers[3] : buffers[2], 3u * sizeof(float), cudaMemcpyDeviceToHost);
        }
        for (float* b : buffers) cudaFree(b);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("side check CUDA error: ") + cudaGetErrorString(err);
            return false;
        }
        return true;
    };

    const auto relativeError = [](const double measured, const double expected) {
        return std::fabs(measured - expected) / std::fabs(expected);
    };
    const auto pushedAlongY = [](const float f[3], const double expectedY) {
        return std::fabs(f[0]) < 1e-3 * std::fabs(expectedY) && std::fabs(f[2]) < 1e-3 * std::fabs(expectedY);
    };
    float f[3];

    // 1. Vertex just OUTSIDE the face (first = small triangle above, second = face).
    if (!runCase(smallTriangle(h, 0.5f), bigFace, true, false, nullptr, f)) return false;
    ++result.casesRun;
    result.vertexOutsideForceY = f[1];
    result.maxRelativeError = std::max(result.maxRelativeError, relativeError(f[1], result.expectedOutside));
    if (f[1] > 0.0f && relativeError(f[1], result.expectedOutside) < 1e-3 && pushedAlongY(f, result.expectedOutside))
        ++result.casesPassed;

    // 2. Vertex just INSIDE the face: still pushed toward +y, and harder.
    if (!runCase(smallTriangle(-h, -0.5f), bigFace, true, false, nullptr, f)) return false;
    ++result.casesRun;
    result.vertexInsideForceY = f[1];
    result.maxRelativeError = std::max(result.maxRelativeError, relativeError(f[1], result.expectedInside));
    if (f[1] > 0.0f && relativeError(f[1], result.expectedInside) < 1e-3 && pushedAlongY(f, result.expectedInside))
        ++result.casesPassed;

    // 3. A tool tip that has sunk into the face (face first, tip second: FV).
    if (!runCase(bigFace, smallTriangle(-h, 0.5f), true, true, nullptr, f)) return false;
    ++result.casesRun;
    result.toolTipInsideForceY = f[1];
    result.maxRelativeError = std::max(result.maxRelativeError, relativeError(f[1], result.expectedInside));
    if (f[1] > 0.0f && relativeError(f[1], result.expectedInside) < 1e-3 && pushedAlongY(f, result.expectedInside))
        ++result.casesPassed;

    // 4. Case 2 under the old unsigned law must push the vertex INWARD - the
    //    failure the side rule fixes. If it didn't, cases 1-3 would prove nothing.
    if (!runCase(smallTriangle(-h, -0.5f), bigFace, false, false, nullptr, f)) return false;
    ++result.casesRun;
    result.unsignedInsideForceY = f[1];
    if (f[1] < 0.0f) ++result.casesPassed;

    // 5. Stiffness sign: nudge the inside vertex OUTWARD by delta; its push must
    //    drop by stiffness * delta (kFactor = 1, i.e. df = (df/dx) dx).
    const float nudge[3] = { 0.0f, delta, 0.0f };
    if (!runCase(smallTriangle(-h, -0.5f), bigFace, true, false, nudge, f)) return false;
    ++result.casesRun;
    result.stiffnessDfY = f[1];
    result.maxRelativeError = std::max(result.maxRelativeError, relativeError(f[1], result.expectedStiffnessDf));
    if (relativeError(f[1], result.expectedStiffnessDf) < 1e-3 && pushedAlongY(f, result.expectedStiffnessDf))
        ++result.casesPassed;

    // 6. Gate 2d, through the public entry point a force field uses: the outside
    //    case gives its force in the pass that computed it, and nothing once a
    //    new collision pass has begun without recomputing the pair (as when the
    //    broad phase drops a pair whose bodies have moved apart).
    {
        const std::vector<BackendTriangleVertex> firstPositions = smallTriangle(h, 0.5f);
        TriangleIndexedSurface first;
        first.positions = firstPositions.data();
        first.vertexCount = 3;
        first.triangleIndices = oneTriangle.data();
        first.triangleCount = 1;
        first.surfaceId = ++nextSurfaceId;
        TriangleIndexedSurface second = first;
        second.positions = bigFace.data();
        second.surfaceId = ++nextSurfaceId;
        std::vector<ProximityContact> unused;
        FeatureBasedProximityStats stats;
        if (!computeFeatureBasedProximityContacts(first, second, grid, proximity, unused, &stats, diagnostic))
        {
            return false;
        }

        float* buffers[4] = {};  // positions first/second, forces first/second
        const std::size_t bytes = 9u * sizeof(float);
        cudaError_t err = cudaSuccess;
        for (float*& b : buffers)
        {
            if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&b), bytes);
            if (err == cudaSuccess) err = cudaMemset(b, 0, bytes);
        }
        if (err == cudaSuccess) err = cudaMemcpy(buffers[0], firstPositions.data(), bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy(buffers[1], bigFace.data(), bytes, cudaMemcpyHostToDevice);

        ContactPenaltyConfig config;
        config.stiffness = stiffness;
        config.contactDistance = contactDistance;
        config.useSurfaceNormals = true;
        const auto forceOnVertex0 = [&](double& outY) -> bool {
            if (err == cudaSuccess) err = cudaMemset(buffers[2], 0, bytes);
            if (err == cudaSuccess) err = cudaMemset(buffers[3], 0, bytes);
            if (err != cudaSuccess) return false;
            if (!accumulateContactPenaltyForces(config, first.surfaceId, second.surfaceId, buffers[2], buffers[3],
                                                nullptr, nullptr, buffers[0], buffers[1], nullptr, diagnostic))
            {
                return false;
            }
            float out[3] = {};
            err = cudaDeviceSynchronize();
            if (err == cudaSuccess) err = cudaMemcpy(out, buffers[2], sizeof(out), cudaMemcpyDeviceToHost);
            outY = out[1];
            return err == cudaSuccess;
        };
        bool ran = forceOnVertex0(result.currentPassForceY);
        if (ran)
        {
            beginContactFrame();   // a new pass that does not recompute this pair
            ran = forceOnVertex0(result.stalePassForceY);
        }
        for (float* b : buffers) cudaFree(b);
        if (!ran)
        {
            if (err != cudaSuccess) diagnostic = std::string("stale-pair check CUDA error: ") + cudaGetErrorString(err);
            return false;
        }
        ++result.casesRun;
        if (relativeError(result.currentPassForceY, result.expectedOutside) < 1e-3 && result.stalePassForceY == 0.0)
            ++result.casesPassed;
    }

    if (validation != nullptr) *validation = result;
    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
