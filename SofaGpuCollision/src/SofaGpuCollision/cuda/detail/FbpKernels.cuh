// FbpKernels.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included only by cuda/GpuCollisionBackend.cu, in dependency order; not
// independently compilable (kernels must be visible to their launch sites
// without -rdc/device-linking, and one TU keeps codegen identical to the
// pre-split monolith). Split from the monolithic backend on 2026-07-03.
// Feature-based proximity: contact struct, Ericson closest-feature math,
// the FBP tri-tri and vertex-triangle kernels, and their host drivers.

namespace
{

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

// Per-pair closest-feature test: cheap exact AABB pre-reject, then 3 VF + 3 FV
// + 9 EE keeping the closest feature under the threshold. Fills every contact
// field EXCEPT the primitive indices (the caller owns those). Returns false
// when the pair is rejected. Extracted verbatim from featureBasedProximityKernel
// (2026-07-12) so the big-cell fused kernel (way 6) can run the identical math
// against shared-memory staged vertices — same test order, bit-identical values.
__device__ __forceinline__ bool fbpComputeClosestFeatureContact(
    const float3 aV[3],
    const float3 bV[3],
    const float distThreshSq,
    const bool computeBarycentrics,
    DeviceProximityContact& c)
{
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
            return false;
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
        return false;
    }

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
    return true;
}

// Append a finished contact to the output array + per-class tallies. Shared by
// the FBP kernel and the way-6 fused kernel (same counter-then-bounds order).
__device__ __forceinline__ void fbpEmitContact(
    const DeviceProximityContact& c,
    DeviceProximityContact* __restrict__ contacts,
    std::uint32_t* __restrict__ contactCount,
    std::uint32_t* __restrict__ overflowCount,
    std::uint32_t* __restrict__ vfCount,
    std::uint32_t* __restrict__ fvCount,
    std::uint32_t* __restrict__ eeCount,
    const std::uint32_t maxContacts)
{
    const std::uint32_t outIdx = atomicAdd(contactCount, 1u);
    if (outIdx >= maxContacts)
    {
        atomicAdd(overflowCount, 1u);
        return;
    }
    contacts[outIdx] = c;
    if (c.featureKind == 0u)      atomicAdd(vfCount, 1u);
    else if (c.featureKind == 1u) atomicAdd(fvCount, 1u);
    else                          atomicAdd(eeCount, 1u);
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

    DeviceProximityContact c;
    if (!fbpComputeClosestFeatureContact(aV, bV, distThreshSq, computeBarycentrics, c))
    {
        continue;
    }
    c.firstPrimitiveIndex = aIdx;
    c.secondPrimitiveIndex = bIdx;
    fbpEmitContact(c, contacts, contactCount, overflowCount, vfCount, fvCount, eeCount, maxContacts);
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

// Download device contacts and convert them to the public ProximityContact.
// Shared by the FBP / v-t / hash / simple-hash / sorted-grid drivers (was five
// copy-pasted blocks). Matches the original blocks exactly, including ignoring
// the memcpy return (the counters were already read back and validated).
void downloadDeviceProximityContacts(
    const DeviceProximityContact* deviceContacts,
    const std::uint32_t hostContactCount,
    const std::uint32_t maxContacts,
    std::vector<SofaGpuCollision::backend::ProximityContact>& contacts,
    SofaGpuCollision::backend::BackendExecutionStats* executionStats)
{
    using SofaGpuCollision::backend::ProximityContact;
    using SofaGpuCollision::backend::ProximityFeatureKind;
    const std::uint32_t copyCount = std::min(hostContactCount, maxContacts);
    if (copyCount == 0)
    {
        return;
    }
    std::vector<DeviceProximityContact> hostBuffer(copyCount);
    cudaMemcpy(hostBuffer.data(), deviceContacts, copyCount * sizeof(DeviceProximityContact), cudaMemcpyDeviceToHost);
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
        out.pointOnFirst  = BackendTriangleVertex { c.pointOnFirst.x,  c.pointOnFirst.y,  c.pointOnFirst.z };
        out.pointOnSecond = BackendTriangleVertex { c.pointOnSecond.x, c.pointOnSecond.y, c.pointOnSecond.z };
        out.normal        = BackendTriangleVertex { c.normal.x,        c.normal.y,        c.normal.z };
        out.signedDistance = c.signedDistance;
        contacts.push_back(out);
    }
}

} // namespace


namespace SofaGpuCollision::backend
{

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
        downloadDeviceProximityContacts(deviceProximityContacts, hostContactCount, proximityConfig.maxContacts, contacts, executionStats);
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
        downloadDeviceProximityContacts(deviceProximityContacts, hostContactCount, proximityConfig.maxContacts, contacts, executionStats);
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


} // namespace SofaGpuCollision::backend
