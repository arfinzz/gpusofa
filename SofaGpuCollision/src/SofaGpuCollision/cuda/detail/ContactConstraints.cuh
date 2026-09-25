// ContactConstraints.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included last: it reads the device contact registry (FbpKernels.cuh) and reuses
// the contact geometry of ContactForces.cuh (contactVertexWeights,
// contactSeparation, computedThisPass).
//
// GPU Lagrangian contact constraints with Coulomb friction: the GPU counterpart of
// SOFA's constraint step (FreeMotionAnimationLoop + BlockGaussSeidelConstraintSolver
// + UnilateralLagrangianConstraint + LinearSolverConstraintCorrection) for one
// deformable body (body 1) touching one rigid body (body 2). Each stage follows
// SOFA's CPU code, so the two can be compared on identical inputs:
//
//   rows     BaseContactLagrangianConstraint::addContact / buildConstraintMatrix:
//            per contact a normal row and two tangent rows (SOFA's t/s basis);
//            body 1 gets -u on the contact point, body 2 gets +u; the contact
//            points are mapped onto body 1's triangle vertices (barycentric) and
//            body 2's rigid DOFs ([u ; r x u]); DOFs held by projective
//            constraints are removed (MechanicalProjectJacobianMatrixVisitor).
//   dfree    BaseContactLagrangianConstraint::getPositionViolation, including its
//            tangential interpolation to the moment of impact. Free points as
//            SOFA has them: body 1 from its free positions, body 2's as
//            p + dt (v + omega x r) (FreeMotionAnimationLoop's x_free = x + dt v_free
//            on mapped states).
//   W        LinearSolverConstraintCorrection::addComplianceInConstraintSpace:
//            W = f1 J1 A1^-1 J1^T + f2 J2 A2^-1 J2^T (f = dt for implicit Euler).
//            A1 is factorized densely on the GPU (cuSOLVER Cholesky, float); only
//            the block of A1^-1 on the touched vertices is formed (one triangular
//            solve and one product). A2 (6x6) is inverted on the host.
//   solve    BlockGaussSeidelConstraintSolver::gaussSeidel_increment with
//            UnilateralConstraintResolutionWithFriction::resolution (or
//            UnilateralConstraintResolution without friction), same error
//            measure, tolerance scaling, SOR and stopping rule.
//   correct  LinearSolverConstraintCorrection::computeMotionCorrection:
//            A dv = J^T lambda for each body.
//
// Contact selection: the narrow phase reports vertex-face, face-vertex and
// edge-edge contacts. For constraints the default keeps one vertex-face contact
// per vertex (the closest face) on each side and drops edge-edge ones: they
// describe the same surfaces many times over, which only makes W larger and
// rank-deficient. The kept contacts are sorted by a key of their features, so
// their order (and with it the Gauss-Seidel sweep order) does not depend on the
// order the narrow phase happened to emit them in.

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace
{

constexpr int kConstraintSolveThreads = 256;
constexpr int kContactGeometryValues = 21;   // P, Q, Pfree, Qfree, n, t, s
constexpr int kConstraintSweepsPerLaunch = 64;

template <class T>
cudaError_t ensureDeviceBuffer(T*& ptr, std::size_t& capacity, const std::size_t needed)
{
    if (ptr != nullptr && needed <= capacity) return cudaSuccess;
    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
        capacity = 0;
    }
    const std::size_t count = std::max<std::size_t>(needed, 16);
    void* raw = nullptr;
    const cudaError_t err = cudaMalloc(&raw, count * sizeof(T));
    if (err != cudaSuccess) return err;
    ptr = static_cast<T*>(raw);
    capacity = count;
    return cudaSuccess;
}

template <class T>
__global__ void constraintFillKernel(T* data, const std::size_t count, const T value)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        data[i] = value;
    }
}

template <class T>
void launchFill(T* data, const std::size_t count, const T value)
{
    if (count == 0) return;
    const std::size_t blocks = std::min<std::size_t>((count + 255) / 256, 4096);
    constraintFillKernel<T><<<static_cast<unsigned>(blocks), 256>>>(data, count, value);
}

// Double atomicAdd for pre-sm_60 targets (the backend compiles for sm_52).
__device__ __forceinline__ void atomicAddDouble(double* address, const double value)
{
    unsigned long long* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *raw;
    unsigned long long assumed;
    do
    {
        assumed = old;
        old = atomicCAS(raw, assumed,
            static_cast<unsigned long long>(__double_as_longlong(value + __longlong_as_double(static_cast<long long>(assumed)))));
    } while (assumed != old);
}

// Body 2 over the free step, as SOFA moves its mapped surface points:
// p_free = p + linear + angular x (p - centre), (linear, angular) = dt * v_free.
struct ConstraintRigidMotion
{
    float3 center;          // at detection: the lever arms of the rows
    float3 freeLinear;
    float3 freeAngular;
    float dofMask[6];       // 0 = DOF held by a projective constraint
};

// The vertex a vertex-face / face-vertex contact is anchored on, whether it
// belongs to the deformable body, and a deterministic tie-break (the face it meets).
__device__ __forceinline__ void constraintAnchorVertex(
    const DeviceProximityContact& c,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    std::uint32_t& vertex,
    bool& onDeformable,
    std::uint32_t& tieKey)
{
    const bool vertexOnFirst = (c.featureKind == 0u);
    const std::uint32_t triangle = vertexOnFirst ? c.firstPrimitiveIndex : c.secondPrimitiveIndex;
    const std::uint32_t local = (vertexOnFirst ? c.firstFeatureLocalIndex : c.secondFeatureLocalIndex) % 3u;
    const std::uint32_t* indices = vertexOnFirst ? firstIndices : secondIndices;
    vertex = indices[3u * triangle + local];
    onDeformable = (vertexOnFirst != swapped);   // the handle's first surface is body 1 unless swapped
    tieKey = vertexOnFirst ? c.secondPrimitiveIndex : c.firstPrimitiveIndex;
}

__device__ __forceinline__ unsigned long long constraintAnchorPriority(const DeviceProximityContact& c, const std::uint32_t tieKey)
{
    return (static_cast<unsigned long long>(__float_as_uint(fmaxf(c.signedDistance, 0.0f))) << 32) |
           static_cast<unsigned long long>(tieKey);
}

// ---- Vertex normal cones (SOFA's LocalMinDistance::testValidity) ------------
// A vertex-face contact is anchored on a vertex, and its direction (from that
// vertex to the other surface) must leave the vertex's own surface: lie in the
// cone of its surrounding faces' normals. A vertex of a flat floor accepts only
// contacts along the floor's normal; without this test, a floor vertex in front
// of a sliding block made a contact with the block's leading edge at 58 degrees,
// which the step's linearised constraint read as a collision and stopped with a
// large impulse. The cone is kept as the mean unit normal N of the faces around
// the vertex and the cosine of the widest of them from N.

__global__ void constraintVertexNormalsKernel(
    const std::uint32_t* __restrict__ indices, const std::uint32_t triangleCount, const float* __restrict__ positions,
    float* __restrict__ normals)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < triangleCount; t += stride)
    {
        const std::uint32_t a = indices[3u * t], b = indices[3u * t + 1u], c = indices[3u * t + 2u];
        const float3 pa = make_float3(positions[3u * a], positions[3u * a + 1u], positions[3u * a + 2u]);
        const float3 pb = make_float3(positions[3u * b], positions[3u * b + 1u], positions[3u * b + 2u]);
        const float3 pc = make_float3(positions[3u * c], positions[3u * c + 1u], positions[3u * c + 2u]);
        const float3 n = cross3(sub3(pb, pa), sub3(pc, pa));
        const float len = sqrtf(dot3(n, n));
        if (len <= 0.0f) continue;
        const float3 u = make_float3(n.x / len, n.y / len, n.z / len);
        const std::uint32_t corners[3] = { a, b, c };
        for (int k = 0; k < 3; ++k)
        {
            atomicAdd(normals + 3u * corners[k], u.x);
            atomicAdd(normals + 3u * corners[k] + 1u, u.y);
            atomicAdd(normals + 3u * corners[k] + 2u, u.z);
        }
    }
}

__global__ void constraintNormalizeKernel(const std::uint32_t count, float* __restrict__ normals)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < count; v += stride)
    {
        const float x = normals[3u * v], y = normals[3u * v + 1u], z = normals[3u * v + 2u];
        const float len = sqrtf(x * x + y * y + z * z);
        const float inv = len > 1e-20f ? 1.0f / len : 0.0f;
        normals[3u * v] = x * inv;
        normals[3u * v + 1u] = y * inv;
        normals[3u * v + 2u] = z * inv;
    }
}

__device__ __forceinline__ void atomicMinFloat(float* address, const float value)
{
    int* raw = reinterpret_cast<int*>(address);
    int old = *raw;
    while (__int_as_float(old) > value)
    {
        const int assumed = old;
        old = atomicCAS(raw, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void constraintVertexConesKernel(
    const std::uint32_t* __restrict__ indices, const std::uint32_t triangleCount, const float* __restrict__ positions,
    const float* __restrict__ normals, float* __restrict__ coneCos)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < triangleCount; t += stride)
    {
        const std::uint32_t a = indices[3u * t], b = indices[3u * t + 1u], c = indices[3u * t + 2u];
        const float3 pa = make_float3(positions[3u * a], positions[3u * a + 1u], positions[3u * a + 2u]);
        const float3 pb = make_float3(positions[3u * b], positions[3u * b + 1u], positions[3u * b + 2u]);
        const float3 pc = make_float3(positions[3u * c], positions[3u * c + 1u], positions[3u * c + 2u]);
        const float3 n = cross3(sub3(pb, pa), sub3(pc, pa));
        const float len = sqrtf(dot3(n, n));
        if (len <= 0.0f) continue;
        const std::uint32_t corners[3] = { a, b, c };
        for (int k = 0; k < 3; ++k)
        {
            const std::uint32_t v = corners[k];
            const float cosine = (n.x * normals[3u * v] + n.y * normals[3u * v + 1u] + n.z * normals[3u * v + 2u]) / len;
            atomicMinFloat(coneCos + v, cosine);
        }
    }
}

// The two surfaces' vertex cones (null normals: no cone test).
struct ConstraintVertexCones
{
    const float* firstNormals { nullptr };
    const float* firstCones { nullptr };
    const float* secondNormals { nullptr };
    const float* secondCones { nullptr };
    float tolerance { 0.05f };   // cosine margin
};

// Whether a vertex-face contact's direction lies in the normal cone of its anchor
// vertex (on the first surface for kind 0, the second for kind 1). Contacts at
// (numerically) zero distance have no direction and are kept.
__device__ __forceinline__ bool constraintInVertexCone(
    const DeviceProximityContact& c, const std::uint32_t vertex, const ConstraintVertexCones& cones)
{
    if (cones.firstNormals == nullptr || c.featureKind == 2u || c.signedDistance <= 1e-7f) return true;
    const bool onFirst = (c.featureKind == 0u);
    const float* normals = onFirst ? cones.firstNormals : cones.secondNormals;
    const float* cone = onFirst ? cones.firstCones : cones.secondCones;
    // The contact normal points from the first surface to the second: from a
    // first-surface vertex outward, or into a second-surface vertex.
    const float sign = onFirst ? 1.0f : -1.0f;
    const float d = sign * (c.normal.x * normals[3u * vertex] + c.normal.y * normals[3u * vertex + 1u] +
                            c.normal.z * normals[3u * vertex + 2u]);
    return d >= cone[vertex] - cones.tolerance;
}

// The anchored vertex's own triangle corner (triangle << 2 | corner): the same
// vertex-face contact is reported once for every triangle around the vertex, so this
// second key keeps exactly one of those identical copies (the lowest, deterministically).
__device__ __forceinline__ std::uint32_t constraintAnchorOwnKey(const DeviceProximityContact& c)
{
    return c.featureKind == 0u ? (c.firstPrimitiveIndex << 2) | (c.firstFeatureLocalIndex & 3u)
                               : (c.secondPrimitiveIndex << 2) | (c.secondFeatureLocalIndex & 3u);
}

// Among the contacts that won a vertex (closest face), the one with the lowest own key.
__global__ void constraintClaimOwnersKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    const ConstraintVertexCones cones,
    const unsigned long long* __restrict__ deformableSlots,
    const unsigned long long* __restrict__ rigidSlots,
    std::uint32_t* __restrict__ deformableOwners,
    std::uint32_t* __restrict__ rigidOwners)
{
    const std::uint32_t total = min(*contactCount, capacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const DeviceProximityContact c = contacts[i];
        if (c.featureKind == 2u) continue;
        std::uint32_t vertex, tieKey;
        bool onDeformable;
        constraintAnchorVertex(c, firstIndices, secondIndices, swapped, vertex, onDeformable, tieKey);
        if (!constraintInVertexCone(c, vertex, cones)) continue;
        const unsigned long long claimed = onDeformable ? deformableSlots[vertex] : rigidSlots[vertex];
        if (claimed != constraintAnchorPriority(c, tieKey)) continue;
        atomicMin(onDeformable ? deformableOwners + vertex : rigidOwners + vertex, constraintAnchorOwnKey(c));
    }
}

__global__ void constraintClaimVerticesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    const ConstraintVertexCones cones,
    unsigned long long* __restrict__ deformableSlots,
    unsigned long long* __restrict__ rigidSlots)
{
    const std::uint32_t total = min(*contactCount, capacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const DeviceProximityContact c = contacts[i];
        if (c.featureKind == 2u) continue;
        std::uint32_t vertex, tieKey;
        bool onDeformable;
        constraintAnchorVertex(c, firstIndices, secondIndices, swapped, vertex, onDeformable, tieKey);
        if (!constraintInVertexCone(c, vertex, cones)) continue;
        atomicMin(onDeformable ? deformableSlots + vertex : rigidSlots + vertex, constraintAnchorPriority(c, tieKey));
    }
}

__global__ void constraintSelectContactsKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ contactCount,
    const std::uint32_t capacity,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    const bool keepAll,
    const ConstraintVertexCones cones,
    const unsigned long long* __restrict__ deformableSlots,
    const unsigned long long* __restrict__ rigidSlots,
    const std::uint32_t* __restrict__ deformableOwners,
    const std::uint32_t* __restrict__ rigidOwners,
    std::uint32_t* __restrict__ selected,
    unsigned long long* __restrict__ sortKeys,
    std::uint32_t* __restrict__ selectedCount)
{
    const std::uint32_t total = min(*contactCount, capacity);
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const DeviceProximityContact c = contacts[i];
        bool keep = keepAll;
        if (!keepAll && c.featureKind != 2u)
        {
            std::uint32_t vertex, tieKey;
            bool onDeformable;
            constraintAnchorVertex(c, firstIndices, secondIndices, swapped, vertex, onDeformable, tieKey);
            const unsigned long long claimed = onDeformable ? deformableSlots[vertex] : rigidSlots[vertex];
            const std::uint32_t owner = onDeformable ? deformableOwners[vertex] : rigidOwners[vertex];
            keep = (claimed == constraintAnchorPriority(c, tieKey)) && owner == constraintAnchorOwnKey(c) &&
                   constraintInVertexCone(c, vertex, cones);
        }
        if (!keep) continue;
        const std::uint32_t slot = atomicAdd(selectedCount, 1u);
        selected[slot] = i;
        sortKeys[slot] = (static_cast<unsigned long long>(c.featureKind & 3u) << 60) |
                         (static_cast<unsigned long long>(c.firstPrimitiveIndex & 0xFFFFFu) << 40) |
                         (static_cast<unsigned long long>(c.secondPrimitiveIndex & 0xFFFFFu) << 20) |
                         (static_cast<unsigned long long>(c.firstFeatureLocalIndex & 3u) << 2) |
                         static_cast<unsigned long long>(c.secondFeatureLocalIndex & 3u);
    }
}

// Body 1's side of a contact: its triangle, index array and vertex weights.
__device__ __forceinline__ void constraintDeformableSide(
    const DeviceProximityContact& c,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    std::uint32_t vertices[3],
    float weights[3])
{
    const std::uint32_t triangle = swapped ? c.secondPrimitiveIndex : c.firstPrimitiveIndex;
    const std::uint32_t* indices = swapped ? secondIndices : firstIndices;
    if (swapped)
        contactVertexWeights(c.featureKind, c.secondFeatureLocalIndex, c.secondBary, false, weights);
    else
        contactVertexWeights(c.featureKind, c.firstFeatureLocalIndex, c.firstBary, true, weights);
    for (int a = 0; a < 3; ++a) vertices[a] = indices[3u * triangle + static_cast<std::uint32_t>(a)];
}

__global__ void constraintFlagVerticesKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ selected,
    const std::uint32_t count,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const bool swapped,
    int* __restrict__ flags)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride)
    {
        const DeviceProximityContact c = contacts[selected[k]];
        std::uint32_t vertices[3];
        float weights[3];
        constraintDeformableSide(c, firstIndices, secondIndices, swapped, vertices, weights);
        for (int a = 0; a < 3; ++a)
        {
            if (weights[a] != 0.0f) flags[vertices[a]] = 1;
        }
    }
}

__global__ void constraintCompactVerticesKernel(
    const int* __restrict__ flags,
    const int* __restrict__ scan,
    const std::uint32_t vertexCount,
    int* __restrict__ vertexCompact,
    int* __restrict__ compactToGlobal)
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < vertexCount; v += stride)
    {
        if (flags[v] != 0)
        {
            vertexCompact[v] = scan[v];
            compactToGlobal[scan[v]] = static_cast<int>(v);
        }
        else
        {
            vertexCompact[v] = -1;
        }
    }
}

// SOFA's tangent basis (BaseContactLagrangianConstraint::addContact). The degenerate
// normal (1,1,1)/sqrt(3), for which SOFA's first guess is parallel to n, gets a
// different first guess instead of SOFA's NaN.
__device__ __forceinline__ void constraintTangentBasis(const float3 n, float3& t, float3& s)
{
    t = make_float3(n.z, n.x, n.y);
    s = cross3(n, t);
    float lenSq = lengthSquared3(s);
    if (lenSq < 1.0e-12f)
    {
        t = make_float3(n.y, -n.x, 0.0f);
        s = cross3(n, t);
        lenSq = lengthSquared3(s);
        if (lenSq < 1.0e-12f)
        {
            t = make_float3(0.0f, n.z, -n.y);
            s = cross3(n, t);
            lenSq = lengthSquared3(s);
        }
    }
    s = mul3(s, rsqrtf(lenSq));
    t = cross3(mul3(n, -1.0f), s);
}

__global__ void constraintBuildRowsKernel(
    const DeviceProximityContact* __restrict__ contacts,
    const std::uint32_t* __restrict__ selected,
    const std::uint32_t count,
    const std::uint32_t* __restrict__ firstIndices,
    const std::uint32_t* __restrict__ secondIndices,
    const float* __restrict__ firstPositions,
    const float* __restrict__ secondPositions,
    const bool swapped,
    const float* __restrict__ deformableFree,       // Vec3f per body-1 vertex
    const unsigned char* __restrict__ deformableMask, // per body-1 vertex, bit c = DOF c free; null = all
    const int* __restrict__ vertexCompact,
    const ConstraintRigidMotion rigid,
    const float* __restrict__ rigidInverse,         // 36, row-major A2^-1
    const float contactDistance,
    const int rowsPerContact,
    int* __restrict__ rowVertexCompact,             // rows*3
    int* __restrict__ rowVertexGlobal,              // rows*3
    float* __restrict__ rowDeformable,              // rows*9
    float* __restrict__ rowRigid,                   // rows*6
    float* __restrict__ rowRigidResponse,           // rows*6 = A2^-1 J2^T
    float* __restrict__ dfree,                      // rows
    float* __restrict__ geometry,                   // count*21
    int* __restrict__ contactVertices,              // count*3
    float* __restrict__ contactWeights)             // count*3
{
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride)
    {
        const DeviceProximityContact c = contacts[selected[k]];

        // Direction from the handle's first surface to its second, side-aware.
        float3 direction;
        float separation;
        contactSeparation(c, firstPositions, firstIndices, secondPositions, secondIndices,
                          contactDistance, direction, separation);

        // SOFA's order: body 1 (deformable) point Q, body 2 (rigid) point P,
        // normal from body 1 to body 2.
        const float3 n = unitOrZero(swapped ? mul3(direction, -1.0f) : direction);
        const float3 q = swapped ? c.pointOnSecond : c.pointOnFirst;
        const float3 p = swapped ? c.pointOnFirst : c.pointOnSecond;

        std::uint32_t vertices[3];
        float weights[3];
        constraintDeformableSide(c, firstIndices, secondIndices, swapped, vertices, weights);

        float3 t, s;
        constraintTangentBasis(n, t, s);

        // Free positions of both contact points.
        float3 qFree = make_float3(0.0f, 0.0f, 0.0f);
        for (int a = 0; a < 3; ++a)
        {
            if (weights[a] == 0.0f) continue;
            const float* x = deformableFree + 3u * vertices[a];
            qFree = add3(qFree, mul3(make_float3(x[0], x[1], x[2]), weights[a]));
        }
        const float3 r = sub3(p, rigid.center);                               // lever arm at detection
        const float3 pFree = add3(add3(p, rigid.freeLinear), cross3(rigid.freeAngular, r));

        // BaseContactLagrangianConstraint::getPositionViolation, line for line.
        const float3 ppFree = sub3(pFree, p);
        const float3 qqFree = sub3(qFree, q);
        const float refDist = sqrtf(lengthSquared3(ppFree)) + sqrtf(lengthSquared3(qqFree));
        const float dn = dot3(sub3(pFree, qFree), n) - contactDistance;
        const float delta = dot3(sub3(p, q), n) - contactDistance;
        float dt = 0.0f, ds = 0.0f;
        if (fabsf(delta) < 0.00001f * refDist && fabsf(dn) < 0.00001f * refDist)
        {
            dt = dot3(ppFree, t) - dot3(qqFree, t);
            ds = dot3(ppFree, s) - dot3(qqFree, s);
        }
        else if (fabsf(delta - dn) > 0.001f * delta)
        {
            const float ratio = delta / (delta - dn);
            if (ratio > 0.0f && ratio < 1.0f)
            {
                const float3 pt = add3(mul3(p, 1.0f - ratio), mul3(pFree, ratio));
                const float3 qt = add3(mul3(q, 1.0f - ratio), mul3(qFree, ratio));
                const float3 ptPFree = sub3(pFree, pt);
                const float3 qtQFree = sub3(qFree, qt);
                dt = dot3(ptPFree, t) - dot3(qtQFree, t);
                ds = dot3(ptPFree, s) - dot3(qtQFree, s);
            }
            else if (dn < 0.0f)
            {
                dt = dot3(ppFree, t) - dot3(qqFree, t);
                ds = dot3(ppFree, s) - dot3(qqFree, s);
            }
        }
        else
        {
            dt = dot3(ppFree, t) - dot3(qqFree, t);
            ds = dot3(ppFree, s) - dot3(qqFree, s);
        }

        for (int l = 0; l < rowsPerContact; ++l)
        {
            const float3 u = (l == 0) ? n : ((l == 1) ? t : s);
            const std::size_t row = static_cast<std::size_t>(k) * rowsPerContact + l;
            for (int a = 0; a < 3; ++a)
            {
                const bool used = weights[a] != 0.0f;
                const unsigned bits = (used && deformableMask != nullptr) ? deformableMask[vertices[a]] : 7u;
                rowVertexGlobal[row * 3 + a] = used ? static_cast<int>(vertices[a]) : -1;
                rowVertexCompact[row * 3 + a] = used ? vertexCompact[vertices[a]] : -1;
                rowDeformable[row * 9 + a * 3 + 0] = (bits & 1u) ? -u.x * weights[a] : 0.0f;
                rowDeformable[row * 9 + a * 3 + 1] = (bits & 2u) ? -u.y * weights[a] : 0.0f;
                rowDeformable[row * 9 + a * 3 + 2] = (bits & 4u) ? -u.z * weights[a] : 0.0f;
            }
            const float3 moment = cross3(r, u);
            float j2[6] = { u.x, u.y, u.z, moment.x, moment.y, moment.z };
            for (int e = 0; e < 6; ++e) j2[e] *= rigid.dofMask[e];
            for (int e = 0; e < 6; ++e)
            {
                rowRigid[row * 6 + e] = j2[e];
                float acc = 0.0f;
                for (int f = 0; f < 6; ++f) acc += rigidInverse[e * 6 + f] * j2[f];
                rowRigidResponse[row * 6 + e] = acc;
            }
            dfree[row] = (l == 0) ? dn : ((l == 1) ? dt : ds);
        }

        float* g = geometry + static_cast<std::size_t>(k) * kContactGeometryValues;
        const float3 values[7] = { p, q, pFree, qFree, n, t, s };
        for (int v = 0; v < 7; ++v)
        {
            g[v * 3 + 0] = values[v].x;
            g[v * 3 + 1] = values[v].y;
            g[v * 3 + 2] = values[v].z;
        }
        for (int a = 0; a < 3; ++a)
        {
            contactVertices[k * 3 + a] = weights[a] != 0.0f ? static_cast<int>(vertices[a]) : -1;
            contactWeights[k * 3 + a] = weights[a];
        }
    }
}

// Right-hand side selecting the touched DOFs: E(:, 3k+c) = unit vector on DOF 3*v_k+c.
__global__ void constraintSelectorKernel(
    const int* __restrict__ compactToGlobal,
    const int touched,
    const int n,
    float* __restrict__ selector)          // n x 3*touched, column-major, zeroed
{
    const int stride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < touched; k += stride)
    {
        const int v = compactToGlobal[k];
        for (int c = 0; c < 3; ++c)
        {
            selector[static_cast<std::size_t>(3 * k + c) * n + (3 * v + c)] = 1.0f;
        }
    }
}

// W(i,j) = f1 * J1_i A1^-1 J1_j^T + f2 * J2_i A2^-1 J2_j^T, with A1^-1 known on the
// touched DOFs only (G, column-major, leading dimension ld = 3*touched).
__global__ void constraintAssembleComplianceKernel(
    const int rows,
    const int* __restrict__ rowVertexCompact,
    const float* __restrict__ rowDeformable,
    const float* __restrict__ rowRigid,
    const float* __restrict__ rowRigidResponse,
    const int* __restrict__ rowBody,                // each row's rigid body
    const float* __restrict__ bodyFactors,          // W's rigid factor per body
    const float* __restrict__ compliance,
    const int ld,
    const float f1,
    float* __restrict__ W)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= rows) return;

    float acc1 = 0.0f;
    for (int a = 0; a < 3; ++a)
    {
        const int pa = rowVertexCompact[i * 3 + a];
        if (pa < 0) continue;
        const float* ci = rowDeformable + static_cast<std::size_t>(i) * 9 + a * 3;
        for (int b = 0; b < 3; ++b)
        {
            const int qb = rowVertexCompact[j * 3 + b];
            if (qb < 0) continue;
            const float* cj = rowDeformable + static_cast<std::size_t>(j) * 9 + b * 3;
            for (int r = 0; r < 3; ++r)
            {
                float tmp = 0.0f;
                for (int col = 0; col < 3; ++col)
                {
                    tmp += compliance[static_cast<std::size_t>(3 * qb + col) * ld + (3 * pa + r)] * cj[col];
                }
                acc1 += ci[r] * tmp;
            }
        }
    }
    // Two rows couple through a rigid body only when they act on the same one.
    float acc2 = 0.0f;
    const int body = rowBody[i];
    if (body == rowBody[j])
        for (int e = 0; e < 6; ++e) acc2 += rowRigid[static_cast<std::size_t>(i) * 6 + e] * rowRigidResponse[static_cast<std::size_t>(j) * 6 + e];
    W[static_cast<std::size_t>(i) * rows + j] = f1 * acc1 + bodyFactors[body] * acc2;
}

__device__ __forceinline__ float constraintWarpSum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ double constraintWarpSumDouble(double v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

struct ConstraintSolveState
{
    int iterations;
    int converged;
    double error;
};

// Per contact, what the local resolution needs from W: its 3x3 diagonal block
// (row-major) and SOFA's two divisors, stored once per solve so the serial part
// of each Gauss-Seidel step reads 11 consecutive floats and does no division.
constexpr int kContactBlockValues = 11;   // 9 block entries, 1/W_nn, 2/(W_tt + W_ss)

__global__ void constraintContactBlocksKernel(
    const float* __restrict__ W,
    const int rows,
    const int rowsPerContact,
    float* __restrict__ blocks)
{
    const int contacts = rows / rowsPerContact;
    const int stride = gridDim.x * blockDim.x;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < contacts; c += stride)
    {
        const int j = c * rowsPerContact;
        float* b = blocks + static_cast<std::size_t>(c) * kContactBlockValues;
        for (int l = 0; l < 3; ++l)
            for (int m = 0; m < 3; ++m)
                b[l * 3 + m] = (l < rowsPerContact && m < rowsPerContact)
                    ? W[static_cast<std::size_t>(j + l) * rows + j + m] : 0.0f;
        b[9] = 1.0f / b[0];
        b[10] = rowsPerContact == 3 ? 2.0f / (b[4] + b[8]) : 0.0f;
    }
}

// One block, persistent: BlockGaussSeidelConstraintSolver::gaussSeidel_increment
// over all contacts, repeated until SOFA's stopping rule holds. The dot products
// W_row . f are split over the block's threads; the local resolution and the
// error measure run on thread 0.
//   Exact = true:  double everywhere, SOFA's own arithmetic (divisions and all);
//                  used to check the kernel against SOFA to machine precision.
//   Exact = false: float, with the precomputed per-contact block and divisors;
//                  the fast mode for simulation.
// Resumable: `state` and `force` persist between launches.
// SharedForce = true: the multipliers live in shared memory (up to about 4,000
// rows); false: in global memory (`force` itself, and `forceShadow` for the float
// copy), for bigger problems. One block either way, so __syncthreads() makes each
// contact's update visible to the whole block before the next contact.
template <int RowsPerContact, bool Exact, int Threads, bool SharedForce>
__global__ void __launch_bounds__(Threads) constraintGaussSeidelKernel(
    const float* __restrict__ W,
    const float* __restrict__ dfree,
    const float* __restrict__ blocks,    // contacts * kContactBlockValues
    const int rows,
    const double mu,
    const double tolerance,              // already scaled by rows when SOFA would
    const int maxIterations,
    const int sweepsThisLaunch,
    const int allVerified,
    const double sor,
    double* force,                       // rows, persistent
    float* forceShadow,                  // rows, used when !SharedForce
    double* forceBeforeSweep,
    ConstraintSolveState* __restrict__ state)
{
    constexpr int warps = Threads / 32;
    extern __shared__ unsigned char sharedRaw[];
    double* f = SharedForce ? reinterpret_cast<double*>(sharedRaw) : force;                  // rows doubles
    float* fs = SharedForce ? reinterpret_cast<float*>(reinterpret_cast<double*>(sharedRaw) + rows)
                            : forceShadow;                                                    // rows floats (shadow)
    __shared__ double partialD[3][warps];
    __shared__ float partialF[3][warps];
    __shared__ int stop;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int k = tid; k < rows; k += Threads)
    {
        if (SharedForce) f[k] = force[k];
        fs[k] = static_cast<float>(force[k]);
    }
    if (tid == 0) stop = (state->converged != 0 || state->iterations >= maxIterations) ? 1 : 0;
    __syncthreads();

    const float muF = static_cast<float>(mu);
    const float tolF = static_cast<float>(tolerance);

    for (int sweep = 0; sweep < sweepsThisLaunch && !stop; ++sweep)
    {
        if (sor != 1.0)
        {
            for (int k = tid; k < rows; k += Threads) forceBeforeSweep[k] = f[k];
            __syncthreads();
        }

        double error = 0.0;           // meaningful on thread 0 only
        float errorF = 0.0f;
        bool verified = true;

        for (int j = 0; j < rows; j += RowsPerContact)
        {
            const float* w0 = W + static_cast<std::size_t>(j) * rows;
            if (Exact)
            {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0;
                for (int k = tid; k < rows; k += Threads)
                {
                    const double fk = f[k];
                    s0 += static_cast<double>(w0[k]) * fk;
                    if (RowsPerContact == 3)
                    {
                        s1 += static_cast<double>(w0[rows + k]) * fk;
                        s2 += static_cast<double>(w0[2 * static_cast<std::size_t>(rows) + k]) * fk;
                    }
                }
                s0 = constraintWarpSumDouble(s0);
                if (RowsPerContact == 3) { s1 = constraintWarpSumDouble(s1); s2 = constraintWarpSumDouble(s2); }
                if (lane == 0) { partialD[0][warp] = s0; partialD[1][warp] = s1; partialD[2][warp] = s2; }
            }
            else
            {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
                for (int k = tid; k < rows; k += Threads)
                {
                    const float fk = fs[k];
                    s0 = fmaf(w0[k], fk, s0);
                    if (RowsPerContact == 3)
                    {
                        s1 = fmaf(w0[rows + k], fk, s1);
                        s2 = fmaf(w0[2 * static_cast<std::size_t>(rows) + k], fk, s2);
                    }
                }
                s0 = constraintWarpSum(s0);
                if (RowsPerContact == 3) { s1 = constraintWarpSum(s1); s2 = constraintWarpSum(s2); }
                if (lane == 0) { partialF[0][warp] = s0; partialF[1][warp] = s1; partialF[2][warp] = s2; }
            }
            __syncthreads();

            if (tid == 0)
            {
                if (Exact)
                {
                    double d[3] = { dfree[j], 0.0, 0.0 };
                    for (int w = 0; w < warps; ++w) d[0] += partialD[0][w];
                    if (RowsPerContact == 3)
                    {
                        d[1] = dfree[j + 1];
                        d[2] = dfree[j + 2];
                        for (int w = 0; w < warps; ++w) { d[1] += partialD[1][w]; d[2] += partialD[2][w]; }
                    }
                    double old[3] = { f[j], 0.0, 0.0 };
                    if (RowsPerContact == 3) { old[1] = f[j + 1]; old[2] = f[j + 2]; }

                    const std::size_t rj = static_cast<std::size_t>(j) * rows;
                    const double w00 = W[rj + j];
                    if (RowsPerContact == 3)
                    {
                        // UnilateralConstraintResolutionWithFriction::resolution
                        const double w01 = W[rj + j + 1];
                        const double w02 = W[rj + j + 2];
                        const double w11 = W[rj + rows + j + 1];
                        const double w22 = W[rj + 2 * static_cast<std::size_t>(rows) + j + 2];
                        double fn = old[0] - d[0] / w00;
                        double ft = old[1];
                        double fsl = old[2];
                        if (fn < 0.0)
                        {
                            fn = 0.0; ft = 0.0; fsl = 0.0;
                        }
                        else
                        {
                            d[1] += w01 * (fn - old[0]);
                            d[2] += w02 * (fn - old[0]);
                            ft -= 2.0 * d[1] / (w11 + w22);
                            fsl -= 2.0 * d[2] / (w11 + w22);
                            const double normFt = sqrt(ft * ft + fsl * fsl);
                            const double limit = mu * fn;
                            if (normFt > limit)
                            {
                                const double factor = limit / normFt;
                                ft *= factor;
                                fsl *= factor;
                            }
                        }
                        f[j] = fn; f[j + 1] = ft; f[j + 2] = fsl;
                        fs[j] = static_cast<float>(fn); fs[j + 1] = static_cast<float>(ft); fs[j + 2] = static_cast<float>(fsl);

                        // Error: sum over the 3 lines of |W_block * delta f|.
                        const double df[3] = { fn - old[0], ft - old[1], fsl - old[2] };
                        double contactError = 0.0;
                        for (int l = 0; l < 3; ++l)
                        {
                            double lineSq = 0.0;
                            for (int m = 0; m < 3; ++m)
                            {
                                const double e = static_cast<double>(W[rj + static_cast<std::size_t>(l) * rows + j + m]) * df[m];
                                lineSq += e * e;
                            }
                            const double lineError = sqrt(lineSq);
                            if (lineError > tolerance) verified = false;
                            contactError += lineError;
                        }
                        error += contactError;
                    }
                    else
                    {
                        // UnilateralConstraintResolution::resolution
                        double fn = old[0] - d[0] / w00;
                        if (fn < 0.0) fn = 0.0;
                        f[j] = fn;
                        fs[j] = static_cast<float>(fn);
                        const double contactError = fabs(w00 * (fn - old[0]));
                        if (contactError > tolerance) verified = false;
                        error += contactError;
                    }
                }
                else
                {
                    const float* b = blocks + static_cast<std::size_t>(j / RowsPerContact) * kContactBlockValues;
                    float d0 = dfree[j];
                    for (int w = 0; w < warps; ++w) d0 += partialF[0][w];
                    const float old0 = fs[j];
                    float fn = old0 - d0 * b[9];
                    if (RowsPerContact == 3)
                    {
                        float d1 = dfree[j + 1], d2 = dfree[j + 2];
                        for (int w = 0; w < warps; ++w) { d1 += partialF[1][w]; d2 += partialF[2][w]; }
                        const float old1 = fs[j + 1], old2 = fs[j + 2];
                        float ft = old1, fsl = old2;
                        if (fn < 0.0f)
                        {
                            fn = 0.0f; ft = 0.0f; fsl = 0.0f;
                        }
                        else
                        {
                            d1 += b[1] * (fn - old0);
                            d2 += b[2] * (fn - old0);
                            ft -= d1 * b[10];
                            fsl -= d2 * b[10];
                            const float normFt = sqrtf(ft * ft + fsl * fsl);
                            const float limit = muF * fn;
                            if (normFt > limit)
                            {
                                const float factor = limit / normFt;
                                ft *= factor;
                                fsl *= factor;
                            }
                        }
                        f[j] = fn; f[j + 1] = ft; f[j + 2] = fsl;
                        fs[j] = fn; fs[j + 1] = ft; fs[j + 2] = fsl;
                        const float df[3] = { fn - old0, ft - old1, fsl - old2 };
                        float contactError = 0.0f;
                        for (int l = 0; l < 3; ++l)
                        {
                            const float e0 = b[l * 3 + 0] * df[0];
                            const float e1 = b[l * 3 + 1] * df[1];
                            const float e2 = b[l * 3 + 2] * df[2];
                            const float lineError = sqrtf(e0 * e0 + e1 * e1 + e2 * e2);
                            if (lineError > tolF) verified = false;
                            contactError += lineError;
                        }
                        errorF += contactError;
                    }
                    else
                    {
                        if (fn < 0.0f) fn = 0.0f;
                        f[j] = fn;
                        fs[j] = fn;
                        const float contactError = fabsf(b[0] * (fn - old0));
                        if (contactError > tolF) verified = false;
                        errorF += contactError;
                    }
                }
            }
            __syncthreads();
        }

        if (sor != 1.0)
        {
            for (int k = tid; k < rows; k += Threads)
            {
                f[k] = sor * f[k] + (1.0 - sor) * forceBeforeSweep[k];
                fs[k] = static_cast<float>(f[k]);
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            if (!Exact) error = errorF;
            state->iterations += 1;
            state->error = error;
            const bool done = allVerified ? verified : (error < tolerance);
            if (done) state->converged = 1;
            stop = (done || state->iterations >= maxIterations) ? 1 : 0;
        }
        __syncthreads();
    }

    if (SharedForce)
    {
        for (int k = tid; k < rows; k += Threads) force[k] = f[k];
    }
}

// J^T lambda for both bodies: body 1 into a full DOF vector, body 2 into 6 values
// (rigidRhs[6] collects the sum of the normal multipliers).
__global__ void constraintImpulseKernel(
    const int rows,
    const int rowsPerContact,
    const int* __restrict__ rowVertexGlobal,
    const float* __restrict__ rowDeformable,
    const float* __restrict__ rowRigid,
    const int* __restrict__ rowBody,
    const double* __restrict__ lambda,
    float* __restrict__ deformableRhs,
    double* __restrict__ rigidRhsAll)       // 7 per rigid body: J^T lambda (6), then the normal sum
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += stride)
    {
        const double l = lambda[i];
        if (l == 0.0) continue;
        double* rigidRhs = rigidRhsAll + 7 * rowBody[i];
        if (i % rowsPerContact == 0) atomicAddDouble(rigidRhs + 6, l);
        for (int a = 0; a < 3; ++a)
        {
            const int v = rowVertexGlobal[i * 3 + a];
            if (v < 0) continue;
            for (int c = 0; c < 3; ++c)
            {
                atomicAdd(deformableRhs + 3 * v + c, rowDeformable[static_cast<std::size_t>(i) * 9 + a * 3 + c] * static_cast<float>(l));
            }
        }
        for (int e = 0; e < 6; ++e) atomicAddDouble(rigidRhs + e, static_cast<double>(rowRigid[static_cast<std::size_t>(i) * 6 + e]) * l);
    }
}

__global__ void constraintDensifyKernel(
    const int n,
    const int* __restrict__ rowPtr,
    const int* __restrict__ columns,
    const float* __restrict__ values,
    float* __restrict__ dense)             // column-major n x n, zeroed
{
    const int stride = gridDim.x * blockDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride)
    {
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; ++p)
        {
            dense[static_cast<std::size_t>(columns[p]) * n + r] = values[p];
        }
    }
}

const char* cublasStatusName(const cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        default: return "error";
    }
}

struct ConstraintEventTimer
{
    cudaEvent_t start { nullptr };
    cudaEvent_t stop { nullptr };
    bool active { false };

    explicit ConstraintEventTimer(const bool enabled)
        : active(enabled)
    {
        if (!active) return;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    double finish()
    {
        if (!active) return 0.0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    ~ConstraintEventTimer()
    {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

} // namespace


namespace SofaGpuCollision::backend
{

struct ConstraintWorkspace
{
    cublasHandle_t blas { nullptr };
    cusolverDnHandle_t solver { nullptr };

    // Body 1: dense Cholesky factor of A1 (lower, column-major).
    int n { 0 };
    float* dense { nullptr };               std::size_t denseCapacity { 0 };
    float* factorWork { nullptr };          std::size_t factorWorkCapacity { 0 };
    int* info { nullptr };                  std::size_t infoCapacity { 0 };
    int* csrRowPtr { nullptr };             std::size_t csrRowPtrCapacity { 0 };
    int* csrColumns { nullptr };            std::size_t csrColumnsCapacity { 0 };
    float* csrValues { nullptr };           std::size_t csrValuesCapacity { 0 };
    std::vector<int> cachedRowPtr;
    std::vector<int> cachedColumns;
    std::vector<float> hostValues;
    bool factorized { false };
    TissueWorkspace* tissue { nullptr };       // body 1 on the GPU: the tissue solver, whose factor solves (not owned)

    // Body 2: A2^-1 (6x6).
    double rigidInverse[36] {};
    float* rigidInverseDevice { nullptr };  std::size_t rigidInverseCapacity { 0 };
    bool rigidReady { false };
    // All rigid bodies (body 0 = the one above, then the additional ones): A^-1
    // (36 per body, host and device), W's rigid factor per body, each row's body,
    // and the additional bodies' last corrections and impulses (6 per body each).
    int bodyCount { 1 };
    std::vector<double> bodyInverse;
    std::vector<double> bodyFactors;
    float* bodyInverseDevice { nullptr };   std::size_t bodyInverseCapacity { 0 };
    float* bodyFactorsDevice { nullptr };   std::size_t bodyFactorsCapacity { 0 };
    int* rowBody { nullptr };               std::size_t rowBodyCapacity { 0 };
    std::vector<double> extraCorrections;
    std::vector<double> extraImpulses;

    // Contact selection.
    unsigned long long* deformableSlots { nullptr }; std::size_t deformableSlotsCapacity { 0 };
    unsigned long long* rigidSlots { nullptr };      std::size_t rigidSlotsCapacity { 0 };
    std::uint32_t* deformableOwners { nullptr };     std::size_t deformableOwnersCapacity { 0 };
    std::uint32_t* rigidOwners { nullptr };          std::size_t rigidOwnersCapacity { 0 };
    float* coneNormals { nullptr };                  std::size_t coneNormalsCapacity { 0 };   // first surface, then second: 3 per vertex
    float* coneCos { nullptr };                      std::size_t coneCosCapacity { 0 };       // 1 per vertex, same layout
    std::uint32_t* selected { nullptr };             std::size_t selectedCapacity { 0 };
    unsigned long long* sortKeys { nullptr };        std::size_t sortKeysCapacity { 0 };
    std::uint32_t* counter { nullptr };              std::size_t counterCapacity { 0 };
    int* flags { nullptr };                          std::size_t flagsCapacity { 0 };
    int* scan { nullptr };                           std::size_t scanCapacity { 0 };
    int* vertexCompact { nullptr };                  std::size_t vertexCompactCapacity { 0 };
    int* compactToGlobal { nullptr };                std::size_t compactToGlobalCapacity { 0 };
    float* freePositions { nullptr };                std::size_t freePositionsCapacity { 0 };
    unsigned char* dofMask { nullptr };              std::size_t dofMaskCapacity { 0 };
    std::vector<unsigned char> uploadedDofMask;      // what dofMask holds (uploaded on change)

    // The constraint problem.
    int contacts { 0 };
    int rows { 0 };
    int rowsPerContact { 3 };
    int touched { 0 };
    double mu { 0.0 };
    int* rowVertexCompact { nullptr };      std::size_t rowVertexCompactCapacity { 0 };
    int* rowVertexGlobal { nullptr };       std::size_t rowVertexGlobalCapacity { 0 };
    float* rowDeformable { nullptr };       std::size_t rowDeformableCapacity { 0 };
    float* rowRigid { nullptr };            std::size_t rowRigidCapacity { 0 };
    float* rowRigidResponse { nullptr };    std::size_t rowRigidResponseCapacity { 0 };
    float* dfree { nullptr };               std::size_t dfreeCapacity { 0 };
    float* geometry { nullptr };            std::size_t geometryCapacity { 0 };
    int* contactVertices { nullptr };       std::size_t contactVerticesCapacity { 0 };
    float* contactWeights { nullptr };      std::size_t contactWeightsCapacity { 0 };
    float* selector { nullptr };            std::size_t selectorCapacity { 0 };
    float* compliance { nullptr };          std::size_t complianceCapacity { 0 };
    float* W { nullptr };                   std::size_t WCapacity { 0 };
    bool complianceReady { false };

    // Solve and correction.
    double* lambda { nullptr };             std::size_t lambdaCapacity { 0 };
    float* lambdaShadow { nullptr };        std::size_t lambdaShadowCapacity { 0 };
    double* lambdaBeforeSweep { nullptr };  std::size_t lambdaBeforeSweepCapacity { 0 };
    float* contactBlocks { nullptr };       std::size_t contactBlocksCapacity { 0 };
    ConstraintSolveState* solveState { nullptr }; std::size_t solveStateCapacity { 0 };
    float* deformableRhs { nullptr };       std::size_t deformableRhsCapacity { 0 };
    double* rigidRhs { nullptr };           std::size_t rigidRhsCapacity { 0 };
    bool solved { false };
    bool deviceCorrectionValid { false };   // deformableRhs holds the last device correction's dv
    float* hostCorrectionDevice { nullptr }; std::size_t hostCorrectionCapacity { 0 };

    ~ConstraintWorkspace()
    {
        if (blas) cublasDestroy(blas);
        if (solver) cusolverDnDestroy(solver);
        void* buffers[] = {
            dense, factorWork, info, csrRowPtr, csrColumns, csrValues, rigidInverseDevice, bodyInverseDevice, bodyFactorsDevice, rowBody,
            deformableSlots, rigidSlots, deformableOwners, rigidOwners, coneNormals, coneCos, selected, sortKeys, counter, flags, scan, vertexCompact,
            compactToGlobal, freePositions, dofMask, rowVertexCompact, rowVertexGlobal, rowDeformable, rowRigid,
            rowRigidResponse, dfree, geometry, contactVertices, contactWeights, selector, compliance, W,
            lambda, lambdaShadow, lambdaBeforeSweep, contactBlocks, solveState, deformableRhs, rigidRhs,
            hostCorrectionDevice };
        for (void* b : buffers)
        {
            if (b) cudaFree(b);
        }
    }
};

namespace
{
// Each rigid body's correction A_b^-1 J_b^T lambda from its impulse (rigidRhsAll: 7
// per body, J^T lambda then the normal sum): body 0's into rigidCorrection and
// impulse (whose normal sum covers every body), the others kept in the workspace.
void rigidBodyResults(ConstraintWorkspace* ws, const std::vector<double>& rigidRhsAll, double rigidCorrection[6],
                      ConstraintImpulse* impulse)
{
    const int bodies = static_cast<int>(rigidRhsAll.size() / 7);
    const std::size_t extra = static_cast<std::size_t>(std::max(bodies - 1, 0));
    ws->extraCorrections.assign(6 * extra, 0.0);
    ws->extraImpulses.assign(6 * extra, 0.0);
    double normalSum = 0.0;
    for (int b = 0; b < bodies; ++b)
    {
        const double* rhs = rigidRhsAll.data() + 7 * b;
        const double* inverse = ws->bodyInverse.size() >= 36u * (b + 1) ? ws->bodyInverse.data() + 36 * b : ws->rigidInverse;
        double* correction = b == 0 ? rigidCorrection : ws->extraCorrections.data() + 6 * (b - 1);
        for (int i = 0; i < 6; ++i)
        {
            double acc = 0.0;
            for (int j = 0; j < 6; ++j) acc += inverse[i * 6 + j] * rhs[j];
            correction[i] = acc;
        }
        if (b > 0)
            for (int e = 0; e < 6; ++e) ws->extraImpulses[6 * (b - 1) + e] = rhs[e];
        normalSum += rhs[6];
    }
    if (impulse != nullptr)
    {
        for (int e = 0; e < 6; ++e) impulse->rigid[e] = rigidRhsAll.empty() ? 0.0 : rigidRhsAll[e];
        impulse->normalSum = normalSum;
    }
}

// b <- A1^-1 b for nrhs columns (single precision, in place): with the tissue
// solver's factor (body 1 on the GPU: band or dense Cholesky, or LU on a step whose
// matrix was not positive definite), else with our own Cholesky (factorizeDeformableSystem).
bool solveWithDeformableFactor(ConstraintWorkspace* ws, const int nrhs, float* b, std::string& diagnostic)
{
    if (ws->tissue != nullptr) return tissueSolveInPlace(ws->tissue, b, nrhs, diagnostic);
    if (cusolverDnSpotrs(ws->solver, CUBLAS_FILL_MODE_LOWER, ws->n, nrhs, ws->dense, ws->n, b, ws->n, ws->info) !=
        CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cusolverDnSpotrs failed to launch.";
        return false;
    }
    diagnostic.clear();
    return true;
}
} // namespace

namespace
{
// cuBLAS and cuSOLVER load their kernels on first use, which stalled the first contact
// step by about 0.8 s. Run each call the contact step makes once, on a small identity
// system, when the workspace is created instead. Failures here are harmless.
void warmUpDenseLibraries(ConstraintWorkspace* ws)
{
    constexpr int n = 256;
    constexpr int m = 64;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* work = nullptr;
    int* info = nullptr;
    bool ok = cudaMalloc(reinterpret_cast<void**>(&a), sizeof(float) * n * n) == cudaSuccess &&
              cudaMalloc(reinterpret_cast<void**>(&b), sizeof(float) * n * m) == cudaSuccess &&
              cudaMalloc(reinterpret_cast<void**>(&c), sizeof(float) * m * m) == cudaSuccess &&
              cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int)) == cudaSuccess;
    if (ok)
    {
        std::vector<float> identity(static_cast<std::size_t>(n) * n, 0.0f);
        for (int i = 0; i < n; ++i) identity[static_cast<std::size_t>(i) * n + i] = 1.0f;
        ok = cudaMemcpy(a, identity.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemset(b, 0, sizeof(float) * n * m) == cudaSuccess;
    }
    int workSize = 0;
    if (ok) ok = cusolverDnSpotrf_bufferSize(ws->solver, CUBLAS_FILL_MODE_LOWER, n, a, n, &workSize) == CUSOLVER_STATUS_SUCCESS &&
                 cudaMalloc(reinterpret_cast<void**>(&work), sizeof(float) * std::max(workSize, 1)) == cudaSuccess;
    if (ok)
    {
        const float one = 1.0f;
        const float zero = 0.0f;
        cusolverDnSpotrf(ws->solver, CUBLAS_FILL_MODE_LOWER, n, a, n, work, workSize, info);
        cusolverDnSpotrs(ws->solver, CUBLAS_FILL_MODE_LOWER, n, m, a, n, b, n, info);
        cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, m, &one, a, n, b, n);
        cublasSgemm(ws->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, b, n, b, n, &zero, c, m);
        cudaDeviceSynchronize();
    }
    for (void* p : { static_cast<void*>(a), static_cast<void*>(b), static_cast<void*>(c), static_cast<void*>(work), static_cast<void*>(info) })
    {
        if (p) cudaFree(p);
    }
    cudaGetLastError();   // a failed warm-up must not leave an error behind
}
} // namespace

ConstraintWorkspace* createConstraintWorkspace(std::string& diagnostic)
{
    auto* ws = new ConstraintWorkspace();
    if (cublasCreate(&ws->blas) != CUBLAS_STATUS_SUCCESS)
    {
        diagnostic = "cuBLAS could not be initialised.";
        delete ws;
        return nullptr;
    }
    if (cusolverDnCreate(&ws->solver) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cuSOLVER could not be initialised.";
        delete ws;
        return nullptr;
    }
    warmUpDenseLibraries(ws);
    diagnostic.clear();
    return ws;
}

void destroyConstraintWorkspace(ConstraintWorkspace* workspace)
{
    delete workspace;
}

bool factorizeDeformableSystem(
    ConstraintWorkspace* ws,
    const HostCsrMatrix& matrix,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    if (matrix.size <= 0 || matrix.rowPtr == nullptr || matrix.columns == nullptr || matrix.values == nullptr)
    {
        diagnostic = "Empty deformable system matrix.";
        return false;
    }
    ConstraintEventTimer timer(timings != nullptr);
    const int n = matrix.size;
    const int nnz = matrix.nonZeros;
    ws->factorized = false;
    ws->tissue = nullptr;
    ws->complianceReady = false;

    cudaError_t err = cudaSuccess;
    const bool sameStructure = ws->n == n &&
        ws->cachedRowPtr.size() == static_cast<std::size_t>(n + 1) &&
        ws->cachedColumns.size() == static_cast<std::size_t>(nnz) &&
        std::equal(ws->cachedRowPtr.begin(), ws->cachedRowPtr.end(), matrix.rowPtr) &&
        std::equal(ws->cachedColumns.begin(), ws->cachedColumns.end(), matrix.columns);
    if (!sameStructure)
    {
        err = ensureDeviceBuffer(ws->csrRowPtr, ws->csrRowPtrCapacity, static_cast<std::size_t>(n + 1));
        if (err == cudaSuccess) err = ensureDeviceBuffer(ws->csrColumns, ws->csrColumnsCapacity, static_cast<std::size_t>(nnz));
        if (err == cudaSuccess) err = cudaMemcpy(ws->csrRowPtr, matrix.rowPtr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy(ws->csrColumns, matrix.columns, sizeof(int) * nnz, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("System matrix structure upload failed: ") + cudaGetErrorString(err);
            return false;
        }
        ws->cachedRowPtr.assign(matrix.rowPtr, matrix.rowPtr + n + 1);
        ws->cachedColumns.assign(matrix.columns, matrix.columns + nnz);
    }
    ws->hostValues.resize(static_cast<std::size_t>(nnz));
    for (int p = 0; p < nnz; ++p) ws->hostValues[p] = static_cast<float>(matrix.values[p]);
    err = ensureDeviceBuffer(ws->csrValues, ws->csrValuesCapacity, static_cast<std::size_t>(nnz));
    if (err == cudaSuccess) err = cudaMemcpy(ws->csrValues, ws->hostValues.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->dense, ws->denseCapacity, static_cast<std::size_t>(n) * n);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->info, ws->infoCapacity, 1);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("System matrix upload failed: ") + cudaGetErrorString(err);
        return false;
    }
    ws->n = n;

    launchFill(ws->dense, static_cast<std::size_t>(n) * n, 0.0f);
    constraintDensifyKernel<<<std::min((n + 255) / 256, 1024), 256>>>(n, ws->csrRowPtr, ws->csrColumns, ws->csrValues, ws->dense);

    int workSize = 0;
    if (cusolverDnSpotrf_bufferSize(ws->solver, CUBLAS_FILL_MODE_LOWER, n, ws->dense, n, &workSize) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cusolverDnSpotrf_bufferSize failed.";
        return false;
    }
    err = ensureDeviceBuffer(ws->factorWork, ws->factorWorkCapacity, static_cast<std::size_t>(std::max(workSize, 1)));
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Factorisation workspace allocation failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (cusolverDnSpotrf(ws->solver, CUBLAS_FILL_MODE_LOWER, n, ws->dense, n, ws->factorWork, workSize, ws->info) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cusolverDnSpotrf failed to launch.";
        return false;
    }
    int info = 0;
    err = cudaMemcpy(&info, ws->info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Cholesky factorisation failed: ") + cudaGetErrorString(err);
        return false;
    }
    if (info != 0)
    {
        diagnostic = "The deformable system matrix is not positive definite (Cholesky stopped at row " + std::to_string(info) + ").";
        return false;
    }
    ws->factorized = true;
    if (timings != nullptr) timings->factorizeMs = timer.finish();
    diagnostic.clear();
    return true;
}

namespace
{
// A 6x6 system matrix's inverse (Gauss-Jordan with partial pivoting); false if singular.
bool invertRigidSystem(const double* matrix, double* inverse)
{
    double a[6][12];
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j) { a[i][j] = matrix[i * 6 + j]; a[i][6 + j] = (i == j) ? 1.0 : 0.0; }
    }
    for (int col = 0; col < 6; ++col)
    {
        int pivot = col;
        for (int r = col + 1; r < 6; ++r) if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
        if (std::fabs(a[pivot][col]) < 1.0e-300) return false;
        if (pivot != col) for (int j = 0; j < 12; ++j) std::swap(a[col][j], a[pivot][j]);
        const double inv = 1.0 / a[col][col];
        for (int j = 0; j < 12; ++j) a[col][j] *= inv;
        for (int r = 0; r < 6; ++r)
        {
            if (r == col) continue;
            const double factor = a[r][col];
            if (factor == 0.0) continue;
            for (int j = 0; j < 12; ++j) a[r][j] -= factor * a[col][j];
        }
    }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) inverse[i * 6 + j] = a[i][6 + j];
    return true;
}

// Every body's inverse to the device (float, 36 per body).
cudaError_t uploadBodyInverses(ConstraintWorkspace* ws)
{
    std::vector<float> inverse(ws->bodyInverse.begin(), ws->bodyInverse.end());
    cudaError_t err = ensureDeviceBuffer(ws->bodyInverseDevice, ws->bodyInverseCapacity, std::max<std::size_t>(inverse.size(), 36));
    if (err == cudaSuccess && !inverse.empty())
        err = cudaMemcpy(ws->bodyInverseDevice, inverse.data(), sizeof(float) * inverse.size(), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rigidInverseDevice, ws->rigidInverseCapacity, 36);
    if (err == cudaSuccess) err = cudaMemcpy(ws->rigidInverseDevice, inverse.data(), sizeof(float) * 36, cudaMemcpyHostToDevice);
    return err;
}
} // namespace

bool setRigidSystem(ConstraintWorkspace* ws, const double matrix[36], std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    if (!invertRigidSystem(matrix, ws->rigidInverse))
    {
        diagnostic = "The rigid body's system matrix is singular.";
        ws->rigidReady = false;
        return false;
    }
    if (ws->bodyInverse.size() < 36) ws->bodyInverse.resize(36);
    std::copy(ws->rigidInverse, ws->rigidInverse + 36, ws->bodyInverse.begin());
    const cudaError_t err = uploadBodyInverses(ws);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Rigid compliance upload failed: ") + cudaGetErrorString(err);
        ws->rigidReady = false;
        return false;
    }
    ws->rigidReady = true;
    diagnostic.clear();
    return true;
}

bool setAdditionalRigidSystems(ConstraintWorkspace* ws, const std::vector<double>& matrices, const std::vector<double>& factors,
                               std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    const std::size_t extra = matrices.size() / 36;
    if (matrices.size() != 36 * extra || factors.size() != extra)
    {
        diagnostic = "Additional rigid systems: 36 matrix values and 1 factor per body.";
        return false;
    }
    ws->bodyInverse.resize(36 * (1 + extra));
    std::copy(ws->rigidInverse, ws->rigidInverse + 36, ws->bodyInverse.begin());
    for (std::size_t b = 0; b < extra; ++b)
    {
        if (!invertRigidSystem(matrices.data() + 36 * b, ws->bodyInverse.data() + 36 * (b + 1)))
        {
            diagnostic = "Rigid body " + std::to_string(b + 1) + "'s system matrix is singular.";
            return false;
        }
    }
    ws->bodyFactors.assign(1 + extra, 0.0);
    for (std::size_t b = 0; b < extra; ++b) ws->bodyFactors[b + 1] = factors[b];
    const cudaError_t err = uploadBodyInverses(ws);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Rigid compliance upload failed: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

bool additionalRigidResults(const ConstraintWorkspace* ws, std::vector<double>& corrections, std::vector<double>& impulses)
{
    corrections.clear();
    impulses.clear();
    if (ws == nullptr) return false;
    corrections = ws->extraCorrections;
    impulses = ws->extraImpulses;
    return true;
}

bool buildContactConstraints(
    ConstraintWorkspace* ws,
    const ConstraintBuildInput& input,
    ConstraintBuildStats* stats,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    if (stats != nullptr) *stats = ConstraintBuildStats {};
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    ws->contacts = 0;
    ws->rows = 0;
    ws->touched = 0;
    ws->complianceReady = false;
    ws->solved = false;
    ws->mu = input.friction;
    ws->rowsPerContact = input.friction > 0.0f ? 3 : 1;

    // The rigid bodies: body 0 from the single fields, then the additional ones.
    std::vector<ConstraintRigidBodyInput> bodies(1);
    bodies[0].surfaceId = input.rigidSurfaceId;
    bodies[0].surfacePositions = input.rigidSurfacePositions;
    bodies[0].vertexCount = input.rigidVertexCount;
    for (int e = 0; e < 3; ++e) bodies[0].center[e] = input.rigidCenter[e];
    for (int e = 0; e < 6; ++e) bodies[0].freeStep[e] = input.rigidFreeStep[e];
    bodies[0].dofMask = input.rigidDofMask;
    bodies.insert(bodies.end(), input.additionalRigidBodies.begin(), input.additionalRigidBodies.end());
    const int bodyCount = static_cast<int>(bodies.size());
    ws->bodyCount = bodyCount;

    if (!ws->rigidReady || ws->bodyInverse.size() < 36u * bodyCount)
    {
        diagnostic = "setRigidSystem (and setAdditionalRigidSystems for more bodies) must be called before building constraints.";
        return false;
    }
    bool complete = input.deformableSurfacePositions != nullptr && input.deformableVertexCount > 0 &&
                    (input.deformableFreePositions != nullptr || input.deformableFreePositionsDevice != nullptr);
    for (const auto& body : bodies) complete = complete && body.surfacePositions != nullptr && body.vertexCount > 0;
    if (!complete)
    {
        diagnostic = "Constraint build input is incomplete (positions or vertex counts missing).";
        return false;
    }

    // Each body's contacts from this collision pass: one segment of the selection each.
    struct Segment
    {
        const RecordedContactHandle* handle;
        bool swapped;
        int body;
        std::uint32_t start;
        std::uint32_t count;
    };
    std::vector<Segment> segments;
    std::size_t capacity = 0;
    std::uint32_t maxRigidVertices = 0;
    for (int b = 0; b < bodyCount; ++b)
    {
        const RecordedContactHandle* handle = nullptr;
        bool swapped = false;
        if (!currentContactsFor(input.deformableSurfaceId, bodies[b].surfaceId, handle, swapped, diagnostic)) return false;
        if (handle == nullptr) continue;
        segments.push_back({ handle, swapped, b, 0u, 0u });
        capacity += handle->capacity;
        maxRigidVertices = std::max(maxRigidVertices, bodies[b].vertexCount);
    }
    if (segments.empty())
    {
        diagnostic.clear();      // no contacts in this collision pass
        return true;
    }

    ConstraintEventTimer timer(timings != nullptr);
    const std::uint32_t nd = input.deformableVertexCount;

    cudaError_t err = ensureDeviceBuffer(ws->deformableSlots, ws->deformableSlotsCapacity, nd);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rigidSlots, ws->rigidSlotsCapacity, maxRigidVertices);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->deformableOwners, ws->deformableOwnersCapacity, nd);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rigidOwners, ws->rigidOwnersCapacity, maxRigidVertices);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->selected, ws->selectedCapacity, capacity);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->sortKeys, ws->sortKeysCapacity, capacity);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->counter, ws->counterCapacity, 1);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->flags, ws->flagsCapacity, nd);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->scan, ws->scanCapacity, nd);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->vertexCompact, ws->vertexCompactCapacity, nd);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->compactToGlobal, ws->compactToGlobalCapacity, nd);
    // Body 1's free positions: already on the GPU, or uploaded from the host.
    const float* freeDevice = static_cast<const float*>(input.deformableFreePositionsDevice);
    if (freeDevice == nullptr)
    {
        if (err == cudaSuccess) err = ensureDeviceBuffer(ws->freePositions, ws->freePositionsCapacity, static_cast<std::size_t>(nd) * 3);
        if (err == cudaSuccess) err = cudaMemcpy(ws->freePositions, input.deformableFreePositions, sizeof(float) * 3 * nd, cudaMemcpyHostToDevice);
        freeDevice = ws->freePositions;
    }
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Constraint selection buffers: ") + cudaGetErrorString(err);
        return false;
    }
    launchFill(ws->flags, nd, 0);

    const bool keepAll = input.filter == ConstraintContactFilter::All;
    constexpr unsigned threads = 256;
    std::uint32_t detectedTotal = 0;
    std::uint32_t start = 0;
    for (auto& segment : segments)
    {
        const RecordedContactHandle* handle = segment.handle;
        const bool swapped = segment.swapped;
        const ConstraintRigidBodyInput& body = bodies[segment.body];
        const std::uint32_t nr = body.vertexCount;
        const unsigned blocks = std::max(1u, std::min((handle->capacity + threads - 1) / threads, 1024u));
        // Claims are per (body-1 vertex, tool): a vertex pinched between two jaws keeps a contact with each.
        launchFill(ws->deformableSlots, nd, ~0ull);
        launchFill(ws->rigidSlots, nr, ~0ull);
        launchFill(ws->deformableOwners, nd, ~0u);
        launchFill(ws->rigidOwners, nr, ~0u);
        launchFill(ws->counter, 1, 0u);

        // Vertex normal cones of both surfaces, at the positions the contacts were found at.
        ConstraintVertexCones cones;
        if (!keepAll && input.vertexConeFilter && handle->firstTriangleCount > 0 && handle->secondTriangleCount > 0)
        {
            const std::uint32_t firstCount = swapped ? nr : nd;
            const std::uint32_t secondCount = swapped ? nd : nr;
            const float* firstPositions = static_cast<const float*>(swapped ? body.surfacePositions : input.deformableSurfacePositions);
            const float* secondPositions = static_cast<const float*>(swapped ? input.deformableSurfacePositions : body.surfacePositions);
            err = ensureDeviceBuffer(ws->coneNormals, ws->coneNormalsCapacity, 3u * static_cast<std::size_t>(firstCount + secondCount));
            if (err == cudaSuccess) err = ensureDeviceBuffer(ws->coneCos, ws->coneCosCapacity, static_cast<std::size_t>(firstCount + secondCount));
            if (err != cudaSuccess)
            {
                diagnostic = std::string("Contact cone buffers: ") + cudaGetErrorString(err);
                return false;
            }
            launchFill(ws->coneNormals, 3u * static_cast<std::size_t>(firstCount + secondCount), 0.0f);
            launchFill(ws->coneCos, static_cast<std::size_t>(firstCount + secondCount), 1.0f);
            float* firstNormals = ws->coneNormals;
            float* secondNormals = ws->coneNormals + 3u * static_cast<std::size_t>(firstCount);
            float* firstCones = ws->coneCos;
            float* secondCones = ws->coneCos + firstCount;
            const unsigned firstTriBlocks = std::max(1u, std::min((handle->firstTriangleCount + threads - 1) / threads, 1024u));
            const unsigned secondTriBlocks = std::max(1u, std::min((handle->secondTriangleCount + threads - 1) / threads, 1024u));
            constraintVertexNormalsKernel<<<firstTriBlocks, threads>>>(handle->firstIndices, handle->firstTriangleCount, firstPositions, firstNormals);
            constraintVertexNormalsKernel<<<secondTriBlocks, threads>>>(handle->secondIndices, handle->secondTriangleCount, secondPositions, secondNormals);
            constraintNormalizeKernel<<<std::max(1u, std::min((firstCount + secondCount + threads - 1) / threads, 1024u)), threads>>>(
                firstCount + secondCount, ws->coneNormals);
            constraintVertexConesKernel<<<firstTriBlocks, threads>>>(handle->firstIndices, handle->firstTriangleCount, firstPositions,
                                                                     firstNormals, firstCones);
            constraintVertexConesKernel<<<secondTriBlocks, threads>>>(handle->secondIndices, handle->secondTriangleCount, secondPositions,
                                                                      secondNormals, secondCones);
            cones.firstNormals = firstNormals;
            cones.firstCones = firstCones;
            cones.secondNormals = secondNormals;
            cones.secondCones = secondCones;
            cones.tolerance = input.vertexConeTolerance;
        }
        if (!keepAll)
        {
            constraintClaimVerticesKernel<<<blocks, threads>>>(
                handle->contacts, handle->countDevice, handle->capacity,
                handle->firstIndices, handle->secondIndices, swapped, cones,
                ws->deformableSlots, ws->rigidSlots);
            constraintClaimOwnersKernel<<<blocks, threads>>>(
                handle->contacts, handle->countDevice, handle->capacity,
                handle->firstIndices, handle->secondIndices, swapped, cones,
                ws->deformableSlots, ws->rigidSlots, ws->deformableOwners, ws->rigidOwners);
        }
        constraintSelectContactsKernel<<<blocks, threads>>>(
            handle->contacts, handle->countDevice, handle->capacity,
            handle->firstIndices, handle->secondIndices, swapped, keepAll, cones,
            ws->deformableSlots, ws->rigidSlots, ws->deformableOwners, ws->rigidOwners,
            ws->selected + start, ws->sortKeys + start, ws->counter);

        std::uint32_t detected = 0;
        std::uint32_t count = 0;
        err = cudaMemcpy(&detected, handle->countDevice, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) err = cudaMemcpy(&count, ws->counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Contact selection: ") + cudaGetErrorString(err);
            return false;
        }
        detectedTotal += std::min(detected, handle->capacity);
        // Order independent of the narrow phase's emission order.
        if (count > 1) thrust::sort_by_key(thrust::device, ws->sortKeys + start, ws->sortKeys + start + count, ws->selected + start);
        segment.start = start;
        segment.count = count;
        start += count;
    }
    const std::uint32_t count = start;
    if (stats != nullptr) stats->detectedContacts = detectedTotal;
    if (count == 0)
    {
        if (timings != nullptr) timings->buildMs = timer.finish();
        diagnostic.clear();
        return true;
    }

    // Touched body-1 vertices (over every tool) and their compact numbering.
    for (const auto& segment : segments)
    {
        if (segment.count == 0) continue;
        const unsigned segmentBlocks = std::max(1u, std::min((segment.count + threads - 1) / threads, 1024u));
        constraintFlagVerticesKernel<<<segmentBlocks, threads>>>(
            segment.handle->contacts, ws->selected + segment.start, segment.count,
            segment.handle->firstIndices, segment.handle->secondIndices, segment.swapped, ws->flags);
    }
    thrust::exclusive_scan(thrust::device, ws->flags, ws->flags + nd, ws->scan);
    const unsigned vertexBlocks = std::max(1u, std::min((nd + threads - 1) / threads, 1024u));
    constraintCompactVerticesKernel<<<vertexBlocks, threads>>>(ws->flags, ws->scan, nd, ws->vertexCompact, ws->compactToGlobal);
    int lastFlag = 0, lastScan = 0;
    err = cudaMemcpy(&lastFlag, ws->flags + (nd - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(&lastScan, ws->scan + (nd - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Vertex compaction: ") + cudaGetErrorString(err);
        return false;
    }
    const int touched = lastFlag + lastScan;

    const int rowsPerContact = ws->rowsPerContact;
    const std::size_t rows = static_cast<std::size_t>(count) * rowsPerContact;
    err = ensureDeviceBuffer(ws->rowVertexCompact, ws->rowVertexCompactCapacity, rows * 3);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rowVertexGlobal, ws->rowVertexGlobalCapacity, rows * 3);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rowDeformable, ws->rowDeformableCapacity, rows * 9);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rowRigid, ws->rowRigidCapacity, rows * 6);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rowRigidResponse, ws->rowRigidResponseCapacity, rows * 6);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rowBody, ws->rowBodyCapacity, rows);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->dfree, ws->dfreeCapacity, rows);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->geometry, ws->geometryCapacity, static_cast<std::size_t>(count) * kContactGeometryValues);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->contactVertices, ws->contactVerticesCapacity, static_cast<std::size_t>(count) * 3);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->contactWeights, ws->contactWeightsCapacity, static_cast<std::size_t>(count) * 3);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Constraint row buffers: ") + cudaGetErrorString(err);
        return false;
    }

    // Body 1's DOF mask changes only when a projective constraint does: upload on change.
    const unsigned char* deviceMask = nullptr;
    if (input.deformableDofMask != nullptr)
    {
        const bool same = ws->uploadedDofMask.size() == nd &&
            std::equal(ws->uploadedDofMask.begin(), ws->uploadedDofMask.end(), input.deformableDofMask);
        if (!same)
        {
            err = ensureDeviceBuffer(ws->dofMask, ws->dofMaskCapacity, nd);
            if (err == cudaSuccess) err = cudaMemcpy(ws->dofMask, input.deformableDofMask, nd, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                ws->uploadedDofMask.clear();
                diagnostic = std::string("DOF mask upload: ") + cudaGetErrorString(err);
                return false;
            }
            ws->uploadedDofMask.assign(input.deformableDofMask, input.deformableDofMask + nd);
        }
        deviceMask = ws->dofMask;
    }

    for (const auto& segment : segments)
    {
        if (segment.count == 0) continue;
        const ConstraintRigidBodyInput& body = bodies[segment.body];
        ConstraintRigidMotion rigid {};
        rigid.center = make_float3(static_cast<float>(body.center[0]), static_cast<float>(body.center[1]),
                                   static_cast<float>(body.center[2]));
        rigid.freeLinear = make_float3(static_cast<float>(body.freeStep[0]), static_cast<float>(body.freeStep[1]),
                                       static_cast<float>(body.freeStep[2]));
        rigid.freeAngular = make_float3(static_cast<float>(body.freeStep[3]), static_cast<float>(body.freeStep[4]),
                                        static_cast<float>(body.freeStep[5]));
        for (int e = 0; e < 6; ++e) rigid.dofMask[e] = ((body.dofMask >> e) & 1u) ? 1.0f : 0.0f;

        const float* firstPositions = static_cast<const float*>(segment.swapped ? body.surfacePositions : input.deformableSurfacePositions);
        const float* secondPositions = static_cast<const float*>(segment.swapped ? input.deformableSurfacePositions : body.surfacePositions);
        const std::size_t r0 = static_cast<std::size_t>(segment.start) * rowsPerContact;
        const unsigned segmentBlocks = std::max(1u, std::min((segment.count + threads - 1) / threads, 1024u));
        constraintBuildRowsKernel<<<segmentBlocks, threads>>>(
            segment.handle->contacts, ws->selected + segment.start, segment.count,
            segment.handle->firstIndices, segment.handle->secondIndices,
            firstPositions, secondPositions, segment.swapped,
            freeDevice, deviceMask, ws->vertexCompact, rigid, ws->bodyInverseDevice + 36u * segment.body,
            input.contactDistance, rowsPerContact,
            ws->rowVertexCompact + r0 * 3, ws->rowVertexGlobal + r0 * 3, ws->rowDeformable + r0 * 9,
            ws->rowRigid + r0 * 6, ws->rowRigidResponse + r0 * 6, ws->dfree + r0,
            ws->geometry + static_cast<std::size_t>(segment.start) * kContactGeometryValues,
            ws->contactVertices + static_cast<std::size_t>(segment.start) * 3,
            ws->contactWeights + static_cast<std::size_t>(segment.start) * 3);
        launchFill(ws->rowBody + r0, static_cast<std::size_t>(segment.count) * rowsPerContact, segment.body);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Constraint row kernel: ") + cudaGetErrorString(err);
        return false;
    }

    ws->contacts = static_cast<int>(count);
    ws->rows = static_cast<int>(rows);
    ws->touched = touched;
    if (stats != nullptr)
    {
        stats->contacts = count;
        stats->rows = static_cast<std::uint32_t>(rows);
        stats->touchedVertices = static_cast<std::uint32_t>(touched);
    }
    if (timings != nullptr) timings->buildMs = timer.finish();
    diagnostic.clear();
    return true;
}

bool assembleContactCompliance(
    ConstraintWorkspace* ws,
    const double deformableFactor,
    const double rigidFactor,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    ws->complianceReady = false;
    if (ws->rows == 0) { diagnostic.clear(); return true; }
    if (!ws->factorized)
    {
        diagnostic = "factorizeDeformableSystem must succeed before the compliance is assembled.";
        return false;
    }
    ConstraintEventTimer timer(timings != nullptr);
    const int n = ws->n;
    const int m = 3 * ws->touched;
    const std::size_t rows = static_cast<std::size_t>(ws->rows);

    cudaError_t err = ensureDeviceBuffer(ws->selector, ws->selectorCapacity, static_cast<std::size_t>(n) * std::max(m, 1));
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->compliance, ws->complianceCapacity, static_cast<std::size_t>(std::max(m, 1)) * std::max(m, 1));
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->W, ws->WCapacity, rows * rows);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Compliance buffers: ") + cudaGetErrorString(err);
        return false;
    }

    if (m > 0)
    {
        // G = E^T A1^-1 E: A1^-1 on the touched DOFs.
        if (ws->tissue != nullptr)
        {
            // The tissue solver's factor (band or dense Cholesky: Y = L^-1 E, G = Y^T Y;
            // or LU on a step whose matrix was not positive definite).
            if (!tissueComplianceBlock(ws->tissue, ws->compactToGlobal, ws->touched, ws->compliance, diagnostic)) return false;
        }
        else
        {
            // Our own dense Cholesky: Y = L^-1 E, then G = Y^T Y.
            launchFill(ws->selector, static_cast<std::size_t>(n) * m, 0.0f);
            constraintSelectorKernel<<<std::max(1, std::min((ws->touched + 255) / 256, 1024)), 256>>>(
                ws->compactToGlobal, ws->touched, n, ws->selector);
            const float one = 1.0f;
            const float zero = 0.0f;
            cublasStatus_t status = cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                                CUBLAS_DIAG_NON_UNIT, n, m, &one, ws->dense, n, ws->selector, n);
            if (status == CUBLAS_STATUS_SUCCESS)
            {
                status = cublasSgemm(ws->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one,
                                     ws->selector, n, ws->selector, n, &zero, ws->compliance, m);
            }
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                diagnostic = std::string("Compliance product failed: cuBLAS ") + cublasStatusName(status);
                return false;
            }
        }
    }

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>((rows + 15) / 16), static_cast<unsigned>((rows + 15) / 16));
    // Each rigid body's factor (body 0: rigidFactor; the others as setAdditionalRigidSystems gave them).
    {
        std::vector<float> factors(static_cast<std::size_t>(std::max(ws->bodyCount, 1)), 0.0f);
        factors[0] = static_cast<float>(rigidFactor);
        for (std::size_t b = 1; b < factors.size() && b < ws->bodyFactors.size(); ++b) factors[b] = static_cast<float>(ws->bodyFactors[b]);
        err = ensureDeviceBuffer(ws->bodyFactorsDevice, ws->bodyFactorsCapacity, factors.size());
        if (err == cudaSuccess)
            err = cudaMemcpy(ws->bodyFactorsDevice, factors.data(), sizeof(float) * factors.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Rigid factors upload: ") + cudaGetErrorString(err);
            return false;
        }
    }
    constraintAssembleComplianceKernel<<<grid, block>>>(
        ws->rows, ws->rowVertexCompact, ws->rowDeformable, ws->rowRigid, ws->rowRigidResponse, ws->rowBody,
        ws->bodyFactorsDevice, ws->compliance, std::max(m, 1), static_cast<float>(deformableFactor), ws->W);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Compliance assembly kernel: ") + cudaGetErrorString(err);
        return false;
    }
    ws->complianceReady = true;
    if (timings != nullptr) timings->complianceMs = timer.finish();
    diagnostic.clear();
    return true;
}

namespace
{
template <int RowsPerContact, bool Exact, int Threads>
void launchGaussSeidel(
    const float* W, const float* dfree, const float* blocks, const int rows, const double mu, const double tolerance,
    const ConstraintSolveConfig& config, const bool sharedForce, double* lambda, float* lambdaShadow,
    double* lambdaBeforeSweep, ConstraintSolveState* state)
{
    if (sharedForce)
    {
        const std::size_t sharedBytes = static_cast<std::size_t>(rows) * (sizeof(double) + sizeof(float));
        constraintGaussSeidelKernel<RowsPerContact, Exact, Threads, true><<<1, Threads, sharedBytes>>>(
            W, dfree, blocks, rows, mu, tolerance, config.maxIterations, kConstraintSweepsPerLaunch,
            config.allVerified ? 1 : 0, config.sor, lambda, lambdaShadow, lambdaBeforeSweep, state);
    }
    else
    {
        constraintGaussSeidelKernel<RowsPerContact, Exact, Threads, false><<<1, Threads, 0>>>(
            W, dfree, blocks, rows, mu, tolerance, config.maxIterations, kConstraintSweepsPerLaunch,
            config.allVerified ? 1 : 0, config.sor, lambda, lambdaShadow, lambdaBeforeSweep, state);
    }
}

// Small problems run on one warp (no cross-warp synchronisation per contact);
// larger ones use more warps so more of each W row is in flight at once.
template <int RowsPerContact, bool Exact>
void launchGaussSeidelSized(
    const float* W, const float* dfree, const float* blocks, const int rows, const double mu, const double tolerance,
    const ConstraintSolveConfig& config, const bool sharedForce, double* lambda, float* lambdaShadow,
    double* lambdaBeforeSweep, ConstraintSolveState* state)
{
    if (rows <= 384)
        launchGaussSeidel<RowsPerContact, Exact, 32>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
    else if (rows <= 1536)
        launchGaussSeidel<RowsPerContact, Exact, 128>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
    else
        launchGaussSeidel<RowsPerContact, Exact, 256>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
}

// lambdaShadow: `rows` floats of device memory, used only above the shared-memory limit.
bool runGaussSeidel(
    const float* W, const float* dfree, float* blocks, const int rows, const int rowsPerContact, const double mu,
    const ConstraintSolveConfig& config, double* lambda, float* lambdaShadow, double* lambdaBeforeSweep,
    ConstraintSolveState* state, ConstraintSolveStats* stats, std::string& diagnostic)
{
    const double tolerance = (config.scaleTolerance && !config.allVerified) ? config.tolerance * rows : config.tolerance;
    const std::size_t sharedBytes = static_cast<std::size_t>(rows) * (sizeof(double) + sizeof(float));
    const std::size_t maxShared = 47 * 1024;   // 48 KB per block, less the kernel's static shared data
    const bool sharedForce = sharedBytes <= maxShared;
    if (!sharedForce && lambdaShadow == nullptr)
    {
        diagnostic = "Gauss-Seidel: no global buffer for a problem above the shared-memory size (" + std::to_string(rows) + " rows).";
        return false;
    }
    const int contacts = rows / rowsPerContact;
    ConstraintEventTimer timer(stats != nullptr);
    constraintContactBlocksKernel<<<std::max(1, std::min((contacts + 255) / 256, 1024)), 256>>>(W, rows, rowsPerContact, blocks);
    ConstraintSolveState initial { 0, 0, 0.0 };
    cudaError_t err = cudaMemcpy(state, &initial, sizeof(initial), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(lambda, 0, sizeof(double) * rows);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Gauss-Seidel setup: ") + cudaGetErrorString(err);
        return false;
    }

    ConstraintSolveState host {};
    while (true)
    {
        if (rowsPerContact == 3)
        {
            if (config.doubleAccumulation)
                launchGaussSeidelSized<3, true>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
            else
                launchGaussSeidelSized<3, false>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
        }
        else
        {
            if (config.doubleAccumulation)
                launchGaussSeidelSized<1, true>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
            else
                launchGaussSeidelSized<1, false>(W, dfree, blocks, rows, mu, tolerance, config, sharedForce, lambda, lambdaShadow, lambdaBeforeSweep, state);
        }
        err = cudaMemcpy(&host, state, sizeof(host), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Gauss-Seidel kernel: ") + cudaGetErrorString(err);
            return false;
        }
        if (host.converged != 0 || host.iterations >= config.maxIterations) break;
    }
    if (stats != nullptr)
    {
        stats->iterations = host.iterations;
        stats->error = host.error;
        stats->converged = host.converged != 0;
        stats->gpuMilliseconds = timer.finish();
    }
    diagnostic.clear();
    return true;
}
} // namespace

bool solveContactConstraints(
    ConstraintWorkspace* ws,
    const ConstraintSolveConfig& config,
    ConstraintSolveStats* stats,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    if (stats != nullptr) *stats = ConstraintSolveStats {};
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    ws->solved = false;
    if (ws->rows == 0) { diagnostic.clear(); return true; }
    if (!ws->complianceReady)
    {
        diagnostic = "assembleContactCompliance must succeed before solving.";
        return false;
    }
    ConstraintEventTimer timer(timings != nullptr);
    const std::size_t rows = static_cast<std::size_t>(ws->rows);
    cudaError_t err = ensureDeviceBuffer(ws->lambda, ws->lambdaCapacity, rows);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->lambdaShadow, ws->lambdaShadowCapacity, rows);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->lambdaBeforeSweep, ws->lambdaBeforeSweepCapacity, rows);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->solveState, ws->solveStateCapacity, 1);
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->contactBlocks, ws->contactBlocksCapacity,
                                                     static_cast<std::size_t>(ws->contacts) * kContactBlockValues);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Solve buffers: ") + cudaGetErrorString(err);
        return false;
    }
    if (!runGaussSeidel(ws->W, ws->dfree, ws->contactBlocks, ws->rows, ws->rowsPerContact, ws->mu, config,
                        ws->lambda, ws->lambdaShadow, ws->lambdaBeforeSweep, ws->solveState, stats, diagnostic))
    {
        return false;
    }
    ws->solved = true;
    if (timings != nullptr) timings->solveMs = timer.finish();
    return true;
}

bool computeContactCorrection(
    ConstraintWorkspace* ws,
    std::vector<float>& deformableCorrection,
    double rigidCorrection[6],
    ConstraintImpulse* impulse,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    for (int e = 0; e < 6; ++e) rigidCorrection[e] = 0.0;
    if (impulse != nullptr) *impulse = ConstraintImpulse {};
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    deformableCorrection.assign(static_cast<std::size_t>(std::max(ws->n, 0)), 0.0f);
    if (ws->rows == 0) { diagnostic.clear(); return true; }
    if (!ws->solved)
    {
        diagnostic = "solveContactConstraints must succeed before the correction.";
        return false;
    }
    ConstraintEventTimer timer(timings != nullptr);
    const int n = ws->n;
    cudaError_t err = ensureDeviceBuffer(ws->deformableRhs, ws->deformableRhsCapacity, static_cast<std::size_t>(n));
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rigidRhs, ws->rigidRhsCapacity, 7);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction buffers: ") + cudaGetErrorString(err);
        return false;
    }
    const int bodies = std::max(ws->bodyCount, 1);
    err = ensureDeviceBuffer(ws->rigidRhs, ws->rigidRhsCapacity, static_cast<std::size_t>(7 * bodies));
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction buffers: ") + cudaGetErrorString(err);
        return false;
    }
    launchFill(ws->deformableRhs, static_cast<std::size_t>(n), 0.0f);
    launchFill(ws->rigidRhs, static_cast<std::size_t>(7 * bodies), 0.0);
    constraintImpulseKernel<<<std::max(1, std::min((ws->rows + 255) / 256, 1024)), 256>>>(
        ws->rows, ws->rowsPerContact, ws->rowVertexGlobal, ws->rowDeformable, ws->rowRigid, ws->rowBody, ws->lambda,
        ws->deformableRhs, ws->rigidRhs);
    if (!solveWithDeformableFactor(ws, 1, ws->deformableRhs, diagnostic)) return false;
    std::vector<double> rigidRhs(static_cast<std::size_t>(7 * bodies));
    err = cudaMemcpy(deformableCorrection.data(), ws->deformableRhs, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(rigidRhs.data(), ws->rigidRhs, sizeof(double) * rigidRhs.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction download: ") + cudaGetErrorString(err);
        return false;
    }
    rigidBodyResults(ws, rigidRhs, rigidCorrection, impulse);
    if (timings != nullptr) timings->correctionMs = timer.finish();
    diagnostic.clear();
    return true;
}

bool useTissueFactor(ConstraintWorkspace* ws, TissueWorkspace* tissue, std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    if (!tissueFactorReady(tissue))
    {
        ws->tissue = nullptr;
        ws->factorized = false;
        diagnostic = "No body-1 factor to use (the tissue step did not factorise its matrix).";
        return false;
    }
    ws->tissue = tissue;
    ws->n = tissueDofCount(tissue);
    ws->factorized = true;
    diagnostic.clear();
    return true;
}

namespace
{
// x = xFree + p dv, v = vFree + q dv, dx = p dv (dv null = no correction).
__global__ void constraintApplyCorrectionKernel(
    const int count, const float* __restrict__ dv, const float* __restrict__ xFree, const float* __restrict__ vFree,
    const float positionFactor, const float velocityFactor, float* __restrict__ x, float* __restrict__ v, float* __restrict__ dx)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const float d = dv != nullptr ? dv[i] : 0.0f;
        x[i] = xFree[i] + positionFactor * d;
        v[i] = vFree[i] + velocityFactor * d;
        if (dx != nullptr) dx[i] = positionFactor * d;
    }
}
} // namespace

bool computeContactCorrectionOnDevice(
    ConstraintWorkspace* ws,
    double rigidCorrection[6],
    ConstraintImpulse* impulse,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    for (int e = 0; e < 6; ++e) rigidCorrection[e] = 0.0;
    if (impulse != nullptr) *impulse = ConstraintImpulse {};
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    ws->deviceCorrectionValid = false;
    if (ws->rows == 0) { diagnostic.clear(); return true; }
    if (!ws->solved)
    {
        diagnostic = "solveContactConstraints must succeed before the correction.";
        return false;
    }
    ConstraintEventTimer timer(timings != nullptr);
    const int n = ws->n;
    cudaError_t err = ensureDeviceBuffer(ws->deformableRhs, ws->deformableRhsCapacity, static_cast<std::size_t>(n));
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->rigidRhs, ws->rigidRhsCapacity, 7);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction buffers: ") + cudaGetErrorString(err);
        return false;
    }
    const int bodies = std::max(ws->bodyCount, 1);
    err = ensureDeviceBuffer(ws->rigidRhs, ws->rigidRhsCapacity, static_cast<std::size_t>(7 * bodies));
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction buffers: ") + cudaGetErrorString(err);
        return false;
    }
    launchFill(ws->deformableRhs, static_cast<std::size_t>(n), 0.0f);
    launchFill(ws->rigidRhs, static_cast<std::size_t>(7 * bodies), 0.0);
    constraintImpulseKernel<<<std::max(1, std::min((ws->rows + 255) / 256, 1024)), 256>>>(
        ws->rows, ws->rowsPerContact, ws->rowVertexGlobal, ws->rowDeformable, ws->rowRigid, ws->rowBody, ws->lambda,
        ws->deformableRhs, ws->rigidRhs);
    if (!solveWithDeformableFactor(ws, 1, ws->deformableRhs, diagnostic)) return false;
    std::vector<double> rigidRhs(static_cast<std::size_t>(7 * bodies));
    err = cudaMemcpy(rigidRhs.data(), ws->rigidRhs, sizeof(double) * rigidRhs.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction download: ") + cudaGetErrorString(err);
        return false;
    }
    ws->deviceCorrectionValid = true;
    rigidBodyResults(ws, rigidRhs, rigidCorrection, impulse);
    if (timings != nullptr) timings->correctionMs = timer.finish();
    diagnostic.clear();
    return true;
}

bool applyContactCorrectionOnDevice(ConstraintWorkspace* ws, const DeviceCorrectionTarget& target, std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    if (target.x == nullptr || target.v == nullptr || target.xFree == nullptr || target.vFree == nullptr || target.vertexCount <= 0)
    {
        diagnostic = "Device correction: missing state pointers.";
        return false;
    }
    const int count = 3 * target.vertexCount;
    const float* dv = nullptr;
    if (target.hostCorrection != nullptr)
    {
        cudaError_t err = ensureDeviceBuffer(ws->hostCorrectionDevice, ws->hostCorrectionCapacity, static_cast<std::size_t>(count));
        if (err == cudaSuccess) err = cudaMemcpy(ws->hostCorrectionDevice, target.hostCorrection, sizeof(float) * count, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Correction upload: ") + cudaGetErrorString(err);
            return false;
        }
        dv = ws->hostCorrectionDevice;
    }
    else if (target.withContacts && ws->deviceCorrectionValid)
    {
        if (ws->n != count)
        {
            diagnostic = "Device correction: body 1 has " + std::to_string(count) + " DOFs, the factor " + std::to_string(ws->n) + ".";
            return false;
        }
        dv = ws->deformableRhs;
    }
    constraintApplyCorrectionKernel<<<std::max(1, std::min((count + 255) / 256, 1024)), 256>>>(
        count, dv, static_cast<const float*>(target.xFree), static_cast<const float*>(target.vFree),
        static_cast<float>(target.positionFactor), static_cast<float>(target.velocityFactor),
        static_cast<float*>(target.x), static_cast<float*>(target.v), static_cast<float*>(target.dx));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction kernel: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

bool downloadDeformableCorrection(ConstraintWorkspace* ws, std::vector<float>& correction, std::string& diagnostic)
{
    correction.clear();
    if (ws == nullptr || !ws->deviceCorrectionValid)
    {
        diagnostic = "No device correction to download.";
        return false;
    }
    correction.resize(static_cast<std::size_t>(ws->n));
    const cudaError_t err = cudaMemcpy(correction.data(), ws->deformableRhs, sizeof(float) * correction.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Correction download: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

bool downloadContactProblem(
    ConstraintWorkspace* ws,
    const bool withCompliance,
    ConstraintProblemSnapshot& snapshot,
    std::string& diagnostic)
{
    snapshot = ConstraintProblemSnapshot {};
    if (ws == nullptr) { diagnostic = "No constraint workspace."; return false; }
    snapshot.contacts = static_cast<std::uint32_t>(ws->contacts);
    snapshot.rows = static_cast<std::uint32_t>(ws->rows);
    snapshot.rowsPerContact = static_cast<std::uint32_t>(ws->rowsPerContact);
    snapshot.touchedVertices = static_cast<std::uint32_t>(ws->touched);
    snapshot.friction = ws->mu;
    if (ws->rows == 0) { diagnostic.clear(); return true; }

    const std::size_t rows = static_cast<std::size_t>(ws->rows);
    const std::size_t contacts = static_cast<std::size_t>(ws->contacts);
    std::vector<float> rowDeformable(rows * 9), rowRigid(rows * 6), dfree(rows), geometry(contacts * kContactGeometryValues), weights(contacts * 3);
    snapshot.rowVertices.resize(rows * 3);
    snapshot.contactVertices.resize(contacts * 3);
    cudaError_t err = cudaMemcpy(snapshot.rowVertices.data(), ws->rowVertexGlobal, sizeof(int) * rows * 3, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(rowDeformable.data(), ws->rowDeformable, sizeof(float) * rows * 9, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(rowRigid.data(), ws->rowRigid, sizeof(float) * rows * 6, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(dfree.data(), ws->dfree, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(geometry.data(), ws->geometry, sizeof(float) * contacts * kContactGeometryValues, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(snapshot.contactVertices.data(), ws->contactVertices, sizeof(int) * contacts * 3, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(weights.data(), ws->contactWeights, sizeof(float) * contacts * 3, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && withCompliance && ws->complianceReady)
    {
        std::vector<float> W(rows * rows);
        err = cudaMemcpy(W.data(), ws->W, sizeof(float) * rows * rows, cudaMemcpyDeviceToHost);
        snapshot.compliance.assign(W.begin(), W.end());
    }
    if (err == cudaSuccess && ws->solved)
    {
        snapshot.lambda.resize(rows);
        err = cudaMemcpy(snapshot.lambda.data(), ws->lambda, sizeof(double) * rows, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Constraint problem download: ") + cudaGetErrorString(err);
        return false;
    }
    snapshot.rowDeformable.assign(rowDeformable.begin(), rowDeformable.end());
    snapshot.rowRigid.assign(rowRigid.begin(), rowRigid.end());
    snapshot.dfree.assign(dfree.begin(), dfree.end());
    snapshot.contactGeometry.assign(geometry.begin(), geometry.end());
    snapshot.contactWeights.assign(weights.begin(), weights.end());
    diagnostic.clear();
    return true;
}

bool solveFrictionProblemOnGpu(
    const int rows,
    const int rowsPerContact,
    const std::vector<double>& W,
    const std::vector<double>& dfree,
    const double mu,
    const ConstraintSolveConfig& config,
    std::vector<double>& lambda,
    ConstraintSolveStats* stats,
    std::string& diagnostic)
{
    lambda.assign(static_cast<std::size_t>(std::max(rows, 0)), 0.0);
    if (rows <= 0) { diagnostic.clear(); return true; }
    if ((rowsPerContact != 1 && rowsPerContact != 3) || rows % rowsPerContact != 0 ||
        W.size() != static_cast<std::size_t>(rows) * rows || dfree.size() != static_cast<std::size_t>(rows))
    {
        diagnostic = "Friction problem has inconsistent sizes.";
        return false;
    }
    std::vector<float> Wf(W.begin(), W.end());
    std::vector<float> dfreeF(dfree.begin(), dfree.end());
    float* dW = nullptr;
    float* dDfree = nullptr;
    float* dBlocks = nullptr;
    double* dLambda = nullptr;
    float* dShadow = nullptr;
    double* dBefore = nullptr;
    ConstraintSolveState* dState = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dW), sizeof(float) * Wf.size());
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dDfree), sizeof(float) * rows);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dBlocks), sizeof(float) * kContactBlockValues * (rows / rowsPerContact));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dLambda), sizeof(double) * rows);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dShadow), sizeof(float) * rows);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dBefore), sizeof(double) * rows);
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&dState), sizeof(ConstraintSolveState));
    if (err == cudaSuccess) err = cudaMemcpy(dW, Wf.data(), sizeof(float) * Wf.size(), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dDfree, dfreeF.data(), sizeof(float) * rows, cudaMemcpyHostToDevice);
    bool ok = err == cudaSuccess;
    if (!ok) diagnostic = std::string("Friction problem upload: ") + cudaGetErrorString(err);
    if (ok) ok = runGaussSeidel(dW, dDfree, dBlocks, rows, rowsPerContact, mu, config, dLambda, dShadow, dBefore, dState, stats, diagnostic);
    if (ok)
    {
        err = cudaMemcpy(lambda.data(), dLambda, sizeof(double) * rows, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { ok = false; diagnostic = std::string("Friction problem download: ") + cudaGetErrorString(err); }
    }
    cudaFree(dW); cudaFree(dDfree); cudaFree(dBlocks); cudaFree(dLambda); cudaFree(dShadow); cudaFree(dBefore); cudaFree(dState);
    return ok;
}

bool computeDenseComplianceOnGpu(
    const HostCsrMatrix& matrix,
    const std::vector<int>& vertices,
    std::vector<double>& compliance,
    ConstraintTimings* timings,
    std::string& diagnostic)
{
    compliance.clear();
    std::string createDiagnostic;
    ConstraintWorkspace* ws = createConstraintWorkspace(createDiagnostic);
    if (ws == nullptr) { diagnostic = createDiagnostic; return false; }
    bool ok = factorizeDeformableSystem(ws, matrix, timings, diagnostic);
    const int m = 3 * static_cast<int>(vertices.size());
    if (ok && m > 0)
    {
        const int n = ws->n;
        cudaError_t err = ensureDeviceBuffer(ws->compactToGlobal, ws->compactToGlobalCapacity, vertices.size());
        if (err == cudaSuccess) err = cudaMemcpy(ws->compactToGlobal, vertices.data(), sizeof(int) * vertices.size(), cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = ensureDeviceBuffer(ws->selector, ws->selectorCapacity, static_cast<std::size_t>(n) * m);
        if (err == cudaSuccess) err = ensureDeviceBuffer(ws->compliance, ws->complianceCapacity, static_cast<std::size_t>(m) * m);
        if (err != cudaSuccess) { ok = false; diagnostic = std::string("Dense compliance buffers: ") + cudaGetErrorString(err); }
        if (ok)
        {
            ConstraintEventTimer timer(timings != nullptr);
            launchFill(ws->selector, static_cast<std::size_t>(n) * m, 0.0f);
            constraintSelectorKernel<<<std::max(1, std::min((static_cast<int>(vertices.size()) + 255) / 256, 1024)), 256>>>(
                ws->compactToGlobal, static_cast<int>(vertices.size()), n, ws->selector);
            const float one = 1.0f;
            const float zero = 0.0f;
            cublasStatus_t status = cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                                CUBLAS_DIAG_NON_UNIT, n, m, &one, ws->dense, n, ws->selector, n);
            if (status == CUBLAS_STATUS_SUCCESS)
                status = cublasSgemm(ws->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, ws->selector, n, ws->selector, n, &zero, ws->compliance, m);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                ok = false;
                diagnostic = std::string("Dense compliance product: cuBLAS ") + cublasStatusName(status);
            }
            else
            {
                std::vector<float> G(static_cast<std::size_t>(m) * m);
                err = cudaMemcpy(G.data(), ws->compliance, sizeof(float) * G.size(), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) { ok = false; diagnostic = std::string("Dense compliance download: ") + cudaGetErrorString(err); }
                else compliance.assign(G.begin(), G.end());
                if (timings != nullptr) timings->complianceMs = timer.finish();
            }
        }
    }
    destroyConstraintWorkspace(ws);
    return ok;
}

} // namespace SofaGpuCollision::backend
