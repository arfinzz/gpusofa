#pragma once

#include <SofaGpuCollision/config.h>

#include <string>
#include <cstdint>
#include <utility>
#include <vector>

namespace sofa::core
{
class CollisionModel;
}

namespace SofaGpuCollision::backend
{

using CollisionModelPair = std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>;
using BroadPhaseIndexPair = std::pair<std::uint32_t, std::uint32_t>;

struct AxisAlignedBoundingBox
{
    float minX { 0.0f };
    float minY { 0.0f };
    float minZ { 0.0f };
    float maxX { 0.0f };
    float maxY { 0.0f };
    float maxZ { 0.0f };
};

struct NarrowPhaseTreePair
{
    std::vector<AxisAlignedBoundingBox> firstTree;
    std::vector<AxisAlignedBoundingBox> secondTree;
};

struct BackendStatus
{
    bool available { false };
    std::string message;
};

struct BackendExecutionStats
{
    double gpuKernelMilliseconds { 0.0 };
    double hostPreparationMilliseconds { 0.0 };
    double backendTrianglePackMilliseconds { 0.0 };
    double hostToDeviceMilliseconds { 0.0 };
    double deviceAllocationMilliseconds { 0.0 };
    double denseGridClearMilliseconds { 0.0 };
    double denseGridCounterClearMilliseconds { 0.0 };
    double denseGridTissueAabbMilliseconds { 0.0 };
    double denseGridToolAabbMilliseconds { 0.0 };
    double denseGridInsertTissueMilliseconds { 0.0 };
    double denseGridInsertToolMilliseconds { 0.0 };
    double denseGridGeneratePairsMilliseconds { 0.0 };
    double denseGridCandidateReadbackMilliseconds { 0.0 };
    double denseGridSortUniqueMilliseconds { 0.0 };
    double denseGridSortUniqueHostMilliseconds { 0.0 };
    double denseGridExactContactMilliseconds { 0.0 };
    double denseGridContactCountReadbackMilliseconds { 0.0 };
    double denseGridContactDownloadMilliseconds { 0.0 };
    double featureBasedProximityKernelMilliseconds { 0.0 };  // FBP narrow pass time (separate from exact-contact)
    std::uint64_t hostToDeviceBytes { 0 };
    std::uint64_t deviceToHostBytes { 0 };
    std::uint64_t deviceAllocationBytes { 0 };
    std::uint32_t kernelLaunchCount { 0 };
    std::uint32_t cudaMemsetCount { 0 };
    std::uint32_t workspaceResizeCount { 0 };
    std::uint32_t inputPrimitiveCount { 0 };
    std::uint32_t outputPairCount { 0 };
    std::uint32_t outputCandidateCount { 0 };
    std::uint32_t outputContactCount { 0 };
    std::uint32_t rawCandidateCount { 0 };
    std::uint32_t uniqueCandidateCount { 0 };
    std::uint32_t hostValidatedUniqueCandidateCount { 0 };
    std::uint32_t gridCellCount { 0 };
    std::uint32_t activeMixedCellCount { 0 };
    std::uint32_t tissueInsertCount { 0 };
    std::uint32_t toolInsertCount { 0 };
    std::uint32_t maxTissueCellOccupancy { 0 };
    std::uint32_t maxToolCellOccupancy { 0 };
    std::uint32_t overflowCount { 0 };
    std::uint32_t hashDedupeProbeOverflowCount { 0 };
    std::uint32_t vfContactCount { 0 };  // feature-based proximity per-class breakdown
    std::uint32_t fvContactCount { 0 };
    std::uint32_t eeContactCount { 0 };
};

struct NarrowPhaseContactCandidate
{
    std::uint32_t pairIndex { 0 };
    std::uint32_t firstLeafIndex { 0 };
    std::uint32_t secondLeafIndex { 0 };
};

struct TriangleVertex
{
    float x { 0.0f };
    float y { 0.0f };
    float z { 0.0f };
};

struct TrianglePrimitive
{
    TriangleVertex p0;
    TriangleVertex p1;
    TriangleVertex p2;
    std::uint32_t triangleIndex { 0 };
};

struct TriangleIndexedSurface
{
    const TriangleVertex* positions { nullptr };
    const TriangleVertex* devicePositions { nullptr };
    std::uint32_t vertexCount { 0 };
    const std::uint32_t* triangleIndices { nullptr };
    std::uint32_t triangleCount { 0 };
    std::uint64_t surfaceId { 0 };
    std::uint64_t topologyVersion { 0 };
};

struct ExactContact
{
    std::uint32_t firstTriangleIndex { 0 };
    std::uint32_t secondTriangleIndex { 0 };
    TriangleVertex pointOnFirst;
    TriangleVertex pointOnSecond;
    TriangleVertex normal;
    float signedDistance { 0.0f };
};

// Feature classifier for proximity contacts. The contact comes from the closest
// pair of features between two triangles. VF means "vertex of A vs face of B",
// FV means "face of A vs vertex of B", EE means edge-edge. The barycentrics are
// always written in the (a0, a1, a2, b0, b1, b2) order, even when one side is a
// vertex (the unused barys are zero).
enum class ProximityFeatureKind : std::uint8_t
{
    VertexFace = 0,  // vertex of "first" surface vs face of "second"
    FaceVertex = 1,  // face of "first" vs vertex of "second"
    EdgeEdge   = 2,  // closest points on two edges
};

struct ProximityContact
{
    std::uint32_t firstPrimitiveIndex { 0 };   // triangle id on the "first" side; for vertex-triangle, vertex id
    std::uint32_t secondPrimitiveIndex { 0 };  // triangle id on the "second" side
    ProximityFeatureKind featureKind { ProximityFeatureKind::VertexFace };
    std::uint8_t firstFeatureLocalIndex { 0 };   // 0..2 vertex/edge index on first triangle
    std::uint8_t secondFeatureLocalIndex { 0 };  // 0..2 vertex/edge index on second triangle
    std::uint8_t reserved { 0 };
    // Barycentric weights for the closest-point pair, suitable for a constraint solver.
    // For VF: firstBarys = (1,0,0) selecting the vertex; secondBarys = (w0,w1,w2) on the face.
    // For FV: firstBarys = (w0,w1,w2); secondBarys = (1,0,0) on the vertex.
    // For EE: firstBarys = (1-s, s, 0) along the chosen edge endpoints; secondBarys = (1-t, t, 0).
    float firstBarycentrics[3] { 1.0f, 0.0f, 0.0f };
    float secondBarycentrics[3] { 1.0f, 0.0f, 0.0f };
    TriangleVertex pointOnFirst;   // world-space closest point on first feature
    TriangleVertex pointOnSecond;  // world-space closest point on second feature
    TriangleVertex normal;         // unit vector from pointOnFirst to pointOnSecond when separated, flipped when penetrating
    float signedDistance { 0.0f }; // negative when interpenetrating along the smallest-feature normal
};

// Vertex cloud input for the generic vertex-triangle path. The cloud is treated
// as a list of points (no edges, no faces). devicePositions takes precedence
// over positions when non-null. Indices into the cloud are emitted in
// firstPrimitiveIndex of the resulting ProximityContact.
struct PointCloudSurface
{
    const TriangleVertex* positions { nullptr };
    const TriangleVertex* devicePositions { nullptr };
    std::uint32_t pointCount { 0 };
    std::uint64_t surfaceId { 0 };
    std::uint64_t topologyVersion { 0 };
};

struct DenseGridConfig
{
    float gridMinX { -8.0f };
    float gridMinY { -2.0f };
    float gridMinZ { -8.0f };
    float gridMaxX { 8.0f };
    float gridMaxY { 4.0f };
    float gridMaxZ { 8.0f };
    std::uint32_t gridResolutionX { 64 };
    std::uint32_t gridResolutionY { 24 };
    std::uint32_t gridResolutionZ { 64 };
    float contactDistance { 0.03f };
    std::uint32_t maxTissueTrianglesPerCell { 64 };
    std::uint32_t maxToolTrianglesPerCell { 32 };
    std::uint32_t maxCandidatePairs { 1000000 };
    bool deduplicatePairs { true };
    bool copyContactsToHost { true };
    bool detailedProfiling { false };
    bool usePinnedHostStaging { true };
    bool useGpuHashDedupe { true };
    bool canonicalPairEmission { false };
    bool validateDedupeOnHost { false };
    bool readCountersWhenContactsStayOnDevice { true };
    bool computeDeviceContactsWhenContactsStayOnDevice { false };
    bool compactActiveCells { false };
    bool batchTriangleInsert { false };
    // Phase 15: generate candidate pairs over tool-occupied (mixed) cells only.
    // The active-cell list is built during the tool insert (no separate scan),
    // then candidate generation launches a small fixed grid over that list
    // instead of one block per grid cell. Distinct from compactActiveCells
    // (which scans all cells and regressed). Mutually exclusive with
    // batchTriangleInsert (which breaks the tissue-before-tool ordering this
    // path relies on).
    //
    // DEFAULT ON as of 2026-05-25: measured 4.3x faster on one-tissue/one-blade
    // and 1.08x on large-tissue/blade, with bit-identical contact output and
    // zero overflow on both. The path is structurally never worse than the
    // all-cells generator (active list <= cellCount, grid-strided).
    bool useToolActiveCellGeneration { true };
};

// Configuration for the feature-based proximity (VF/EE) narrow phase. Reuses
// the dense-grid broad cull (so the existing DenseGridConfig drives the cell
// machinery) and adds proximity-specific knobs on top.
struct FeatureBasedProximityConfig
{
    float contactDistance { 0.03f };       // proximity threshold; pairs farther than this are dropped
    bool emitOnePerPair { true };          // keep only the closest feature pair per triangle pair
    bool computeBarycentrics { true };     // populate firstBarycentrics/secondBarycentrics
    bool keepContactsOnDevice { true };    // skip D2H of contact array (detection-only)
    bool readContactCounter { false };     // copy back the contact count for validation/profiling
    std::uint32_t maxContacts { 1000000 }; // contact-output capacity
};

struct FeatureBasedProximityStats
{
    std::uint32_t candidatePairCount { 0 };
    std::uint32_t emittedContactCount { 0 };
    std::uint32_t vfContactCount { 0 };
    std::uint32_t fvContactCount { 0 };
    std::uint32_t eeContactCount { 0 };
    float minSignedDistance { 0.0f };
};

// Experimental (experiment/hash-prefixsum-broadphase branch): configuration for
// the spatial-hash + prefix-sum work-expansion broad cull. This is an ALTERNATIVE
// to the dense-grid broad cull, intended for large-tissue + large-tool scenes
// where the dense grid's fixed cell array wastes memory and the block-per-cell
// candidate generation load-balances poorly. The narrow phase is unchanged (it
// reuses the same feature-based proximity kernel on the generated pairs).
struct HashPrefixSumConfig
{
    // Number of slots in the open-addressing spatial hash table. 0 = auto-derive
    // from the input sizes. Must be a power of two when non-zero; the backend
    // rounds up to the next power of two regardless.
    std::uint32_t hashTableSize { 0 };
    // Maximum linear-probe steps before an insertion gives up (counts as overflow).
    std::uint32_t maxProbe { 64 };
};

struct HashPrefixSumStats
{
    std::uint32_t hashTableSize { 0 };
    std::uint32_t occupiedSlotCount { 0 };       // hash slots that ended up non-empty
    std::uint32_t rawPairCount { 0 };            // total pairs before dedup (the prefix-sum total)
    std::uint32_t uniquePairCount { 0 };         // after dedup
    std::uint32_t hashProbeOverflowCount { 0 };  // insertions that exceeded maxProbe
    std::uint32_t bucketOverflowCount { 0 };     // triangles dropped from full slot buckets
};

SOFA_GPU_COLLISION_API BackendStatus probe();

SOFA_GPU_COLLISION_API bool computeBroadPhasePairs(
    const std::vector<AxisAlignedBoundingBox>& boxes,
    std::vector<BroadPhaseIndexPair>& pairs,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

SOFA_GPU_COLLISION_API bool prefilterNarrowPhasePairs(
    const std::vector<NarrowPhaseTreePair>& inputTrees,
    std::vector<std::uint32_t>& survivingPairIndices,
    std::vector<NarrowPhaseContactCandidate>* contactCandidates,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

SOFA_GPU_COLLISION_API bool computeExactTriangleContacts(
    const std::vector<TrianglePrimitive>& firstTriangles,
    const std::vector<TrianglePrimitive>& secondTriangles,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

SOFA_GPU_COLLISION_API bool computeDenseGridTriangleContacts(
    const std::vector<TrianglePrimitive>& tissueTriangles,
    const std::vector<TrianglePrimitive>& toolTriangles,
    const DenseGridConfig& config,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

SOFA_GPU_COLLISION_API bool computeDenseGridIndexedTriangleContacts(
    const TriangleIndexedSurface& tissueSurface,
    const TriangleIndexedSurface& toolSurface,
    const DenseGridConfig& config,
    std::vector<ExactContact>& contacts,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

// Feature-based proximity (VF + EE) for triangle-triangle. The dense grid is
// used as a broad cull (same DenseGridConfig knobs as the exact-contact path).
// Each surviving candidate pair runs 6 VF + 9 EE tests; the closest feature is
// kept when its distance is below FeatureBasedProximityConfig.contactDistance.
// Output is a flat ProximityContact array suitable for direct consumption by a
// CUDA constraint solver (barycentrics + signed distance + normal).
SOFA_GPU_COLLISION_API bool computeFeatureBasedProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

// Generic vertex-triangle proximity. The point cloud is broad-culled against
// the triangle mesh via the same dense grid (point AABB is the inflated
// vertex). Each candidate (vertex, triangle) pair runs one closest-point-on-
// triangle test. Used for two front-ends: (a) self-collision of a single mesh
// (cloud = mesh vertex set, triangleSurface = same mesh), (b) cross-model
// PointCollisionModel vs CudaTriangleCollisionModel.
SOFA_GPU_COLLISION_API bool computeFeatureBasedVertexTriangleContacts(
    const PointCloudSurface& pointCloud,
    const TriangleIndexedSurface& triangleSurface,
    const DenseGridConfig& gridConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

// Experimental (experiment/hash-prefixsum-broadphase branch): triangle-triangle
// feature-based proximity using a spatial-hash + prefix-sum broad cull instead
// of the dense grid. The grid geometry (bounds, resolution, contact distance,
// per-cell capacities) still comes from DenseGridConfig — the cells are the same;
// only the storage (hash table vs dense array) and the candidate generation
// (prefix-sum work expansion vs block-per-cell) differ. The narrow phase reuses
// the same FBP kernel, so contact output is identical to
// computeFeatureBasedProximityContacts. Intended for large-tissue + large-tool
// scenes. Independent of DenseGridWorkspace (its own HashGridWorkspace).
SOFA_GPU_COLLISION_API bool computeHashPrefixSumProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const HashPrefixSumConfig& hashConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    HashPrefixSumStats* hashStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

} // namespace SofaGpuCollision::backend
