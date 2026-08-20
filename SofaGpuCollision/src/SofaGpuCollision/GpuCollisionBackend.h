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
    double hashGridResetMilliseconds { 0.0 };
    double hashGridPairHashClearMilliseconds { 0.0 };
    double hashGridInsertTissueMilliseconds { 0.0 };
    double hashGridInsertToolMilliseconds { 0.0 };
    double hashGridPairCountMilliseconds { 0.0 };
    double hashGridScanMilliseconds { 0.0 };
    double hashGridGeneratePairsMilliseconds { 0.0 };
    double hashGridProximityCounterClearMilliseconds { 0.0 };
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
    std::uint32_t occupiedSlotCount { 0 };       // compact occupied cell buckets
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

// Experimental "4th way" (simple direct-bucket spatial hash). Like
// computeHashPrefixSumProximityContacts it replaces the dense grid with an
// open-addressing spatial hash of cells (MurmurHash3 fmix64 + linear probing),
// but it stores triangles DIRECTLY into per-cell buckets in a single insert pass
// — no separate mark/compact/fill build (each table slot IS a bucket). It is
// best-effort: a cell holding more than maxTissue/maxToolTrianglesPerCell
// entries drops the surplus (reported in HashPrefixSumStats::bucketOverflowCount).
// Reuses the same FBP narrow kernel, candidate dedup and CUDA graph as the other
// paths. Independent workspace from the optimised hash path.
SOFA_GPU_COLLISION_API bool computeSimpleHashProximityContacts(
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

// Experimental "5th way" (sorted-grid / tiled-binning broad cull, Green's
// particle-grid method adapted to triangles). Each triangle is expanded into
// (cellKey, triangleId) incidences (key = cellId*2 + meshTag, so tissue sorts
// before tool within a cell), the incidences are sorted by key, and the scanned
// per-key histogram gives every cell's contiguous run — no per-cell capacity
// caps, so NO best-effort drops (unlike the simple hash). Candidate pairs are
// generated one block per mixed cell with the tool run staged in shared memory.
struct SortedGridConfig
{
    // Sort engine: false (default) = hand-rolled one-pass counting sort
    // (histogram -> single-block chained scan -> scatter; the scanned histogram
    // doubles as the cell-run starts, and the repo stays CUB-free on this
    // route). true = cub::DeviceRadixSort::SortPairs over the padded incidence
    // buffer (textbook route; re-adds the CUB dependency).
    bool useCubRadixSort { false };
    // Dedup: false (default) = home-cell exactly-once emission (a pair is
    // emitted only from the cell containing the min corner of its two AABBs'
    // intersection — no dedup hash table at all). true = the same atomicCAS
    // pair-hash + touched-slot machinery as the hash ways.
    bool usePairHashDedup { false };
    // Incidence buffer capacity = scale * (firstTris + secondTris). Overflowed
    // incidences are dropped and counted. Default 16: the SOFA hash scene's
    // triangles overlap ~8.6 cells on average (8x measured 8,080 dropped
    // incidences there, 2026-07-03), so 16x gives ~2x headroom. Watch the
    // incidenceOverflowCount stat when changing scenes.
    std::uint32_t incidenceCapacityScale { 16 };
};

struct SortedGridStats
{
    std::uint32_t binCount { 0 };                 // 2 * cellCount (cell x meshTag)
    std::uint32_t incidenceCount { 0 };           // (cell, triangle) entries this frame
    std::uint32_t mixedCellCount { 0 };           // cells holding both tissue and tool
    std::uint32_t uniquePairCount { 0 };          // candidate pairs after dedup
    std::uint32_t incidenceOverflowCount { 0 };   // dropped incidences (capacity)
    std::uint32_t pairOverflowCount { 0 };        // dropped pairs (maxCandidatePairs)
};

SOFA_GPU_COLLISION_API bool computeSortedGridProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const SortedGridConfig& sortedConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    SortedGridStats* sortedStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

// Experimental "6th way" (big-cell / small-cell FUSED generation + narrow
// phase). The small cells are the ordinary dense-grid cells; a big cell groups
// bigCellFactor^3 of them. A per-big-cell CSR table of (triangle, small cell)
// entries is built (count -> scan -> fill; the scan input is factor^3 smaller
// than way 5's), then ONE kernel runs one block per mixed big cell: it stages
// the tool side into shared memory (ids + AABBs + all three vertices),
// organizes it into per-small-cell runs with an in-shared counting sort, and
// sweeps the tissue entries — AABB pre-cull + way-5's home-cell exactly-once
// rule at small-cell granularity (so the surviving pair set is identical to
// way 5's), then the identical FBP closest-feature math inline. There is no
// intermediate candidate-pair list and no separate FBP launch: contacts are
// appended straight to the single output array.
struct BigCellConfig
{
    // Big cell edge length in small cells. Power of two, at most 4 (a local
    // small-cell id is packed into 6 bits of each entry). Default 2, measured
    // 2026-07-12: factor 4 collapses block-level parallelism (48 mixed big
    // cells on the 200k bench -> 1.52 ms) while factor 2 keeps the GPU fed
    // (1.09 ms; best or tied-best on both the 80k and 200k benches).
    std::uint32_t bigCellFactor { 2 };
    // Tool entries staged in shared memory per chunk (1..256). Oversized big
    // cells loop over chunks; forcing this low exercises the chunk path.
    std::uint32_t toolTileCapacity { 256 };
    // Entry buffer capacity = scale * (firstTris + secondTris), same sizing
    // rationale as SortedGridConfig::incidenceCapacityScale.
    std::uint32_t entryCapacityScale { 16 };
    // false (default) = CSR bucket lists (count -> scan -> fill; the big-cell
    // id is a perfect direct index). true = literal per-big-cell open-
    // addressing hash multi-map build, kept as a toggle so the two build
    // strategies can be A/B'd with numbers.
    bool useHashTableBuild { false };
    // Hash-build only: open-addressing slots per (big cell, side) region.
    // Rounded up to a power of two, clamped to [64, 4096]. A region that fills
    // up drops entries (BigCellStats::buildOverflowCount) — the fixed-capacity
    // memory waste is inherent to the per-big-cell hash layout and is part of
    // what the A/B measures.
    std::uint32_t hashSlotsPerBigCell { 1024 };
    // CSR-build shared-memory privatization A/B ("populate in shared, merge
    // later" — each block takes a FIXED chunk of triangles, builds its chunk's
    // contribution in shared memory, then merges to the global histogram/CSR
    // with per-bin instead of per-entry global atomics):
    //   0 = off (direct global atomics, the baseline)
    //   1 = shared HASH TABLE (insert-or-count keyed by bin, staged entries)
    //   2 = shared SORTED LIST (stage entries, in-shared bitonic sort by bin,
    //       one reservation per run, run-writer scatter)
    // Entries that overflow the shared staging fall back to the direct global
    // path (BigCellStats::sharedSpillCount) — contacts are identical either
    // way. Mutually exclusive with useHashTableBuild.
    // DEFAULT 1 (shared hash) as of 2026-07-15, measured: -12% at 80k / -26%
    // at 200k full-pipeline vs direct atomics (and the block-grouped entry
    // order speeds the fused consumer too); the shared SORTED LIST lost ~2.6x
    // (the in-shared bitonic sort costs more than every atomic it avoids).
    std::uint32_t sharedBuildMode { 1 };
    // Diagnostic-only instrumentation for the fused consumer. This launches a
    // separately compiled diagnostic kernel that samples clock64() around
    // major block phases and pair filters and counts every rejection stage.
    // It intentionally perturbs execution, disables CUDA-graph replay, and is
    // therefore OFF by default. CUDA-event pipeline timings remain available
    // through DenseGridConfig::detailedProfiling without enabling this flag.
    bool profileFusedInternals { false };
};

struct BigCellStats
{
    std::uint32_t bigCellCount { 0 };
    std::uint32_t entryCount { 0 };            // (small cell, triangle) entries this frame
    std::uint32_t mixedBigCellCount { 0 };     // big cells holding both tissue and tool
    std::uint32_t pairsTestedCount { 0 };      // pairs surviving AABB + home-cell (parity: way-5 uniquePairCount)
    std::uint32_t entryOverflowCount { 0 };    // dropped entries (capacity)
    std::uint32_t buildOverflowCount { 0 };    // hash-build slot-region overflow (0 on CSR)
    std::uint32_t sharedSpillCount { 0 };      // shared-build entries that fell back to the direct global path

    // CUDA-event elapsed times. Populated when detailedProfiling is enabled
    // (profileFusedInternals implies it). Values are serialized stream time,
    // not CPU wall time, and include event-record overhead between stages.
    double resetMilliseconds { 0.0 };
    double buildClearMilliseconds { 0.0 };     // hash build only
    double eventMarkerGapMilliseconds { 0.0 }; // CSR no-op marker: estimates one event boundary cost
    double firstCountOrInsertMilliseconds { 0.0 };
    double secondCountOrInsertMilliseconds { 0.0 };
    double scanMilliseconds { 0.0 };           // CSR only; ~0 for hash build
    double firstFillMilliseconds { 0.0 };      // CSR only; ~0 for hash build
    double secondFillMilliseconds { 0.0 };     // CSR only; ~0 for hash build
    double mixedCellBuildMilliseconds { 0.0 };
    double proximityResetMilliseconds { 0.0 };
    double fusedKernelMilliseconds { 0.0 };
    double totalPipelineMilliseconds { 0.0 };

    // Runtime resource/occupancy data for the production fused specialization.
    std::uint32_t fusedGridBlocks { 0 };
    std::uint32_t fusedBlockThreads { 0 };
    std::uint32_t fusedRegistersPerThread { 0 };
    std::uint32_t fusedStaticSharedBytes { 0 };
    std::uint32_t fusedLocalBytesPerThread { 0 };
    std::uint32_t fusedMaxThreadsPerBlock { 0 };
    std::uint32_t fusedActiveBlocksPerSm { 0 };
    std::uint32_t deviceMultiprocessorCount { 0 };
    std::uint32_t deviceMaxThreadsPerSm { 0 };
    std::uint32_t deviceWarpSize { 0 };
    std::uint32_t deviceClockRateKHz { 0 };
    double fusedTheoreticalOccupancyPercent { 0.0 };
    std::uint32_t profiledRegistersPerThread { 0 };
    std::uint32_t profiledStaticSharedBytes { 0 };
    std::uint32_t profiledLocalBytesPerThread { 0 };
    std::uint32_t profiledActiveBlocksPerSm { 0 };
    double profiledTheoreticalOccupancyPercent { 0.0 };

    // Diagnostic specialization counters. Cycle fields are sums over blocks
    // (major phases) or participating threads (pair filters); they are work
    // attribution, not elapsed time, because many blocks/threads overlap.
    std::uint64_t profiledBigCellIterations { 0 };
    std::uint64_t profiledTileIterations { 0 };
    std::uint64_t profiledToolEntriesStaged { 0 };
    std::uint64_t profiledTissueEntriesVisited { 0 };
    std::uint64_t profiledSmallCellPairVisits { 0 };
    std::uint64_t profiledInflatedAabbRejects { 0 };
    std::uint64_t profiledHomeCellRejects { 0 };
    std::uint64_t profiledRawAabbRejects { 0 };
    std::uint64_t profiledFbpCalls { 0 };
    std::uint64_t profiledFbpNoContact { 0 };
    std::uint64_t profiledTileSetupBlockCycles { 0 };
    std::uint64_t profiledBinPrefixBlockCycles { 0 };
    std::uint64_t profiledToolGatherBlockCycles { 0 };
    std::uint64_t profiledTissueSweepBlockCycles { 0 };
    std::uint64_t profiledInflatedAabbThreadCycles { 0 };
    std::uint64_t profiledHomeCellThreadCycles { 0 };
    std::uint64_t profiledRawAabbThreadCycles { 0 };
    std::uint64_t profiledFbpThreadCycles { 0 };
    std::uint64_t profiledToolSortScatterThreadCycles { 0 };
    std::uint64_t profiledToolDataLoadThreadCycles { 0 };
    std::uint64_t profiledContactEmitThreadCycles { 0 };
};

// ============================================================================
// Contact consumption (Tier 1, 2026-07-15)
// ----------------------------------------------------------------------------
// Every proximity entry point above leaves its contacts in a DEVICE buffer and
// records a handle to it. `accumulateContactPenaltyForces` turns that buffer
// into forces on the two surfaces' vertices WITHOUT the contacts ever reaching
// the host: one kernel decodes each contact's feature (VF/FV/EE) into per-
// triangle-vertex weights, evaluates a proximity penalty law, and scatter-adds
// into the caller's device force vectors.
//
// The caller passes DEVICE pointers to SOFA's CudaVec3f force/velocity vectors
// (obtained via CudaVector::deviceWrite()/deviceRead() — never the host
// accessors, which would copy the state back and defeat the whole point).
//
// The contact struct layout stays private to the CUDA translation unit (the
// §7 API-boundary rule): callers never see a contact, only the resulting forces.
// ============================================================================
struct ContactPenaltyConfig
{
    // Penalty law: F = stiffness * max(0, contactDistance - distance)
    //                - damping * (relative normal velocity), clamped to >= 0.
    // contactDistance must match the value the contacts were generated with,
    // otherwise the force turns on at the wrong separation.
    float stiffness { 1000.0f };
    float damping { 0.0f };
    float contactDistance { 0.03f };
};

struct ContactPenaltyStats
{
    std::uint32_t contactCount { 0 };        // contacts in the device buffer
    std::uint32_t activeContactCount { 0 };  // those actually producing force
};

// Accumulate penalty forces from the last proximity result for (firstSurfaceId,
// secondSurfaceId). Returns false (with a diagnostic) if no handle was recorded
// for that pair this frame — e.g. the pair produced no contacts, or a different
// pair ran last.
//
// deviceFirstForces / deviceSecondForces: Vec3f* device pointers, accumulated
// into (never overwritten). Velocity pointers may be null when damping == 0.
SOFA_GPU_COLLISION_API bool accumulateContactPenaltyForces(
    const ContactPenaltyConfig& config,
    std::uint64_t firstSurfaceId,
    std::uint64_t secondSurfaceId,
    void* deviceFirstForces,
    void* deviceSecondForces,
    const void* deviceFirstVelocities,
    const void* deviceSecondVelocities,
    ContactPenaltyStats* stats,
    std::string& diagnostic);

// Stiffness-times-dx for implicit integration (SOFA's addDForce). Applies
// K = kFactor * stiffness * (n outer n) for each active contact, projected
// through the same per-vertex weights.
SOFA_GPU_COLLISION_API bool accumulateContactPenaltyDForces(
    const ContactPenaltyConfig& config,
    std::uint64_t firstSurfaceId,
    std::uint64_t secondSurfaceId,
    float kFactor,
    void* deviceFirstDForces,
    void* deviceSecondDForces,
    const void* deviceFirstDx,
    const void* deviceSecondDx,
    std::string& diagnostic);

// Self-validation for the contact-force path (Gates 1 and 2). Runs the GPU
// penalty kernel into freshly allocated device force vectors, recomputes the
// same forces on the host from the downloaded contacts, and reports:
//   * maxAbsErrorVsReference  — GPU vs CPU reference (Gate 1: must be ~0)
//   * netForceMagnitude       — sum of ALL forces over both bodies
//                               (Gate 2, Newton's third law: must be ~0)
// Only used by tests/benchmarks: it synchronises and allocates, so it is not on
// any simulation path.
struct ContactForceValidation
{
    std::uint32_t contactCount { 0 };
    std::uint32_t activeContactCount { 0 };
    double maxAbsErrorVsReference { 0.0 };
    double maxReferenceMagnitude { 0.0 };
    double netForceMagnitude { 0.0 };      // |sum of every force vector|
    double totalForceMagnitude { 0.0 };    // sum of |force| — the scale to judge the two above against
    // Gate 1b — INDEPENDENT validation of the barycentric-weight decoding.
    // Gate 1 alone is partly circular: the host reference re-uses the same
    // VF/FV/EE weight convention as the kernel, so a wrong convention would pass
    // both. This instead reconstructs each contact point from the decoded
    // weights and the triangle's vertex positions, and compares it against the
    // pointOnFirst / pointOnSecond that the COLLISION kernel computed by a
    // completely different route (closest-feature math). Agreement means the
    // weights genuinely address the right vertices with the right coefficients.
    double maxContactPointError { 0.0 };
    double maxWeightSumError { 0.0 };      // weights must be a partition of unity
    bool contactPointCheckRan { false };   // false when host positions were unavailable
};

SOFA_GPU_COLLISION_API bool validateContactPenaltyForces(
    const ContactPenaltyConfig& config,
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    ContactForceValidation* validation,
    std::string& diagnostic);

SOFA_GPU_COLLISION_API bool computeBigCellFusedProximityContacts(
    const TriangleIndexedSurface& firstSurface,
    const TriangleIndexedSurface& secondSurface,
    const DenseGridConfig& gridConfig,
    const BigCellConfig& bigConfig,
    const FeatureBasedProximityConfig& proximityConfig,
    std::vector<ProximityContact>& contacts,
    FeatureBasedProximityStats* proximityStats,
    BigCellStats* bigStats,
    std::string& diagnostic,
    BackendExecutionStats* executionStats = nullptr);

} // namespace SofaGpuCollision::backend
