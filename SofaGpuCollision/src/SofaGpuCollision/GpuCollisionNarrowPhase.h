#pragma once

#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/config.h>

#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>
#include <sofa/core/objectmodel/Data.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SofaGpuCollision
{

class SOFA_GPU_COLLISION_API GpuCollisionNarrowPhase
    : public sofa::component::collision::detection::algorithm::BVHNarrowPhase
{
public:
    SOFA_CLASS(GpuCollisionNarrowPhase, sofa::component::collision::detection::algorithm::BVHNarrowPhase);

    GpuCollisionNarrowPhase();
    ~GpuCollisionNarrowPhase() override = default;

    void init() override;
    void beginNarrowPhase() override;
    void addCollisionPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& cmPair) override;
    void endNarrowPhase() override;

private:
    using DataBool = sofa::core::objectmodel::Data<bool>;
    using DataDouble = sofa::core::objectmodel::Data<double>;
    using DataUInt = sofa::core::objectmodel::Data<unsigned int>;
    using CollisionModelPair = std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>;

    DataBool d_enableGpu;
    DataBool d_allowCpuFallback;
    DataBool d_logBackendStatus;
    DataBool d_useDenseGrid;
    DataBool d_deduplicatePairs;
    DataBool d_copyContactsToHost;
    DataBool d_detailedProfiling;
    DataBool d_usePinnedHostStaging;
    DataBool d_useGpuHashDedupe;
    DataBool d_canonicalPairEmission;
    DataBool d_cacheTriangleTopology;
    DataBool d_useIndexedDenseGridInput;
    DataBool d_useDirectDevicePositions;
    DataBool d_validateTriangleTopologyCache;
    DataBool d_readCountersWhenContactsStayOnDevice;
    DataBool d_computeDeviceContactsWhenContactsStayOnDevice;
    DataBool d_compactActiveCells;
    DataBool d_batchTriangleInsert;
    DataBool d_useToolActiveCellGeneration;
    DataBool d_useHashPrefixSumGeneration;
    DataUInt d_hashTableSize;
    DataBool d_useFeatureBasedProximity;
    DataBool d_useVertexTriangleProximity;       // routes self-collision pairs through computeFeatureBasedVertexTriangleContacts
    DataBool d_proximityComputeBarycentrics;
    DataBool d_proximityReadContactCounter;
    DataBool d_proximityKeepContactsOnDevice;
    DataUInt d_proximityMaxContacts;
    DataUInt d_proximityCounterReadbackInterval;  // 0 = never except when explicitly requested; N = every Nth frame
    DataUInt d_minGpuPairCount;
    DataDouble d_gridMinX;
    DataDouble d_gridMinY;
    DataDouble d_gridMinZ;
    DataDouble d_gridMaxX;
    DataDouble d_gridMaxY;
    DataDouble d_gridMaxZ;
    DataUInt d_gridResolutionX;
    DataUInt d_gridResolutionY;
    DataUInt d_gridResolutionZ;
    DataDouble d_contactDistance;
    DataUInt d_maxTissueTrianglesPerCell;
    DataUInt d_maxToolTrianglesPerCell;
    DataUInt d_maxCandidatePairs;

#ifdef SOFAGPUCOLLISION_WITH_CUDA
    struct CachedTriangleTopology
    {
        std::size_t triangleCount { 0 };
        std::size_t vertexCount { 0 };
        std::uint64_t topologyHash { 0 };
        std::uint64_t hitCount { 0 };
        std::uint64_t missCount { 0 };
        bool seenThisFrame { false };
        std::vector<std::uint32_t> triangleIndices;
        std::vector<backend::TriangleVertex> positionScratch;
        std::vector<backend::TrianglePrimitive> triangleScratch;
    };

    bool extractCudaTriangles(
        sofa::core::CollisionModel* collisionModel,
        sofa::core::CollisionModel*& outModel,
        const std::vector<backend::TrianglePrimitive>*& outTriangles);
    bool extractCudaIndexedSurface(
        sofa::core::CollisionModel* collisionModel,
        sofa::core::CollisionModel*& outModel,
        backend::TriangleIndexedSurface& outSurface);
    // Self-collision helper: extracts the indexed surface AND a point cloud that
    // shares the same device positions and the same surfaceId. The matching
    // surfaceId triggers own-corner exclusion in featureBasedVertexTriangleProximityKernel
    // so that a vertex isn't flagged as colliding with its own adjacent faces.
    bool extractCudaSelfCollisionSurfaces(
        sofa::core::CollisionModel* collisionModel,
        sofa::core::CollisionModel*& outModel,
        backend::TriangleIndexedSurface& outTriangleSurface,
        backend::PointCloudSurface& outPointCloud);
    // Cross-model point-cloud extractor: detects CudaPointCollisionModel and
    // pulls its device positions directly from the underlying MechanicalState.
    // The resulting surfaceId is the point-model pointer so it differs from
    // any TriangleIndexedSurface — the kernel's own-corner exclusion stays off.
    bool extractCudaPointCloud(
        sofa::core::CollisionModel* collisionModel,
        sofa::core::CollisionModel*& outModel,
        backend::PointCloudSurface& outPointCloud);
    void markTriangleTopologyCacheUnused();
    void pruneTriangleTopologyCache();

    std::unordered_map<const void*, CachedTriangleTopology> m_triangleTopologyCache;
#endif

    std::vector<CollisionModelPair> m_pendingPairs;
    bool m_backendAvailable { false };
    bool m_reportedFallback { false };
    std::uint64_t m_frameCounter { 0 };  // for sampled counter readback
};

} // namespace SofaGpuCollision
