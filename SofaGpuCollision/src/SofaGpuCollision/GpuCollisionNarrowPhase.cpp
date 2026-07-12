#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuCollisionNarrowPhase.h>
#include <SofaGpuCollision/GpuPipelineProfiling.h>

#ifdef SOFAGPUCOLLISION_WITH_CUDA
#include <SofaCUDA/component/collision/geometry/CudaTriangleModel.h>
#include <SofaCUDA/component/collision/geometry/CudaPointModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#endif

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SOFAGPUCOLLISION_HAS_NVTX 1
#else
#define SOFAGPUCOLLISION_HAS_NVTX 0
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

// Bridge ProximityContact -> ExactContact so the SOFA DetectionOutput
// publication path stays unchanged (was duplicated per dispatch branch).
void repackProximityContactsAsExact(
    const std::vector<SofaGpuCollision::backend::ProximityContact>& proximityContacts,
    std::vector<SofaGpuCollision::backend::ExactContact>& exactContacts)
{
    exactContacts.clear();
    exactContacts.reserve(proximityContacts.size());
    for (const auto& pc : proximityContacts)
    {
        exactContacts.push_back(SofaGpuCollision::backend::ExactContact {
            pc.firstPrimitiveIndex,
            pc.secondPrimitiveIndex,
            pc.pointOnFirst,
            pc.pointOnSecond,
            pc.normal,
            pc.signedDistance,
        });
    }
}

} // namespace

namespace SofaGpuCollision
{

namespace
{

using CubeCollisionModel = sofa::component::collision::geometry::CubeCollisionModel;

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

#ifdef SOFAGPUCOLLISION_WITH_CUDA
using CudaTriangleCollisionModel = sofa::gpu::cuda::CudaTriangleCollisionModel;
using CudaPointCollisionModel = sofa::gpu::cuda::CudaPointCollisionModel;
using CudaTriangleOutputVector =
    sofa::core::collision::TDetectionOutputVector<CudaTriangleCollisionModel, CudaTriangleCollisionModel>;
// Cross-model output vector: point cloud (first) vs triangle mesh (second).
// SOFA's TDetectionOutputVector is templated on any two CollisionModel types,
// so this just works for the (CudaPoint, CudaTriangle) combination.
using CudaPointTriangleOutputVector =
    sofa::core::collision::TDetectionOutputVector<CudaPointCollisionModel, CudaTriangleCollisionModel>;
#endif

bool tryExtractBoundingTree(
    sofa::core::CollisionModel* collisionModel,
    std::vector<backend::AxisAlignedBoundingBox>& outTree)
{
    outTree.clear();

    auto* cubeModel = dynamic_cast<CubeCollisionModel*>(collisionModel == nullptr ? nullptr : collisionModel->getFirst());
    if (cubeModel == nullptr)
    {
        return false;
    }

    cubeModel->computeBoundingTree();

    if (cubeModel->empty() || cubeModel->getSize() == 0)
    {
        return false;
    }

    sofa::type::vector<std::pair<sofa::type::Vec3, sofa::type::Vec3>> rawBoundingTree;
    cubeModel->getBoundingTree(rawBoundingTree);

    if (rawBoundingTree.empty())
    {
        return false;
    }

    outTree.reserve(rawBoundingTree.size());
    for (const auto& [min, max] : rawBoundingTree)
    {
        outTree.push_back(backend::AxisAlignedBoundingBox {
            static_cast<float>(min[0]),
            static_cast<float>(min[1]),
            static_cast<float>(min[2]),
            static_cast<float>(max[0]),
            static_cast<float>(max[1]),
            static_cast<float>(max[2]),
        });
    }

    return true;
}

void accumulateBackendStats(profiling::StageSnapshot& stageSnapshot, const backend::BackendExecutionStats& stats)
{
    stageSnapshot.gpuUsed = true;
    stageSnapshot.gpuKernelMilliseconds += stats.gpuKernelMilliseconds;
    stageSnapshot.hostPreparationMilliseconds += stats.hostPreparationMilliseconds;
    stageSnapshot.backendTrianglePackMilliseconds += stats.backendTrianglePackMilliseconds;
    stageSnapshot.hostToDeviceMilliseconds += stats.hostToDeviceMilliseconds;
    stageSnapshot.deviceAllocationMilliseconds += stats.deviceAllocationMilliseconds;
    stageSnapshot.denseGridClearMilliseconds += stats.denseGridClearMilliseconds;
    stageSnapshot.denseGridCounterClearMilliseconds += stats.denseGridCounterClearMilliseconds;
    stageSnapshot.denseGridTissueAabbMilliseconds += stats.denseGridTissueAabbMilliseconds;
    stageSnapshot.denseGridToolAabbMilliseconds += stats.denseGridToolAabbMilliseconds;
    stageSnapshot.denseGridInsertTissueMilliseconds += stats.denseGridInsertTissueMilliseconds;
    stageSnapshot.denseGridInsertToolMilliseconds += stats.denseGridInsertToolMilliseconds;
    stageSnapshot.denseGridGeneratePairsMilliseconds += stats.denseGridGeneratePairsMilliseconds;
    stageSnapshot.denseGridCandidateReadbackMilliseconds += stats.denseGridCandidateReadbackMilliseconds;
    stageSnapshot.denseGridSortUniqueMilliseconds += stats.denseGridSortUniqueMilliseconds;
    stageSnapshot.denseGridSortUniqueHostMilliseconds += stats.denseGridSortUniqueHostMilliseconds;
    stageSnapshot.denseGridExactContactMilliseconds += stats.denseGridExactContactMilliseconds;
    stageSnapshot.denseGridContactCountReadbackMilliseconds += stats.denseGridContactCountReadbackMilliseconds;
    stageSnapshot.denseGridContactDownloadMilliseconds += stats.denseGridContactDownloadMilliseconds;
    stageSnapshot.hashGridResetMilliseconds += stats.hashGridResetMilliseconds;
    stageSnapshot.hashGridPairHashClearMilliseconds += stats.hashGridPairHashClearMilliseconds;
    stageSnapshot.hashGridInsertTissueMilliseconds += stats.hashGridInsertTissueMilliseconds;
    stageSnapshot.hashGridInsertToolMilliseconds += stats.hashGridInsertToolMilliseconds;
    stageSnapshot.hashGridPairCountMilliseconds += stats.hashGridPairCountMilliseconds;
    stageSnapshot.hashGridScanMilliseconds += stats.hashGridScanMilliseconds;
    stageSnapshot.hashGridGeneratePairsMilliseconds += stats.hashGridGeneratePairsMilliseconds;
    stageSnapshot.hashGridProximityCounterClearMilliseconds += stats.hashGridProximityCounterClearMilliseconds;
    stageSnapshot.featureBasedProximityKernelMilliseconds += stats.featureBasedProximityKernelMilliseconds;
    stageSnapshot.vfContactCount += stats.vfContactCount;
    stageSnapshot.fvContactCount += stats.fvContactCount;
    stageSnapshot.eeContactCount += stats.eeContactCount;
    stageSnapshot.hostToDeviceBytes += stats.hostToDeviceBytes;
    stageSnapshot.deviceToHostBytes += stats.deviceToHostBytes;
    stageSnapshot.deviceAllocationBytes += stats.deviceAllocationBytes;
    stageSnapshot.kernelLaunchCount += stats.kernelLaunchCount;
    stageSnapshot.cudaMemsetCount += stats.cudaMemsetCount;
    stageSnapshot.workspaceResizeCount += stats.workspaceResizeCount;
    stageSnapshot.inputPrimitiveCount += stats.inputPrimitiveCount;
    stageSnapshot.outputPairCount += stats.outputPairCount;
    stageSnapshot.outputCandidateCount += stats.outputCandidateCount;
    stageSnapshot.outputContactCount += stats.outputContactCount;
    stageSnapshot.rawCandidateCount += stats.rawCandidateCount;
    stageSnapshot.uniqueCandidateCount += stats.uniqueCandidateCount;
    stageSnapshot.gridCellCount += stats.gridCellCount;
    stageSnapshot.activeMixedCellCount += stats.activeMixedCellCount;
    stageSnapshot.tissueInsertCount += stats.tissueInsertCount;
    stageSnapshot.toolInsertCount += stats.toolInsertCount;
    stageSnapshot.maxTissueCellOccupancy = std::max(stageSnapshot.maxTissueCellOccupancy, stats.maxTissueCellOccupancy);
    stageSnapshot.maxToolCellOccupancy = std::max(stageSnapshot.maxToolCellOccupancy, stats.maxToolCellOccupancy);
    stageSnapshot.overflowCount += stats.overflowCount;
    stageSnapshot.hashDedupeProbeOverflowCount += stats.hashDedupeProbeOverflowCount;
}

#ifdef SOFAGPUCOLLISION_WITH_CUDA
void publishCudaTriangleContacts(
    sofa::core::collision::NarrowPhaseDetection* narrowPhase,
    CudaTriangleCollisionModel* firstModel,
    CudaTriangleCollisionModel* secondModel,
    const std::vector<backend::ExactContact>& contacts)
{
    if (firstModel == nullptr || secondModel == nullptr || contacts.empty())
    {
        return;
    }

    auto*& outputsBase = narrowPhase->getDetectionOutputs(firstModel, secondModel);
    if (outputsBase == nullptr)
    {
        outputsBase = new CudaTriangleOutputVector();
    }

    auto* outputs = static_cast<CudaTriangleOutputVector*>(outputsBase);
    outputs->clear();

    constexpr std::uint64_t kTriangleIndexMask = 0xffffffffull;
    for (const auto& contact : contacts)
    {
        sofa::core::collision::DetectionOutput output;
        output.elem = std::make_pair(
            sofa::core::CollisionElementIterator(firstModel, static_cast<sofa::Index>(contact.firstTriangleIndex)),
            sofa::core::CollisionElementIterator(secondModel, static_cast<sofa::Index>(contact.secondTriangleIndex)));
        output.id = static_cast<sofa::core::collision::DetectionOutput::ContactId>(
            ((static_cast<std::uint64_t>(contact.firstTriangleIndex) & kTriangleIndexMask) << 32) |
            (static_cast<std::uint64_t>(contact.secondTriangleIndex) & kTriangleIndexMask));
        output.point[0] = sofa::type::Vec3(contact.pointOnFirst.x, contact.pointOnFirst.y, contact.pointOnFirst.z);
        output.point[1] = sofa::type::Vec3(contact.pointOnSecond.x, contact.pointOnSecond.y, contact.pointOnSecond.z);
        output.normal = sofa::type::Vec3(contact.normal.x, contact.normal.y, contact.normal.z);
        output.value = static_cast<double>(contact.signedDistance);
        output.deltaT = 0.0;
        outputs->push_back(output);
    }
}

// Cross-model publication: (CudaPointCollisionModel, CudaTriangleCollisionModel).
// Used by the v-t cross-model dispatch when copyContactsToHost is true.
// The ProximityContact convention from computeFeatureBasedVertexTriangleContacts is:
//   firstPrimitiveIndex  = vertex id on the point cloud
//   secondPrimitiveIndex = triangle id on the triangle mesh
//   featureKind          = VertexFace always
//   pointOnFirst         = vertex world-space position
//   pointOnSecond        = closest point on the triangle face
//   normal               = unit vector from pointOnFirst toward pointOnSecond
//   signedDistance       = proximity distance (>= 0)
//
// We assume groupSize=1 on the CudaPointCollisionModel — i.e. each "element"
// is a single vertex, so the element index equals the vertex index. The
// header's CudaPoint::i0() multiplies by groupSize, so for non-default
// groupSize the caller is responsible for the index mapping.
void publishCudaPointTriangleContacts(
    sofa::core::collision::NarrowPhaseDetection* narrowPhase,
    CudaPointCollisionModel* pointModel,
    CudaTriangleCollisionModel* triangleModel,
    const std::vector<backend::ProximityContact>& contacts)
{
    if (pointModel == nullptr || triangleModel == nullptr || contacts.empty())
    {
        return;
    }

    auto*& outputsBase = narrowPhase->getDetectionOutputs(pointModel, triangleModel);
    if (outputsBase == nullptr)
    {
        outputsBase = new CudaPointTriangleOutputVector();
    }
    auto* outputs = static_cast<CudaPointTriangleOutputVector*>(outputsBase);
    outputs->clear();

    constexpr std::uint64_t kIndexMask = 0xffffffffull;
    for (const auto& pc : contacts)
    {
        sofa::core::collision::DetectionOutput output;
        output.elem = std::make_pair(
            sofa::core::CollisionElementIterator(pointModel, static_cast<sofa::Index>(pc.firstPrimitiveIndex)),
            sofa::core::CollisionElementIterator(triangleModel, static_cast<sofa::Index>(pc.secondPrimitiveIndex)));
        output.id = static_cast<sofa::core::collision::DetectionOutput::ContactId>(
            ((static_cast<std::uint64_t>(pc.firstPrimitiveIndex) & kIndexMask) << 32) |
            (static_cast<std::uint64_t>(pc.secondPrimitiveIndex) & kIndexMask));
        output.point[0] = sofa::type::Vec3(pc.pointOnFirst.x, pc.pointOnFirst.y, pc.pointOnFirst.z);
        output.point[1] = sofa::type::Vec3(pc.pointOnSecond.x, pc.pointOnSecond.y, pc.pointOnSecond.z);
        output.normal = sofa::type::Vec3(pc.normal.x, pc.normal.y, pc.normal.z);
        output.value = static_cast<double>(pc.signedDistance);
        output.deltaT = 0.0;
        outputs->push_back(output);
    }
}
#endif

} // namespace

int GpuCollisionNarrowPhaseClass = sofa::core::RegisterObject(
    "GPU-first narrow phase pruning with SOFA BVH contact generation fallback.")
    .add<GpuCollisionNarrowPhase>();

GpuCollisionNarrowPhase::GpuCollisionNarrowPhase()
    : sofa::component::collision::detection::algorithm::BVHNarrowPhase()
    , d_enableGpu(initData(&d_enableGpu, true, "enableGPU", "Try to execute the GPU narrow phase backend."))
    , d_allowCpuFallback(initData(&d_allowCpuFallback, true, "allowCPUFallback", "Use the SOFA CPU narrow phase if GPU execution is unavailable."))
    , d_logBackendStatus(initData(&d_logBackendStatus, true, "logBackendStatus", "Log the selected narrow phase backend during init."))
    , d_useDenseGrid(initData(&d_useDenseGrid, true, "useDenseGrid", "Use dense 3D grid primitive candidate generation before exact GPU contacts."))
    , d_deduplicatePairs(initData(&d_deduplicatePairs, true, "deduplicatePairs", "Deduplicate dense-grid candidate pairs before exact contact tests."))
    , d_copyContactsToHost(initData(&d_copyContactsToHost, true, "copyContactsToHost", "Copy exact GPU contacts back to SOFA DetectionOutput. Disable only for detection-only GPU benchmarks without CPU collision response."))
    , d_detailedProfiling(initData(&d_detailedProfiling, false, "detailedProfiling", "Collect per-stage CUDA event timings, NVTX ranges, and dense-grid occupancy stats. This adds synchronizations and should be enabled only for profiling runs."))
    , d_usePinnedHostStaging(initData(&d_usePinnedHostStaging, true, "usePinnedHostStaging", "Use pinned host staging buffers for dense-grid triangle uploads when CUDA pinned allocation succeeds."))
    , d_useGpuHashDedupe(initData(&d_useGpuHashDedupe, true, "useGpuHashDedupe", "Deduplicate dense-grid candidate pairs with a GPU hash table instead of the legacy global sort/unique path."))
    , d_canonicalPairEmission(initData(&d_canonicalPairEmission, false, "canonicalPairEmission", "Emit each tissue/tool candidate pair only from its canonical overlap cell to reduce duplicate pairs before sort/unique."))
    , d_cacheTriangleTopology(initData(&d_cacheTriangleTopology, true, "cacheTriangleTopology", "Cache stable SOFA triangle topology and rebuild only vertex-position data each frame."))
    , d_useIndexedDenseGridInput(initData(&d_useIndexedDenseGridInput, true, "useIndexedDenseGridInput", "Use cached indexed triangle input for dense-grid backend so stable topology uploads only indices on change and positions each frame."))
    , d_useDirectDevicePositions(initData(&d_useDirectDevicePositions, true, "useDirectDevicePositions", "Use SOFA CUDA vector device pointers directly for indexed dense-grid input when available."))
    , d_validateTriangleTopologyCache(initData(&d_validateTriangleTopologyCache, false, "validateTriangleTopologyCache", "Hash triangle topology every frame to detect same-size topology changes. Leave disabled for fastest static-topology scenes."))
    , d_readCountersWhenContactsStayOnDevice(initData(&d_readCountersWhenContactsStayOnDevice, false, "readCountersWhenContactsStayOnDevice", "Read dense-grid counters back to CPU when exact contacts remain device-side. Disable for the fastest detection-only path."))
    , d_computeDeviceContactsWhenContactsStayOnDevice(initData(&d_computeDeviceContactsWhenContactsStayOnDevice, false, "computeDeviceContactsWhenContactsStayOnDevice", "Compute exact contact records on the GPU when contacts are not copied to CPU. Disable for fastest detection-only candidate generation."))
    , d_compactActiveCells(initData(&d_compactActiveCells, false, "compactActiveCells", "Compact mixed dense-grid cells before candidate generation. Experimental; disabled by default because it regressed the current GTX 1650 Ti benchmark path."))
    , d_batchTriangleInsert(initData(&d_batchTriangleInsert, false, "batchTriangleInsert", "Insert tissue and tool triangles with one dense-grid launch when supported. Experimental; disabled by default because separate indexed inserts are faster for the current benchmark path."))
    , d_useToolActiveCellGeneration(initData(&d_useToolActiveCellGeneration, true, "useToolActiveCellGeneration", "Phase 15 (DEFAULT ON since 2026-05-25): generate candidate pairs over tool-occupied (mixed) cells only. The active-cell list is built during the tool insert (no separate scan), then candidate generation launches a small fixed grid over that list instead of one block per grid cell. Measured 4.3x faster on one-tissue/one-blade and 1.08x on large-tissue, bit-identical contacts, never a regression. Mutually exclusive with batchTriangleInsert."))
    , d_useHashPrefixSumGeneration(initData(&d_useHashPrefixSumGeneration, false, "useHashPrefixSumGeneration", "EXPERIMENTAL (experiment/hash-prefixsum-broadphase): replace the dense-grid broad cull with a spatial-hash + prefix-sum work-expansion broad cull for the tri-tri FBP path. Intended for large-tissue + large-tool scenes. The narrow phase (FBP kernel) is unchanged, so contacts are identical. Default off."))
    , d_useSimpleHashGeneration(initData(&d_useSimpleHashGeneration, false, "useSimpleHashGeneration", "EXPERIMENTAL '4th way': replace the dense-grid broad cull with a SIMPLE direct-bucket spatial hash for the tri-tri FBP path. Triangles are stored straight into per-cell hash buckets in one insert pass (no mark/compact/fill). Best-effort on per-cell overflow. Same FBP narrow kernel, so contacts match the other paths when there is no overflow. Default off. Mutually exclusive with useHashPrefixSumGeneration (hash takes precedence)."))
    , d_useSortedGridGeneration(initData(&d_useSortedGridGeneration, false, "useSortedGridGeneration", "EXPERIMENTAL '5th way': replace the dense-grid broad cull with a SORTED-GRID (tiled binning) broad cull for the tri-tri FBP path — expand (cell, triangle) incidences, sort them by cell, generate pairs from each cell's contiguous run with the tool run staged in shared memory. NO per-cell capacity caps (no best-effort drops). Same FBP narrow kernel -> identical contacts. Default off. Precedence: hash > simple hash > sorted grid."))
    , d_sortedGridUseCubSort(initData(&d_sortedGridUseCubSort, false, "sortedGridUseCubSort", "Sorted-grid sort engine: false (default) = hand-rolled one-pass counting sort (CUB-free); true = cub::DeviceRadixSort over the padded incidence buffer."))
    , d_sortedGridUsePairHashDedup(initData(&d_sortedGridUsePairHashDedup, false, "sortedGridUsePairHashDedup", "Sorted-grid dedup: false (default) = home-cell exactly-once emission (no dedup hash table; also pre-culls AABB-disjoint pairs, so candidate counts may be lower while contacts stay identical); true = the same atomicCAS pair-hash as the hash ways (reproduces their candidate set exactly)."))
    , d_useBigCellFusedGeneration(initData(&d_useBigCellFusedGeneration, false, "useBigCellFusedGeneration", "EXPERIMENTAL '6th way': big-cell / small-cell FUSED generation + narrow phase. Small cells are grouped into big cells (bigCellFactor^3); a per-big-cell CSR table of (triangle, small cell) entries is built, then ONE kernel per mixed big cell stages the tool side (ids + AABBs + vertices) in shared memory and runs the FBP closest-feature math inline — no intermediate pair list, no separate FBP launch. Same home-cell dedup rule as the sorted grid -> identical contacts. Default off. Precedence: big cell > hash > simple hash > sorted grid."))
    , d_bigCellFactor(initData(&d_bigCellFactor, static_cast<unsigned int>(2), "bigCellFactor", "Big-cell edge length in small cells for useBigCellFusedGeneration. Power of two, at most 4. Default 2 (measured best: factor 4 starves block-level parallelism on blade-concentrated scenes)."))
    , d_bigCellToolTile(initData(&d_bigCellToolTile, static_cast<unsigned int>(256), "bigCellToolTile", "Tool entries staged in shared memory per chunk for useBigCellFusedGeneration (1..256). Lower values exercise the oversized-big-cell chunk loop."))
    , d_bigCellUseHashBuild(initData(&d_bigCellUseHashBuild, false, "bigCellUseHashBuild", "Big-cell table build strategy A/B: false (default) = CSR bucket lists (count -> scan -> fill); true = literal per-big-cell open-addressing hash multi-map build."))
    , d_bigCellHashSlots(initData(&d_bigCellHashSlots, static_cast<unsigned int>(1024), "bigCellHashSlots", "Hash-build only: open-addressing slots per (big cell, side) region. Rounded to a power of two, clamped to [64, 4096]."))
    , d_hashTableSize(initData(&d_hashTableSize, static_cast<unsigned int>(0), "hashTableSize", "Hash table slot count for useHashPrefixSumGeneration / useSimpleHashGeneration. 0 = auto-derive (~4 slots per input triangle). Rounded up to a power of two."))
    , d_useFeatureBasedProximity(initData(&d_useFeatureBasedProximity, false, "useFeatureBasedProximity", "Replace SAT-style exact triangle intersection with feature-based proximity (VF + EE) using Ericson closest-point math. Outputs barycentric weights for a CUDA constraint solver."))
    , d_useVertexTriangleProximity(initData(&d_useVertexTriangleProximity, false, "useVertexTriangleProximity", "When set together with useFeatureBasedProximity, route self-collision pairs (pair.first == pair.second on a CudaTriangleCollisionModel) through the vertex-triangle proximity kernel. Useful for surgical self-collision such as cutting/tearing."))
    , d_proximityComputeBarycentrics(initData(&d_proximityComputeBarycentrics, true, "proximityComputeBarycentrics", "Populate barycentric weights in each ProximityContact. Set false only when the consumer does not need barys (saves a few writes per contact)."))
    , d_proximityReadContactCounter(initData(&d_proximityReadContactCounter, false, "proximityReadContactCounter", "Read back the proximity contact count and per-class counters (vf/fv/ee) for profiling. Adds ~20 D2H bytes per frame."))
    , d_proximityKeepContactsOnDevice(initData(&d_proximityKeepContactsOnDevice, true, "proximityKeepContactsOnDevice", "Keep proximity contacts in device memory (constraint-solver-ready). Disable to copy contacts into SOFA DetectionOutput for the CPU collision response path."))
    , d_proximityMaxContacts(initData(&d_proximityMaxContacts, static_cast<unsigned int>(1000000), "proximityMaxContacts", "Capacity of the feature-based proximity contact output buffer."))
    , d_proximityCounterReadbackInterval(initData(&d_proximityCounterReadbackInterval, static_cast<unsigned int>(0), "proximityCounterReadbackInterval", "When > 0, read back proximity counters every Nth frame (sampled mode). 0 means read only when proximityReadContactCounter is true. Lets long profiling runs collect VF/FV/EE breakdowns without paying a sync every frame."))
    , d_minGpuPairCount(initData(&d_minGpuPairCount, static_cast<unsigned int>(8), "minGPUPairCount", "Minimum number of candidate pairs before the GPU narrow-phase prefilter is worth using."))
    , d_gridMinX(initData(&d_gridMinX, -8.0, "gridMinX", "Dense grid minimum X coordinate."))
    , d_gridMinY(initData(&d_gridMinY, -2.0, "gridMinY", "Dense grid minimum Y coordinate."))
    , d_gridMinZ(initData(&d_gridMinZ, -8.0, "gridMinZ", "Dense grid minimum Z coordinate."))
    , d_gridMaxX(initData(&d_gridMaxX, 8.0, "gridMaxX", "Dense grid maximum X coordinate."))
    , d_gridMaxY(initData(&d_gridMaxY, 4.0, "gridMaxY", "Dense grid maximum Y coordinate."))
    , d_gridMaxZ(initData(&d_gridMaxZ, 8.0, "gridMaxZ", "Dense grid maximum Z coordinate."))
    , d_gridResolutionX(initData(&d_gridResolutionX, static_cast<unsigned int>(64), "gridResolutionX", "Dense grid X cell count."))
    , d_gridResolutionY(initData(&d_gridResolutionY, static_cast<unsigned int>(24), "gridResolutionY", "Dense grid Y cell count."))
    , d_gridResolutionZ(initData(&d_gridResolutionZ, static_cast<unsigned int>(64), "gridResolutionZ", "Dense grid Z cell count."))
    , d_contactDistance(initData(&d_contactDistance, 0.03, "contactDistance", "Inflation distance for dense-grid primitive AABBs."))
    , d_maxTissueTrianglesPerCell(initData(&d_maxTissueTrianglesPerCell, static_cast<unsigned int>(64), "maxTissueTrianglesPerCell", "Dense-grid tissue bucket capacity per cell."))
    , d_maxToolTrianglesPerCell(initData(&d_maxToolTrianglesPerCell, static_cast<unsigned int>(32), "maxToolTrianglesPerCell", "Dense-grid tool bucket capacity per cell."))
    , d_maxCandidatePairs(initData(&d_maxCandidatePairs, static_cast<unsigned int>(1000000), "maxCandidatePairs", "Dense-grid candidate-pair output capacity."))
{
}

#ifdef SOFAGPUCOLLISION_WITH_CUDA
void GpuCollisionNarrowPhase::markTriangleTopologyCacheUnused()
{
    for (auto& [_, cacheEntry] : m_triangleTopologyCache)
    {
        cacheEntry.seenThisFrame = false;
    }
}

void GpuCollisionNarrowPhase::pruneTriangleTopologyCache()
{
    for (auto it = m_triangleTopologyCache.begin(); it != m_triangleTopologyCache.end();)
    {
        if (!it->second.seenThisFrame)
        {
            it = m_triangleTopologyCache.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool GpuCollisionNarrowPhase::extractCudaIndexedSurface(
    sofa::core::CollisionModel* collisionModel,
    sofa::core::CollisionModel*& outModel,
    backend::TriangleIndexedSurface& outSurface)
{
    outSurface = backend::TriangleIndexedSurface {};
    auto* cudaModel = dynamic_cast<CudaTriangleCollisionModel*>(
        collisionModel == nullptr ? nullptr : collisionModel->getLast());
    outModel = cudaModel;
    if (cudaModel == nullptr || cudaModel->empty())
    {
        return false;
    }

    const auto& positions = cudaModel->getX();
    const auto& triangles = cudaModel->getTriangles();
    if (positions.empty() || triangles.empty())
    {
        return false;
    }

    auto& cacheEntry = m_triangleTopologyCache[static_cast<const void*>(cudaModel)];
    cacheEntry.seenThisFrame = true;

    auto computeTopologyHash = [&]() {
        std::uint64_t hash = 1469598103934665603ull;
        const auto mix = [&](const std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(static_cast<std::uint64_t>(positions.size()));
        mix(static_cast<std::uint64_t>(triangles.size()));
        for (const auto& triangle : triangles)
        {
            mix(static_cast<std::uint64_t>(triangle[0]));
            mix(static_cast<std::uint64_t>(triangle[1]));
            mix(static_cast<std::uint64_t>(triangle[2]));
        }
        return hash;
    };

    const bool sizeChanged =
        cacheEntry.vertexCount != positions.size() ||
        cacheEntry.triangleCount != triangles.size() ||
        cacheEntry.triangleIndices.size() != triangles.size() * 3u;
    const std::uint64_t topologyHash =
        (sizeChanged || d_validateTriangleTopologyCache.getValue()) ? computeTopologyHash() : cacheEntry.topologyHash;
    const bool topologyChanged = sizeChanged || cacheEntry.topologyHash != topologyHash;

    if (topologyChanged)
    {
        cacheEntry.vertexCount = positions.size();
        cacheEntry.triangleCount = triangles.size();
        cacheEntry.topologyHash = topologyHash;
        cacheEntry.triangleIndices.resize(triangles.size() * 3u);
        for (std::uint32_t triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
        {
            const auto i0 = static_cast<std::uint32_t>(triangles[triangleIndex][0]);
            const auto i1 = static_cast<std::uint32_t>(triangles[triangleIndex][1]);
            const auto i2 = static_cast<std::uint32_t>(triangles[triangleIndex][2]);
            if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
            {
                cacheEntry.triangleIndices.clear();
                return false;
            }
            cacheEntry.triangleIndices[3u * triangleIndex + 0u] = i0;
            cacheEntry.triangleIndices[3u * triangleIndex + 1u] = i1;
            cacheEntry.triangleIndices[3u * triangleIndex + 2u] = i2;
        }
        cacheEntry.missCount += 1;
    }
    else
    {
        cacheEntry.hitCount += 1;
    }

    const backend::TriangleVertex* directDevicePositions = nullptr;
    if (d_useDirectDevicePositions.getValue())
    {
        const auto* sofaDevicePositions = positions.deviceRead();
        using SofaHostPosition = std::remove_cv_t<std::remove_reference_t<decltype(positions[0])>>;
        static_assert(sizeof(SofaHostPosition) == sizeof(backend::TriangleVertex));
        static_assert(alignof(SofaHostPosition) == alignof(backend::TriangleVertex));
        directDevicePositions = reinterpret_cast<const backend::TriangleVertex*>(sofaDevicePositions);
    }

    if (directDevicePositions == nullptr)
    {
        cacheEntry.positionScratch.resize(positions.size());
        for (std::uint32_t vertexIndex = 0; vertexIndex < positions.size(); ++vertexIndex)
        {
            const auto& position = positions[vertexIndex];
            cacheEntry.positionScratch[vertexIndex] = backend::TriangleVertex {
                static_cast<float>(position[0]),
                static_cast<float>(position[1]),
                static_cast<float>(position[2]),
            };
        }
        outSurface.positions = cacheEntry.positionScratch.data();
    }
    else
    {
        outSurface.positions = nullptr;
        outSurface.devicePositions = directDevicePositions;
    }

    outSurface.vertexCount = static_cast<std::uint32_t>(positions.size());
    outSurface.triangleIndices = cacheEntry.triangleIndices.data();
    outSurface.triangleCount = static_cast<std::uint32_t>(cacheEntry.triangleCount);
    outSurface.surfaceId = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cudaModel));
    outSurface.topologyVersion = cacheEntry.topologyHash;
    return (outSurface.positions != nullptr || outSurface.devicePositions != nullptr) &&
        outSurface.triangleIndices != nullptr;
}

bool GpuCollisionNarrowPhase::extractCudaPointCloud(
    sofa::core::CollisionModel* collisionModel,
    sofa::core::CollisionModel*& outModel,
    backend::PointCloudSurface& outPointCloud)
{
    outPointCloud = backend::PointCloudSurface {};
    auto* pointModel = dynamic_cast<CudaPointCollisionModel*>(
        collisionModel == nullptr ? nullptr : collisionModel->getLast());
    outModel = pointModel;
    if (pointModel == nullptr)
    {
        return false;
    }
    auto* mstate = pointModel->getMechanicalState();
    if (mstate == nullptr)
    {
        return false;
    }
    // Pull positions directly from the CUDA mechanical state. The returned
    // VecCoord type is sofa::gpu::cuda::CudaVec3fTypes::VecCoord which lives
    // on the device; .deviceRead() yields the GPU pointer we hand to the
    // dense-grid backend with zero per-frame H2D traffic.
    const auto* positionsData = mstate->read(sofa::core::vec_id::read_access::position);
    if (positionsData == nullptr)
    {
        return false;
    }
    const auto& positions = positionsData->getValue();
    if (positions.empty())
    {
        return false;
    }
    using SofaHostPosition = std::remove_cv_t<std::remove_reference_t<decltype(positions[0])>>;
    static_assert(sizeof(SofaHostPosition) == sizeof(backend::TriangleVertex),
                  "CudaVec3f host element must match backend TriangleVertex layout");
    static_assert(alignof(SofaHostPosition) == alignof(backend::TriangleVertex),
                  "CudaVec3f host element must match backend TriangleVertex alignment");

    outPointCloud.positions = nullptr;
    outPointCloud.devicePositions =
        reinterpret_cast<const backend::TriangleVertex*>(positions.deviceRead());
    outPointCloud.pointCount = static_cast<std::uint32_t>(positions.size());
    // The surfaceId is the point-model pointer cast to uint64. This guarantees
    // a different id from any TriangleIndexedSurface (which uses its own model
    // pointer), so the kernel's own-corner exclusion stays off for cross-model
    // pairs — vertices are not erroneously filtered against triangles of an
    // unrelated mesh.
    outPointCloud.surfaceId = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointModel));
    // topologyVersion is unused by the vertex-triangle backend for the point
    // side (only the triangle side's surfaceId/version drives the index-cache
    // decision). Keep it deterministic at 0 so a future caching variant can
    // detect a "fresh frame" signal explicitly.
    outPointCloud.topologyVersion = 0u;
    return outPointCloud.pointCount > 0u;
}

bool GpuCollisionNarrowPhase::extractCudaSelfCollisionSurfaces(
    sofa::core::CollisionModel* collisionModel,
    sofa::core::CollisionModel*& outModel,
    backend::TriangleIndexedSurface& outTriangleSurface,
    backend::PointCloudSurface& outPointCloud)
{
    outPointCloud = backend::PointCloudSurface {};
    if (!extractCudaIndexedSurface(collisionModel, outModel, outTriangleSurface))
    {
        return false;
    }
    // Mirror the triangle surface's vertex storage into the point cloud. Both
    // share the same surfaceId so the kernel-side own-corner exclusion fires.
    outPointCloud.positions = outTriangleSurface.positions;
    outPointCloud.devicePositions = outTriangleSurface.devicePositions;
    outPointCloud.pointCount = outTriangleSurface.vertexCount;
    outPointCloud.surfaceId = outTriangleSurface.surfaceId;
    outPointCloud.topologyVersion = outTriangleSurface.topologyVersion;
    return (outPointCloud.positions != nullptr || outPointCloud.devicePositions != nullptr) &&
        outPointCloud.pointCount > 0u;
}

bool GpuCollisionNarrowPhase::extractCudaTriangles(
    sofa::core::CollisionModel* collisionModel,
    sofa::core::CollisionModel*& outModel,
    const std::vector<backend::TrianglePrimitive>*& outTriangles)
{
    outTriangles = nullptr;
    auto* cudaModel = dynamic_cast<CudaTriangleCollisionModel*>(
        collisionModel == nullptr ? nullptr : collisionModel->getLast());
    outModel = cudaModel;
    if (cudaModel == nullptr || cudaModel->empty())
    {
        return false;
    }

    const auto& positions = cudaModel->getX();
    const auto& triangles = cudaModel->getTriangles();
    if (positions.empty() || triangles.empty())
    {
        return false;
    }

    const auto fillPackedTriangles = [&](std::vector<backend::TrianglePrimitive>& destination,
                                         const std::vector<std::uint32_t>* cachedIndices,
                                         const std::vector<backend::TriangleVertex>* cachedPositions) {
        destination.resize(triangles.size());
        for (std::uint32_t triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
        {
            const std::uint32_t i0 = cachedIndices == nullptr
                ? static_cast<std::uint32_t>(triangles[triangleIndex][0])
                : (*cachedIndices)[3u * triangleIndex + 0u];
            const std::uint32_t i1 = cachedIndices == nullptr
                ? static_cast<std::uint32_t>(triangles[triangleIndex][1])
                : (*cachedIndices)[3u * triangleIndex + 1u];
            const std::uint32_t i2 = cachedIndices == nullptr
                ? static_cast<std::uint32_t>(triangles[triangleIndex][2])
                : (*cachedIndices)[3u * triangleIndex + 2u];
            const std::size_t positionCount = cachedPositions == nullptr ? positions.size() : cachedPositions->size();
            if (i0 >= positionCount || i1 >= positionCount || i2 >= positionCount)
            {
                destination.clear();
                return false;
            }

            if (cachedPositions == nullptr)
            {
                const auto& p0 = positions[i0];
                const auto& p1 = positions[i1];
                const auto& p2 = positions[i2];
                destination[triangleIndex] = backend::TrianglePrimitive {
                    backend::TriangleVertex { static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2]) },
                    backend::TriangleVertex { static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]) },
                    backend::TriangleVertex { static_cast<float>(p2[0]), static_cast<float>(p2[1]), static_cast<float>(p2[2]) },
                    triangleIndex,
                };
            }
            else
            {
                destination[triangleIndex] = backend::TrianglePrimitive {
                    (*cachedPositions)[i0],
                    (*cachedPositions)[i1],
                    (*cachedPositions)[i2],
                    triangleIndex,
                };
            }
        }
        return true;
    };

    if (!d_cacheTriangleTopology.getValue())
    {
        auto& scratchEntry = m_triangleTopologyCache[static_cast<const void*>(cudaModel)];
        scratchEntry.seenThisFrame = true;
        if (!fillPackedTriangles(scratchEntry.triangleScratch, nullptr, nullptr))
        {
            return false;
        }
        outTriangles = &scratchEntry.triangleScratch;
        return true;
    }

    auto& cacheEntry = m_triangleTopologyCache[static_cast<const void*>(cudaModel)];
    cacheEntry.seenThisFrame = true;

    auto computeTopologyHash = [&]() {
        std::uint64_t hash = 1469598103934665603ull;
        const auto mix = [&](const std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(static_cast<std::uint64_t>(positions.size()));
        mix(static_cast<std::uint64_t>(triangles.size()));
        for (const auto& triangle : triangles)
        {
            mix(static_cast<std::uint64_t>(triangle[0]));
            mix(static_cast<std::uint64_t>(triangle[1]));
            mix(static_cast<std::uint64_t>(triangle[2]));
        }
        return hash;
    };

    const bool sizeChanged =
        cacheEntry.vertexCount != positions.size() ||
        cacheEntry.triangleCount != triangles.size() ||
        cacheEntry.triangleIndices.size() != triangles.size() * 3u;
    const std::uint64_t topologyHash =
        (sizeChanged || d_validateTriangleTopologyCache.getValue()) ? computeTopologyHash() : cacheEntry.topologyHash;
    const bool topologyChanged = sizeChanged || cacheEntry.topologyHash != topologyHash;

    if (topologyChanged)
    {
        cacheEntry.vertexCount = positions.size();
        cacheEntry.triangleCount = triangles.size();
        cacheEntry.topologyHash = topologyHash;
        cacheEntry.triangleIndices.resize(triangles.size() * 3u);
        for (std::uint32_t triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
        {
            cacheEntry.triangleIndices[3u * triangleIndex + 0u] =
                static_cast<std::uint32_t>(triangles[triangleIndex][0]);
            cacheEntry.triangleIndices[3u * triangleIndex + 1u] =
                static_cast<std::uint32_t>(triangles[triangleIndex][1]);
            cacheEntry.triangleIndices[3u * triangleIndex + 2u] =
                static_cast<std::uint32_t>(triangles[triangleIndex][2]);
        }
        cacheEntry.missCount += 1;
    }
    else
    {
        cacheEntry.hitCount += 1;
    }

    cacheEntry.positionScratch.resize(positions.size());
    for (std::uint32_t vertexIndex = 0; vertexIndex < positions.size(); ++vertexIndex)
    {
        const auto& position = positions[vertexIndex];
        cacheEntry.positionScratch[vertexIndex] = backend::TriangleVertex {
            static_cast<float>(position[0]),
            static_cast<float>(position[1]),
            static_cast<float>(position[2]),
        };
    }

    if (!fillPackedTriangles(cacheEntry.triangleScratch, &cacheEntry.triangleIndices, &cacheEntry.positionScratch))
    {
        return false;
    }
    outTriangles = &cacheEntry.triangleScratch;
    return true;
}
#endif

void GpuCollisionNarrowPhase::init()
{
    sofa::component::collision::detection::algorithm::BVHNarrowPhase::init();

    const auto status = backend::probe();
    m_backendAvailable = status.available;
    m_reportedFallback = false;

    if (d_logBackendStatus.getValue())
    {
        if (status.available)
        {
            msg_info() << "[GpuCollisionNarrowPhase] " << status.message;
        }
        else
        {
            msg_warning() << "[GpuCollisionNarrowPhase] " << status.message;
        }
    }
}

void GpuCollisionNarrowPhase::beginNarrowPhase()
{
    m_pendingPairs.clear();
    ++m_frameCounter;
#ifdef SOFAGPUCOLLISION_WITH_CUDA
    markTriangleTopologyCacheUnused();
#endif
    sofa::component::collision::detection::algorithm::BVHNarrowPhase::beginNarrowPhase();
}

void GpuCollisionNarrowPhase::addCollisionPair(
    const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& cmPair)
{
    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(cmPair);
        return;
    }

    m_pendingPairs.push_back(cmPair);
}

void GpuCollisionNarrowPhase::endNarrowPhase()
{
    const auto phaseStart = std::chrono::steady_clock::now();
    ScopedNvtxRange narrowPhaseRange("SOFA narrow phase", d_detailedProfiling.getValue());
    profiling::StageSnapshot stageSnapshot;

    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
        stageSnapshot.cpuFallbackUsed = true;
        stageSnapshot.wallMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
        profiling::recordNarrowPhase(stageSnapshot);
        return;
    }

    auto runLegacyPath = [&](const std::vector<CollisionModelPair>& legacyPairs) {
        if (legacyPairs.empty())
        {
            return;
        }

        std::string diagnostic;
        std::vector<std::uint32_t> gpuEligibleIndices;
        std::vector<backend::NarrowPhaseTreePair> gpuTreePairs;
        std::vector<backend::NarrowPhaseContactCandidate> gpuContactCandidates;

        for (std::uint32_t i = 0; i < legacyPairs.size(); ++i)
        {
            backend::NarrowPhaseTreePair treePair;
            const bool firstOk = tryExtractBoundingTree(legacyPairs[i].first, treePair.firstTree);
            const bool secondOk = tryExtractBoundingTree(legacyPairs[i].second, treePair.secondTree);
            if (firstOk && secondOk)
            {
                gpuEligibleIndices.push_back(i);
                gpuTreePairs.push_back(std::move(treePair));
            }
        }

        stageSnapshot.inputPrimitiveCount += static_cast<std::uint32_t>(gpuTreePairs.size());

        if (gpuTreePairs.size() < d_minGpuPairCount.getValue())
        {
            stageSnapshot.cpuFallbackUsed = true;
            for (const auto& pair : legacyPairs)
            {
                sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(pair);
            }
            sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
            return;
        }

        std::vector<std::uint32_t> survivingGpuPairIndices;
        backend::BackendExecutionStats backendStats;
        const bool gpuSucceeded =
            m_backendAvailable &&
            backend::prefilterNarrowPhasePairs(
                gpuTreePairs,
                survivingGpuPairIndices,
                &gpuContactCandidates,
                diagnostic,
                &backendStats);

        if (gpuSucceeded)
        {
            accumulateBackendStats(stageSnapshot, backendStats);
            std::vector<bool> keepPair(legacyPairs.size(), true);
            for (const auto gpuEligibleIndex : gpuEligibleIndices)
            {
                keepPair[gpuEligibleIndex] = false;
            }

            for (const auto localPairIndex : survivingGpuPairIndices)
            {
                keepPair[gpuEligibleIndices[localPairIndex]] = true;
            }

            for (std::size_t i = 0; i < legacyPairs.size(); ++i)
            {
                if (keepPair[i])
                {
                    sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(legacyPairs[i]);
                }
            }
            sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
            return;
        }

        if (!diagnostic.empty() && !m_reportedFallback)
        {
            msg_warning() << "[GpuCollisionNarrowPhase] " << diagnostic;
            m_reportedFallback = true;
        }

        if (d_allowCpuFallback.getValue())
        {
            stageSnapshot.cpuFallbackUsed = true;
            for (const auto& pair : legacyPairs)
            {
                sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(pair);
            }
        }

        sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
    };

    std::vector<CollisionModelPair> legacyPairs;
    legacyPairs.reserve(m_pendingPairs.size());

#ifdef SOFAGPUCOLLISION_WITH_CUDA
    for (const auto& pair : m_pendingPairs)
    {
        sofa::core::CollisionModel* firstModel = nullptr;
        sofa::core::CollisionModel* secondModel = nullptr;
        const std::vector<backend::TrianglePrimitive>* firstTriangles = nullptr;
        const std::vector<backend::TrianglePrimitive>* secondTriangles = nullptr;
        backend::TriangleIndexedSurface firstIndexedSurface;
        backend::TriangleIndexedSurface secondIndexedSurface;
        const bool tryIndexedDenseGrid =
            d_useDenseGrid.getValue() &&
            d_cacheTriangleTopology.getValue() &&
            d_useIndexedDenseGridInput.getValue();
        bool usingIndexedDenseGrid = false;

        // -------------------------------------------------------------------
        // Cross-model vertex-triangle (Phase 12, cross-model variant).
        // Triggers when:
        //   - useFeatureBasedProximity AND useVertexTriangleProximity are set
        //   - one side of the pair is a CudaPointCollisionModel
        //   - the other side is a CudaTriangleCollisionModel
        // We dispatch v-t with the point cloud on one side and the triangle
        // surface on the other. surfaceIds differ (point-model ptr vs
        // triangle-model ptr) so own-corner exclusion stays OFF.
        // -------------------------------------------------------------------
        if (d_useFeatureBasedProximity.getValue() &&
            d_useVertexTriangleProximity.getValue() &&
            tryIndexedDenseGrid &&
            pair.first != pair.second && pair.first != nullptr && pair.second != nullptr)
        {
            auto* firstLast = pair.first->getLast();
            auto* secondLast = pair.second->getLast();
            auto* firstAsPoint = dynamic_cast<CudaPointCollisionModel*>(firstLast);
            auto* secondAsPoint = dynamic_cast<CudaPointCollisionModel*>(secondLast);
            auto* firstAsTri = dynamic_cast<CudaTriangleCollisionModel*>(firstLast);
            auto* secondAsTri = dynamic_cast<CudaTriangleCollisionModel*>(secondLast);

            sofa::core::CollisionModel* pointSideCm = nullptr;
            sofa::core::CollisionModel* triSideCm = nullptr;
            if (firstAsPoint != nullptr && secondAsTri != nullptr)
            {
                pointSideCm = pair.first;
                triSideCm = pair.second;
            }
            else if (firstAsTri != nullptr && secondAsPoint != nullptr)
            {
                pointSideCm = pair.second;
                triSideCm = pair.first;
            }

            if (pointSideCm != nullptr && triSideCm != nullptr)
            {
                sofa::core::CollisionModel* outPointModel = nullptr;
                sofa::core::CollisionModel* outTriModel = nullptr;
                backend::PointCloudSurface crossPointCloud;
                backend::TriangleIndexedSurface crossTriangleSurface;
                const auto xmExtractionStart = std::chrono::steady_clock::now();
                bool pointOk = false;
                bool triOk = false;
                {
                    ScopedNvtxRange r("cross-model v-t extraction", d_detailedProfiling.getValue());
                    pointOk = extractCudaPointCloud(pointSideCm, outPointModel, crossPointCloud);
                    triOk = extractCudaIndexedSurface(triSideCm, outTriModel, crossTriangleSurface);
                }
                const double xmExtractionMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - xmExtractionStart).count();
                stageSnapshot.sofaTriangleExtractionMilliseconds += xmExtractionMs;
                stageSnapshot.hostPreparationMilliseconds += xmExtractionMs;

                if (pointOk && triOk)
                {
                    backend::DenseGridConfig denseGridConfig;
                    denseGridConfig.gridMinX = static_cast<float>(d_gridMinX.getValue());
                    denseGridConfig.gridMinY = static_cast<float>(d_gridMinY.getValue());
                    denseGridConfig.gridMinZ = static_cast<float>(d_gridMinZ.getValue());
                    denseGridConfig.gridMaxX = static_cast<float>(d_gridMaxX.getValue());
                    denseGridConfig.gridMaxY = static_cast<float>(d_gridMaxY.getValue());
                    denseGridConfig.gridMaxZ = static_cast<float>(d_gridMaxZ.getValue());
                    denseGridConfig.gridResolutionX = d_gridResolutionX.getValue();
                    denseGridConfig.gridResolutionY = d_gridResolutionY.getValue();
                    denseGridConfig.gridResolutionZ = d_gridResolutionZ.getValue();
                    denseGridConfig.contactDistance = static_cast<float>(d_contactDistance.getValue());
                    denseGridConfig.maxTissueTrianglesPerCell = d_maxTissueTrianglesPerCell.getValue();
                    denseGridConfig.maxToolTrianglesPerCell = d_maxToolTrianglesPerCell.getValue();
                    denseGridConfig.maxCandidatePairs = d_maxCandidatePairs.getValue();
                    denseGridConfig.deduplicatePairs = d_deduplicatePairs.getValue();
                    denseGridConfig.copyContactsToHost = d_copyContactsToHost.getValue();
                    denseGridConfig.detailedProfiling = d_detailedProfiling.getValue();
                    denseGridConfig.usePinnedHostStaging = d_usePinnedHostStaging.getValue();
                    denseGridConfig.useGpuHashDedupe = d_useGpuHashDedupe.getValue();
                    denseGridConfig.canonicalPairEmission = false;
                    denseGridConfig.readCountersWhenContactsStayOnDevice = false;
                    denseGridConfig.computeDeviceContactsWhenContactsStayOnDevice = false;
                    denseGridConfig.compactActiveCells = false;
                    denseGridConfig.batchTriangleInsert = false;

                    backend::FeatureBasedProximityConfig proximityConfig;
                    proximityConfig.contactDistance = static_cast<float>(d_contactDistance.getValue());
                    proximityConfig.emitOnePerPair = true;
                    proximityConfig.computeBarycentrics = d_proximityComputeBarycentrics.getValue();
                    // If the user wants contacts published to SOFA, they must
                    // be downloaded to host. Override keepContactsOnDevice in
                    // that case.
                    proximityConfig.keepContactsOnDevice =
                        d_proximityKeepContactsOnDevice.getValue() && !d_copyContactsToHost.getValue();
                    const auto readbackInterval = d_proximityCounterReadbackInterval.getValue();
                    const bool sampledReadFrame =
                        readbackInterval > 0 && (m_frameCounter % readbackInterval == 0);
                    proximityConfig.readContactCounter =
                        d_proximityReadContactCounter.getValue() ||
                        d_detailedProfiling.getValue() ||
                        sampledReadFrame;
                    proximityConfig.maxContacts = d_proximityMaxContacts.getValue();

                    std::vector<backend::ProximityContact> proximityContacts;
                    backend::FeatureBasedProximityStats proximityStats;
                    backend::BackendExecutionStats xmBackendStats;
                    std::string xmDiagnostic;
                    const bool xmOk = backend::computeFeatureBasedVertexTriangleContacts(
                        crossPointCloud,
                        crossTriangleSurface,
                        denseGridConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        xmDiagnostic,
                        &xmBackendStats);

                    if (xmOk)
                    {
                        accumulateBackendStats(stageSnapshot, xmBackendStats);
                        if (!proximityContacts.empty() && d_copyContactsToHost.getValue())
                        {
                            // CPU publication for cross-model (point, triangle).
                            // Uses the typed TDetectionOutputVector<CudaPoint, CudaTriangle>
                            // specialization. The downstream contact-response
                            // handler must be registered for this model-pair
                            // type; if it isn't, SOFA will simply ignore the
                            // outputs.
                            const auto publishStart = std::chrono::steady_clock::now();
                            auto* pointCudaModel = dynamic_cast<CudaPointCollisionModel*>(outPointModel);
                            auto* triCudaModel = dynamic_cast<CudaTriangleCollisionModel*>(outTriModel);
                            publishCudaPointTriangleContacts(this, pointCudaModel, triCudaModel, proximityContacts);
                            stageSnapshot.sofaDetectionOutputPublishMilliseconds += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - publishStart).count();
                        }
                        // Successfully handled this pair via cross-model v-t;
                        // skip the rest of the per-pair extraction/dispatch.
                        continue;
                    }
                    else if (!xmDiagnostic.empty() && !m_reportedFallback)
                    {
                        msg_warning() << "[GpuCollisionNarrowPhase] cross-model vertex-triangle path rejected: "
                                      << xmDiagnostic << " Falling back to legacy pipeline for this pair.";
                        m_reportedFallback = true;
                    }
                    // Cross-model v-t was requested but failed; fall through
                    // to the legacy CPU path below so SOFA still gets some
                    // contact data.
                    legacyPairs.push_back(pair);
                    continue;
                }
            }
        }

        bool firstOk = false;
        bool secondOk = false;
        const auto extractionStart = std::chrono::steady_clock::now();
        {
            ScopedNvtxRange extractionRange("CPU triangle extraction", d_detailedProfiling.getValue());
            if (tryIndexedDenseGrid)
            {
                firstOk = extractCudaIndexedSurface(pair.first, firstModel, firstIndexedSurface);
                secondOk = extractCudaIndexedSurface(pair.second, secondModel, secondIndexedSurface);
                usingIndexedDenseGrid = firstOk && secondOk;
            }
            else
            {
                firstOk = extractCudaTriangles(pair.first, firstModel, firstTriangles);
                secondOk = extractCudaTriangles(pair.second, secondModel, secondTriangles);
            }
        }
        const double extractionMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - extractionStart).count();
        stageSnapshot.sofaTriangleExtractionMilliseconds += extractionMs;
        stageSnapshot.hostPreparationMilliseconds += extractionMs;
        if (!firstOk || !secondOk)
        {
            legacyPairs.push_back(pair);
            continue;
        }

        backend::BackendExecutionStats backendStats;
        std::vector<backend::ExactContact> exactContacts;
        std::string diagnostic;

        bool exactSucceeded = false;
        if (m_backendAvailable && d_useDenseGrid.getValue())
        {
            backend::DenseGridConfig denseGridConfig;
            denseGridConfig.gridMinX = static_cast<float>(d_gridMinX.getValue());
            denseGridConfig.gridMinY = static_cast<float>(d_gridMinY.getValue());
            denseGridConfig.gridMinZ = static_cast<float>(d_gridMinZ.getValue());
            denseGridConfig.gridMaxX = static_cast<float>(d_gridMaxX.getValue());
            denseGridConfig.gridMaxY = static_cast<float>(d_gridMaxY.getValue());
            denseGridConfig.gridMaxZ = static_cast<float>(d_gridMaxZ.getValue());
            denseGridConfig.gridResolutionX = d_gridResolutionX.getValue();
            denseGridConfig.gridResolutionY = d_gridResolutionY.getValue();
            denseGridConfig.gridResolutionZ = d_gridResolutionZ.getValue();
            denseGridConfig.contactDistance = static_cast<float>(d_contactDistance.getValue());
            denseGridConfig.maxTissueTrianglesPerCell = d_maxTissueTrianglesPerCell.getValue();
            denseGridConfig.maxToolTrianglesPerCell = d_maxToolTrianglesPerCell.getValue();
            denseGridConfig.maxCandidatePairs = d_maxCandidatePairs.getValue();
            denseGridConfig.deduplicatePairs = d_deduplicatePairs.getValue();
            denseGridConfig.copyContactsToHost = d_copyContactsToHost.getValue();
            denseGridConfig.detailedProfiling = d_detailedProfiling.getValue();
            denseGridConfig.usePinnedHostStaging = d_usePinnedHostStaging.getValue();
            denseGridConfig.useGpuHashDedupe = d_useGpuHashDedupe.getValue();
            denseGridConfig.canonicalPairEmission = d_canonicalPairEmission.getValue();
            denseGridConfig.readCountersWhenContactsStayOnDevice =
                d_readCountersWhenContactsStayOnDevice.getValue() || d_detailedProfiling.getValue();
            denseGridConfig.computeDeviceContactsWhenContactsStayOnDevice =
                d_computeDeviceContactsWhenContactsStayOnDevice.getValue();
            denseGridConfig.compactActiveCells = d_compactActiveCells.getValue();
            denseGridConfig.batchTriangleInsert = d_batchTriangleInsert.getValue();
            denseGridConfig.useToolActiveCellGeneration = d_useToolActiveCellGeneration.getValue();

            // Build the shared FBP/v-t proximity config once.
            backend::FeatureBasedProximityConfig proximityConfig;
            proximityConfig.contactDistance = static_cast<float>(d_contactDistance.getValue());
            proximityConfig.emitOnePerPair = true;
            proximityConfig.computeBarycentrics = d_proximityComputeBarycentrics.getValue();
            // copyContactsToHost implies the host-side contacts vector must be
            // populated, so keepContactsOnDevice must be false in that mode.
            proximityConfig.keepContactsOnDevice =
                d_proximityKeepContactsOnDevice.getValue() && !d_copyContactsToHost.getValue();
            // Sampled counter readback: opt in via interval, or always when the
            // explicit flag or detailed-profiling mode is on.
            const auto readbackInterval = d_proximityCounterReadbackInterval.getValue();
            const bool sampledReadFrame =
                readbackInterval > 0 && (m_frameCounter % readbackInterval == 0);
            proximityConfig.readContactCounter =
                d_proximityReadContactCounter.getValue() ||
                d_detailedProfiling.getValue() ||
                sampledReadFrame;
            proximityConfig.maxContacts = d_proximityMaxContacts.getValue();

            // Self-collision vertex-triangle path. Triggers only when:
            //   - the user opted into both useFeatureBasedProximity AND useVertexTriangleProximity
            //   - the pair is (cm, cm) — i.e. broad phase emitted self-collision
            //   - we successfully extracted the indexed surface above
            const bool isSelfCollisionPair =
                pair.first == pair.second && pair.first != nullptr;
            const bool routeToVertexTriangle =
                usingIndexedDenseGrid &&
                d_useFeatureBasedProximity.getValue() &&
                d_useVertexTriangleProximity.getValue() &&
                isSelfCollisionPair;

            if (routeToVertexTriangle)
            {
                // Reuse firstIndexedSurface for both the triangle mesh and the
                // point cloud — they share device positions and surfaceId. The
                // matching surfaceId trips own-corner exclusion in the kernel.
                backend::PointCloudSurface selfPointCloud;
                selfPointCloud.positions       = firstIndexedSurface.positions;
                selfPointCloud.devicePositions = firstIndexedSurface.devicePositions;
                selfPointCloud.pointCount      = firstIndexedSurface.vertexCount;
                selfPointCloud.surfaceId       = firstIndexedSurface.surfaceId;
                selfPointCloud.topologyVersion = firstIndexedSurface.topologyVersion;

                std::vector<backend::ProximityContact> proximityContacts;
                backend::FeatureBasedProximityStats proximityStats;
                exactSucceeded = backend::computeFeatureBasedVertexTriangleContacts(
                    selfPointCloud,
                    firstIndexedSurface,
                    denseGridConfig,
                    proximityConfig,
                    proximityContacts,
                    &proximityStats,
                    diagnostic,
                    &backendStats);

                if (exactSucceeded)
                {
                    // Convert ProximityContact -> ExactContact for the existing
                    // SOFA publication path. The (vertex, triangle) pair is mapped
                    // so that firstTriangleIndex carries the vertex id and
                    // secondTriangleIndex carries the triangle id; this matches the
                    // backend's convention for vertex-triangle output.
                    // (firstPrimitiveIndex = vertex id, secondPrimitiveIndex = triangle id)
                    repackProximityContactsAsExact(proximityContacts, exactContacts);
                }
            }
            else if (usingIndexedDenseGrid && d_useFeatureBasedProximity.getValue())
            {
                // Feature-based proximity (VF + EE) path. Outputs ProximityContact
                // with barycentric weights consumable by a CUDA constraint solver.
                std::vector<backend::ProximityContact> proximityContacts;
                backend::FeatureBasedProximityStats proximityStats;
                if (d_useBigCellFusedGeneration.getValue())
                {
                    // 6th way: big-cell / small-cell fused generation + narrow
                    // phase. Home-cell dedup at small-cell granularity + the
                    // identical FBP math inline -> contacts match the other paths.
                    backend::BigCellConfig bigConfig;
                    bigConfig.bigCellFactor = d_bigCellFactor.getValue();
                    bigConfig.toolTileCapacity = d_bigCellToolTile.getValue();
                    bigConfig.useHashTableBuild = d_bigCellUseHashBuild.getValue();
                    bigConfig.hashSlotsPerBigCell = d_bigCellHashSlots.getValue();
                    backend::BigCellStats bigStats;
                    exactSucceeded = backend::computeBigCellFusedProximityContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        bigConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        &bigStats,
                        diagnostic,
                        &backendStats);
                }
                else if (d_useHashPrefixSumGeneration.getValue())
                {
                    // EXPERIMENTAL: spatial-hash + prefix-sum broad cull instead
                    // of the dense grid. Same FBP narrow kernel -> identical contacts.
                    backend::HashPrefixSumConfig hashConfig;
                    hashConfig.hashTableSize = d_hashTableSize.getValue();
                    backend::HashPrefixSumStats hashStats;
                    exactSucceeded = backend::computeHashPrefixSumProximityContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        hashConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        &hashStats,
                        diagnostic,
                        &backendStats);
                }
                else if (d_useSimpleHashGeneration.getValue())
                {
                    // 4th way: simple direct-bucket spatial hash. Same FBP narrow
                    // kernel -> contacts match the dense/hash paths (no overflow).
                    backend::HashPrefixSumConfig hashConfig;
                    hashConfig.hashTableSize = d_hashTableSize.getValue();
                    backend::HashPrefixSumStats hashStats;
                    exactSucceeded = backend::computeSimpleHashProximityContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        hashConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        &hashStats,
                        diagnostic,
                        &backendStats);
                }
                else if (d_useSortedGridGeneration.getValue())
                {
                    // 5th way: sorted-grid (tiled binning) broad cull. Same FBP
                    // narrow kernel -> contacts identical to the other paths.
                    backend::SortedGridConfig sortedConfig;
                    sortedConfig.useCubRadixSort = d_sortedGridUseCubSort.getValue();
                    sortedConfig.usePairHashDedup = d_sortedGridUsePairHashDedup.getValue();
                    backend::SortedGridStats sortedStats;
                    exactSucceeded = backend::computeSortedGridProximityContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        sortedConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        &sortedStats,
                        diagnostic,
                        &backendStats);
                }
                else
                {
                    exactSucceeded = backend::computeFeatureBasedProximityContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        proximityConfig,
                        proximityContacts,
                        &proximityStats,
                        diagnostic,
                        &backendStats);
                }

                // Re-pack ProximityContact -> ExactContact so the rest of the
                // SOFA DetectionOutput publication code path stays unchanged.
                // The barycentric weights live device-side for the constraint
                // solver; this CPU-side bridge is only used when the user
                // explicitly disabled proximityKeepContactsOnDevice.
                if (exactSucceeded)
                {
                    repackProximityContactsAsExact(proximityContacts, exactContacts);
                }
            }
            else if (usingIndexedDenseGrid)
            {
                exactSucceeded =
                    backend::computeDenseGridIndexedTriangleContacts(
                        firstIndexedSurface,
                        secondIndexedSurface,
                        denseGridConfig,
                        exactContacts,
                        diagnostic,
                        &backendStats);
            }
            else
            {
                exactSucceeded =
                    backend::computeDenseGridTriangleContacts(
                        *firstTriangles,
                        *secondTriangles,
                        denseGridConfig,
                        exactContacts,
                        diagnostic,
                        &backendStats);
            }

            if (!exactSucceeded && !diagnostic.empty() && !m_reportedFallback)
            {
                msg_warning() << "[GpuCollisionNarrowPhase] dense-grid path rejected: " << diagnostic
                              << " Falling back to exact all-pairs GPU contacts for correctness.";
                m_reportedFallback = true;
            }
        }

        if (!exactSucceeded)
        {
            if (firstTriangles == nullptr || secondTriangles == nullptr ||
                firstTriangles->empty() || secondTriangles->empty())
            {
                sofa::core::CollisionModel* ignoredFirstModel = nullptr;
                sofa::core::CollisionModel* ignoredSecondModel = nullptr;
                firstOk = extractCudaTriangles(pair.first, ignoredFirstModel, firstTriangles);
                secondOk = extractCudaTriangles(pair.second, ignoredSecondModel, secondTriangles);
                if (!firstOk || !secondOk)
                {
                    legacyPairs.push_back(pair);
                    continue;
                }
            }
            backendStats = backend::BackendExecutionStats {};
            exactContacts.clear();
            diagnostic.clear();
            exactSucceeded =
                m_backendAvailable &&
                backend::computeExactTriangleContacts(
                    *firstTriangles,
                    *secondTriangles,
                    exactContacts,
                    diagnostic,
                    &backendStats);
        }

        if (exactSucceeded)
        {
            accumulateBackendStats(stageSnapshot, backendStats);
            const auto publishStart = std::chrono::steady_clock::now();
            auto* firstCudaModel = dynamic_cast<CudaTriangleCollisionModel*>(firstModel);
            auto* secondCudaModel = dynamic_cast<CudaTriangleCollisionModel*>(secondModel);
            publishCudaTriangleContacts(this, firstCudaModel, secondCudaModel, exactContacts);
            stageSnapshot.sofaDetectionOutputPublishMilliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - publishStart).count();
            continue;
        }

        if (!diagnostic.empty() && !m_reportedFallback)
        {
            msg_warning() << "[GpuCollisionNarrowPhase] " << diagnostic;
            m_reportedFallback = true;
        }
        legacyPairs.push_back(pair);
    }
#else
    legacyPairs = m_pendingPairs;
#endif

    if (!legacyPairs.empty())
    {
        runLegacyPath(legacyPairs);
    }

#ifdef SOFAGPUCOLLISION_WITH_CUDA
    pruneTriangleTopologyCache();
#endif

    _zeroCollision = m_outputsMap.empty();
    stageSnapshot.wallMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
    // Derived: host-side sync overhead = whatever wall time was not GPU kernel time.
    // For the FBP path with the sync fix this collapses to a few hundred microseconds;
    // historically (pre-fix) this was the main wall-vs-kernel gap and worth surfacing.
    const double wallMinusKernel =
        stageSnapshot.wallMilliseconds - stageSnapshot.gpuKernelMilliseconds;
    stageSnapshot.hostSynchronizationMilliseconds = wallMinusKernel > 0.0 ? wallMinusKernel : 0.0;
    profiling::recordNarrowPhase(stageSnapshot);
}

} // namespace SofaGpuCollision
