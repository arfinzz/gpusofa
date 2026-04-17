#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuCollisionNarrowPhase.h>
#include <SofaGpuCollision/GpuPipelineProfiling.h>

#ifdef SOFAGPUCOLLISION_WITH_CUDA
#include <SofaCUDA/component/collision/geometry/CudaTriangleModel.h>
#endif

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace SofaGpuCollision
{

namespace
{

using CubeCollisionModel = sofa::component::collision::geometry::CubeCollisionModel;

#ifdef SOFAGPUCOLLISION_WITH_CUDA
using CudaTriangleCollisionModel = sofa::gpu::cuda::CudaTriangleCollisionModel;
using CudaTriangleOutputVector =
    sofa::core::collision::TDetectionOutputVector<CudaTriangleCollisionModel, CudaTriangleCollisionModel>;
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

#ifdef SOFAGPUCOLLISION_WITH_CUDA
bool tryExtractCudaTriangles(
    sofa::core::CollisionModel* collisionModel,
    CudaTriangleCollisionModel*& outModel,
    std::vector<backend::TrianglePrimitive>& outTriangles)
{
    outTriangles.clear();
    outModel = dynamic_cast<CudaTriangleCollisionModel*>(collisionModel == nullptr ? nullptr : collisionModel->getLast());
    if (outModel == nullptr || outModel->empty())
    {
        return false;
    }

    const auto& positions = outModel->getX();
    const auto& triangles = outModel->getTriangles();
    if (positions.empty() || triangles.empty())
    {
        return false;
    }

    outTriangles.reserve(triangles.size());
    for (std::uint32_t triangleIndex = 0; triangleIndex < triangles.size(); ++triangleIndex)
    {
        const auto& triangle = triangles[triangleIndex];
        const auto& p0 = positions[triangle[0]];
        const auto& p1 = positions[triangle[1]];
        const auto& p2 = positions[triangle[2]];
        outTriangles.push_back(backend::TrianglePrimitive {
            backend::TriangleVertex { static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2]) },
            backend::TriangleVertex { static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2]) },
            backend::TriangleVertex { static_cast<float>(p2[0]), static_cast<float>(p2[1]), static_cast<float>(p2[2]) },
            triangleIndex,
        });
    }

    return true;
}
#endif

void accumulateBackendStats(profiling::StageSnapshot& stageSnapshot, const backend::BackendExecutionStats& stats)
{
    stageSnapshot.gpuUsed = true;
    stageSnapshot.gpuKernelMilliseconds += stats.gpuKernelMilliseconds;
    stageSnapshot.hostToDeviceBytes += stats.hostToDeviceBytes;
    stageSnapshot.deviceToHostBytes += stats.deviceToHostBytes;
    stageSnapshot.deviceAllocationBytes += stats.deviceAllocationBytes;
    stageSnapshot.kernelLaunchCount += stats.kernelLaunchCount;
    stageSnapshot.inputPrimitiveCount += stats.inputPrimitiveCount;
    stageSnapshot.outputPairCount += stats.outputPairCount;
    stageSnapshot.outputCandidateCount += stats.outputCandidateCount;
}

#ifdef SOFAGPUCOLLISION_WITH_CUDA
void publishCudaTriangleContacts(
    sofa::core::collision::NarrowPhaseDetection* narrowPhase,
    CudaTriangleCollisionModel* firstModel,
    CudaTriangleCollisionModel* secondModel,
    const std::vector<backend::ExactContact>& contacts)
{
    if (contacts.empty())
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
    , d_minGpuPairCount(initData(&d_minGpuPairCount, static_cast<unsigned int>(8), "minGPUPairCount", "Minimum number of candidate pairs before the GPU narrow-phase prefilter is worth using."))
{
}

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
        CudaTriangleCollisionModel* firstModel = nullptr;
        CudaTriangleCollisionModel* secondModel = nullptr;
        std::vector<backend::TrianglePrimitive> firstTriangles;
        std::vector<backend::TrianglePrimitive> secondTriangles;

        const bool firstOk = tryExtractCudaTriangles(pair.first, firstModel, firstTriangles);
        const bool secondOk = tryExtractCudaTriangles(pair.second, secondModel, secondTriangles);
        if (!firstOk || !secondOk)
        {
            legacyPairs.push_back(pair);
            continue;
        }

        backend::BackendExecutionStats backendStats;
        std::vector<backend::ExactContact> exactContacts;
        std::string diagnostic;
        const bool exactSucceeded =
            m_backendAvailable &&
            backend::computeExactTriangleContacts(
                firstTriangles,
                secondTriangles,
                exactContacts,
                diagnostic,
                &backendStats);

        if (exactSucceeded)
        {
            accumulateBackendStats(stageSnapshot, backendStats);
            publishCudaTriangleContacts(this, firstModel, secondModel, exactContacts);
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

    _zeroCollision = m_outputsMap.empty();
    stageSnapshot.wallMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
    profiling::recordNarrowPhase(stageSnapshot);
}

} // namespace SofaGpuCollision
