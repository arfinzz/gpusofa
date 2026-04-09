#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuCollisionNarrowPhase.h>

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#include <utility>
#include <vector>

namespace SofaGpuCollision
{

namespace
{

using CubeCollisionModel = sofa::component::collision::geometry::CubeCollisionModel;

bool tryExtractBoundingTree(
    sofa::core::CollisionModel* collisionModel,
    std::vector<backend::AxisAlignedBoundingBox>& outTree)
{
    outTree.clear();

    auto* cubeModel = dynamic_cast<CubeCollisionModel*>(collisionModel == nullptr ? nullptr : collisionModel->getFirst());
    if (cubeModel == nullptr || cubeModel->empty() || cubeModel->getSize() == 0)
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
    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
        return;
    }

    std::string diagnostic;
    std::vector<std::uint32_t> gpuEligibleIndices;
    std::vector<backend::NarrowPhaseTreePair> gpuTreePairs;

    for (std::uint32_t i = 0; i < m_pendingPairs.size(); ++i)
    {
        backend::NarrowPhaseTreePair treePair;
        const bool firstOk = tryExtractBoundingTree(m_pendingPairs[i].first, treePair.firstTree);
        const bool secondOk = tryExtractBoundingTree(m_pendingPairs[i].second, treePair.secondTree);
        if (firstOk && secondOk)
        {
            gpuEligibleIndices.push_back(i);
            gpuTreePairs.push_back(std::move(treePair));
        }
    }

    if (gpuTreePairs.size() < d_minGpuPairCount.getValue())
    {
        for (const auto& pair : m_pendingPairs)
        {
            sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(pair);
        }
        sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
        return;
    }

    std::vector<std::uint32_t> survivingGpuPairIndices;
    const bool gpuSucceeded =
        m_backendAvailable && backend::prefilterNarrowPhasePairs(gpuTreePairs, survivingGpuPairIndices, diagnostic);

    if (gpuSucceeded)
    {
        std::vector<bool> keepPair(m_pendingPairs.size(), true);
        for (const auto gpuEligibleIndex : gpuEligibleIndices)
        {
            keepPair[gpuEligibleIndex] = false;
        }

        for (const auto localPairIndex : survivingGpuPairIndices)
        {
            keepPair[gpuEligibleIndices[localPairIndex]] = true;
        }

        for (std::size_t i = 0; i < m_pendingPairs.size(); ++i)
        {
            if (keepPair[i])
            {
                sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(m_pendingPairs[i]);
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
        for (const auto& pair : m_pendingPairs)
        {
            sofa::component::collision::detection::algorithm::BVHNarrowPhase::addCollisionPair(pair);
        }
    }

    sofa::component::collision::detection::algorithm::BVHNarrowPhase::endNarrowPhase();
}

} // namespace SofaGpuCollision
