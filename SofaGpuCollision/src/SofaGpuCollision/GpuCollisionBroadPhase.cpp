#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuCollisionBroadPhase.h>

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace SofaGpuCollision
{

namespace
{

using Cube = sofa::component::collision::geometry::Cube;
using CubeCollisionModel = sofa::component::collision::geometry::CubeCollisionModel;

struct CollisionModelEntry
{
    sofa::core::CollisionModel* firstCollisionModel { nullptr };
    sofa::core::CollisionModel* lastCollisionModel { nullptr };
    bool gpuEligible { false };
    backend::AxisAlignedBoundingBox box;
};

bool tryExtractRootAabb(
    sofa::core::CollisionModel* rootCollisionModel,
    backend::AxisAlignedBoundingBox& outBox)
{
    auto* cubeModel = dynamic_cast<CubeCollisionModel*>(rootCollisionModel);
    if (cubeModel == nullptr || cubeModel->empty() || cubeModel->getSize() == 0)
    {
        return false;
    }

    const Cube rootCube(cubeModel, 0);
    const auto& min = rootCube.minVect();
    const auto& max = rootCube.maxVect();

    outBox.minX = static_cast<float>(min[0]);
    outBox.minY = static_cast<float>(min[1]);
    outBox.minZ = static_cast<float>(min[2]);
    outBox.maxX = static_cast<float>(max[0]);
    outBox.maxY = static_cast<float>(max[1]);
    outBox.maxZ = static_cast<float>(max[2]);
    return true;
}

} // namespace

int GpuCollisionBroadPhaseClass = sofa::core::RegisterObject(
    "GPU-first broad phase collision detection with a safe CPU fallback.")
    .add<GpuCollisionBroadPhase>();

GpuCollisionBroadPhase::GpuCollisionBroadPhase()
    : sofa::component::collision::detection::algorithm::BruteForceBroadPhase()
    , d_enableGpu(initData(&d_enableGpu, true, "enableGPU", "Try to execute the GPU broad phase backend."))
    , d_allowCpuFallback(initData(&d_allowCpuFallback, true, "allowCPUFallback", "Use the SOFA CPU broad phase if GPU execution is unavailable."))
    , d_logBackendStatus(initData(&d_logBackendStatus, true, "logBackendStatus", "Log the selected broad phase backend during init."))
{
}

void GpuCollisionBroadPhase::init()
{
    sofa::component::collision::detection::algorithm::BruteForceBroadPhase::init();

    const auto status = backend::probe();
    m_backendAvailable = status.available;
    m_reportedFallback = false;

    if (d_logBackendStatus.getValue())
    {
        if (status.available)
        {
            msg_info() << "[GpuCollisionBroadPhase] " << status.message;
        }
        else
        {
            msg_warning() << "[GpuCollisionBroadPhase] " << status.message;
        }
    }
}

void GpuCollisionBroadPhase::beginBroadPhase()
{
    m_pendingModels.clear();
    sofa::component::collision::detection::algorithm::BruteForceBroadPhase::beginBroadPhase();
}

void GpuCollisionBroadPhase::addCollisionModel(sofa::core::CollisionModel* cm)
{
    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BruteForceBroadPhase::addCollisionModel(cm);
        return;
    }

    if (cm != nullptr)
    {
        m_pendingModels.push_back(cm);
    }
}

void GpuCollisionBroadPhase::endBroadPhase()
{
    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BruteForceBroadPhase::endBroadPhase();
        return;
    }

    auto appendPotentialPair = [this](
        sofa::core::CollisionModel* firstA,
        sofa::core::CollisionModel* lastA,
        sofa::core::CollisionModel* firstB,
        sofa::core::CollisionModel* lastB) -> void
    {
        if (!lastA->isSimulated() && !lastB->isSimulated())
        {
            return;
        }

        if (!sofa::component::collision::detection::algorithm::BruteForceBroadPhase::keepCollisionBetween(lastA, lastB))
        {
            return;
        }

        bool swapModels = false;
        auto* cm1 = firstA;
        auto* cm2 = firstB;

        auto* intersector = this->intersectionMethod->findIntersector(cm1, cm2, swapModels);
        if (intersector == nullptr)
        {
            return;
        }

        if (swapModels)
        {
            std::swap(cm1, cm2);
        }

        if (!intersector->canIntersect(cm1->begin(), cm2->begin(), this->intersectionMethod))
        {
            return;
        }

        this->cmPairs.emplace_back(cm1, cm2);
    };

    std::vector<CollisionModelEntry> entries;
    entries.reserve(m_pendingModels.size());

    for (auto* cm : m_pendingModels)
    {
        if (cm == nullptr || cm->empty())
        {
            continue;
        }

        CollisionModelEntry entry;
        entry.firstCollisionModel = cm;
        entry.lastCollisionModel = cm->getLast();
        entry.gpuEligible = tryExtractRootAabb(cm->getFirst(), entry.box);
        entries.push_back(entry);

        if (doesSelfCollide(cm))
        {
            this->cmPairs.emplace_back(cm, cm);
        }
    }

    std::vector<std::uint32_t> gpuEntryIndices;
    std::vector<backend::AxisAlignedBoundingBox> gpuBoxes;
    gpuEntryIndices.reserve(entries.size());
    gpuBoxes.reserve(entries.size());

    for (std::uint32_t i = 0; i < entries.size(); ++i)
    {
        if (entries[i].gpuEligible)
        {
            gpuEntryIndices.push_back(i);
            gpuBoxes.push_back(entries[i].box);
        }
    }

    bool gpuSucceeded = false;
    std::string diagnostic;
    std::vector<backend::BroadPhaseIndexPair> gpuPairs;

    if (m_backendAvailable && gpuBoxes.size() >= 2)
    {
        gpuSucceeded = backend::computeBroadPhasePairs(gpuBoxes, gpuPairs, diagnostic);
    }
    else if (m_backendAvailable)
    {
        gpuSucceeded = true;
    }

    if (!gpuSucceeded && !diagnostic.empty() && !m_reportedFallback)
    {
        msg_warning() << "[GpuCollisionBroadPhase] " << diagnostic;
        m_reportedFallback = true;
    }

    if (!gpuSucceeded && !d_allowCpuFallback.getValue())
    {
        this->cmPairs.clear();
        return;
    }

    if (gpuSucceeded)
    {
        for (const auto& pair : gpuPairs)
        {
            const auto entryIndexA = gpuEntryIndices[pair.first];
            const auto entryIndexB = gpuEntryIndices[pair.second];
            const auto& a = entries[entryIndexA];
            const auto& b = entries[entryIndexB];
            appendPotentialPair(a.firstCollisionModel, a.lastCollisionModel, b.firstCollisionModel, b.lastCollisionModel);
        }
    }

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            if (gpuSucceeded && entries[i].gpuEligible && entries[j].gpuEligible)
            {
                continue;
            }

            const auto& a = entries[i];
            const auto& b = entries[j];
            appendPotentialPair(a.firstCollisionModel, a.lastCollisionModel, b.firstCollisionModel, b.lastCollisionModel);
        }
    }
}

} // namespace SofaGpuCollision
