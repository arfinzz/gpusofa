#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuCollisionBroadPhase.h>
#include <SofaGpuCollision/GpuPipelineProfiling.h>

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

namespace SofaGpuCollision
{

namespace
{

using Cube = sofa::component::collision::geometry::Cube;
using CubeCollisionModel = sofa::component::collision::geometry::CubeCollisionModel;
using TriangleCollisionModel = sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>;

struct CollisionModelEntry
{
    sofa::core::CollisionModel* firstCollisionModel { nullptr };
    sofa::core::CollisionModel* lastCollisionModel { nullptr };
    bool gpuEligible { false };
    backend::AxisAlignedBoundingBox box;
};

bool tryExtractRootAabb(
    sofa::core::CollisionModel* collisionModel,
    backend::AxisAlignedBoundingBox& outBox)
{
    auto* triangleModel = dynamic_cast<TriangleCollisionModel*>(collisionModel == nullptr ? nullptr : collisionModel->getLast());
    if (triangleModel != nullptr && !triangleModel->empty())
    {
        const auto& positions = triangleModel->getX();
        const auto& triangles = triangleModel->getTriangles();
        if (!positions.empty() && !triangles.empty())
        {
            const auto& firstTriangle = triangles.front();
            const auto& seed = positions[firstTriangle[0]];

            float minX = static_cast<float>(seed[0]);
            float minY = static_cast<float>(seed[1]);
            float minZ = static_cast<float>(seed[2]);
            float maxX = minX;
            float maxY = minY;
            float maxZ = minZ;

            for (const auto& triangle : triangles)
            {
                for (unsigned int vertexOffset = 0; vertexOffset < 3; ++vertexOffset)
                {
                    const auto& p = positions[triangle[vertexOffset]];
                    minX = std::min(minX, static_cast<float>(p[0]));
                    minY = std::min(minY, static_cast<float>(p[1]));
                    minZ = std::min(minZ, static_cast<float>(p[2]));
                    maxX = std::max(maxX, static_cast<float>(p[0]));
                    maxY = std::max(maxY, static_cast<float>(p[1]));
                    maxZ = std::max(maxZ, static_cast<float>(p[2]));
                }
            }

            outBox.minX = minX;
            outBox.minY = minY;
            outBox.minZ = minZ;
            outBox.maxX = maxX;
            outBox.maxY = maxY;
            outBox.maxZ = maxZ;
            return true;
        }
    }

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
    , d_logBoxesOnce(initData(&d_logBoxesOnce, false, "logBoxesOnce", "Log the first-frame root AABBs collected for GPU broad phase."))
    , d_useObjectAabbCulling(initData(&d_useObjectAabbCulling, false, "useObjectAabbCulling", "Use object-level GPU AABB culling before narrow phase. Disable this for one tissue/one tool surgical scenes."))
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
    const auto phaseStart = std::chrono::steady_clock::now();
    profiling::StageSnapshot stageSnapshot;

    if (!d_enableGpu.getValue())
    {
        sofa::component::collision::detection::algorithm::BruteForceBroadPhase::endBroadPhase();
        stageSnapshot.cpuFallbackUsed = true;
        stageSnapshot.wallMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
        profiling::recordBroadPhase(stageSnapshot);
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

    if (!d_useObjectAabbCulling.getValue())
    {
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
            entries.push_back(entry);

            if (doesSelfCollide(cm))
            {
                this->cmPairs.emplace_back(cm, cm);
            }
        }

        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                const auto& a = entries[i];
                const auto& b = entries[j];
                appendPotentialPair(a.firstCollisionModel, a.lastCollisionModel, b.firstCollisionModel, b.lastCollisionModel);
            }
        }

        stageSnapshot.inputPrimitiveCount = static_cast<std::uint32_t>(entries.size());
        stageSnapshot.outputPairCount = static_cast<std::uint32_t>(this->cmPairs.size());
        stageSnapshot.wallMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
        profiling::recordBroadPhase(stageSnapshot);
        return;
    }

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

    if (d_logBoxesOnce.getValue() && !m_loggedBoxes)
    {
        m_loggedBoxes = true;
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            if (!entry.gpuEligible)
            {
                msg_info() << "[GpuCollisionBroadPhase] model[" << i << "] gpuEligible=false";
                continue;
            }

            msg_info() << "[GpuCollisionBroadPhase] model[" << i << "] "
                       << "min=(" << entry.box.minX << ", " << entry.box.minY << ", " << entry.box.minZ << ") "
                       << "max=(" << entry.box.maxX << ", " << entry.box.maxY << ", " << entry.box.maxZ << ")";
        }
    }

    stageSnapshot.inputPrimitiveCount = static_cast<std::uint32_t>(gpuBoxes.size());

    bool gpuSucceeded = false;
    std::string diagnostic;
    std::vector<backend::BroadPhaseIndexPair> gpuPairs;
    backend::BackendExecutionStats backendStats;

    if (m_backendAvailable && gpuBoxes.size() >= 2)
    {
        gpuSucceeded = backend::computeBroadPhasePairs(gpuBoxes, gpuPairs, diagnostic, &backendStats);
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
        stageSnapshot.cpuFallbackUsed = true;
        stageSnapshot.wallMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
        profiling::recordBroadPhase(stageSnapshot);
        return;
    }

    if (gpuSucceeded)
    {
        stageSnapshot.gpuUsed = true;
        stageSnapshot.gpuKernelMilliseconds = backendStats.gpuKernelMilliseconds;
        stageSnapshot.hostToDeviceBytes = backendStats.hostToDeviceBytes;
        stageSnapshot.deviceToHostBytes = backendStats.deviceToHostBytes;
        stageSnapshot.deviceAllocationBytes = backendStats.deviceAllocationBytes;
        stageSnapshot.kernelLaunchCount = backendStats.kernelLaunchCount;
        stageSnapshot.inputPrimitiveCount = backendStats.inputPrimitiveCount;
        stageSnapshot.outputPairCount = backendStats.outputPairCount;
        for (const auto& pair : gpuPairs)
        {
            const auto entryIndexA = gpuEntryIndices[pair.first];
            const auto entryIndexB = gpuEntryIndices[pair.second];
            const auto& a = entries[entryIndexA];
            const auto& b = entries[entryIndexB];
            appendPotentialPair(a.firstCollisionModel, a.lastCollisionModel, b.firstCollisionModel, b.lastCollisionModel);
        }
    }
    else
    {
        stageSnapshot.cpuFallbackUsed = true;
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

    stageSnapshot.wallMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
    profiling::recordBroadPhase(stageSnapshot);
}

} // namespace SofaGpuCollision
