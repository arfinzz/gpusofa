#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>
#include <sofa/core/objectmodel/Data.h>

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
    using CollisionModelPair = std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>;

    DataBool d_enableGpu;
    DataBool d_allowCpuFallback;
    DataBool d_logBackendStatus;
    sofa::core::objectmodel::Data<unsigned int> d_minGpuPairCount;

    std::vector<CollisionModelPair> m_pendingPairs;
    bool m_backendAvailable { false };
    bool m_reportedFallback { false };
};

} // namespace SofaGpuCollision
