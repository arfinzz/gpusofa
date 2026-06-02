#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/component/collision/detection/algorithm/BruteForceBroadPhase.h>
#include <sofa/core/objectmodel/Data.h>

#include <vector>

namespace SofaGpuCollision
{

class SOFA_GPU_COLLISION_API GpuCollisionBroadPhase
    : public sofa::component::collision::detection::algorithm::BruteForceBroadPhase
{
public:
    SOFA_CLASS(GpuCollisionBroadPhase, sofa::component::collision::detection::algorithm::BruteForceBroadPhase);

    GpuCollisionBroadPhase();
    ~GpuCollisionBroadPhase() override = default;

    void init() override;
    void beginBroadPhase() override;
    void addCollisionModel(sofa::core::CollisionModel* cm) override;
    void endBroadPhase() override;

private:
    using DataBool = sofa::core::objectmodel::Data<bool>;

    DataBool d_enableGpu;
    DataBool d_allowCpuFallback;
    DataBool d_logBackendStatus;
    DataBool d_logBoxesOnce;
    DataBool d_useObjectAabbCulling;

    std::vector<sofa::core::CollisionModel*> m_pendingModels;
    bool m_backendAvailable { false };
    bool m_reportedFallback { false };
    bool m_loggedBoxes { false };
};

} // namespace SofaGpuCollision
