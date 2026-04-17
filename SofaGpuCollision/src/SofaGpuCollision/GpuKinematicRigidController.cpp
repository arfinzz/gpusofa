#include <SofaGpuCollision/GpuKinematicRigidController.h>

#include <algorithm>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/accessor/WriteAccessor.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace SofaGpuCollision
{

int GpuKinematicRigidControllerClass = sofa::core::RegisterObject(
    "Deterministic native C++ rigid-path controller for benchmark blades.")
    .add<GpuKinematicRigidController>();

GpuKinematicRigidController::GpuKinematicRigidController()
    : sofa::core::objectmodel::BaseObject()
    , d_startPosition(initData(&d_startPosition, sofa::type::vector<double> { 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0 }, "startPosition", "Initial rigid pose [x y z qx qy qz qw]."))
    , d_settleSteps(initData(&d_settleSteps, 100, "settleSteps", "Settle steps before motion starts."))
    , d_descendSteps(initData(&d_descendSteps, 350, "descendSteps", "Steps spent moving down into the tissue."))
    , d_sweepSteps(initData(&d_sweepSteps, 700, "sweepSteps", "Steps spent sweeping in X."))
    , d_liftSteps(initData(&d_liftSteps, 200, "liftSteps", "Steps spent lifting out of the tissue."))
    , d_totalDown(initData(&d_totalDown, 2.6, "totalDown", "Total downward displacement."))
    , d_totalSweepX(initData(&d_totalSweepX, 2.5, "totalSweepX", "Total sweep displacement in X before sweepSign is applied."))
    , d_totalLift(initData(&d_totalLift, 2.0, "totalLift", "Total lift displacement."))
    , d_sweepSign(initData(&d_sweepSign, 1.0, "sweepSign", "Sign applied to the X sweep direction."))
{
    f_listening.setValue(true);
}

void GpuKinematicRigidController::init()
{
    sofa::core::objectmodel::BaseObject::init();

    auto* context = this->getContext();
    if (context != nullptr)
    {
        m_rigidDofs = context->get<RigidMechanicalObject>(sofa::core::objectmodel::BaseContext::Local);
    }

    if (m_rigidDofs == nullptr)
    {
        msg_error() << "[GpuKinematicRigidController] No local MechanicalObject<Rigid3d> was found in this node.";
    }
}

void GpuKinematicRigidController::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent*>(event) != nullptr)
    {
        onAnimateBegin();
    }
}

sofa::type::vector<double> GpuKinematicRigidController::computePose() const
{
    const auto& start = d_startPosition.getValue();
    const double x0 = start.size() > 0 ? start[0] : 0.0;
    const double y0 = start.size() > 1 ? start[1] : 0.0;
    const double z0 = start.size() > 2 ? start[2] : 0.0;
    const double qx = start.size() > 3 ? start[3] : 0.0;
    const double qy = start.size() > 4 ? start[4] : 0.0;
    const double qz = start.size() > 5 ? start[5] : 0.0;
    const double qw = start.size() > 6 ? start[6] : 1.0;

    const int s0 = d_settleSteps.getValue();
    const int s1 = s0 + d_descendSteps.getValue();
    const int s2 = s1 + d_sweepSteps.getValue();
    const int s3 = s2 + d_liftSteps.getValue();

    auto lerp = [](const double a, const double b, const double t) { return a + (b - a) * t; };

    double x = x0;
    double y = y0;

    if (m_currentStep < s0)
    {
    }
    else if (m_currentStep < s1)
    {
        const double t = static_cast<double>(m_currentStep - s0) / std::max(1, d_descendSteps.getValue());
        y = lerp(y0, y0 - d_totalDown.getValue(), t);
    }
    else if (m_currentStep < s2)
    {
        const double t = static_cast<double>(m_currentStep - s1) / std::max(1, d_sweepSteps.getValue());
        y = y0 - d_totalDown.getValue();
        x = lerp(x0, x0 + d_sweepSign.getValue() * d_totalSweepX.getValue(), t);
    }
    else if (m_currentStep < s3)
    {
        const double t = static_cast<double>(m_currentStep - s2) / std::max(1, d_liftSteps.getValue());
        x = x0 + d_sweepSign.getValue() * d_totalSweepX.getValue();
        y = lerp(y0 - d_totalDown.getValue(), y0 - d_totalDown.getValue() + d_totalLift.getValue(), t);
    }
    else
    {
        x = x0 + d_sweepSign.getValue() * d_totalSweepX.getValue();
        y = y0 - d_totalDown.getValue() + d_totalLift.getValue();
    }

    return sofa::type::vector<double> { x, y, z0, qx, qy, qz, qw };
}

void GpuKinematicRigidController::onAnimateBegin()
{
    if (m_rigidDofs == nullptr)
    {
        return;
    }

    using PositionData = sofa::core::objectmodel::Data<typename RigidMechanicalObject::VecCoord>;
    auto* positionData = dynamic_cast<PositionData*>(m_rigidDofs->findData("position"));
    if (positionData == nullptr)
    {
        msg_error() << "[GpuKinematicRigidController] Could not access the rigid position data.";
        return;
    }

    sofa::helper::WriteAccessor<PositionData> positions(*positionData);
    if (positions.empty())
    {
        return;
    }

    const auto pose = computePose();
    const auto count = std::min(pose.size(), static_cast<std::size_t>(positions[0].size()));
    for (std::size_t i = 0; i < count; ++i)
    {
        positions[0][i] = pose[i];
    }

    ++m_currentStep;
}

} // namespace SofaGpuCollision
