#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/vector.h>

namespace sofa::component::statecontainer
{
template<class DataTypes>
class MechanicalObject;
}

namespace SofaGpuCollision
{

class SOFA_GPU_COLLISION_API GpuKinematicRigidController
    : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(GpuKinematicRigidController, sofa::core::objectmodel::BaseObject);

    GpuKinematicRigidController();
    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

private:
    using DataDouble = sofa::core::objectmodel::Data<double>;
    using DataInt = sofa::core::objectmodel::Data<int>;
    using DataVector = sofa::core::objectmodel::Data<sofa::type::vector<double>>;
    using RigidMechanicalObject = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types>;

    void onAnimateBegin();
    sofa::type::vector<double> computePose() const;

    DataVector d_startPosition;
    DataInt d_settleSteps;
    DataInt d_descendSteps;
    DataInt d_sweepSteps;
    DataInt d_liftSteps;
    DataDouble d_totalDown;
    DataDouble d_totalSweepX;
    DataDouble d_totalLift;
    DataDouble d_sweepSign;

    int m_currentStep { 0 };
    RigidMechanicalObject* m_rigidDofs { nullptr };
};

} // namespace SofaGpuCollision
