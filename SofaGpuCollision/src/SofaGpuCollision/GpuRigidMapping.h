#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/gpu/cuda/CudaTypes.h>

#include <string>
#include <vector>

namespace SofaGpuCollision
{

// ============================================================================
// A rigid body's surface on the GPU: Rigid3d pose -> CudaVec3f points.
//
// What SofaCUDA's RigidMapping<Rigid3d,CudaVec3f> does, with a correct force
// mapping. SofaCUDA's GPU kernel for the forces (RigidMappingCuda3f_applyJT_kernel
// in SOFA v25.12) writes per-block partial FORCES into the torque slots after its
// reduction, so a body whose surface receives forces (penalty contact, a force
// field on the surface) gets a wrong torque, and rotates wrongly or blows up.
//
// The points are the child state's initial positions, in the body's own frame
// (as with SofaCUDA's RigidMapping). Positions and velocities are mapped on the
// GPU from the pose; the surface forces are summed on the GPU (in double) into a
// force and a torque about the body's centre, which is read back (48 bytes).
// Constraint rows (SOFA's CPU constraint solvers) are mapped on the CPU, as
// RigidMapping maps them.
// ============================================================================
class SOFA_GPU_COLLISION_API GpuRigidMapping
    : public sofa::core::Mapping<sofa::defaulttype::Rigid3dTypes, sofa::gpu::cuda::CudaVec3fTypes>
{
public:
    SOFA_CLASS(GpuRigidMapping, SOFA_TEMPLATE2(sofa::core::Mapping, sofa::defaulttype::Rigid3dTypes, sofa::gpu::cuda::CudaVec3fTypes));

    using In = sofa::defaulttype::Rigid3dTypes;
    using Out = sofa::gpu::cuda::CudaVec3fTypes;

    GpuRigidMapping();

    void init() override;
    void apply(const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    void applyJ(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    void applyJT(const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    void applyJT(const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in) override;

    sofa::core::objectmodel::Data<unsigned int> d_index;   // which rigid DOF of the input

private:
    void reportFailure(const std::string& what, const std::string& diagnostic);

    Out::VecCoord m_points;    // local coordinates (device-resident CudaVector)
    Out::VecCoord m_rotated;   // R p of the last apply
    bool m_ready { false };
    bool m_warned { false };
};

} // namespace SofaGpuCollision
