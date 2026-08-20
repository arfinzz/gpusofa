#pragma once

#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/config.h>

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/gpu/cuda/CudaTypes.h>

#include <cstdint>
#include <string>

namespace SofaGpuCollision
{

// ============================================================================
// Tier 1 — the contact CONSUMER.
//
// GpuCollisionNarrowPhase leaves its contacts in a device buffer. This force
// field turns that buffer into forces on the two surfaces' vertices without the
// contacts, positions, velocities or forces ever visiting the host:
//
//   collision detection (GPU)  ->  device contact buffer
//                                        |
//                              this component's kernel
//                                        v
//                        += into SOFA's CudaVec3f force vectors (GPU)
//
// The whole point is the accessor discipline: force/velocity/dx are reached via
// CudaVector::deviceWrite()/deviceRead(), NEVER through helper::ReadAccessor or
// WriteAccessor. Those call hostRead()/hostWrite(), which copy the entire state
// vector back to the host and mark the device copy stale — one such call
// anywhere in the frame silently reintroduces the per-frame transfer this
// project exists to remove.
//
// Response law (proximity penalty, matching how the contacts were generated —
// they only exist within contactDistance in the first place):
//
//     depth = contactDistance - distance
//     F     = max(0, stiffness * depth - damping * relativeNormalVelocity)
//
// Use with DefaultAnimationLoop, whose step order is collision -> integrate, so
// the buffer produced during collision is read while the solver evaluates forces.
// ============================================================================
class SOFA_GPU_COLLISION_API CudaContactPenaltyForceField
    : public sofa::core::behavior::PairInteractionForceField<sofa::gpu::cuda::CudaVec3fTypes>
{
public:
    using DataTypes = sofa::gpu::cuda::CudaVec3fTypes;
    using Inherit = sofa::core::behavior::PairInteractionForceField<DataTypes>;
    SOFA_CLASS(CudaContactPenaltyForceField, SOFA_TEMPLATE(sofa::core::behavior::PairInteractionForceField, DataTypes));

    using VecCoord = DataTypes::VecCoord;
    using VecDeriv = DataTypes::VecDeriv;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using Real = DataTypes::Real;

    CudaContactPenaltyForceField();
    ~CudaContactPenaltyForceField() override = default;

    void init() override;

    void addForce(
        const sofa::core::MechanicalParams* mparams,
        DataVecDeriv& f1, DataVecDeriv& f2,
        const DataVecCoord& x1, const DataVecCoord& x2,
        const DataVecDeriv& v1, const DataVecDeriv& v2) override;

    void addDForce(
        const sofa::core::MechanicalParams* mparams,
        DataVecDeriv& df1, DataVecDeriv& df2,
        const DataVecDeriv& dx1, const DataVecDeriv& dx2) override;

    SReal getPotentialEnergy(
        const sofa::core::MechanicalParams* mparams,
        const DataVecCoord& x1, const DataVecCoord& x2) const override;

private:
    using DataReal = sofa::core::objectmodel::Data<Real>;
    using DataBool = sofa::core::objectmodel::Data<bool>;
    using DataUInt = sofa::core::objectmodel::Data<unsigned int>;

    DataReal d_stiffness;
    DataReal d_damping;
    DataReal d_contactDistance;
    DataBool d_useDamping;
    DataBool d_reportStats;          ///< costs one sync per frame; off by default
    DataUInt d_firstSurfaceIdOverride;   ///< 0 = derive from the mechanical state pointer
    DataUInt d_secondSurfaceIdOverride;

    /// Surface ids must match the ones the narrow phase recorded with the
    /// contacts. GpuCollisionNarrowPhase derives them from the collision-model
    /// pointer, so this component resolves the same way.
    std::uint64_t resolveSurfaceId(const sofa::core::behavior::MechanicalState<DataTypes>* state, bool first) const;

    /// Resolved once in init(): resolveSurfaceId walks the scene graph, which is
    /// far too expensive to repeat twice per addForce, every frame.
    std::uint64_t m_firstSurfaceId { 0 };
    std::uint64_t m_secondSurfaceId { 0 };

    bool m_reportedFailure { false };
    std::uint32_t m_lastContactCount { 0 };
    std::uint32_t m_lastActiveContactCount { 0 };
    std::uint32_t m_lastLoggedActiveCount { 0xffffffffu };
};

} // namespace SofaGpuCollision
