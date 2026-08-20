#include <SofaGpuCollision/CudaContactPenaltyForceField.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/logging/Messaging.h>

#ifdef SOFAGPUCOLLISION_WITH_CUDA
#include <SofaCUDA/component/collision/geometry/CudaTriangleModel.h>
#endif

namespace SofaGpuCollision
{

int CudaContactPenaltyForceFieldClass = sofa::core::RegisterObject(
    "GPU contact response: turns the device-resident contact buffer produced by "
    "GpuCollisionNarrowPhase into penalty forces without any device-to-host transfer.")
    .add<CudaContactPenaltyForceField>();

CudaContactPenaltyForceField::CudaContactPenaltyForceField()
    : d_stiffness(initData(&d_stiffness, static_cast<Real>(1000.0), "stiffness",
        "Penalty stiffness: F = stiffness * (contactDistance - distance). Higher is stiffer contact but needs a smaller timestep."))
    , d_damping(initData(&d_damping, static_cast<Real>(0.0), "damping",
        "Damping applied along the contact normal, opposing the approach velocity. Reduces bounce; 0 disables the velocity gather entirely."))
    , d_contactDistance(initData(&d_contactDistance, static_cast<Real>(0.03), "contactDistance",
        "Separation at which the penalty force starts. MUST match the contactDistance the contacts were generated with, otherwise the force turns on at the wrong distance."))
    , d_useDamping(initData(&d_useDamping, false, "useDamping",
        "Enable the damping term. Requires gathering velocities at each contact, so it costs extra device reads."))
    , d_reportStats(initData(&d_reportStats, false, "reportStats",
        "Read back contact/active counts each frame for diagnostics. Costs one synchronisation per frame; leave off in production."))
    , d_firstSurfaceIdOverride(initData(&d_firstSurfaceIdOverride, static_cast<unsigned int>(0), "firstSurfaceId",
        "Override the first surface id. 0 = derive from the CudaTriangleCollisionModel found in the first mechanical state's context (the same rule the narrow phase uses)."))
    , d_secondSurfaceIdOverride(initData(&d_secondSurfaceIdOverride, static_cast<unsigned int>(0), "secondSurfaceId",
        "Override the second surface id. 0 = derive from context."))
{
}

std::uint64_t CudaContactPenaltyForceField::resolveSurfaceId(
    const sofa::core::behavior::MechanicalState<DataTypes>* state, const bool first) const
{
    const unsigned int override = first
        ? d_firstSurfaceIdOverride.getValue()
        : d_secondSurfaceIdOverride.getValue();
    if (override != 0u)
    {
        return static_cast<std::uint64_t>(override);
    }

#ifdef SOFAGPUCOLLISION_WITH_CUDA
    if (state != nullptr && state->getContext() != nullptr)
    {
        // The narrow phase keys contacts by the CudaTriangleCollisionModel
        // pointer (extractCudaIndexedSurface: surfaceId = (uintptr_t)cudaModel),
        // so resolve the very same object from this state's node.
        //
        // SearchDown, because the collision surface commonly lives in a CHILD
        // node of the mechanical state (a tet volume with its boundary triangles
        // in a sub-node is the standard layout) — searching only the local node
        // finds the tool but silently misses the tissue.
        auto* model = state->getContext()->template get<sofa::gpu::cuda::CudaTriangleCollisionModel>(
            sofa::core::objectmodel::BaseContext::SearchDown);
        if (model != nullptr)
        {
            return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model));
        }
    }
#else
    SOFA_UNUSED(state);
#endif
    return 0u;
}

void CudaContactPenaltyForceField::init()
{
    Inherit::init();

    // Resolve ONCE. resolveSurfaceId walks the scene graph (get<T>(SearchDown)),
    // which is far too expensive to repeat twice per addForce, every frame.
    m_firstSurfaceId = resolveSurfaceId(this->mstate1, true);
    m_secondSurfaceId = resolveSurfaceId(this->mstate2, false);

    const auto firstId = m_firstSurfaceId;
    const auto secondId = m_secondSurfaceId;
    if (firstId == 0u || secondId == 0u)
    {
        msg_warning() << "Could not resolve both surface ids from context "
                         "(first=" << firstId << ", second=" << secondId << "). "
                         "Attach this force field to the nodes holding the CudaTriangleCollisionModels, "
                         "or set firstSurfaceId/secondSurfaceId explicitly. No contact forces will be applied.";
    }
}

void CudaContactPenaltyForceField::addForce(
    const sofa::core::MechanicalParams* /*mparams*/,
    DataVecDeriv& f1, DataVecDeriv& f2,
    const DataVecCoord& /*x1*/, const DataVecCoord& /*x2*/,
    const DataVecDeriv& v1, const DataVecDeriv& v2)
{
    const auto firstId = m_firstSurfaceId;
    const auto secondId = m_secondSurfaceId;
    if (firstId == 0u || secondId == 0u)
    {
        return;
    }

    backend::ContactPenaltyConfig config;
    config.stiffness = static_cast<float>(d_stiffness.getValue());
    config.damping = d_useDamping.getValue() ? static_cast<float>(d_damping.getValue()) : 0.0f;
    config.contactDistance = static_cast<float>(d_contactDistance.getValue());

    // deviceWrite()/deviceRead() ONLY. helper::WriteAccessor would call
    // hostWrite(), copying the whole force vector to the host and invalidating
    // the device copy — exactly the per-frame transfer this component exists
    // to avoid.
    VecDeriv& force1 = *f1.beginEdit();
    VecDeriv& force2 = *f2.beginEdit();
    void* deviceForce1 = force1.deviceWrite();
    void* deviceForce2 = force2.deviceWrite();

    const void* deviceVel1 = nullptr;
    const void* deviceVel2 = nullptr;
    if (config.damping > 0.0f)
    {
        deviceVel1 = v1.getValue().deviceRead();
        deviceVel2 = v2.getValue().deviceRead();
    }

    backend::ContactPenaltyStats stats;
    std::string diagnostic;
    const bool ok = backend::accumulateContactPenaltyForces(
        config, firstId, secondId,
        deviceForce1, deviceForce2,
        deviceVel1, deviceVel2,
        d_reportStats.getValue() ? &stats : nullptr,
        diagnostic);

    f1.endEdit();
    f2.endEdit();

    if (!ok)
    {
        if (!m_reportedFailure)
        {
            msg_warning() << "GPU contact penalty forces skipped: " << diagnostic
                          << " (further occurrences suppressed)";
            m_reportedFailure = true;
        }
        return;
    }

    if (d_reportStats.getValue())
    {
        m_lastContactCount = stats.contactCount;
        m_lastActiveContactCount = stats.activeContactCount;
        // Log on change so a run shows the contact history without one line per
        // frame. Per-INSTANCE state: a function-local `static` would be shared by
        // every force field in the scene and would survive a scene reload,
        // silently suppressing another instance's first report.
        if (m_lastActiveContactCount != m_lastLoggedActiveCount)
        {
            m_lastLoggedActiveCount = m_lastActiveContactCount;
            msg_info() << "contacts=" << m_lastContactCount
                       << " active=" << m_lastActiveContactCount;
        }
    }
}

void CudaContactPenaltyForceField::addDForce(
    const sofa::core::MechanicalParams* mparams,
    DataVecDeriv& df1, DataVecDeriv& df2,
    const DataVecDeriv& dx1, const DataVecDeriv& dx2)
{
    const auto firstId = m_firstSurfaceId;
    const auto secondId = m_secondSurfaceId;
    if (firstId == 0u || secondId == 0u)
    {
        return;
    }

    backend::ContactPenaltyConfig config;
    config.stiffness = static_cast<float>(d_stiffness.getValue());
    config.damping = 0.0f;  // the damping term's derivative is handled by SOFA's b-factor
    config.contactDistance = static_cast<float>(d_contactDistance.getValue());

    const auto kFactor = static_cast<float>(
        sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue()));

    VecDeriv& dforce1 = *df1.beginEdit();
    VecDeriv& dforce2 = *df2.beginEdit();

    std::string diagnostic;
    const bool ok = backend::accumulateContactPenaltyDForces(
        config, firstId, secondId, kFactor,
        dforce1.deviceWrite(), dforce2.deviceWrite(),
        dx1.getValue().deviceRead(), dx2.getValue().deviceRead(),
        diagnostic);

    df1.endEdit();
    df2.endEdit();

    if (!ok && !m_reportedFailure)
    {
        msg_warning() << "GPU contact penalty dforces skipped: " << diagnostic
                      << " (further occurrences suppressed)";
        m_reportedFailure = true;
    }
}

SReal CudaContactPenaltyForceField::getPotentialEnergy(
    const sofa::core::MechanicalParams* /*mparams*/,
    const DataVecCoord& /*x1*/, const DataVecCoord& /*x2*/) const
{
    // Would require a device reduction over the contact buffer plus a readback
    // — i.e. exactly the synchronisation this path avoids. SOFA only needs this
    // for energy diagnostics, not for the simulation itself.
    return 0.0;
}

} // namespace SofaGpuCollision
