#include <SofaGpuCollision/GpuRigidMapping.h>

#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/Mapping.inl>
#include <sofa/helper/logging/Messaging.h>

namespace SofaGpuCollision
{

int GpuRigidMappingClass = sofa::core::RegisterObject(
    "A rigid body's surface on the GPU (Rigid3d -> CudaVec3f), with a correct force/torque mapping "
    "(SofaCUDA's RigidMapping<Rigid3d,CudaVec3f> maps surface forces to a wrong torque in SOFA v25.12).")
    .add<GpuRigidMapping>();

GpuRigidMapping::GpuRigidMapping()
    : d_index(initData(&d_index, 0u, "index", "Index of the rigid DOF of the input state."))
{
}

void GpuRigidMapping::reportFailure(const std::string& what, const std::string& diagnostic)
{
    if (m_warned) return;
    m_warned = true;
    msg_error() << what << " failed: " << diagnostic << " (further failures are not reported).";
}

void GpuRigidMapping::init()
{
    // The child's initial positions are the points in the body's own frame.
    const auto* child = this->toModel.get();
    if (child == nullptr || this->fromModel.get() == nullptr)
    {
        msg_error() << "GpuRigidMapping needs a Rigid3d input and a CudaVec3f output state.";
        return;
    }
    const auto& positions = child->read(sofa::core::vec_id::read_access::position)->getValue();
    m_points.resize(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) m_points[i] = positions[i];
    m_rotated.resize(positions.size());
    m_ready = true;
    Inherit1::init();
}

void GpuRigidMapping::apply(const sofa::core::MechanicalParams*, OutDataVecCoord& dOut, const InDataVecCoord& dIn)
{
    if (!m_ready) return;
    const auto& in = dIn.getValue();
    const auto& pose = in[d_index.getValue()];
    double rotation[9];
    {
        sofa::type::Mat<3, 3, double> r;
        pose.writeRotationMatrix(r);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) rotation[3 * i + j] = r[i][j];
    }
    const double translation[3] = { pose.getCenter()[0], pose.getCenter()[1], pose.getCenter()[2] };
    auto& out = *dOut.beginEdit();
    const int n = static_cast<int>(m_points.size());
    if (out.size() != m_points.size()) out.resize(m_points.size());
    std::string diagnostic;
    if (!backend::rigidMappingApply(n, rotation, translation, m_points.deviceRead(), out.deviceWrite(), m_rotated.deviceWrite(), diagnostic))
        reportFailure("positions", diagnostic);
    dOut.endEdit();
}

void GpuRigidMapping::applyJ(const sofa::core::MechanicalParams*, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn)
{
    if (!m_ready) return;
    const auto& in = dIn.getValue();
    const auto& d = in[d_index.getValue()];
    const double velocity[3] = { d.getVCenter()[0], d.getVCenter()[1], d.getVCenter()[2] };
    const double angular[3] = { d.getVOrientation()[0], d.getVOrientation()[1], d.getVOrientation()[2] };
    auto& out = *dOut.beginEdit();
    if (out.size() != m_points.size()) out.resize(m_points.size());
    std::string diagnostic;
    if (!backend::rigidMappingApplyJ(static_cast<int>(m_points.size()), velocity, angular, m_rotated.deviceRead(),
                                     out.deviceWrite(), false, diagnostic))
        reportFailure("velocities", diagnostic);
    dOut.endEdit();
}

void GpuRigidMapping::applyJT(const sofa::core::MechanicalParams*, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn)
{
    if (!m_ready) return;
    const auto& in = dIn.getValue();
    if (in.size() < m_points.size()) return;
    double forceAndTorque[6];
    std::string diagnostic;
    if (!backend::rigidMappingApplyJT(static_cast<int>(m_points.size()), m_rotated.deviceRead(), in.deviceRead(),
                                      forceAndTorque, diagnostic))
    {
        reportFailure("forces", diagnostic);
        return;
    }
    auto& out = *dOut.beginEdit();
    auto& d = out[d_index.getValue()];
    for (int c = 0; c < 3; ++c)
    {
        d.getVCenter()[c] += forceAndTorque[c];
        d.getVOrientation()[c] += forceAndTorque[3 + c];
    }
    dOut.endEdit();
}

void GpuRigidMapping::applyJT(const sofa::core::ConstraintParams*, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn)
{
    // Constraint rows (SOFA's CPU constraint solvers), as RigidMapping maps them:
    // each row's entries u_i on the points become [sum u_i ; sum r_i x u_i] on the
    // body. The rows are few and sparse, so this runs on the CPU; the rotated
    // points are read back only when there are rows. (GpuContactConstraintSolver
    // builds the body's rows itself and does not call this.)
    if (!m_ready) return;
    const auto& in = dIn.getValue();
    if (in.begin() == in.end()) return;
    const auto& rotated = m_rotated;   // const access: one download of R p
    auto& out = *dOut.beginEdit();
    for (auto rowIt = in.begin(); rowIt != in.end(); ++rowIt)
    {
        In::Deriv::Pos v;
        In::Deriv::Rot omega;
        bool any = false;
        for (auto colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
        {
            const auto i = static_cast<std::size_t>(colIt.index());
            if (i >= rotated.size()) continue;
            const auto& f = colIt.val();
            const sofa::type::Vec3d force(f[0], f[1], f[2]);
            const sofa::type::Vec3d r(rotated[i][0], rotated[i][1], rotated[i][2]);
            v += force;
            omega += sofa::type::cross(r, force);
            any = true;
        }
        if (any)
        {
            auto o = out.writeLine(rowIt.index());
            o.addCol(d_index.getValue(), In::Deriv(v, omega));
        }
    }
    dOut.endEdit();
}

} // namespace SofaGpuCollision
