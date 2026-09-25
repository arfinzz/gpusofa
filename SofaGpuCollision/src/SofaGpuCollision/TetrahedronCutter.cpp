#include <SofaGpuCollision/TetrahedronCutter.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <algorithm>
#include <cmath>

namespace SofaGpuCollision
{

int TetrahedronCutterClass = sofa::core::RegisterObject(
    "Cutting by element removal: a straight blade moving in a plane removes the tetrahedra it has passed through "
    "(rest-configuration centroids, through SOFA's TetrahedronSetTopologyModifier).")
    .add<TetrahedronCutter>();

using sofa::type::Vec3d;

TetrahedronCutter::TetrahedronCutter()
    : d_restPositions(initData(&d_restPositions, "restPositions",
        "The mesh's rest positions (the tetrahedra's centroids are taken there)."))
    , d_planePoint(initData(&d_planePoint, Vec3d(0.0, 0.0, 0.0), "planePoint", "A point of the blade's plane."))
    , d_planeNormal(initData(&d_planeNormal, Vec3d(1.0, 0.0, 0.0), "planeNormal", "The blade's plane normal."))
    , d_cutDirection(initData(&d_cutDirection, Vec3d(0.0, -1.0, 0.0), "cutDirection",
        "The direction the cutting edge moves in, in the plane (made perpendicular to the normal)."))
    , d_kerf(initData(&d_kerf, 0.0, "kerf", "Half-width of the removed slab around the plane."))
    , d_edgeStart(initData(&d_edgeStart, 0.0, "edgeStart",
        "The edge's position along cutDirection from planePoint at startTime."))
    , d_edgeStop(initData(&d_edgeStop, 0.0, "edgeStop", "The edge stops here (along cutDirection from planePoint)."))
    , d_speed(initData(&d_speed, 0.0, "speed", "The edge's speed along cutDirection (m/s)."))
    , d_startTime(initData(&d_startTime, 0.0, "startTime", "When the edge starts moving."))
    , d_widthMin(initData(&d_widthMin, -1e30, "widthMin",
        "The blade's extent along its edge (planeNormal x cutDirection), from planePoint: lower end."))
    , d_widthMax(initData(&d_widthMax, 1e30, "widthMax", "Upper end of the blade's extent along its edge."))
    , d_removedCount(initData(&d_removedCount, 0, "removedCount", "OUTPUT: tetrahedra removed so far."))
    , d_removedLastStep(initData(&d_removedLastStep, 0, "removedLastStep", "OUTPUT: tetrahedra removed before the last step."))
    , l_topology(initLink("topology", "The tetrahedral topology to cut (default: the one in this node)."))
{
    d_removedCount.setReadOnly(true);
    d_removedLastStep.setReadOnly(true);
    f_listening.setValue(true);
}

void TetrahedronCutter::init()
{
    BaseObject::init();
    if (l_topology.empty()) l_topology.set(this->getContext()->getMeshTopologyLink());
    auto* topology = l_topology.get();
    if (topology != nullptr) topology->getContext()->get(m_modifier);
    if (topology == nullptr || m_modifier == nullptr)
    {
        msg_error() << "TetrahedronCutter needs a tetrahedral topology and a TetrahedronSetTopologyModifier in its node.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    const auto& rest = d_restPositions.getValue();
    for (const auto& tet : topology->getTetrahedra())
        for (int k = 0; k < 4; ++k)
            if (static_cast<std::size_t>(tet[k]) >= rest.size())
            {
                msg_error() << "restPositions has " << rest.size() << " points; the topology uses vertex " << tet[k] << ".";
                d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
    m_normal = d_planeNormal.getValue();
    m_normal.normalize();
    m_direction = d_cutDirection.getValue();
    m_direction -= m_normal * sofa::type::dot(m_direction, m_normal);
    if (m_direction.norm() < 1e-12)
    {
        msg_error() << "cutDirection must not be parallel to planeNormal.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    m_direction.normalize();
    m_along = sofa::type::cross(m_normal, m_direction);
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void TetrahedronCutter::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent*>(event) != nullptr &&
        d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Valid)
        cut();
}

void TetrahedronCutter::cut()
{
    // The edge's position at the end of the coming step.
    const double t = this->getContext()->getTime() + this->getContext()->getDt();
    double s = d_edgeStart.getValue();
    if (t > d_startTime.getValue()) s += d_speed.getValue() * (t - d_startTime.getValue());
    const double stop = d_edgeStop.getValue();
    s = d_speed.getValue() >= 0.0 ? std::min(s, stop) : std::max(s, stop);

    const auto& rest = d_restPositions.getValue();
    const Vec3d p0 = d_planePoint.getValue();
    const double kerf = d_kerf.getValue();
    const double wmin = d_widthMin.getValue();
    const double wmax = d_widthMax.getValue();
    const bool forward = d_speed.getValue() >= 0.0;
    sofa::type::vector<sofa::core::topology::BaseMeshTopology::TetrahedronID> cut;
    const auto& tets = l_topology->getTetrahedra();
    for (std::size_t i = 0; i < tets.size(); ++i)
    {
        Vec3d c(0.0, 0.0, 0.0);
        for (int k = 0; k < 4; ++k) c += rest[tets[i][k]];
        c *= 0.25;
        const Vec3d r = c - p0;
        const double along = sofa::type::dot(r, m_direction);
        const double across = sofa::type::dot(r, m_along);
        if (std::fabs(sofa::type::dot(r, m_normal)) <= kerf && (forward ? along <= s : along >= s) &&
            across >= wmin && across <= wmax)
            cut.push_back(static_cast<sofa::core::topology::BaseMeshTopology::TetrahedronID>(i));
    }
    d_removedLastStep.setValue(static_cast<int>(cut.size()));
    if (cut.empty()) return;
    m_modifier->removeTetrahedra(cut, true);
    d_removedCount.setValue(d_removedCount.getValue() + static_cast<int>(cut.size()));
}

} // namespace SofaGpuCollision
