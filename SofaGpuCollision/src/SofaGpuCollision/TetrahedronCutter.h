#pragma once

#include <SofaGpuCollision/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

namespace sofa::component::topology::container::dynamic
{
class TetrahedronSetTopologyModifier;
}

namespace SofaGpuCollision
{

// ============================================================================
// Cutting by element removal: a straight blade moving in a plane removes every
// tetrahedron it has passed through.
//
// The blade lies in the plane through planePoint with normal planeNormal. Its
// cutting edge is a straight line in that plane, perpendicular to cutDirection,
// and it moves along cutDirection: at time t it is at
//   s(t) = edgeStart + speed (t - startTime)   (clamped to edgeStop)
// along cutDirection from planePoint. A tetrahedron is removed once the edge has
// passed its centroid, s(t) >= cutDirection . (c - planePoint), if the centroid lies
// within `kerf` of the plane and between widthMin and widthMax along the edge
// (planeNormal x cutDirection).
//
// Centroids are taken in the rest configuration (restPositions), so a CPU scene and
// a GPU scene with the same mesh remove exactly the same tetrahedra in the same
// steps, whatever small differences their deformation has. Removal goes through
// SOFA's TetrahedronSetTopologyModifier (with removeIsolatedItems), so every
// component that follows topology changes sees it: SOFA's force fields and mass,
// the surface's Tetra2TriangleTopologicalMapping, and GpuTissueSolver.
// A vertex left with no tetrahedron would be removed from the state by SOFA,
// which GpuTissueSolver refuses: cuts should remove whole layers of cells (a kerf of
// one cell), so that every vertex keeps a tetrahedron.
// ============================================================================
class SOFA_GPU_COLLISION_API TetrahedronCutter : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(TetrahedronCutter, sofa::core::objectmodel::BaseObject);

    TetrahedronCutter();

    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

    sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3d>> d_restPositions;
    sofa::core::objectmodel::Data<sofa::type::Vec3d> d_planePoint;
    sofa::core::objectmodel::Data<sofa::type::Vec3d> d_planeNormal;
    sofa::core::objectmodel::Data<sofa::type::Vec3d> d_cutDirection;
    sofa::core::objectmodel::Data<double> d_kerf;
    sofa::core::objectmodel::Data<double> d_edgeStart;
    sofa::core::objectmodel::Data<double> d_edgeStop;
    sofa::core::objectmodel::Data<double> d_speed;
    sofa::core::objectmodel::Data<double> d_startTime;
    sofa::core::objectmodel::Data<double> d_widthMin;
    sofa::core::objectmodel::Data<double> d_widthMax;
    sofa::core::objectmodel::Data<int> d_removedCount;       // output: tetrahedra removed so far
    sofa::core::objectmodel::Data<int> d_removedLastStep;    // output

    sofa::core::objectmodel::SingleLink<TetrahedronCutter, sofa::core::topology::BaseMeshTopology,
                                        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK>
        l_topology;

private:
    void cut();

    sofa::component::topology::container::dynamic::TetrahedronSetTopologyModifier* m_modifier { nullptr };
    sofa::type::Vec3d m_normal;
    sofa::type::Vec3d m_direction;
    sofa::type::Vec3d m_along;   // the edge's direction: normal x direction
};

} // namespace SofaGpuCollision
