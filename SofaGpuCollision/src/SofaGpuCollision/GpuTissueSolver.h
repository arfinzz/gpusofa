#pragma once

#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace SofaGpuCollision
{

// ============================================================================
// A tetrahedral (visco)hyperelastic tissue integrated entirely on the GPU.
//
// An ODE solver for the node of a CudaVec3f MechanicalObject with tetrahedra.
// It takes the place, stage for stage, of this CPU set-up:
//   EulerImplicitSolver + SparseLDLSolver
//   MeshMatrixMass (massDensity, not lumped)
//   the material, one or more of these force fields on the same mesh:
//     TetrahedronHyperelasticityFEMForceField (SOFA core): Ogden, NeoHookean,
//       StableNeoHookean, StVenantKirchhoff or MooneyRivlin  (hyperelasticMaterial)
//     TetrahedronViscoHyperelasticityFEMForceField, SLSOgdenFirstOrder (ogdenParameters)
//     TetrahedronViscoelasticityFEMForceField, MaxwellFirstOrder       (maxwellParameters)
//   FixedProjectiveConstraint (read from the node)
//   ConstantForceField loads (read from the node; SOFA's own addForce gives the
//     nodal forces, re-read whenever the load's data change)
//   cutting: tetrahedra removed from the topology (TetrahedronCutter, or any change
//     that only removes tetrahedra) are followed before the next step: the elements
//     and the mass are rebuilt for the remaining ones, as SOFA's force fields and
//     MeshMatrixMass update theirs, and each keeps its viscous state. Vertices must
//     not be removed.
// Each step: the material's forces and per-edge stiffness in double, the system
// A = M - h^2 K and the RHS h (f + h K v), a Cholesky factorisation of A in single
// precision (factorization: "band" = the vertices renumbered by reverse
// Cuthill-McKee so A is a band matrix, which cut into blocks as wide as the band is
// block tridiagonal: only those blocks are stored (2 n w floats, not n^2) and a
// block Cholesky works on them; "dense" = cuSOLVER's Cholesky of the whole matrix; LU with
// pivoting on a step where A is not positive definite, which SOFA's LDL solver
// also gets through) with iterative refinement in double, and v_free = v + dv,
// x_free = x + h v_free. The state never leaves the GPU. The factor of A stays for
// GpuContactConstraintSolver, which needs A^-1 for the contact compliance and
// correction (deformableGpuSolver link).
//
// compareWithCpu: a hidden replica of the CPU set-up above (SOFA's own
// components, not part of the scene graph) is given the same state every step
// and runs SOFA's free motion; the differences (forces, system matrix, velocity
// change, free positions) and both sides' times go to compareFile. The replica's
// linear solver also serves GpuContactConstraintSolver's CPU comparison.
//
// ogdenEigenvectors: SLSOgdenFirstOrder builds C^(alpha/2-1) from C's eigenvalues
// and eigenvectors, but in SOFA v25.12 its call to Eigen asks for no eigenvectors
// (Eigen::SelfAdjointEigenSolver(C, true): 'true' is read as options = 1), so it
// pairs the sorted eigenvalues with C's scaled lower triangle instead. "sofa"
// (default) reproduces that, so results match SOFA's CPU components; "exact" uses
// the true eigenvectors (the Ogden material as written; the replica then differs).
// ============================================================================
class SOFA_GPU_COLLISION_API GpuTissueSolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(GpuTissueSolver, sofa::core::behavior::OdeSolver);

    using StateTypes = sofa::gpu::cuda::CudaVec3fTypes;

    GpuTissueSolver();
    ~GpuTissueSolver() override;

    void init() override;
    void solve(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult,
               sofa::core::MultiVecDerivId vResult) override;

    // As EulerImplicitSolver.
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override;
    SReal getSolutionIntegrationFactor(int outputDerivative) const override;
    SReal getVelocityIntegrationFactor() const override { return 1.0; }
    SReal getPositionIntegrationFactor() const override { return this->getContext()->getDt(); }

    // For GpuContactConstraintSolver.
    sofa::core::behavior::MechanicalState<StateTypes>* tissueState() const { return m_state; }
    const std::vector<unsigned char>& fixedDofs() const { return m_fixedDofs; }   // bit c set = DOF c fixed
    backend::TissueWorkspace* workspace() const { return m_workspace; }            // its factor solves for the contact
    void updateMonitor(const void* xDevice);                                        // after the step's correction
    sofa::core::behavior::LinearSolver* cpuReplicaLinearSolver() const;             // compareWithCpu only

    // Parameters.
    sofa::core::objectmodel::Data<std::string> d_hyperelasticMaterial;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_hyperelasticParameters;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_ogdenParameters;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_maxwellParameters;
    sofa::core::objectmodel::Data<std::string> d_ogdenEigenvectors;
    sofa::core::objectmodel::Data<std::string> d_ogdenTangent;
    sofa::core::objectmodel::Data<sofa::type::vector<int>> d_partialFixedIndices;
    sofa::core::objectmodel::Data<sofa::type::vector<int>> d_partialFixedMasks;
    sofa::core::objectmodel::Data<SReal> d_massDensity;
    sofa::core::objectmodel::Data<SReal> d_rayleighStiffness;
    sofa::core::objectmodel::Data<SReal> d_rayleighMass;
    sofa::core::objectmodel::Data<int> d_refinementSteps;
    sofa::core::objectmodel::Data<std::string> d_factorization;
    sofa::core::objectmodel::Data<int> d_bandPanel;
    sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3d>> d_restPositions;
    sofa::core::objectmodel::Data<int> d_monitorVertex;
    sofa::core::objectmodel::Data<bool> d_measureTimes;
    sofa::core::objectmodel::Data<bool> d_compareWithCpu;
    sofa::core::objectmodel::Data<int> d_compareEvery;
    sofa::core::objectmodel::Data<std::string> d_compareFile;

    // Outputs.
    sofa::core::objectmodel::Data<SReal> d_minVolumeRatio;
    sofa::core::objectmodel::Data<sofa::type::Vec3d> d_monitorPosition;
    sofa::core::objectmodel::Data<SReal> d_stepGpuMilliseconds;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_stageMilliseconds;
    sofa::core::objectmodel::Data<int> d_luFallbackSteps;
    sofa::core::objectmodel::Data<int> d_bandwidth;

    sofa::core::objectmodel::SingleLink<GpuTissueSolver, sofa::core::topology::BaseMeshTopology,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_topology;

private:
    struct CpuReplica;

    bool computeFixedDofs(std::string& diagnostic);
    bool collectLoads(std::string& diagnostic);
    bool updateExternalForces(std::string& diagnostic);
    bool createWorkspace(std::string& diagnostic);
    bool followTopologyChanges(std::string& diagnostic);
    bool createReplica(std::string& diagnostic);
    void compareWithCpu(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult,
                        sofa::core::MultiVecDerivId vResult, double gpuMs);
    void reportFailure(const std::string& stage, const std::string& diagnostic);

    backend::TissueWorkspace* m_workspace { nullptr };
    sofa::core::behavior::MechanicalState<StateTypes>* m_state { nullptr };
    sofa::core::topology::BaseMeshTopology* m_topology { nullptr };
    std::vector<double> m_restPositions;
    std::vector<double> m_startX, m_startV;   // compareWithCpu: the step's start state
    std::vector<unsigned char> m_fixedDofs;
    std::vector<sofa::core::behavior::ForceField<StateTypes>*> m_loads;   // the node's ConstantForceFields
    std::size_t m_loadSignature { 0 };                                     // their data counters when last applied
    bool m_loadsApplied { false };
    std::unique_ptr<CpuReplica> m_cpu;
    std::ofstream m_compareStream;
    int m_steps { 0 };
    int m_core { 0 };   // backend::TissueCoreMaterial of hyperelasticMaterial
    bool m_failed { false };
    // Cutting (tetrahedra removed from the topology): the workspace's tetrahedra and
    // edges as created, to renumber the remaining tetrahedra's edges and carry their state.
    std::vector<std::array<int, 4>> m_tetrahedra;
    std::vector<int> m_edges;                                   // 2 per edge
    std::unordered_map<std::uint64_t, int> m_edgeIndex;         // vertex pair -> edge
    int m_isolatedVertices { 0 };
    std::vector<std::string> m_warned;
};

} // namespace SofaGpuCollision
