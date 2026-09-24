#pragma once

#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace SofaGpuCollision
{

// ============================================================================
// A tetrahedral viscoelastic tissue integrated entirely on the GPU.
//
// An ODE solver for the node of a CudaVec3f MechanicalObject with tetrahedra.
// It takes the place, stage for stage, of this CPU set-up:
//   EulerImplicitSolver + SparseLDLSolver
//   MeshMatrixMass (massDensity, not lumped)
//   TetrahedronViscoHyperelasticityFEMForceField, SLSOgdenFirstOrder  (ogdenParameters)
//   TetrahedronViscoelasticityFEMForceField, MaxwellFirstOrder        (maxwellParameters)
//   FixedProjectiveConstraint (read from the node)
// Each step: the material's forces and per-edge stiffness in double, the system
// A = M - h^2 K and the RHS h (f + h K v), a dense Cholesky of A (single
// precision) with iterative refinement in double, and v_free = v + dv,
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
    const float* deviceFactor(int& size) const;                                     // null before the first step
    void updateMonitor(const void* xDevice);                                        // after the step's correction
    sofa::core::behavior::LinearSolver* cpuReplicaLinearSolver() const;             // compareWithCpu only

    // Parameters.
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_ogdenParameters;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_maxwellParameters;
    sofa::core::objectmodel::Data<std::string> d_ogdenEigenvectors;
    sofa::core::objectmodel::Data<SReal> d_massDensity;
    sofa::core::objectmodel::Data<SReal> d_rayleighStiffness;
    sofa::core::objectmodel::Data<SReal> d_rayleighMass;
    sofa::core::objectmodel::Data<int> d_refinementSteps;
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

    sofa::core::objectmodel::SingleLink<GpuTissueSolver, sofa::core::topology::BaseMeshTopology,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_topology;

private:
    struct CpuReplica;

    bool computeFixedDofs(std::string& diagnostic);
    bool createWorkspace(std::string& diagnostic);
    bool createReplica(std::string& diagnostic);
    void compareWithCpu(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult,
                        sofa::core::MultiVecDerivId vResult, double gpuMs);
    void reportFailure(const std::string& stage, const std::string& diagnostic);

    backend::TissueWorkspace* m_workspace { nullptr };
    sofa::core::behavior::MechanicalState<StateTypes>* m_state { nullptr };
    sofa::core::topology::BaseMeshTopology* m_topology { nullptr };
    std::vector<double> m_restPositions;
    std::vector<unsigned char> m_fixedDofs;
    std::unique_ptr<CpuReplica> m_cpu;
    std::ofstream m_compareStream;
    int m_steps { 0 };
    bool m_failed { false };
    std::vector<std::string> m_warned;
};

} // namespace SofaGpuCollision
