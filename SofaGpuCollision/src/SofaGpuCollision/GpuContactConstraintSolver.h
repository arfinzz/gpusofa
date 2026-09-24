#pragma once

#include <SofaGpuCollision/GpuCollisionBackend.h>
#include <SofaGpuCollision/GpuTissueSolver.h>
#include <SofaGpuCollision/config.h>

#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/type/Vec.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace SofaGpuCollision
{

// ============================================================================
// Constraint-based contact response on the GPU: Lagrange multipliers with
// Coulomb friction, for one deformable body touching one rigid body.
//
// It takes the place of SOFA's constraint solver in a FreeMotionAnimationLoop.
// Each step, after the free motion and the GPU collision detection:
//   build     constraint rows from the GPU contacts of this collision pass:
//             normal + 2 tangents per contact (SOFA's UnilateralLagrangian-
//             Constraint rows and free-violation formula), minus the DOFs held
//             by projective constraints;
//   compliance W = dt (J1 A1^-1 J1^T + J2 A2^-1 J2^T), A = each body's implicit
//             system matrix, taken from its linear solver after the free motion
//             (what LinearSolverConstraintCorrection does). A1 is factorized on
//             the GPU (dense Cholesky);
//   solve     SOFA's block Gauss-Seidel with its friction resolution, on the GPU;
//   correct   x = x_free + dt A^-1 J^T lambda, v = v_free + A^-1 J^T lambda.
// Every stage follows SOFA's CPU code (see cuda/detail/ContactConstraints.cuh).
//
// compareWithCpu: every compareEvery-th step with contacts, the same contacts go
// through SOFA's CPU pipeline too, stage by stage: rows and violations from
// SOFA's UnilateralLagrangianConstraint, W from each body's SOFA linear solver
// (addJMInvJt on the free motion's factorization), multipliers from SOFA's
// BlockGaussSeidelConstraintSolver, and the correction A^-1 J^T lambda in double.
// SOFA's solver also runs on the GPU's own W and violations, which separates a
// solver difference from a compliance difference. Differences and both sides'
// times go to a CSV file, one row per compared step.
//
// response="cpu": SOFA's CPU pipeline above drives the simulation instead of
// the GPU's (the GPU still detects the contacts and still runs, for the CSV).
// Two runs of one scene, response="gpu" and response="cpu", then differ only in
// who computed the contact response.
//
// Scene requirements: FreeMotionAnimationLoop; each body has its own ODE solver
// and a direct linear solver (SparseLDLSolver: its matrix is read after the free
// motion); body 1's GPU collision surface (CudaVec3f) has body 1's vertex
// numbering (IdentityMapping); body 2's GPU surface is rigidly mapped
// (RigidMapping Rigid3d -> CudaVec3f); both surfaces wound with outward normals;
// GpuCollisionNarrowPhase keeps contacts on the device, with its contactDistance
// at least this component's (typically the alarm distance). Only these two bodies
// get the free motion + correction; other solved bodies are not handled.
//
// Body 1 on the GPU (deformableGpuSolver: a GpuTissueSolver instead of the
// deformableState / LinearSolver / OdeSolver links): its state is CudaVec3f, the
// free positions are read in place, the tissue step's own Cholesky factor gives
// the compliance (no matrix copy, no second factorisation), and the correction is
// written straight into the state on the GPU. Its CPU comparison then uses the
// tissue solver's CPU replica (its compareWithCpu must be on too).
// ============================================================================
class SOFA_GPU_COLLISION_API GpuContactConstraintSolver : public sofa::core::behavior::ConstraintSolver
{
public:
    SOFA_CLASS(GpuContactConstraintSolver, sofa::core::behavior::ConstraintSolver);

    using DeformableTypes = sofa::defaulttype::Vec3dTypes;
    using RigidTypes = sofa::defaulttype::Rigid3dTypes;
    using SurfaceTypes = sofa::gpu::cuda::CudaVec3fTypes;
    using Wrench = sofa::type::Vec<6, SReal>;

    GpuContactConstraintSolver();
    ~GpuContactConstraintSolver() override;

    void init() override;
    void cleanup() override;

    bool prepareStates(const sofa::core::ConstraintParams*, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2) override;
    bool buildSystem(const sofa::core::ConstraintParams*, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2) override;
    bool solveSystem(const sofa::core::ConstraintParams*, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2) override;
    bool applyCorrection(const sofa::core::ConstraintParams*, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2) override;

    sofa::core::MultiVecDerivId getLambda() const override { return m_lambdaId; }
    sofa::core::MultiVecDerivId getDx() const override { return m_dxId; }
    void removeConstraintCorrection(sofa::core::behavior::BaseConstraintCorrection*) override {}

    // Parameters (the same names and defaults as SOFA's constraint solvers and
    // FrictionContactConstraint where they exist).
    sofa::core::objectmodel::Data<SReal> d_friction;
    sofa::core::objectmodel::Data<SReal> d_contactDistance;
    sofa::core::objectmodel::Data<SReal> d_tolerance;
    sofa::core::objectmodel::Data<int> d_maxIterations;
    sofa::core::objectmodel::Data<bool> d_scaleTolerance;
    sofa::core::objectmodel::Data<bool> d_allVerified;
    sofa::core::objectmodel::Data<SReal> d_sor;
    sofa::core::objectmodel::Data<std::string> d_contactFilter;
    sofa::core::objectmodel::Data<bool> d_exactArithmetic;
    sofa::core::objectmodel::Data<std::string> d_response;
    sofa::core::objectmodel::Data<bool> d_compareWithCpu;
    sofa::core::objectmodel::Data<int> d_compareEvery;
    sofa::core::objectmodel::Data<std::string> d_compareFile;
    sofa::core::objectmodel::Data<bool> d_measureTimes;

    // Outputs (read-only).
    sofa::core::objectmodel::Data<int> d_currentContacts;
    sofa::core::objectmodel::Data<int> d_currentConstraints;
    sofa::core::objectmodel::Data<int> d_currentIterations;
    sofa::core::objectmodel::Data<SReal> d_currentError;
    sofa::core::objectmodel::Data<SReal> d_normalImpulse;
    sofa::core::objectmodel::Data<Wrench> d_rigidContactForce;
    sofa::core::objectmodel::Data<SReal> d_stepGpuMilliseconds;
    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_stageMilliseconds;

    // The two bodies.
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::MechanicalState<DeformableTypes>,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_deformableState;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::MechanicalState<SurfaceTypes>,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_deformableSurface;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::LinearSolver,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_deformableLinearSolver;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::OdeSolver,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_deformableOdeSolver;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::MechanicalState<RigidTypes>,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_rigidState;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::MechanicalState<SurfaceTypes>,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_rigidSurface;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::LinearSolver,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_rigidLinearSolver;
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, sofa::core::behavior::OdeSolver,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_rigidOdeSolver;
    // Body 1 on the GPU: replaces deformableState, deformableLinearSolver and deformableOdeSolver.
    sofa::core::objectmodel::SingleLink<GpuContactConstraintSolver, GpuTissueSolver,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH | sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_deformableGpuSolver;

private:
    struct CpuReference;   // SOFA's CPU pipeline (defined in the .cpp)

    bool linksReady() const;
    bool gpuMode() const { return l_deformableGpuSolver.get() != nullptr; }
    std::size_t deformableVertexCount() const;
    sofa::core::behavior::OdeSolver* deformableOde() const;
    sofa::core::behavior::LinearSolver* deformableLinear() const;   // GPU mode: the tissue solver's CPU replica's
    void updateDofMasks();
    bool readDeformableMatrix(sofa::core::behavior::LinearSolver* solver, backend::HostCsrMatrix& matrix, std::string& diagnostic);
    bool readRigidMatrix(double matrix[36], std::string& diagnostic);
    void applyMotion(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2);
    void applyDeformableMotionOnDevice(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId res1,
                                       sofa::core::MultiVecId res2, bool useAppliedCorrection);
    // SOFA's CPU pipeline on this step's contacts; fills m_applied* with its
    // correction when `apply`, writes a comparison row when `writeRow`.
    bool runCpuPipeline(const sofa::core::ConstraintParams* cParams, bool apply, bool writeRow);
    void reportFailure(const std::string& stage, const std::string& diagnostic,
                       const std::string& consequence = "the bodies keep their free motion this step");

    backend::ConstraintWorkspace* m_workspace { nullptr };
    sofa::core::behavior::MechanicalState<SurfaceTypes>* m_gpuState { nullptr };   // body 1's state in GPU mode
    std::uint64_t m_deformableSurfaceId { 0 };
    std::uint64_t m_rigidSurfaceId { 0 };
    sofa::core::MultiVecDerivId m_lambdaId { sofa::core::MultiVecDerivId::null() };
    sofa::core::MultiVecDerivId m_dxId { sofa::core::MultiVecDerivId::null() };

    // This step's state.
    bool m_haveRows { false };
    bool m_solved { false };
    bool m_corrected { false };
    backend::ConstraintBuildStats m_buildStats;
    backend::ConstraintSolveStats m_solveStats;
    backend::ConstraintTimings m_timings;
    backend::ConstraintImpulse m_impulse;
    double m_factorDeformable { 0.0 };
    double m_factorRigid { 0.0 };
    double m_rigidMatrix[36] {};
    std::vector<float> m_freePositions;
    std::vector<float> m_deformableCorrection;   // the GPU's dv
    double m_rigidCorrection[6] {};
    // What applyMotion applies: the GPU's correction, or SOFA's CPU pipeline's
    // with response="cpu".
    std::vector<double> m_appliedDeformable;
    double m_appliedRigid[6] {};
    double m_appliedImpulse[6] {};
    double m_appliedNormalImpulse { 0.0 };

    // DOFs held by projective constraints (bit set = free), refreshed every step.
    std::vector<std::uint8_t> m_deformableMask;
    bool m_deformableMaskUsed { false };
    std::uint8_t m_rigidMask { 0x3F };

    // Body 1's matrix in scalar CSR form.
    std::vector<int> m_csrRowPtr;
    std::vector<int> m_csrColumns;
    std::vector<double> m_csrValues;

    std::unique_ptr<CpuReference> m_cpu;
    std::ofstream m_compareStream;
    int m_contactSteps { 0 };
    std::set<std::string> m_warnedStages;
};

} // namespace SofaGpuCollision
