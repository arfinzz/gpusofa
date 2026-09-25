#include <SofaGpuCollision/GpuContactConstraintSolver.h>

#include <sofa/component/constraint/lagrangian/model/UnilateralConstraintResolution.h>
#include <sofa/component/constraint/lagrangian/model/UnilateralLagrangianConstraint.h>
#include <sofa/component/constraint/lagrangian/solver/BlockGaussSeidelConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintProblem.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseMatrixLinearSystem.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>

#include <SofaCUDA/component/collision/geometry/CudaTriangleModel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace SofaGpuCollision
{

int GpuContactConstraintSolverClass = sofa::core::RegisterObject(
    "Constraint-based contact response on the GPU (Lagrange multipliers with Coulomb friction) for a "
    "deformable body touching a rigid body: SOFA's constraint pipeline stage for stage, on the GPU.")
    .add<GpuContactConstraintSolver>();

namespace
{

using sofa::core::objectmodel::BaseContext;
using sofa::component::constraint::lagrangian::solver::BlockGaussSeidelConstraintSolver;
using sofa::component::constraint::lagrangian::solver::GenericConstraintProblem;
using Vec3d = sofa::type::Vec3d;

double nowMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The id the narrow phase recorded the surface's contacts under: the address of
// the CudaTriangleCollisionModel at or below the surface state's node (the same
// rule as GpuCollisionNarrowPhase and CudaContactPenaltyForceField).
std::uint64_t surfaceIdOf(const sofa::core::BaseState* state)
{
    if (state == nullptr || state->getContext() == nullptr) return 0;
    auto* model = state->getContext()->template get<sofa::gpu::cuda::CudaTriangleCollisionModel>(BaseContext::SearchDown);
    return model ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model)) : 0;
}

void clearMultiVec(BaseContext* context, const sofa::core::ConstraintParams* cParams, const sofa::core::MultiVecDerivId id)
{
    sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor clear(
        cParams, id, sofa::core::ConstMultiVecDerivId::null(), sofa::core::ConstMultiVecDerivId::null(), 1.0);
    clear.setMapped(true);
    context->executeVisitor(&clear);
}

// BaseConstraintCorrection::correctionFactor (protected in SOFA): the factor
// LinearSolverConstraintCorrection puts in front of J A^-1 J^T.
SReal correctionFactor(const sofa::core::behavior::OdeSolver* solver, const sofa::core::ConstraintOrder order)
{
    if (solver == nullptr) return 1.0;
    switch (order)
    {
        case sofa::core::ConstraintOrder::POS_AND_VEL:
        case sofa::core::ConstraintOrder::POS:
            return solver->getPositionIntegrationFactor();
        case sofa::core::ConstraintOrder::ACC:
        case sofa::core::ConstraintOrder::VEL:
            return solver->getVelocityIntegrationFactor();
        default:
            return 1.0;
    }
}

double maxAbs(const std::vector<double>& v)
{
    double m = 0.0;
    for (const double x : v) m = std::max(m, std::fabs(x));
    return m;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// A x = b for a 6x6 A (row-major), partial pivoting.
bool solveDense6(const double A[36], const double b[6], double x[6])
{
    double m[6][7];
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j) m[i][j] = A[i * 6 + j];
        m[i][6] = b[i];
    }
    for (int col = 0; col < 6; ++col)
    {
        int pivot = col;
        for (int r = col + 1; r < 6; ++r)
            if (std::fabs(m[r][col]) > std::fabs(m[pivot][col])) pivot = r;
        if (std::fabs(m[pivot][col]) < 1e-300) return false;
        if (pivot != col)
            for (int j = 0; j < 7; ++j) std::swap(m[col][j], m[pivot][j]);
        for (int r = col + 1; r < 6; ++r)
        {
            const double f = m[r][col] / m[col][col];
            for (int j = col; j < 7; ++j) m[r][j] -= f * m[col][j];
        }
    }
    for (int i = 5; i >= 0; --i)
    {
        double acc = m[i][6];
        for (int j = i + 1; j < 6; ++j) acc -= m[i][j] * x[j];
        x[i] = acc / m[i][i];
    }
    return true;
}

// SOFA's block Gauss-Seidel (the real code) on a given problem.
struct ReferenceSolve
{
    std::vector<double> lambda;
    int sweeps { 0 };
    double ms { 0.0 };
};

ReferenceSolve solveWithSofa(BlockGaussSeidelConstraintSolver* solver, const int rows, const int rowsPerContact,
                             const double mu, const std::vector<double>& W, const std::vector<double>& dfree,
                             const double tolerance, const int maxIterations, const bool scaleTolerance,
                             const bool allVerified, const double sor)
{
    using sofa::component::constraint::lagrangian::model::UnilateralConstraintResolution;
    using sofa::component::constraint::lagrangian::model::UnilateralConstraintResolutionWithFriction;
    GenericConstraintProblem problem(solver);
    problem.clear(rows);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < rows; ++j) problem.W.set(i, j, W[static_cast<std::size_t>(i) * rows + j]);
        problem.dFree.set(i, dfree[i]);
        problem.f.set(i, 0.0);
    }
    for (int i = 0; i < rows; i += rowsPerContact)
    {
        if (rowsPerContact == 3)
            problem.constraintsResolutions[i] = new UnilateralConstraintResolutionWithFriction(mu);
        else
            problem.constraintsResolutions[i] = new UnilateralConstraintResolution();
    }
    problem.scaleTolerance = scaleTolerance;
    problem.allVerified = allVerified;
    problem.sor = sor;
    const double t0 = nowMs();
    problem.solveTimed(tolerance, maxIterations, 0.0);
    ReferenceSolve result;
    result.ms = nowMs() - t0;
    result.sweeps = problem.currentIterations - 1;   // SOFA reports sweeps + 1
    result.lambda.resize(rows);
    for (int i = 0; i < rows; ++i) result.lambda[i] = problem.f[i];
    return result;
}

// Lowest normal-row violation after the solve, d = dfree + W lambda: how far
// below the contact distance the surfaces end up (negative = overlap left).
double lowestNormalViolation(const int rows, const int rowsPerContact, const std::vector<double>& W,
                             const std::vector<double>& dfree, const std::vector<double>& lambda)
{
    double lowest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < rows; i += rowsPerContact)
    {
        double d = dfree[i];
        for (int j = 0; j < rows; ++j) d += W[static_cast<std::size_t>(i) * rows + j] * lambda[j];
        lowest = std::min(lowest, d);
    }
    return lowest;
}

} // namespace

// SOFA's CPU pipeline for compareWithCpu: SOFA's own contact constraint on two
// point sets holding the contact points, SOFA's Gauss-Seidel, and a double LDL^T
// of body 1's matrix for the correction.
struct GpuContactConstraintSolver::CpuReference
{
    using Points = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3dTypes>;
    using Contact = sofa::component::constraint::lagrangian::model::UnilateralLagrangianConstraint<sofa::defaulttype::Vec3dTypes>;

    Points::SPtr bodyPoints;    // SOFA's Q: body 1's contact points
    Points::SPtr rigidPoints;   // SOFA's P: body 2's contact points
    Contact::SPtr constraint;
    BlockGaussSeidelConstraintSolver::SPtr solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    std::vector<int> analyzedRowPtr;
    std::vector<int> analyzedColumns;

    CpuReference()
    {
        bodyPoints = sofa::core::objectmodel::New<Points>();
        rigidPoints = sofa::core::objectmodel::New<Points>();
        constraint = sofa::core::objectmodel::New<Contact>(bodyPoints.get(), rigidPoints.get());
        solver = sofa::core::objectmodel::New<BlockGaussSeidelConstraintSolver>();
    }
};

GpuContactConstraintSolver::GpuContactConstraintSolver()
    : d_friction(initData(&d_friction, 0.0_sreal, "friction",
        "Coulomb friction coefficient mu. 0 = frictionless (one row per contact instead of three)."))
    , d_contactDistance(initData(&d_contactDistance, 0.0_sreal, "contactDistance",
        "Gap kept between the surfaces: SOFA's contactDistance, the constraint's d0. The narrow phase must "
        "report contacts up to at least this distance (usually the alarm distance)."))
    , d_tolerance(initData(&d_tolerance, 0.001_sreal, "tolerance",
        "Gauss-Seidel stopping tolerance, as in SOFA's constraint solvers."))
    , d_maxIterations(initData(&d_maxIterations, 1000, "maxIterations", "Maximal number of Gauss-Seidel sweeps."))
    , d_scaleTolerance(initData(&d_scaleTolerance, true, "scaleTolerance",
        "Multiply the tolerance by the number of constraint rows (SOFA's default)."))
    , d_allVerified(initData(&d_allVerified, false, "allVerified",
        "Stop only when every constraint's own error is below the tolerance."))
    , d_sor(initData(&d_sor, 1.0_sreal, "sor", "Successive over-relaxation factor (1 = plain Gauss-Seidel)."))
    , d_contactFilter(initData(&d_contactFilter, std::string("vertexFace"), "contactFilter",
        "vertexFace: one vertex-face contact per vertex (its closest face) on each side, no edge-edge contacts. "
        "all: every contact the narrow phase reported (a larger, rank-deficient problem)."))
    , d_vertexConeFilter(initData(&d_vertexConeFilter, true, "vertexConeFilter",
        "vertexFace: keep a contact only if its direction leaves its anchor vertex's surface, within the cone of "
        "the faces around that vertex, as SOFA's LocalMinDistance filters point contacts. Without it, a flat "
        "floor's vertex in front of a sliding block made an oblique contact with the block's edge, which the "
        "step's linearised constraint read as a collision."))
    , d_vertexConeTolerance(initData(&d_vertexConeTolerance, 0.05_sreal, "vertexConeTolerance",
        "Cosine margin of vertexConeFilter (0.05: about 18 degrees around a flat surface's normal)."))
    , d_exactArithmetic(initData(&d_exactArithmetic, false, "exactArithmetic",
        "Run the Gauss-Seidel in double with SOFA's own arithmetic (slow; for checks). Default: float."))
    , d_response(initData(&d_response, std::string("gpu"), "response",
        "gpu: the GPU's contact response moves the bodies. cpu: SOFA's CPU pipeline does, on the same "
        "GPU-detected contacts (its contact constraint, addJMInvJt compliance, Gauss-Seidel, and the "
        "correction in double); the GPU pipeline still runs alongside, for the comparison. Slow: for validation."))
    , d_compareWithCpu(initData(&d_compareWithCpu, false, "compareWithCpu",
        "Also run SOFA's CPU pipeline on the same contacts, stage by stage, and write the differences and "
        "both sides' times to compareFile. Slow: for validation runs."))
    , d_compareEvery(initData(&d_compareEvery, 1, "compareEvery",
        "With compareWithCpu: compare every N-th step that has contacts."))
    , d_compareFile(initData(&d_compareFile, std::string("gpu_constraint_compare.csv"), "compareFile",
        "CSV file written when compareWithCpu is on."))
    , d_dumpContactsFile(initData(&d_dumpContactsFile, std::string(), "dumpContactsFile",
        "Diagnostics, with compareWithCpu: append every compared step's contacts (body-1 vertices and weights, "
        "points, normal, gap) to this CSV file. Empty: off."))
    , d_measureTimes(initData(&d_measureTimes, false, "measureTimes",
        "Time each GPU stage with CUDA events (adds a synchronisation per stage)."))
    , d_currentContacts(initData(&d_currentContacts, 0, "currentContacts", "OUTPUT: contacts kept as constraints this step."))
    , d_currentConstraints(initData(&d_currentConstraints, 0, "currentConstraints", "OUTPUT: constraint rows this step."))
    , d_currentIterations(initData(&d_currentIterations, 0, "currentIterations", "OUTPUT: Gauss-Seidel sweeps this step."))
    , d_currentError(initData(&d_currentError, 0.0_sreal, "currentError", "OUTPUT: Gauss-Seidel error at the last sweep."))
    , d_normalImpulse(initData(&d_normalImpulse, 0.0_sreal, "normalImpulse",
        "OUTPUT: sum of the normal multipliers this step (an impulse, N s; divide by dt for a force)."))
    , d_rigidContactForce(initData(&d_rigidContactForce, Wrench(), "rigidContactForce",
        "OUTPUT: contact force and torque (about its centre) on the rigid body this step, J2^T lambda / dt."))
    , d_stepGpuMilliseconds(initData(&d_stepGpuMilliseconds, 0.0_sreal, "stepGpuMilliseconds",
        "OUTPUT: GPU time of this step's constraint stages (only with measureTimes or compareWithCpu)."))
    , d_stageMilliseconds(initData(&d_stageMilliseconds, "stageMilliseconds",
        "OUTPUT: this step's GPU time per stage: rows, factorization, compliance, Gauss-Seidel, correction "
        "(only with measureTimes or compareWithCpu)."))
    , l_deformableState(initLink("deformableState", "Body 1: the deformable body's mechanical state (Vec3d)."))
    , l_deformableSurface(initLink("deformableSurface",
        "Body 1's GPU collision surface state (CudaVec3f) with body 1's vertex numbering (IdentityMapping)."))
    , l_deformableLinearSolver(initLink("deformableLinearSolver",
        "Body 1's direct linear solver; its system matrix is read after the free motion."))
    , l_deformableOdeSolver(initLink("deformableOdeSolver", "Body 1's ODE solver (for the integration factors)."))
    , l_rigidState(initLink("rigidState", "Body 2: the rigid body's mechanical state (Rigid3d)."))
    , l_rigidSurface(initLink("rigidSurface", "Body 2's GPU collision surface state (CudaVec3f, RigidMapping)."))
    , l_rigidLinearSolver(initLink("rigidLinearSolver", "Body 2's direct linear solver (6x6 system matrix)."))
    , l_rigidOdeSolver(initLink("rigidOdeSolver", "Body 2's ODE solver (for the integration factors)."))
    , l_deformableGpuSolver(initLink("deformableGpuSolver",
        "Body 1 on the GPU: its GpuTissueSolver, instead of deformableState, deformableLinearSolver and deformableOdeSolver."))
    , l_additionalRigidStates(initLink("additionalRigidStates",
        "More rigid bodies touching body 1 (other tools, a grasper's second jaw): their mechanical states (Rigid3d)."))
    , l_additionalRigidSurfaces(initLink("additionalRigidSurfaces", "Their GPU collision surface states (CudaVec3f), in the same order."))
    , l_additionalRigidLinearSolvers(initLink("additionalRigidLinearSolvers", "Their direct linear solvers, in the same order."))
    , l_additionalRigidOdeSolvers(initLink("additionalRigidOdeSolvers", "Their ODE solvers, in the same order."))
    , d_additionalRigidContactForces(initData(&d_additionalRigidContactForces, "additionalRigidContactForces",
        "OUTPUT: the contact force and torque on each additional rigid body this step (as rigidContactForce)."))
{
    d_additionalRigidContactForces.setReadOnly(true);
    d_currentContacts.setReadOnly(true);
    d_currentConstraints.setReadOnly(true);
    d_currentIterations.setReadOnly(true);
    d_currentError.setReadOnly(true);
    d_normalImpulse.setReadOnly(true);
    d_rigidContactForce.setReadOnly(true);
    d_stepGpuMilliseconds.setReadOnly(true);
    d_stageMilliseconds.setReadOnly(true);
}

GpuContactConstraintSolver::~GpuContactConstraintSolver()
{
    backend::destroyConstraintWorkspace(m_workspace);
    m_workspace = nullptr;
}

bool GpuContactConstraintSolver::linksReady() const
{
    const bool body1 = gpuMode() || (l_deformableState && l_deformableLinearSolver && l_deformableOdeSolver);
    return body1 && l_deformableSurface && l_rigidState && l_rigidSurface && l_rigidLinearSolver && l_rigidOdeSolver;
}

std::size_t GpuContactConstraintSolver::deformableVertexCount() const
{
    if (gpuMode()) return m_gpuState ? m_gpuState->getSize() : 0;
    return l_deformableState ? l_deformableState->getSize() : 0;
}

sofa::core::behavior::OdeSolver* GpuContactConstraintSolver::deformableOde() const
{
    return gpuMode() ? static_cast<sofa::core::behavior::OdeSolver*>(l_deformableGpuSolver.get()) : l_deformableOdeSolver.get();
}

sofa::core::behavior::LinearSolver* GpuContactConstraintSolver::deformableLinear() const
{
    return gpuMode() ? l_deformableGpuSolver->cpuReplicaLinearSolver() : l_deformableLinearSolver.get();
}

void GpuContactConstraintSolver::init()
{
    Inherit1::init();
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    if (!linksReady())
    {
        msg_error() << "Links missing. Needed: deformableSurface, rigidState, rigidSurface, rigidLinearSolver, "
                       "rigidOdeSolver, and for body 1 either deformableGpuSolver or deformableState + "
                       "deformableLinearSolver + deformableOdeSolver.";
        return;
    }
    if (gpuMode())
    {
        m_gpuState = dynamic_cast<sofa::core::behavior::MechanicalState<SurfaceTypes>*>(
            l_deformableGpuSolver->getContext()->getMechanicalState());
        if (m_gpuState == nullptr)
        {
            msg_error() << "deformableGpuSolver's node has no CudaVec3f MechanicalObject.";
            return;
        }
    }
    if (l_deformableSurface->getSize() != deformableVertexCount())
    {
        msg_error() << "The deformable body's GPU surface state must have the body's vertex numbering "
                       "(IdentityMapping): " << l_deformableSurface->getSize() << " vs " << deformableVertexCount() << " points.";
        return;
    }
    if (l_rigidState->getSize() != 1)
    {
        msg_error() << "The rigid state must hold exactly one rigid body (it has " << l_rigidState->getSize() << ").";
        return;
    }
    m_deformableSurfaceId = surfaceIdOf(l_deformableSurface.get());
    m_rigidSurfaceId = surfaceIdOf(l_rigidSurface.get());
    if (m_deformableSurfaceId == 0 || m_rigidSurfaceId == 0)
    {
        msg_error() << "No CudaTriangleCollisionModel found at or below one of the surface states.";
        return;
    }
    // More tools: four lists of the same length, one rigid body each.
    const std::size_t extra = l_additionalRigidStates.size();
    if (l_additionalRigidSurfaces.size() != extra || l_additionalRigidLinearSolvers.size() != extra ||
        l_additionalRigidOdeSolvers.size() != extra)
    {
        msg_error() << "additionalRigidStates, additionalRigidSurfaces, additionalRigidLinearSolvers and "
                       "additionalRigidOdeSolvers must list the same number of bodies.";
        return;
    }
    m_additional.assign(extra, AdditionalRigid {});
    for (std::size_t b = 0; b < extra; ++b)
    {
        auto* state = l_additionalRigidStates.get(b);
        auto* surface = l_additionalRigidSurfaces.get(b);
        if (state == nullptr || surface == nullptr || l_additionalRigidLinearSolvers.get(b) == nullptr ||
            l_additionalRigidOdeSolvers.get(b) == nullptr || state->getSize() != 1)
        {
            msg_error() << "Additional rigid body " << b + 1 << ": a link is missing or its state does not hold one rigid body.";
            return;
        }
        m_additional[b].surfaceId = surfaceIdOf(surface);
        if (m_additional[b].surfaceId == 0)
        {
            msg_error() << "Additional rigid body " << b + 1 << ": no CudaTriangleCollisionModel at or below its surface state.";
            return;
        }
    }
    if (extra > 0 && (d_compareWithCpu.getValue() || d_response.getValue() == "cpu"))
    {
        msg_warning() << "compareWithCpu and response=\"cpu\" handle one rigid body: turned off with "
                      << extra + 1 << " rigid bodies.";
        d_compareWithCpu.setValue(false);
        d_response.setValue(std::string("gpu"));
    }

    std::string diagnostic;
    m_workspace = backend::createConstraintWorkspace(diagnostic);
    if (m_workspace == nullptr)
    {
        msg_error() << "GPU constraint workspace: " << diagnostic;
        return;
    }

    sofa::simulation::common::VectorOperations vop(sofa::core::execparams::defaultInstance(), this->getContext());
    {
        sofa::core::behavior::MultiVecDeriv lambda(&vop, m_lambdaId);
        lambda.realloc(&vop, false, true, sofa::core::VecIdProperties{ "lambda", GetClass()->className });
        m_lambdaId = lambda.id();
    }
    {
        sofa::core::behavior::MultiVecDeriv dx(&vop, m_dxId);
        dx.realloc(&vop, false, true, sofa::core::VecIdProperties{ "constraint_dx", GetClass()->className });
        m_dxId = dx.id();
    }

    if (d_response.getValue() != "gpu" && d_response.getValue() != "cpu")
    {
        msg_error() << "response must be 'gpu' or 'cpu', not '" << d_response.getValue() << "'.";
        return;
    }
    if (d_response.getValue() == "cpu") m_cpu = std::make_unique<CpuReference>();
    if (d_compareWithCpu.getValue())
    {
        m_compareStream.open(d_compareFile.getValue());
        if (!m_compareStream)
        {
            msg_warning() << "Could not open " << d_compareFile.getValue() << "; compareWithCpu disabled.";
        }
        else
        {
            if (!m_cpu) m_cpu = std::make_unique<CpuReference>();
            m_compareStream <<
                "time,detected_contacts,contacts,rows,touched_vertices,"
                "rows_max_diff,dfree_max_diff_m,"
                "compliance_rel_diff,lambda_rel_diff,lambda_rel_diff_same_input,"
                "sweeps_gpu,sweeps_cpu,sweeps_cpu_same_input,"
                "final_violation_min_gpu_m,final_violation_min_cpu_m,"
                "correction_diff_m,correction_diff_same_lambda_m,correction_max_m,rigid_correction_diff_m,"
                "contact_force_y_gpu_N,contact_force_y_cpu_N,normal_impulse_gpu,normal_impulse_cpu,"
                "gpu_build_ms,gpu_factorize_ms,gpu_compliance_ms,gpu_solve_ms,gpu_correction_ms,"
                "cpu_rows_ms,cpu_compliance_ms,cpu_solve_ms,cpu_correction_ms,cpu_reference_factorize_ms,applied\n";
        }
    }
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void GpuContactConstraintSolver::cleanup()
{
    if (!m_lambdaId.isNull() || !m_dxId.isNull())
    {
        sofa::simulation::common::VectorOperations vop(sofa::core::execparams::defaultInstance(), this->getContext());
        if (!m_lambdaId.isNull()) vop.v_free(m_lambdaId, false, true);
        if (!m_dxId.isNull()) vop.v_free(m_dxId, false, true);
    }
    Inherit1::cleanup();
}

void GpuContactConstraintSolver::reportFailure(const std::string& stage, const std::string& diagnostic,
                                               const std::string& consequence)
{
    // Said once per stage, loudly, rather than every frame.
    if (m_warnedStages.insert(stage).second)
    {
        msg_warning() << "GPU contact constraints: " << stage << " failed: " << diagnostic << " -- " << consequence
                      << ". Further occurrences at this stage are not reported.";
    }
}

std::uint8_t GpuContactConstraintSolver::rigidMaskOf(sofa::core::behavior::MechanicalState<RigidTypes>* state, bool& general)
{
    // A rigid body's DOFs its projective constraints hold: project a vector of ones.
    const sofa::core::MechanicalParams* mparams = sofa::core::mechanicalparams::defaultInstance();
    sofa::core::objectmodel::Data<RigidTypes::VecDeriv> ones;
    ones.setValue(RigidTypes::VecDeriv(state->getSize(), RigidTypes::Deriv(Vec3d(1.0, 1.0, 1.0), Vec3d(1.0, 1.0, 1.0))));
    bool used = false;
    for (auto* constraint : state->getContext()->getObjects<sofa::core::behavior::ProjectiveConstraintSet<RigidTypes>>(BaseContext::Local))
    {
        if (constraint->getMState() != state || !constraint->isActive()) continue;
        constraint->projectResponse(mparams, ones);
        used = true;
    }
    std::uint8_t mask = 0x3F;
    if (used)
    {
        const auto& d = ones.getValue()[0];
        mask = 0;
        for (int e = 0; e < 3; ++e)
        {
            if (d.getVCenter()[e] != 0.0) mask |= static_cast<std::uint8_t>(1u << e);
            if (d.getVOrientation()[e] != 0.0) mask |= static_cast<std::uint8_t>(1u << (3 + e));
            if (d.getVCenter()[e] != 0.0 && d.getVCenter()[e] != 1.0) general = true;
            if (d.getVOrientation()[e] != 0.0 && d.getVOrientation()[e] != 1.0) general = true;
        }
    }
    return mask;
}

void GpuContactConstraintSolver::updateDofMasks()
{
    // Which DOFs the projective constraints hold: project a vector of ones.
    // FixedProjectiveConstraint / PartialFixedProjectiveConstraint zero the held
    // entries, and SOFA removes the same entries from the constraint rows
    // (MechanicalProjectJacobianMatrixVisitor).
    const sofa::core::MechanicalParams* mparams = sofa::core::mechanicalparams::defaultInstance();
    bool general = false;
    if (gpuMode())
    {
        // The tissue solver found its fixed DOFs once (bit set = fixed).
        const auto& fixed = l_deformableGpuSolver->fixedDofs();
        m_deformableMaskUsed = false;
        m_deformableMask.assign(fixed.size(), 7);
        for (std::size_t i = 0; i < fixed.size(); ++i)
        {
            m_deformableMask[i] = static_cast<std::uint8_t>(~fixed[i] & 7u);
            if (fixed[i] != 0) m_deformableMaskUsed = true;
        }
    }
    else
    {
        auto* state = l_deformableState.get();
        sofa::core::objectmodel::Data<DeformableTypes::VecDeriv> ones;
        ones.setValue(DeformableTypes::VecDeriv(state->getSize(), DeformableTypes::Deriv(1.0, 1.0, 1.0)));
        m_deformableMaskUsed = false;
        for (auto* constraint : state->getContext()->getObjects<sofa::core::behavior::ProjectiveConstraintSet<DeformableTypes>>(BaseContext::Local))
        {
            if (constraint->getMState() != state || !constraint->isActive()) continue;
            constraint->projectResponse(mparams, ones);
            m_deformableMaskUsed = true;
        }
        if (m_deformableMaskUsed)
        {
            const auto& projected = ones.getValue();
            m_deformableMask.assign(projected.size(), 0);
            for (std::size_t i = 0; i < projected.size(); ++i)
            {
                for (int c = 0; c < 3; ++c)
                {
                    if (projected[i][c] != 0.0) m_deformableMask[i] |= static_cast<std::uint8_t>(1u << c);
                    if (projected[i][c] != 0.0 && projected[i][c] != 1.0) general = true;
                }
            }
        }
    }
    m_rigidMask = rigidMaskOf(l_rigidState.get(), general);
    for (std::size_t b = 0; b < m_additional.size(); ++b) m_additional[b].mask = rigidMaskOf(l_additionalRigidStates.get(b), general);
    if (general)
    {
        reportFailure("projective constraints",
                      "a projective constraint that is not a fixed-DOF constraint (e.g. a projection onto a plane) "
                      "acts on one of the bodies",
                      "only fixed DOFs are removed from the contact rows; the contacts are still solved");
    }
}

bool GpuContactConstraintSolver::readDeformableMatrix(sofa::core::behavior::LinearSolver* solver,
                                                      backend::HostCsrMatrix& matrix, std::string& diagnostic)
{
    auto* system = solver != nullptr ? solver->getLinearSystem() : nullptr;
    sofa::linearalgebra::BaseMatrix* base = system != nullptr ? system->getSystemBaseMatrix() : nullptr;
    if (base == nullptr)
    {
        diagnostic = "the deformable body's linear solver has no assembled system matrix (a direct solver is needed)";
        return false;
    }
    const int n = static_cast<int>(base->rowSize());
    if (n != 3 * static_cast<int>(deformableVertexCount()))
    {
        diagnostic = "the deformable system matrix has " + std::to_string(n) + " rows, expected " +
                     std::to_string(3 * deformableVertexCount());
        return false;
    }

    using Block3 = sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Mat<3, 3, SReal>>;
    using Scalar = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    m_csrRowPtr.assign(static_cast<std::size_t>(n) + 1, 0);
    if (auto* m = dynamic_cast<Block3*>(base))
    {
        m->compress();
        for (std::size_t xi = 0; xi < m->rowIndex.size(); ++xi)
        {
            const int blockRow = static_cast<int>(m->rowIndex[xi]);
            const int blocks = static_cast<int>(m->rowBegin[xi + 1] - m->rowBegin[xi]);
            for (int r = 0; r < 3; ++r) m_csrRowPtr[3 * blockRow + r + 1] = 3 * blocks;
        }
        for (int r = 0; r < n; ++r) m_csrRowPtr[r + 1] += m_csrRowPtr[r];
        m_csrColumns.resize(static_cast<std::size_t>(m_csrRowPtr[n]));
        m_csrValues.resize(static_cast<std::size_t>(m_csrRowPtr[n]));
        for (std::size_t xi = 0; xi < m->rowIndex.size(); ++xi)
        {
            const int blockRow = static_cast<int>(m->rowIndex[xi]);
            for (int r = 0; r < 3; ++r)
            {
                int p = m_csrRowPtr[3 * blockRow + r];
                for (auto b = m->rowBegin[xi]; b < m->rowBegin[xi + 1]; ++b)
                {
                    const int blockCol = static_cast<int>(m->colsIndex[b]);
                    const auto& block = m->colsValue[b];
                    for (int c = 0; c < 3; ++c)
                    {
                        m_csrColumns[p] = 3 * blockCol + c;
                        m_csrValues[p] = block[r][c];
                        ++p;
                    }
                }
            }
        }
    }
    else if (auto* s = dynamic_cast<Scalar*>(base))
    {
        s->compress();
        for (std::size_t xi = 0; xi < s->rowIndex.size(); ++xi)
        {
            m_csrRowPtr[s->rowIndex[xi] + 1] = static_cast<int>(s->rowBegin[xi + 1] - s->rowBegin[xi]);
        }
        for (int r = 0; r < n; ++r) m_csrRowPtr[r + 1] += m_csrRowPtr[r];
        m_csrColumns.resize(static_cast<std::size_t>(m_csrRowPtr[n]));
        m_csrValues.resize(static_cast<std::size_t>(m_csrRowPtr[n]));
        for (std::size_t xi = 0; xi < s->rowIndex.size(); ++xi)
        {
            int p = m_csrRowPtr[s->rowIndex[xi]];
            for (auto b = s->rowBegin[xi]; b < s->rowBegin[xi + 1]; ++b)
            {
                m_csrColumns[p] = static_cast<int>(s->colsIndex[b]);
                m_csrValues[p] = s->colsValue[b];
                ++p;
            }
        }
    }
    else
    {
        diagnostic = "the deformable system matrix is not a CompressedRowSparseMatrix (use SparseLDLSolver with "
                     "CompressedRowSparseMatrixMat3x3d or CompressedRowSparseMatrixd)";
        return false;
    }
    matrix.size = n;
    matrix.nonZeros = m_csrRowPtr[n];
    matrix.rowPtr = m_csrRowPtr.data();
    matrix.columns = m_csrColumns.data();
    matrix.values = m_csrValues.data();
    return true;
}

bool GpuContactConstraintSolver::readRigidMatrix(double matrix[36], std::string& diagnostic)
{
    return readRigidMatrixOf(l_rigidLinearSolver.get(), matrix, diagnostic);
}

bool GpuContactConstraintSolver::readRigidMatrixOf(sofa::core::behavior::LinearSolver* solver, double matrix[36],
                                                   std::string& diagnostic)
{
    auto* system = solver != nullptr ? solver->getLinearSystem() : nullptr;
    sofa::linearalgebra::BaseMatrix* base = system != nullptr ? system->getSystemBaseMatrix() : nullptr;
    if (base == nullptr || base->rowSize() != 6 || base->colSize() != 6)
    {
        diagnostic = "the rigid body's linear solver has no assembled 6x6 system matrix (a direct solver is needed)";
        return false;
    }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) matrix[i * 6 + j] = base->element(i, j);
    return true;
}

bool GpuContactConstraintSolver::prepareStates(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId, sofa::core::MultiVecId)
{
    sofa::simulation::common::VectorOperations vop(cParams, this->getContext());
    {
        sofa::core::behavior::MultiVecDeriv lambda(&vop, m_lambdaId);
        lambda.realloc(&vop, false, true, sofa::core::VecIdProperties{ "lambda", GetClass()->className });
        m_lambdaId = lambda.id();
        clearMultiVec(this->getContext(), cParams, m_lambdaId);
    }
    {
        sofa::core::behavior::MultiVecDeriv dx(&vop, m_dxId);
        dx.realloc(&vop, false, true, sofa::core::VecIdProperties{ "constraint_dx", GetClass()->className });
        m_dxId = dx.id();
        clearMultiVec(this->getContext(), cParams, m_dxId);
    }
    m_haveRows = false;
    m_solved = false;
    m_corrected = false;
    m_buildStats = backend::ConstraintBuildStats {};
    m_solveStats = backend::ConstraintSolveStats {};
    m_timings = backend::ConstraintTimings {};
    m_impulse = backend::ConstraintImpulse {};
    return true;
}

bool GpuContactConstraintSolver::buildSystem(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId, sofa::core::MultiVecId)
{
    // Always true: applyCorrection must still run, to move the bodies to their
    // free motion when there is nothing to correct.
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid) return true;
    if (cParams->constOrder() != sofa::core::ConstraintOrder::POS_AND_VEL)
    {
        reportFailure("constraint order", "only POS_AND_VEL (FreeMotionAnimationLoop's default) is supported");
        return true;
    }
    const bool timing = d_measureTimes.getValue() || m_compareStream.is_open();
    backend::ConstraintTimings* timings = timing ? &m_timings : nullptr;
    std::string diagnostic;

    if (!readRigidMatrix(m_rigidMatrix, diagnostic) || !backend::setRigidSystem(m_workspace, m_rigidMatrix, diagnostic))
    {
        reportFailure("rigid body system", diagnostic);
        return true;
    }
    if (!m_additional.empty())
    {
        std::vector<double> matrices(36 * m_additional.size());
        std::vector<double> factors(m_additional.size());
        for (std::size_t b = 0; b < m_additional.size(); ++b)
        {
            if (!readRigidMatrixOf(l_additionalRigidLinearSolvers.get(b), m_additional[b].matrix, diagnostic))
            {
                reportFailure("rigid body system", "additional body " + std::to_string(b + 1) + ": " + diagnostic);
                return true;
            }
            std::copy(m_additional[b].matrix, m_additional[b].matrix + 36, matrices.begin() + 36 * b);
            m_additional[b].factor = correctionFactor(l_additionalRigidOdeSolvers.get(b), cParams->constOrder());
            factors[b] = m_additional[b].factor;
        }
        if (!backend::setAdditionalRigidSystems(m_workspace, matrices, factors, diagnostic))
        {
            reportFailure("rigid body system", diagnostic);
            return true;
        }
    }
    updateDofMasks();

    const void* freeDevice = nullptr;
    if (gpuMode())
    {
        // Body 1 on the GPU: its free positions, in place.
        freeDevice = cParams->readX(m_gpuState)->getValue().deviceRead();
    }
    else
    {
        const auto& deformableFree = cParams->readX(l_deformableState.get())->getValue();
        m_freePositions.resize(3 * deformableFree.size());
        for (std::size_t i = 0; i < deformableFree.size(); ++i)
        {
            for (int c = 0; c < 3; ++c) m_freePositions[3 * i + c] = static_cast<float>(deformableFree[i][c]);
        }
    }
    const auto& rigidCurrent = l_rigidState->read(sofa::core::vec_id::read_access::position)->getValue()[0];
    const auto& rigidFreeVelocity = cParams->readV(l_rigidState.get())->getValue()[0];
    const SReal dt = this->getContext()->getDt();

    backend::ConstraintBuildInput input;
    input.deformableSurfaceId = m_deformableSurfaceId;
    input.rigidSurfaceId = m_rigidSurfaceId;
    input.deformableSurfacePositions = l_deformableSurface->read(sofa::core::vec_id::read_access::position)->getValue().deviceRead();
    input.rigidSurfacePositions = l_rigidSurface->read(sofa::core::vec_id::read_access::position)->getValue().deviceRead();
    input.deformableVertexCount = static_cast<std::uint32_t>(l_deformableSurface->getSize());
    input.rigidVertexCount = static_cast<std::uint32_t>(l_rigidSurface->getSize());
    input.deformableFreePositions = gpuMode() ? nullptr : m_freePositions.data();
    input.deformableFreePositionsDevice = freeDevice;
    for (int e = 0; e < 3; ++e)
    {
        input.rigidCenter[e] = rigidCurrent.getCenter()[e];
        input.rigidFreeStep[e] = dt * rigidFreeVelocity.getVCenter()[e];
        input.rigidFreeStep[3 + e] = dt * rigidFreeVelocity.getVOrientation()[e];
    }
    input.deformableDofMask = m_deformableMaskUsed ? m_deformableMask.data() : nullptr;
    input.rigidDofMask = m_rigidMask;
    input.contactDistance = static_cast<float>(d_contactDistance.getValue());
    input.friction = static_cast<float>(d_friction.getValue());
    input.filter = d_contactFilter.getValue() == "all" ? backend::ConstraintContactFilter::All
                                                       : backend::ConstraintContactFilter::VertexFace;
    input.vertexConeFilter = d_vertexConeFilter.getValue();
    input.vertexConeTolerance = static_cast<float>(d_vertexConeTolerance.getValue());
    for (std::size_t b = 0; b < m_additional.size(); ++b)
    {
        auto* state = l_additionalRigidStates.get(b);
        auto* surface = l_additionalRigidSurfaces.get(b);
        const auto& current = state->read(sofa::core::vec_id::read_access::position)->getValue()[0];
        const auto& freeVelocity = cParams->readV(state)->getValue()[0];
        backend::ConstraintRigidBodyInput body;
        body.surfaceId = m_additional[b].surfaceId;
        body.surfacePositions = surface->read(sofa::core::vec_id::read_access::position)->getValue().deviceRead();
        body.vertexCount = static_cast<std::uint32_t>(surface->getSize());
        for (int e = 0; e < 3; ++e)
        {
            body.center[e] = current.getCenter()[e];
            body.freeStep[e] = dt * freeVelocity.getVCenter()[e];
            body.freeStep[3 + e] = dt * freeVelocity.getVOrientation()[e];
        }
        body.dofMask = m_additional[b].mask;
        input.additionalRigidBodies.push_back(body);
    }
    if (!backend::buildContactConstraints(m_workspace, input, &m_buildStats, timings, diagnostic))
    {
        reportFailure("constraint rows", diagnostic);
        return true;
    }
    d_currentContacts.setValue(static_cast<int>(m_buildStats.contacts));
    d_currentConstraints.setValue(static_cast<int>(m_buildStats.rows));
    if (m_buildStats.rows == 0) return true;

    m_factorDeformable = correctionFactor(deformableOde(), cParams->constOrder());
    m_factorRigid = correctionFactor(l_rigidOdeSolver.get(), cParams->constOrder());
    if (gpuMode())
    {
        // The tissue step's own factor of A1: no copy, no second factorisation
        // (band or dense Cholesky, or LU on a step where A1 was not positive definite).
        if (!backend::useTissueFactor(m_workspace, l_deformableGpuSolver->workspace(), diagnostic))
        {
            reportFailure("deformable body system", diagnostic);
            return true;
        }
    }
    else
    {
        backend::HostCsrMatrix matrix;
        if (!readDeformableMatrix(l_deformableLinearSolver.get(), matrix, diagnostic) ||
            !backend::factorizeDeformableSystem(m_workspace, matrix, timings, diagnostic))
        {
            reportFailure("deformable body system", diagnostic);
            return true;
        }
    }
    if (!backend::assembleContactCompliance(m_workspace, m_factorDeformable, m_factorRigid, timings, diagnostic))
    {
        reportFailure("compliance", diagnostic);
        return true;
    }
    m_haveRows = true;
    return true;
}

bool GpuContactConstraintSolver::solveSystem(const sofa::core::ConstraintParams*, sofa::core::MultiVecId, sofa::core::MultiVecId)
{
    d_currentIterations.setValue(0);
    d_currentError.setValue(0.0);
    if (!m_haveRows) return true;

    const bool timing = d_measureTimes.getValue() || m_compareStream.is_open();
    backend::ConstraintSolveConfig config;
    config.maxIterations = d_maxIterations.getValue();
    config.tolerance = d_tolerance.getValue();
    config.scaleTolerance = d_scaleTolerance.getValue();
    config.allVerified = d_allVerified.getValue();
    config.sor = d_sor.getValue();
    config.doubleAccumulation = d_exactArithmetic.getValue();
    std::string diagnostic;
    if (!backend::solveContactConstraints(m_workspace, config, &m_solveStats, timing ? &m_timings : nullptr, diagnostic))
    {
        reportFailure("Gauss-Seidel", diagnostic);
        return true;
    }
    m_solved = true;
    d_currentIterations.setValue(m_solveStats.iterations);
    d_currentError.setValue(m_solveStats.error);
    return true;
}

bool GpuContactConstraintSolver::applyCorrection(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2)
{
    if (!linksReady()) return true;
    const bool timing = d_measureTimes.getValue() || m_compareStream.is_open();
    const bool cpuResponse = d_response.getValue() == "cpu";

    const std::size_t deformableSize = deformableVertexCount();
    m_deformableCorrection.assign(3 * deformableSize, 0.0f);
    for (double& v : m_rigidCorrection) v = 0.0;
    m_impulse = backend::ConstraintImpulse {};
    for (auto& body : m_additional)
    {
        for (double& v : body.correction) v = 0.0;
        for (double& v : body.impulse) v = 0.0;
    }
    if (m_solved)
    {
        std::string diagnostic;
        const bool ok = gpuMode()
            // Body 1 on the GPU: dv stays on the device.
            ? backend::computeContactCorrectionOnDevice(m_workspace, m_rigidCorrection, &m_impulse,
                                                        timing ? &m_timings : nullptr, diagnostic)
            : backend::computeContactCorrection(m_workspace, m_deformableCorrection, m_rigidCorrection, &m_impulse,
                                                timing ? &m_timings : nullptr, diagnostic) &&
                  m_deformableCorrection.size() == 3 * deformableSize;
        if (ok)
        {
            m_corrected = true;
            if (!m_additional.empty())
            {
                std::vector<double> corrections, impulses;
                backend::additionalRigidResults(m_workspace, corrections, impulses);
                for (std::size_t b = 0; b < m_additional.size() && 6 * (b + 1) <= corrections.size(); ++b)
                    for (int e = 0; e < 6; ++e)
                    {
                        m_additional[b].correction[e] = corrections[6 * b + e];
                        m_additional[b].impulse[e] = impulses[6 * b + e];
                    }
            }
        }
        else
        {
            reportFailure("correction", diagnostic);
            m_deformableCorrection.assign(3 * deformableSize, 0.0f);
            for (double& v : m_rigidCorrection) v = 0.0;
            m_impulse = backend::ConstraintImpulse {};
        }
    }

    bool writeRow = false;
    if (m_corrected)
    {
        ++m_contactSteps;
        const int every = std::max(1, d_compareEvery.getValue());
        writeRow = m_compareStream.is_open() && (cpuResponse || (m_contactSteps - 1) % every == 0);
        if (gpuMode() && (cpuResponse || writeRow))
        {
            // The comparison needs the GPU's dv on the host (only then).
            std::string diagnostic;
            if (!backend::downloadDeformableCorrection(m_workspace, m_deformableCorrection, diagnostic))
                reportFailure("correction download", diagnostic, "the comparison uses a zero correction");
        }
    }

    // By default the GPU's correction is applied.
    m_appliedDeformable.assign(m_deformableCorrection.begin(), m_deformableCorrection.end());
    for (int e = 0; e < 6; ++e)
    {
        m_appliedRigid[e] = m_rigidCorrection[e];
        m_appliedImpulse[e] = m_impulse.rigid[e];
    }
    m_appliedNormalImpulse = m_impulse.normalSum;

    // SOFA's CPU pipeline reads the bodies' positions at detection, so it runs
    // before the correction moves them.
    bool appliedFromCpu = false;
    if (m_corrected && (cpuResponse || writeRow))
    {
        const bool complete = runCpuPipeline(cParams, cpuResponse, writeRow);
        if (!complete && cpuResponse)
        {
            reportFailure("CPU response", "SOFA's CPU pipeline did not complete on this step",
                          "the GPU's correction is applied instead");
        }
        appliedFromCpu = complete && cpuResponse;
    }

    if (gpuMode() && cParams->constOrder() == sofa::core::ConstraintOrder::POS_AND_VEL)
        applyDeformableMotionOnDevice(cParams, res1, res2, appliedFromCpu);
    applyMotion(cParams, res1, res2);

    const SReal dt = this->getContext()->getDt();
    Wrench force;
    for (int e = 0; e < 6; ++e) force[e] = m_appliedImpulse[e] / dt;
    d_rigidContactForce.setValue(force);
    if (!m_additional.empty())
    {
        sofa::type::vector<Wrench> forces(m_additional.size());
        for (std::size_t b = 0; b < m_additional.size(); ++b)
            for (int e = 0; e < 6; ++e) forces[b][e] = m_additional[b].impulse[e] / dt;
        d_additionalRigidContactForces.setValue(forces);
    }
    d_normalImpulse.setValue(m_appliedNormalImpulse);
    d_stepGpuMilliseconds.setValue(m_timings.buildMs + m_timings.factorizeMs + m_timings.complianceMs +
                                   m_timings.solveMs + m_timings.correctionMs);
    d_stageMilliseconds.setValue({ m_timings.buildMs, m_timings.factorizeMs, m_timings.complianceMs,
                                   m_timings.solveMs, m_timings.correctionMs });
    return true;
}

void GpuContactConstraintSolver::applyMotion(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId res1, sofa::core::MultiVecId res2)
{
    // x = x_free + positionFactor dv, v = v_free + velocityFactor dv, dx = positionFactor dv:
    // LinearSolverConstraintCorrection::applyMotionCorrection, for both bodies
    // (dv = 0 without contacts: the bodies take their free motion).
    if (cParams->constOrder() != sofa::core::ConstraintOrder::POS_AND_VEL)
    {
        // VEL order: res1 is the velocity; keep the free velocity.
        if (!gpuMode())
        {
            auto* deformable = l_deformableState.get();
            auto v1 = sofa::helper::getWriteAccessor(*deformable->write(sofa::core::MultiVecDerivId(res1).getId(deformable)));
            v1.wref() = cParams->readV(deformable)->getValue();
        }
        std::vector<sofa::core::behavior::MechanicalState<RigidTypes>*> rigids { l_rigidState.get() };
        for (std::size_t b = 0; b < l_additionalRigidStates.size(); ++b) rigids.push_back(l_additionalRigidStates.get(b));
        for (auto* rigid : rigids)
        {
            auto v2 = sofa::helper::getWriteAccessor(*rigid->write(sofa::core::MultiVecDerivId(res1).getId(rigid)));
            v2.wref() = cParams->readV(rigid)->getValue();
        }
        return;
    }
    if (!gpuMode())   // body 1 on the GPU is corrected by applyDeformableMotionOnDevice
    {
        auto* state = l_deformableState.get();
        const SReal positionFactor = l_deformableOdeSolver->getPositionIntegrationFactor();
        const SReal velocityFactor = l_deformableOdeSolver->getVelocityIntegrationFactor();
        auto x = sofa::helper::getWriteAccessor(*state->write(sofa::core::MultiVecCoordId(res1).getId(state)));
        auto v = sofa::helper::getWriteAccessor(*state->write(sofa::core::MultiVecDerivId(res2).getId(state)));
        auto dx = sofa::helper::getWriteAccessor(*state->write(m_dxId.getId(state)));
        const auto& xFree = cParams->readX(state)->getValue();
        const auto& vFree = cParams->readV(state)->getValue();
        const std::size_t size = xFree.size();
        x.resize(size);
        v.resize(size);
        dx.resize(size);
        m_appliedDeformable.resize(3 * size, 0.0);
        for (std::size_t i = 0; i < size; ++i)
        {
            const DeformableTypes::Deriv correction(m_appliedDeformable[3 * i], m_appliedDeformable[3 * i + 1],
                                                    m_appliedDeformable[3 * i + 2]);
            const DeformableTypes::Deriv dxi = correction * positionFactor;
            x[i] = xFree[i] + dxi;
            v[i] = vFree[i] + correction * velocityFactor;
            dx[i] = dxi;
        }
    }
    applyRigidMotion(cParams, res1, res2, l_rigidState.get(), l_rigidOdeSolver.get(), m_appliedRigid, m_appliedImpulse);
    for (std::size_t b = 0; b < m_additional.size(); ++b)
        applyRigidMotion(cParams, res1, res2, l_additionalRigidStates.get(b), l_additionalRigidOdeSolvers.get(b),
                         m_additional[b].correction, m_additional[b].impulse);
}

void GpuContactConstraintSolver::applyRigidMotion(const sofa::core::ConstraintParams* cParams, sofa::core::MultiVecId res1,
                                                  sofa::core::MultiVecId res2,
                                                  sofa::core::behavior::MechanicalState<RigidTypes>* state,
                                                  sofa::core::behavior::OdeSolver* ode, const double correctionValues[6],
                                                  const double impulseValues[6])
{
    const SReal positionFactor = ode->getPositionIntegrationFactor();
    const SReal velocityFactor = ode->getVelocityIntegrationFactor();
    auto x = sofa::helper::getWriteAccessor(*state->write(sofa::core::MultiVecCoordId(res1).getId(state)));
    auto v = sofa::helper::getWriteAccessor(*state->write(sofa::core::MultiVecDerivId(res2).getId(state)));
    auto dx = sofa::helper::getWriteAccessor(*state->write(m_dxId.getId(state)));
    auto lambda = sofa::helper::getWriteAccessor(*state->write(m_lambdaId.getId(state)));
    const auto& xFree = cParams->readX(state)->getValue();
    const auto& vFree = cParams->readV(state)->getValue();
    x.resize(1);
    v.resize(1);
    dx.resize(1);
    lambda.resize(1);
    RigidTypes::Deriv correction;
    RigidTypes::Deriv impulse;
    for (int e = 0; e < 3; ++e)
    {
        correction.getVCenter()[e] = correctionValues[e];
        correction.getVOrientation()[e] = correctionValues[3 + e];
        impulse.getVCenter()[e] = impulseValues[e];
        impulse.getVOrientation()[e] = impulseValues[3 + e];
    }
    const RigidTypes::Deriv dxi = correction * positionFactor;
    x[0] = xFree[0] + dxi;
    v[0] = vFree[0] + correction * velocityFactor;
    dx[0] = dxi;
    // SOFA stores J^T lambda (an impulse) in the lambda vector
    // (GenericConstraintSolver::storeConstraintLambdas). Only the rigid bodies' are
    // stored: body 1's would cost a download of a full DOF vector.
    lambda[0] = impulse;
}

void GpuContactConstraintSolver::applyDeformableMotionOnDevice(const sofa::core::ConstraintParams* cParams,
                                                              sofa::core::MultiVecId res1, sofa::core::MultiVecId res2,
                                                              const bool useAppliedCorrection)
{
    // Body 1 on the GPU: x = x_free + h dv, v = v_free + dv, dx = h dv, on the device.
    auto* state = m_gpuState;
    const std::size_t n = state->getSize();
    auto* xData = state->write(sofa::core::MultiVecCoordId(res1).getId(state));
    auto* vData = state->write(sofa::core::MultiVecDerivId(res2).getId(state));
    auto* dxData = state->write(m_dxId.getId(state));
    backend::DeviceCorrectionTarget target;
    target.xFree = cParams->readX(state)->getValue().deviceRead();
    target.vFree = cParams->readV(state)->getValue().deviceRead();
    auto& x = *xData->beginEdit();
    auto& v = *vData->beginEdit();
    auto& dx = *dxData->beginEdit();
    if (dx.size() != n) dx.resize(n);
    target.x = x.deviceWrite();
    target.v = v.deviceWrite();
    target.dx = dx.deviceWrite();
    target.positionFactor = deformableOde()->getPositionIntegrationFactor();
    target.velocityFactor = deformableOde()->getVelocityIntegrationFactor();
    target.vertexCount = static_cast<int>(n);
    target.withContacts = m_corrected;
    std::vector<float> hostCorrection;
    if (useAppliedCorrection)
    {
        hostCorrection.assign(m_appliedDeformable.begin(), m_appliedDeformable.end());
        hostCorrection.resize(3 * n, 0.0f);
        target.hostCorrection = hostCorrection.data();
    }
    std::string diagnostic;
    if (!backend::applyContactCorrectionOnDevice(m_workspace, target, diagnostic))
        reportFailure("correction on the GPU", diagnostic, "the tissue keeps its previous position this step");
    xData->endEdit();
    vData->endEdit();
    dxData->endEdit();
    l_deformableGpuSolver->updateMonitor(target.x);
}

bool GpuContactConstraintSolver::runCpuPipeline(const sofa::core::ConstraintParams* cParams, const bool apply, const bool writeRow)
{
    std::string diagnostic;
    backend::ConstraintProblemSnapshot snap;
    if (!m_cpu || !backend::downloadContactProblem(m_workspace, true, snap, diagnostic) || snap.rows == 0 ||
        snap.lambda.size() != snap.rows || snap.compliance.size() != static_cast<std::size_t>(snap.rows) * snap.rows)
    {
        return false;
    }
    CpuReference& cpu = *m_cpu;
    const int rows = static_cast<int>(snap.rows);
    const int rowsPerContact = static_cast<int>(snap.rowsPerContact);
    const int contacts = static_cast<int>(snap.contacts);
    if (!d_dumpContactsFile.getValue().empty())
    {
        if (!m_dumpStream.is_open())
        {
            m_dumpStream.open(d_dumpContactsFile.getValue());
            m_dumpStream << "time,contact,v0,v1,v2,w0,w1,w2,px,py,pz,qx,qy,qz,nx,ny,nz,gap,lambda_n\n";
        }
        const double time = this->getContext()->getTime();
        for (int k = 0; k < contacts; ++k)
        {
            const double* g = snap.contactGeometry.data() + static_cast<std::size_t>(k) * 21;
            const double gap = g[12] * (g[3] - g[0]) + g[13] * (g[4] - g[1]) + g[14] * (g[5] - g[2]);
            m_dumpStream << time << ',' << k;
            for (int a = 0; a < 3; ++a) m_dumpStream << ',' << snap.contactVertices[static_cast<std::size_t>(k) * 3 + a];
            for (int a = 0; a < 3; ++a) m_dumpStream << ',' << snap.contactWeights[static_cast<std::size_t>(k) * 3 + a];
            for (int a = 0; a < 6; ++a) m_dumpStream << ',' << g[a];
            for (int a = 12; a < 15; ++a) m_dumpStream << ',' << g[a];
            const std::size_t row = static_cast<std::size_t>(k) * rowsPerContact;
            m_dumpStream << ',' << gap << ',' << (row < snap.lambda.size() ? snap.lambda[row] : 0.0) << '\n';
        }
        m_dumpStream.flush();
    }
    const int n = 3 * static_cast<int>(deformableVertexCount());
    const SReal dt = this->getContext()->getDt();
    sofa::core::behavior::LinearSolver* deformableSolver = deformableLinear();
    if (deformableSolver == nullptr)
    {
        reportFailure("CPU comparison", "body 1's CPU linear solver is missing (with deformableGpuSolver, turn on the "
                      "tissue solver's compareWithCpu: its CPU replica provides it)", "no comparison");
        return false;
    }
    if (gpuMode())
    {
        // Body 1's system matrix for the double-precision correction reference.
        backend::HostCsrMatrix matrix;
        std::string readDiagnostic;
        if (!readDeformableMatrix(deformableSolver, matrix, readDiagnostic))
        {
            reportFailure("CPU comparison", readDiagnostic, "no comparison");
            return false;
        }
    }

    // ---- 1. rows and free violations: SOFA's UnilateralLagrangianConstraint on
    // the same contacts (points, normal, body-1 vertices and weights), with the
    // free points computed here in double from the bodies' states.
    double t0 = nowMs();
    std::vector<Vec3d> x(n / 3), xFree(n / 3);
    if (gpuMode())
    {
        // Host copies of the GPU state (comparison only).
        const auto& xg = m_gpuState->read(sofa::core::vec_id::read_access::position)->getValue();
        const auto& xfg = cParams->readX(m_gpuState)->getValue();
        for (int i = 0; i < n / 3; ++i)
        {
            x[i] = Vec3d(xg[i][0], xg[i][1], xg[i][2]);
            xFree[i] = Vec3d(xfg[i][0], xfg[i][1], xfg[i][2]);
        }
    }
    else
    {
        const auto& xs = l_deformableState->read(sofa::core::vec_id::read_access::position)->getValue();
        const auto& xfs = cParams->readX(l_deformableState.get())->getValue();
        for (int i = 0; i < n / 3; ++i)
        {
            x[i] = xs[i];
            xFree[i] = xfs[i];
        }
    }
    const auto& rigidX = l_rigidState->read(sofa::core::vec_id::read_access::position)->getValue()[0];
    const auto& rigidVFree = cParams->readV(l_rigidState.get())->getValue()[0];
    const Vec3d center = rigidX.getCenter();
    const Vec3d linearStep = rigidVFree.getVCenter() * dt;
    const Vec3d angularStep = rigidVFree.getVOrientation() * dt;

    cpu.bodyPoints->resize(contacts);
    cpu.rigidPoints->resize(contacts);
    std::vector<Vec3d> contactP(contacts);
    {
        auto q = sofa::helper::getWriteOnlyAccessor(*cpu.bodyPoints->write(sofa::core::vec_id::write_access::position));
        auto qFree = sofa::helper::getWriteOnlyAccessor(*cpu.bodyPoints->write(sofa::core::vec_id::write_access::freePosition));
        auto p = sofa::helper::getWriteOnlyAccessor(*cpu.rigidPoints->write(sofa::core::vec_id::write_access::position));
        auto pFree = sofa::helper::getWriteOnlyAccessor(*cpu.rigidPoints->write(sofa::core::vec_id::write_access::freePosition));
        q.resize(contacts);
        qFree.resize(contacts);
        p.resize(contacts);
        pFree.resize(contacts);
        for (int k = 0; k < contacts; ++k)
        {
            const double* g = snap.contactGeometry.data() + static_cast<std::size_t>(k) * 21;
            const Vec3d P(g[0], g[1], g[2]);
            Vec3d Q, QFree;
            for (int a = 0; a < 3; ++a)
            {
                const int v = snap.contactVertices[static_cast<std::size_t>(k) * 3 + a];
                if (v < 0) continue;
                const double w = snap.contactWeights[static_cast<std::size_t>(k) * 3 + a];
                Q += x[v] * w;
                QFree += xFree[v] * w;
            }
            q[k] = Q;
            qFree[k] = QFree;
            p[k] = P;
            // FreeMotionAnimationLoop: a rigidly mapped point moves by dt (v + omega x r).
            pFree[k] = P + linearStep + sofa::type::cross(angularStep, P - center);
            contactP[k] = P;
        }
    }
    cpu.constraint->clear(contacts);
    {
        const auto& q = cpu.bodyPoints->read(sofa::core::vec_id::read_access::position)->getValue();
        const auto& qFree = cpu.bodyPoints->read(sofa::core::vec_id::read_access::freePosition)->getValue();
        const auto& p = cpu.rigidPoints->read(sofa::core::vec_id::read_access::position)->getValue();
        const auto& pFree = cpu.rigidPoints->read(sofa::core::vec_id::read_access::freePosition)->getValue();
        const sofa::component::constraint::lagrangian::model::UnilateralLagrangianContactParameters parameters(snap.friction);
        for (int k = 0; k < contacts; ++k)
        {
            const double* g = snap.contactGeometry.data() + static_cast<std::size_t>(k) * 21;
            Vec3d normal(g[12], g[13], g[14]);
            normal.normalize();
            cpu.constraint->addContact(parameters, normal, p[k], q[k], d_contactDistance.getValue(), k, k, pFree[k], qFree[k]);
        }
    }
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec3dTypes::MatrixDeriv> rowsOnQ;
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec3dTypes::MatrixDeriv> rowsOnP;
    unsigned int constraintIndex = 0;
    cpu.constraint->buildConstraintMatrix(cParams, rowsOnQ, rowsOnP, constraintIndex,
        *cpu.bodyPoints->read(sofa::core::vec_id::read_access::position),
        *cpu.rigidPoints->read(sofa::core::vec_id::read_access::position));
    sofa::linearalgebra::FullVector<SReal> violation(rows);
    violation.clear();
    cpu.constraint->getConstraintViolation(cParams, &violation,
        *cpu.bodyPoints->read(sofa::core::vec_id::read_access::position),
        *cpu.rigidPoints->read(sofa::core::vec_id::read_access::position),
        *cpu.bodyPoints->read(sofa::core::vec_id::read_access::velocity),
        *cpu.rigidPoints->read(sofa::core::vec_id::read_access::velocity));

    // Map the rows onto the DOFs as the mappings would: body 1 barycentrically
    // onto its triangle's vertices, body 2 through RigidMapping ([u ; r x u]);
    // then drop the DOFs held by projective constraints.
    std::vector<double> cpuDeformable(static_cast<std::size_t>(rows) * 9, 0.0);
    std::vector<double> cpuRigid(static_cast<std::size_t>(rows) * 6, 0.0);
    std::vector<double> cpuDfree(rows, 0.0);
    for (int r = 0; r < rows && r < static_cast<int>(constraintIndex); ++r) cpuDfree[r] = violation[r];
    for (auto rowIt = rowsOnQ.getValue().begin(); rowIt != rowsOnQ.getValue().end(); ++rowIt)
    {
        const int r = static_cast<int>(rowIt.index());
        if (r >= rows) continue;
        for (auto colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
        {
            const int k = static_cast<int>(colIt.index());
            const Vec3d u = colIt.val();
            for (int a = 0; a < 3; ++a)
            {
                const int v = snap.contactVertices[static_cast<std::size_t>(k) * 3 + a];
                if (v < 0) continue;
                const double w = snap.contactWeights[static_cast<std::size_t>(k) * 3 + a];
                const unsigned bits = m_deformableMaskUsed ? m_deformableMask[v] : 7u;
                for (int c = 0; c < 3; ++c)
                    cpuDeformable[static_cast<std::size_t>(r) * 9 + a * 3 + c] += ((bits >> c) & 1u) ? u[c] * w : 0.0;
            }
        }
    }
    for (auto rowIt = rowsOnP.getValue().begin(); rowIt != rowsOnP.getValue().end(); ++rowIt)
    {
        const int r = static_cast<int>(rowIt.index());
        if (r >= rows) continue;
        for (auto colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
        {
            const int k = static_cast<int>(colIt.index());
            const Vec3d u = colIt.val();
            const Vec3d moment = sofa::type::cross(contactP[k] - center, u);
            const double j2[6] = { u[0], u[1], u[2], moment[0], moment[1], moment[2] };
            for (int e = 0; e < 6; ++e)
                cpuRigid[static_cast<std::size_t>(r) * 6 + e] += ((m_rigidMask >> e) & 1u) ? j2[e] : 0.0;
        }
    }
    const double cpuRowsMs = nowMs() - t0;
    const double rowsDiff = std::max(maxAbsDiff(cpuDeformable, snap.rowDeformable), maxAbsDiff(cpuRigid, snap.rowRigid));
    const double dfreeDiff = maxAbsDiff(cpuDfree, snap.dfree);

    // ---- 2. compliance: each body's SOFA linear solver, on the free motion's
    // factorization (LinearSolverConstraintCorrection::addComplianceInConstraintSpace).
    sofa::linearalgebra::SparseMatrix<SReal> J1;
    sofa::linearalgebra::SparseMatrix<SReal> J2;
    J1.resize(rows, n);
    J2.resize(rows, 6);
    for (int r = 0; r < rows; ++r)
    {
        const int k = r / rowsPerContact;
        for (int a = 0; a < 3; ++a)
        {
            const int v = snap.contactVertices[static_cast<std::size_t>(k) * 3 + a];
            if (v < 0) continue;
            for (int c = 0; c < 3; ++c)
            {
                const double value = cpuDeformable[static_cast<std::size_t>(r) * 9 + a * 3 + c];
                if (value != 0.0) J1.add(r, 3 * v + c, value);
            }
        }
        for (int e = 0; e < 6; ++e)
        {
            const double value = cpuRigid[static_cast<std::size_t>(r) * 6 + e];
            if (value != 0.0) J2.add(r, e, value);
        }
    }
    sofa::linearalgebra::LPtrFullMatrix<SReal> sofaW;
    sofaW.resize(rows, rows);
    sofaW.clear();
    t0 = nowMs();
    deformableSolver->getLinearSystem()->setSystemSolution(sofa::core::MultiVecDerivId::null());
    deformableSolver->addJMInvJt(&sofaW, &J1, m_factorDeformable);
    l_rigidLinearSolver->getLinearSystem()->setSystemSolution(sofa::core::MultiVecDerivId::null());
    l_rigidLinearSolver->addJMInvJt(&sofaW, &J2, m_factorRigid);
    const double cpuComplianceMs = nowMs() - t0;
    std::vector<double> cpuW(static_cast<std::size_t>(rows) * rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < rows; ++j) cpuW[static_cast<std::size_t>(i) * rows + j] = sofaW.element(i, j);
    const double complianceRel = maxAbsDiff(snap.compliance, cpuW) / std::max(maxAbs(cpuW), 1e-300);

    // ---- 3. multipliers: SOFA's BlockGaussSeidelConstraintSolver, on the CPU
    // problem and on the GPU's own W and violations (the solver alone).
    const ReferenceSolve cpuSolve = solveWithSofa(cpu.solver.get(), rows, rowsPerContact, snap.friction, cpuW, cpuDfree,
        d_tolerance.getValue(), d_maxIterations.getValue(), d_scaleTolerance.getValue(), d_allVerified.getValue(), d_sor.getValue());
    const ReferenceSolve sameInput = solveWithSofa(cpu.solver.get(), rows, rowsPerContact, snap.friction, snap.compliance, snap.dfree,
        d_tolerance.getValue(), d_maxIterations.getValue(), d_scaleTolerance.getValue(), d_allVerified.getValue(), d_sor.getValue());
    const double lambdaRel = maxAbsDiff(snap.lambda, cpuSolve.lambda) / std::max(maxAbs(cpuSolve.lambda), 1e-300);
    const double lambdaRelSameInput = maxAbsDiff(snap.lambda, sameInput.lambda) / std::max(maxAbs(sameInput.lambda), 1e-300);
    const double violationGpu = lowestNormalViolation(rows, rowsPerContact, snap.compliance, snap.dfree, snap.lambda);
    const double violationCpu = lowestNormalViolation(rows, rowsPerContact, cpuW, cpuDfree, cpuSolve.lambda);

    // ---- 4. correction dv = A^-1 J^T lambda, in double: body 1 by an LDL^T of
    // the same matrix the GPU factorized, body 2 by its 6x6.
    double factorizeMs = std::numeric_limits<double>::quiet_NaN();
    double correctionMs = std::numeric_limits<double>::quiet_NaN();
    double correctionDiff = std::numeric_limits<double>::quiet_NaN();
    double correctionDiffSameLambda = std::numeric_limits<double>::quiet_NaN();
    double correctionMax = std::numeric_limits<double>::quiet_NaN();
    const SReal positionFactor = deformableOde()->getPositionIntegrationFactor();
    bool deformableSolved = false;
    Eigen::VectorXd dvCpu;
    {
        const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>> rowMajor(
            n, n, static_cast<int>(m_csrValues.size()), m_csrRowPtr.data(), m_csrColumns.data(), m_csrValues.data());
        const Eigen::SparseMatrix<double> A = rowMajor;
        if (cpu.analyzedRowPtr != m_csrRowPtr || cpu.analyzedColumns != m_csrColumns)
        {
            cpu.ldlt.analyzePattern(A);
            cpu.analyzedRowPtr = m_csrRowPtr;
            cpu.analyzedColumns = m_csrColumns;
        }
        t0 = nowMs();
        cpu.ldlt.factorize(A);
        factorizeMs = nowMs() - t0;
        if (cpu.ldlt.info() == Eigen::Success)
        {
            Eigen::VectorXd rhsCpu = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd rhsGpu = Eigen::VectorXd::Zero(n);
            for (int r = 0; r < rows; ++r)
            {
                const int k = r / rowsPerContact;
                for (int a = 0; a < 3; ++a)
                {
                    const int v = snap.contactVertices[static_cast<std::size_t>(k) * 3 + a];
                    if (v < 0) continue;
                    for (int c = 0; c < 3; ++c)
                    {
                        const std::size_t e = static_cast<std::size_t>(r) * 9 + a * 3 + c;
                        rhsCpu[3 * v + c] += cpuDeformable[e] * cpuSolve.lambda[r];
                        rhsGpu[3 * v + c] += snap.rowDeformable[e] * snap.lambda[r];
                    }
                }
            }
            t0 = nowMs();
            dvCpu = cpu.ldlt.solve(rhsCpu);
            correctionMs = nowMs() - t0;
            deformableSolved = cpu.ldlt.info() == Eigen::Success;
            const Eigen::VectorXd dvSameLambda = cpu.ldlt.solve(rhsGpu);
            correctionDiff = 0.0;
            correctionDiffSameLambda = 0.0;
            correctionMax = 0.0;
            for (int i = 0; i < n; ++i)
            {
                correctionDiff = std::max(correctionDiff, std::fabs(m_deformableCorrection[i] - dvCpu[i]));
                correctionDiffSameLambda = std::max(correctionDiffSameLambda, std::fabs(m_deformableCorrection[i] - dvSameLambda[i]));
                correctionMax = std::max(correctionMax, std::fabs(dvCpu[i]));
            }
            correctionDiff *= positionFactor;
            correctionDiffSameLambda *= positionFactor;
            correctionMax *= positionFactor;
        }
    }
    double rigidImpulseCpu[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int r = 0; r < rows; ++r)
        for (int e = 0; e < 6; ++e) rigidImpulseCpu[e] += cpuRigid[static_cast<std::size_t>(r) * 6 + e] * cpuSolve.lambda[r];
    double rigidCorrectionCpu[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double rigidDiff = std::numeric_limits<double>::quiet_NaN();
    const bool rigidSolved = solveDense6(m_rigidMatrix, rigidImpulseCpu, rigidCorrectionCpu);
    if (rigidSolved)
    {
        rigidDiff = 0.0;
        for (int e = 0; e < 6; ++e) rigidDiff = std::max(rigidDiff, std::fabs(m_rigidCorrection[e] - rigidCorrectionCpu[e]));
        rigidDiff *= l_rigidOdeSolver->getPositionIntegrationFactor();
    }
    double normalImpulseCpu = 0.0;
    for (int r = 0; r < rows; r += rowsPerContact) normalImpulseCpu += cpuSolve.lambda[r];

    const bool complete = deformableSolved && rigidSolved;
    if (apply && complete)
    {
        m_appliedDeformable.assign(dvCpu.data(), dvCpu.data() + n);
        for (int e = 0; e < 6; ++e)
        {
            m_appliedRigid[e] = rigidCorrectionCpu[e];
            m_appliedImpulse[e] = rigidImpulseCpu[e];
        }
        m_appliedNormalImpulse = normalImpulseCpu;
    }
    if (!writeRow) return complete;

    m_compareStream << this->getContext()->getTime() << ',' << m_buildStats.detectedContacts << ',' << contacts << ','
                    << rows << ',' << snap.touchedVertices << ','
                    << rowsDiff << ',' << dfreeDiff << ','
                    << complianceRel << ',' << lambdaRel << ',' << lambdaRelSameInput << ','
                    << m_solveStats.iterations << ',' << cpuSolve.sweeps << ',' << sameInput.sweeps << ','
                    << violationGpu << ',' << violationCpu << ','
                    << correctionDiff << ',' << correctionDiffSameLambda << ',' << correctionMax << ',' << rigidDiff << ','
                    << m_impulse.rigid[1] / dt << ',' << rigidImpulseCpu[1] / dt << ','
                    << m_impulse.normalSum << ',' << normalImpulseCpu << ','
                    << m_timings.buildMs << ',' << m_timings.factorizeMs << ',' << m_timings.complianceMs << ','
                    << m_timings.solveMs << ',' << m_timings.correctionMs << ','
                    << cpuRowsMs << ',' << cpuComplianceMs << ',' << cpuSolve.ms << ',' << correctionMs << ','
                    << factorizeMs << ',' << (apply && complete ? "cpu" : "gpu") << '\n';
    m_compareStream.flush();
    return complete;
}

} // namespace SofaGpuCollision
