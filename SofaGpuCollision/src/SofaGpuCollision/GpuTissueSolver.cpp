#include <SofaGpuCollision/GpuTissueSolver.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseMatrixLinearSystem.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Tetrahedron.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace SofaGpuCollision
{

int GpuTissueSolverClass = sofa::core::RegisterObject(
    "Implicit Euler step of a tetrahedral viscoelastic Ogden tissue, entirely on the GPU (material, "
    "consistent mass, fixed DOFs, direct solve), following SOFA's CPU components stage for stage.")
    .add<GpuTissueSolver>();

namespace
{

using Vec3dTypes = sofa::defaulttype::Vec3dTypes;
using sofa::core::objectmodel::BaseContext;

double nowMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string joinNumbers(const std::vector<double>& values)
{
    std::ostringstream out;
    out << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? " " : "") << values[i];
    return out.str();
}

// A scalar CSR copy of a SOFA system matrix (block 3x3 or scalar CRS).
bool toCsr(sofa::linearalgebra::BaseMatrix* base, std::vector<int>& rowPtr, std::vector<int>& columns, std::vector<double>& values)
{
    using Block3 = sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Mat<3, 3, SReal>>;
    using Scalar = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    if (base == nullptr) return false;
    const int n = static_cast<int>(base->rowSize());
    rowPtr.assign(static_cast<std::size_t>(n) + 1, 0);
    if (auto* m = dynamic_cast<Block3*>(base))
    {
        m->compress();
        for (std::size_t xi = 0; xi < m->rowIndex.size(); ++xi)
        {
            const int blocks = static_cast<int>(m->rowBegin[xi + 1] - m->rowBegin[xi]);
            for (int r = 0; r < 3; ++r) rowPtr[3 * m->rowIndex[xi] + r + 1] = 3 * blocks;
        }
        for (int r = 0; r < n; ++r) rowPtr[r + 1] += rowPtr[r];
        columns.resize(static_cast<std::size_t>(rowPtr[n]));
        values.resize(static_cast<std::size_t>(rowPtr[n]));
        for (std::size_t xi = 0; xi < m->rowIndex.size(); ++xi)
        {
            for (int r = 0; r < 3; ++r)
            {
                int p = rowPtr[3 * m->rowIndex[xi] + r];
                for (auto b = m->rowBegin[xi]; b < m->rowBegin[xi + 1]; ++b)
                    for (int c = 0; c < 3; ++c)
                    {
                        columns[p] = 3 * static_cast<int>(m->colsIndex[b]) + c;
                        values[p] = m->colsValue[b][r][c];
                        ++p;
                    }
            }
        }
        return true;
    }
    if (auto* s = dynamic_cast<Scalar*>(base))
    {
        s->compress();
        for (std::size_t xi = 0; xi < s->rowIndex.size(); ++xi)
            rowPtr[s->rowIndex[xi] + 1] = static_cast<int>(s->rowBegin[xi + 1] - s->rowBegin[xi]);
        for (int r = 0; r < n; ++r) rowPtr[r + 1] += rowPtr[r];
        columns.resize(static_cast<std::size_t>(rowPtr[n]));
        values.resize(static_cast<std::size_t>(rowPtr[n]));
        for (std::size_t xi = 0; xi < s->rowIndex.size(); ++xi)
        {
            int p = rowPtr[s->rowIndex[xi]];
            for (auto b = s->rowBegin[xi]; b < s->rowBegin[xi + 1]; ++b)
            {
                columns[p] = static_cast<int>(s->colsIndex[b]);
                values[p] = s->colsValue[b];
                ++p;
            }
        }
        return true;
    }
    return false;
}

// max |A - B| over both patterns (rows merged by column).
double csrMaxDiff(const std::vector<int>& ra, const std::vector<int>& ca, const std::vector<double>& va,
                  const std::vector<int>& rb, const std::vector<int>& cb, const std::vector<double>& vb)
{
    double m = 0.0;
    const std::size_t rows = std::min(ra.size(), rb.size()) - 1;
    for (std::size_t r = 0; r < rows; ++r)
    {
        int i = ra[r], j = rb[r];
        while (i < ra[r + 1] || j < rb[r + 1])
        {
            if (j >= rb[r + 1] || (i < ra[r + 1] && ca[i] < cb[j])) { m = std::max(m, std::fabs(va[i])); ++i; }
            else if (i >= ra[r + 1] || cb[j] < ca[i]) { m = std::max(m, std::fabs(vb[j])); ++j; }
            else { m = std::max(m, std::fabs(va[i] - vb[j])); ++i; ++j; }
        }
    }
    return m;
}

double maxAbsOf(const std::vector<double>& v)
{
    double m = 0.0;
    for (const double x : v) m = std::max(m, std::fabs(x));
    return m;
}

} // namespace

// SOFA's own CPU tissue (the scene's CPU set-up), outside the scene graph, for compareWithCpu.
struct GpuTissueSolver::CpuReplica
{
    sofa::simulation::NodeSPtr node;
    sofa::core::behavior::MechanicalState<Vec3dTypes>* state { nullptr };
    sofa::core::behavior::OdeSolver* odeSolver { nullptr };
    sofa::core::behavior::LinearSolver* linearSolver { nullptr };
};

GpuTissueSolver::GpuTissueSolver()
    : d_ogdenParameters(initData(&d_ogdenParameters, "ogdenParameters",
        "SLSOgdenFirstOrder's ParameterSet: mu1 alpha1 G1 tau k0 (as TetrahedronViscoHyperelasticityFEMForceField)."))
    , d_maxwellParameters(initData(&d_maxwellParameters, "maxwellParameters",
        "MaxwellFirstOrder's ParameterSet: G1 tau lambda (as TetrahedronViscoelasticityFEMForceField). Empty: no Maxwell branch."))
    , d_ogdenEigenvectors(initData(&d_ogdenEigenvectors, std::string("sofa"), "ogdenEigenvectors",
        "sofa: C^p built as SofaViscoElastic v25.12 builds it (its Eigen call computes no eigenvectors and pairs the "
        "sorted eigenvalues with C's scaled lower triangle), so results match SOFA's CPU components. "
        "exact: the true eigenvectors (the Ogden material as written)."))
    , d_massDensity(initData(&d_massDensity, 1.0_sreal, "massDensity", "Mass density (MeshMatrixMass, not lumped)."))
    , d_rayleighStiffness(initData(&d_rayleighStiffness, 0.0_sreal, "rayleighStiffness", "Rayleigh damping, stiffness part (EulerImplicitSolver)."))
    , d_rayleighMass(initData(&d_rayleighMass, 0.0_sreal, "rayleighMass", "Rayleigh damping, mass part (EulerImplicitSolver)."))
    , d_refinementSteps(initData(&d_refinementSteps, 1, "refinementSteps",
        "Iterative refinement steps (in double) after the single-precision Cholesky solve."))
    , d_restPositions(initData(&d_restPositions, "restPositions",
        "Rest positions in double precision (the state keeps them in single). Empty: the state's rest positions."))
    , d_monitorVertex(initData(&d_monitorVertex, -1, "monitorVertex",
        "Vertex whose position is reported in monitorPosition, with minVolumeRatio (-1: off)."))
    , d_measureTimes(initData(&d_measureTimes, false, "measureTimes", "Time each GPU stage with CUDA events."))
    , d_compareWithCpu(initData(&d_compareWithCpu, false, "compareWithCpu",
        "Run SOFA's CPU components (EulerImplicitSolver, SparseLDLSolver, SofaViscoElastic, MeshMatrixMass, "
        "FixedProjectiveConstraint) on the same state every step and write the differences to compareFile. Slow."))
    , d_compareEvery(initData(&d_compareEvery, 1, "compareEvery",
        "With compareWithCpu: write a comparison every N-th step (the CPU replica still runs every step)."))
    , d_compareFile(initData(&d_compareFile, std::string("gpu_tissue_compare.csv"), "compareFile", "CSV file for compareWithCpu."))
    , d_minVolumeRatio(initData(&d_minVolumeRatio, 1.0_sreal, "minVolumeRatio",
        "OUTPUT: smallest det(F) (volume / rest volume) over the tetrahedra, at the end of the step."))
    , d_monitorPosition(initData(&d_monitorPosition, "monitorPosition", "OUTPUT: monitorVertex's position at the end of the step."))
    , d_stepGpuMilliseconds(initData(&d_stepGpuMilliseconds, 0.0_sreal, "stepGpuMilliseconds",
        "OUTPUT: GPU time of the step (with measureTimes)."))
    , d_stageMilliseconds(initData(&d_stageMilliseconds, "stageMilliseconds",
        "OUTPUT: GPU time per stage: material, assembly, factorisation, solve (with measureTimes)."))
    , l_topology(initLink("topology", "The tetrahedral topology (default: the node's)."))
{
    d_minVolumeRatio.setReadOnly(true);
    d_monitorPosition.setReadOnly(true);
    d_stepGpuMilliseconds.setReadOnly(true);
    d_stageMilliseconds.setReadOnly(true);
}

GpuTissueSolver::~GpuTissueSolver()
{
    backend::destroyTissueWorkspace(m_workspace);
    m_workspace = nullptr;
}

void GpuTissueSolver::reportFailure(const std::string& stage, const std::string& diagnostic)
{
    if (std::find(m_warned.begin(), m_warned.end(), stage) != m_warned.end()) return;
    m_warned.push_back(stage);
    msg_error() << "GPU tissue step: " << stage << " failed: " << diagnostic
                << " -- the tissue keeps its state. Further failures at this stage are not reported.";
}

void GpuTissueSolver::init()
{
    Inherit1::init();
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    m_state = dynamic_cast<sofa::core::behavior::MechanicalState<StateTypes>*>(this->getContext()->getMechanicalState());
    if (m_state == nullptr)
    {
        msg_error() << "GpuTissueSolver needs a CudaVec3f MechanicalObject in its node.";
        return;
    }
    m_topology = l_topology ? l_topology.get() : this->getContext()->getMeshTopology();
    if (m_topology == nullptr || m_topology->getNbTetrahedra() == 0)
    {
        msg_error() << "GpuTissueSolver needs a tetrahedral topology in its node.";
        return;
    }
    const auto& ogden = d_ogdenParameters.getValue();
    const auto& maxwell = d_maxwellParameters.getValue();
    if (ogden.size() != 5 || (!maxwell.empty() && maxwell.size() != 3))
    {
        msg_error() << "ogdenParameters needs 5 values (mu1 alpha1 G1 tau k0), maxwellParameters 3 (G1 tau lambda) or none.";
        return;
    }
    if (d_ogdenEigenvectors.getValue() != "sofa" && d_ogdenEigenvectors.getValue() != "exact")
    {
        msg_error() << "ogdenEigenvectors must be 'sofa' or 'exact'.";
        return;
    }
    const std::size_t n = m_state->getSize();
    const auto& rest = d_restPositions.getValue();
    m_restPositions.resize(3 * n);
    if (!rest.empty())
    {
        if (rest.size() != n)
        {
            msg_error() << "restPositions has " << rest.size() << " points, the state " << n << ".";
            return;
        }
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) m_restPositions[3 * i + c] = rest[i][c];
    }
    else
    {
        const auto& r = m_state->read(sofa::core::vec_id::read_access::restPosition)->getValue();
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) m_restPositions[3 * i + c] = r[i][c];
    }
    if (d_compareWithCpu.getValue())
    {
        m_compareStream.open(d_compareFile.getValue());
        if (!m_compareStream) msg_warning() << "Could not open " << d_compareFile.getValue() << "; compareWithCpu disabled.";
        else
            m_compareStream << "time,force_rel_diff,matrix_rel_diff,dv_rel_diff,dv_max_abs_diff,dv_max,"
                               "xfree_max_abs_diff_m,vfree_max_abs_diff,gpu_step_ms,gpu_material_ms,gpu_assembly_ms,"
                               "gpu_factorize_ms,gpu_solve_ms,cpu_free_motion_ms\n";
    }
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

bool GpuTissueSolver::computeFixedDofs(std::string& diagnostic)
{
    // Which DOFs the node's projective constraints hold: project a vector of ones
    // (once; on the GPU for CudaFixedProjectiveConstraint, read back here).
    const std::size_t n = m_state->getSize();
    m_fixedDofs.assign(n, 0);
    sofa::core::objectmodel::Data<StateTypes::VecDeriv> ones;
    {
        auto& vec = *ones.beginEdit();
        vec.resize(n);
        for (std::size_t i = 0; i < n; ++i) vec[i] = StateTypes::Deriv(1.0f, 1.0f, 1.0f);
        ones.endEdit();
    }
    bool general = false;
    const sofa::core::MechanicalParams* mparams = sofa::core::mechanicalparams::defaultInstance();
    for (auto* constraint : this->getContext()->getObjects<sofa::core::behavior::ProjectiveConstraintSet<StateTypes>>(BaseContext::Local))
    {
        if (constraint->getMState() != m_state || !constraint->isActive()) continue;
        constraint->projectResponse(mparams, ones);
    }
    const auto& projected = ones.getValue();
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
        {
            if (projected[i][c] == 0.0f) m_fixedDofs[i] |= static_cast<unsigned char>(1u << c);
            else if (projected[i][c] != 1.0f) general = true;
        }
    if (general)
    {
        diagnostic = "a projective constraint in the node is not a fixed-DOF constraint; only fixed DOFs are supported";
        return false;
    }
    diagnostic.clear();
    return true;
}

bool GpuTissueSolver::createWorkspace(std::string& diagnostic)
{
    backend::TissueSetup setup;
    const std::size_t n = m_state->getSize();
    setup.vertexCount = static_cast<int>(n);
    const auto& tets = m_topology->getTetrahedra();
    const auto& edges = m_topology->getEdges();
    setup.tetrahedra.resize(4 * tets.size());
    setup.tetrahedronEdges.resize(6 * tets.size());
    setup.tetrahedronEdgeSides.resize(6 * tets.size());
    setup.edges.resize(2 * edges.size());
    for (std::size_t e = 0; e < edges.size(); ++e)
    {
        setup.edges[2 * e] = static_cast<int>(edges[e][0]);
        setup.edges[2 * e + 1] = static_cast<int>(edges[e][1]);
    }
    setup.restPositions = m_restPositions;
    setup.vertexMass.assign(n, 0.0);
    setup.edgeMass.assign(edges.size(), 0.0);
    const double density = d_massDensity.getValue();
    for (std::size_t t = 0; t < tets.size(); ++t)
    {
        const auto& ta = tets[t];
        const auto te = m_topology->getEdgesInTetrahedron(static_cast<sofa::Index>(t));
        for (int j = 0; j < 4; ++j) setup.tetrahedra[4 * t + j] = static_cast<int>(ta[j]);
        for (int j = 0; j < 6; ++j)
        {
            // updateTangentMatrix: k is the local vertex at the global edge's first end.
            const auto local = m_topology->getLocalEdgesInTetrahedron(static_cast<sofa::Index>(j));
            int k = static_cast<int>(local[0]);
            int l = static_cast<int>(local[1]);
            if (edges[te[j]][0] != ta[k]) std::swap(k, l);
            setup.tetrahedronEdges[6 * t + j] = static_cast<int>(te[j]);
            setup.tetrahedronEdgeSides[6 * t + j] = static_cast<unsigned char>(k | (l << 2));
        }
        // MeshMatrixMass (tetrahedra, not lumped): rho V / 10 per vertex, rho V / 20 per edge.
        sofa::type::Vec3d p[4];
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c) p[j][c] = m_restPositions[3 * ta[j] + c];
        const double volume = sofa::geometry::Tetrahedron::volume(p[0], p[1], p[2], p[3]);
        for (int j = 0; j < 4; ++j) setup.vertexMass[ta[j]] += density * volume / 10.0;
        for (int j = 0; j < 6; ++j) setup.edgeMass[te[j]] += density * volume / 20.0;
    }
    setup.gravityMass.resize(n);
    for (std::size_t i = 0; i < n; ++i) setup.gravityMass[i] = setup.vertexMass[i] * 2.5;   // MeshMatrixMass's lumping coefficient (tetrahedra)
    setup.fixedDofs = m_fixedDofs;
    const auto& ogden = d_ogdenParameters.getValue();
    const auto& maxwell = d_maxwellParameters.getValue();
    setup.material.hasOgden = true;
    setup.material.ogdenMu1 = ogden[0];
    setup.material.ogdenAlpha1 = ogden[1];
    setup.material.ogdenG1 = ogden[2];
    setup.material.ogdenTau = ogden[3];
    setup.material.ogdenK0 = ogden[4];
    setup.material.ogdenSofaEigenvectors = d_ogdenEigenvectors.getValue() == "sofa";
    setup.material.hasMaxwell = maxwell.size() == 3;
    if (setup.material.hasMaxwell)
    {
        setup.material.maxwellG1 = maxwell[0];
        setup.material.maxwellTau = maxwell[1];
        setup.material.maxwellLambda = maxwell[2];
    }
    const auto g = this->getContext()->getGravity();
    for (int c = 0; c < 3; ++c) setup.gravity[c] = g[c];
    m_workspace = backend::createTissueWorkspace(setup, diagnostic);
    return m_workspace != nullptr;
}

bool GpuTissueSolver::createReplica(std::string& diagnostic)
{
    m_cpu = std::make_unique<CpuReplica>();
    m_cpu->node = sofa::simulation::getSimulation()->createNewNode("GpuTissueSolverCpuReplica");
    auto* node = m_cpu->node.get();
    node->setDt(this->getContext()->getDt());
    node->setGravity(this->getContext()->getGravity());

    auto create = [node](const char* name, const char* className,
                         const std::vector<std::pair<std::string, std::string>>& attributes) {
        sofa::core::objectmodel::BaseObjectDescription desc(name, className);
        for (const auto& a : attributes) desc.setAttribute(a.first, a.second);
        return sofa::core::ObjectFactory::CreateObject(node, &desc);
    };
    std::vector<double> tetValues;
    for (const auto& t : m_topology->getTetrahedra())
        for (int j = 0; j < 4; ++j) tetValues.push_back(t[j]);
    std::ostringstream tetrahedra;
    for (std::size_t i = 0; i < tetValues.size(); ++i) tetrahedra << (i ? " " : "") << static_cast<long>(tetValues[i]);
    std::ostringstream fixed;
    int partial = 0;
    for (std::size_t i = 0; i < m_fixedDofs.size(); ++i)
    {
        if (m_fixedDofs[i] == 7) fixed << (fixed.tellp() > 0 ? " " : "") << i;
        else if (m_fixedDofs[i] != 0) ++partial;
    }
    if (partial > 0)
    {
        diagnostic = "the CPU replica supports only fully fixed vertices";
        return false;
    }
    const auto& ogden = d_ogdenParameters.getValue();
    const auto& maxwell = d_maxwellParameters.getValue();
    std::ostringstream ogdenSet, maxwellSet;
    ogdenSet << std::setprecision(17) << ogden[0] << ' ' << ogden[1] << ' ' << ogden[2] << ' ' << ogden[3] << ' ' << ogden[4];
    if (maxwell.size() == 3) maxwellSet << std::setprecision(17) << maxwell[0] << ' ' << maxwell[1] << ' ' << maxwell[2];

    create("odeSolver", "EulerImplicitSolver", { { "rayleighStiffness", std::to_string(d_rayleighStiffness.getValue()) },
                                                  { "rayleighMass", std::to_string(d_rayleighMass.getValue()) } });
    create("linearSolver", "SparseLDLSolver", { { "template", "CompressedRowSparseMatrixMat3x3d" } });
    create("dofs", "MechanicalObject", { { "template", "Vec3d" }, { "position", joinNumbers(m_restPositions) } });
    create("topology", "TetrahedronSetTopologyContainer", { { "tetrahedra", tetrahedra.str() } });
    create("mass", "MeshMatrixMass", { { "massDensity", std::to_string(d_massDensity.getValue()) } });
    create("elastic", "TetrahedronViscoHyperelasticityFEMForceField",
           { { "template", "Vec3d" }, { "materialName", "SLSOgdenFirstOrder" }, { "ParameterSet", ogdenSet.str() } });
    if (maxwell.size() == 3)
        create("viscous", "TetrahedronViscoelasticityFEMForceField",
               { { "template", "Vec3d" }, { "materialName", "MaxwellFirstOrder" }, { "ParameterSet", maxwellSet.str() } });
    if (fixed.tellp() > 0) create("fixed", "FixedProjectiveConstraint", { { "indices", fixed.str() } });
    sofa::simulation::node::init(node);

    m_cpu->state = dynamic_cast<sofa::core::behavior::MechanicalState<Vec3dTypes>*>(node->getMechanicalState());
    m_cpu->odeSolver = node->get<sofa::core::behavior::OdeSolver>();
    m_cpu->linearSolver = node->get<sofa::core::behavior::LinearSolver>();
    if (m_cpu->state == nullptr || m_cpu->odeSolver == nullptr || m_cpu->linearSolver == nullptr)
    {
        diagnostic = "the CPU replica could not be built (are SofaViscoElastic and SOFA's direct solvers loaded?)";
        return false;
    }
    diagnostic.clear();
    return true;
}

void GpuTissueSolver::solve(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult,
                            sofa::core::MultiVecDerivId vResult)
{
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid || m_failed) return;
    std::string diagnostic;
    if (m_workspace == nullptr)
    {
        if (!computeFixedDofs(diagnostic) || !createWorkspace(diagnostic))
        {
            msg_error() << "GPU tissue set-up failed: " << diagnostic;
            m_failed = true;
            return;
        }
        if (m_compareStream.is_open() && !createReplica(diagnostic))
        {
            msg_warning() << "compareWithCpu disabled: " << diagnostic;
            m_compareStream.close();
            m_cpu.reset();
        }
    }

    auto* state = m_state;
    const void* x = state->read(sofa::core::vec_id::read_access::position)->getValue().deviceRead();
    const void* v = state->read(sofa::core::vec_id::read_access::velocity)->getValue().deviceRead();
    auto* xFreeData = state->write(xResult.getId(state));
    auto* vFreeData = state->write(vResult.getId(state));
    auto& xFree = *xFreeData->beginEdit();
    auto& vFree = *vFreeData->beginEdit();
    if (xFree.size() != state->getSize()) xFree.resize(state->getSize());
    if (vFree.size() != state->getSize()) vFree.resize(state->getSize());
    void* xf = xFree.deviceWrite();
    void* vf = vFree.deviceWrite();

    backend::TissueStepConfig config;
    config.dt = dt;
    config.rayleighMass = d_rayleighMass.getValue();
    config.rayleighStiffness = d_rayleighStiffness.getValue();
    config.refinementSteps = std::max(0, d_refinementSteps.getValue());
    backend::TissueTimings timings;
    const bool timing = d_measureTimes.getValue() || m_compareStream.is_open();
    const bool ok = backend::tissueFreeMotion(m_workspace, x, v, xf, vf, config, timing ? &timings : nullptr, diagnostic);
    xFreeData->endEdit();
    vFreeData->endEdit();
    if (!ok)
    {
        reportFailure("free motion", diagnostic);
        return;
    }
    const double gpuMs = timings.materialMs + timings.assembleMs + timings.factorizeMs + timings.solveMs;
    d_stepGpuMilliseconds.setValue(gpuMs);
    d_stageMilliseconds.setValue({ timings.materialMs, timings.assembleMs, timings.factorizeMs, timings.solveMs });
    ++m_steps;
    if (m_cpu) compareWithCpu(params, dt, xResult, vResult, gpuMs);
}

void GpuTissueSolver::compareWithCpu(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult,
                                     sofa::core::MultiVecDerivId vResult, const double gpuMs)
{
    // The replica starts from the GPU's state at the step's start (read back here)
    // and runs SOFA's own free motion every step, which also keeps its viscous
    // strains in step with the GPU's.
    const std::size_t n = m_state->getSize();
    const auto& xGpu = m_state->read(sofa::core::vec_id::read_access::position)->getValue();
    const auto& vGpu = m_state->read(sofa::core::vec_id::read_access::velocity)->getValue();
    std::vector<double> v0(3 * n);
    {
        auto xr = sofa::helper::getWriteOnlyAccessor(*m_cpu->state->write(sofa::core::vec_id::write_access::position));
        auto vr = sofa::helper::getWriteOnlyAccessor(*m_cpu->state->write(sofa::core::vec_id::write_access::velocity));
        xr.resize(n);
        vr.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c)
            {
                xr[i][c] = xGpu[i][c];
                vr[i][c] = vGpu[i][c];
                v0[3 * i + c] = vGpu[i][c];
            }
    }
    const double t0 = nowMs();
    m_cpu->odeSolver->solve(params, dt);   // in place: position -> x_free, velocity -> v_free
    const double cpuMs = nowMs() - t0;

    const int every = std::max(1, d_compareEvery.getValue());
    if ((m_steps - 1) % every != 0) return;

    std::string diagnostic;
    backend::TissueStepSnapshot snap;
    if (!backend::downloadTissueStep(m_workspace, true, snap, diagnostic)) return;

    const auto& fCpu = m_cpu->state->read(sofa::core::vec_id::read_access::force)->getValue();
    const auto& xFreeCpu = m_cpu->state->read(sofa::core::vec_id::read_access::position)->getValue();
    const auto& vFreeCpu = m_cpu->state->read(sofa::core::vec_id::read_access::velocity)->getValue();
    const auto& xFreeGpu = m_state->read(xResult.getId(m_state))->getValue();
    const auto& vFreeGpu = m_state->read(vResult.getId(m_state))->getValue();

    double forceDiff = 0.0, forceMax = 0.0, dvDiff = 0.0, dvMax = 0.0, xDiff = 0.0, vDiff = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            const std::size_t k = 3 * i + c;
            forceDiff = std::max(forceDiff, std::fabs(snap.force[k] - fCpu[i][c]));
            forceMax = std::max(forceMax, std::fabs(fCpu[i][c]));
            const double dvCpu = vFreeCpu[i][c] - v0[k];
            dvDiff = std::max(dvDiff, std::fabs(snap.dv[k] - dvCpu));
            dvMax = std::max(dvMax, std::fabs(dvCpu));
            xDiff = std::max(xDiff, std::fabs(static_cast<double>(xFreeGpu[i][c]) - xFreeCpu[i][c]));
            vDiff = std::max(vDiff, std::fabs(static_cast<double>(vFreeGpu[i][c]) - vFreeCpu[i][c]));
        }
    }
    double matrixRel = std::numeric_limits<double>::quiet_NaN();
    {
        std::vector<int> rowPtr, columns;
        std::vector<double> values;
        auto* system = m_cpu->linearSolver->getLinearSystem();
        if (system != nullptr && toCsr(system->getSystemBaseMatrix(), rowPtr, columns, values) && rowPtr.size() == snap.rowPtr.size())
            matrixRel = csrMaxDiff(snap.rowPtr, snap.columns, snap.values, rowPtr, columns, values) / std::max(maxAbsOf(values), 1e-300);
    }
    backend::TissueTimings timings;
    const auto& stages = d_stageMilliseconds.getValue();
    if (stages.size() == 4)
    {
        timings.materialMs = stages[0];
        timings.assembleMs = stages[1];
        timings.factorizeMs = stages[2];
        timings.solveMs = stages[3];
    }
    m_compareStream << this->getContext()->getTime() << ',' << forceDiff / std::max(forceMax, 1e-300) << ',' << matrixRel << ','
                    << dvDiff / std::max(dvMax, 1e-300) << ',' << dvDiff << ',' << dvMax << ',' << xDiff << ',' << vDiff << ','
                    << gpuMs << ',' << timings.materialMs << ',' << timings.assembleMs << ',' << timings.factorizeMs << ','
                    << timings.solveMs << ',' << cpuMs << '\n';
    m_compareStream.flush();
}

const float* GpuTissueSolver::deviceFactor(int& size) const
{
    return backend::tissueFactor(m_workspace, size);
}

void GpuTissueSolver::updateMonitor(const void* xDevice)
{
    const int vertex = d_monitorVertex.getValue();
    if (m_workspace == nullptr || vertex < 0 || xDevice == nullptr) return;
    double minVolume = 0.0;
    double position[3] = { 0.0, 0.0, 0.0 };
    std::string diagnostic;
    if (backend::tissueMonitor(m_workspace, xDevice, vertex, minVolume, position, diagnostic))
    {
        d_minVolumeRatio.setValue(minVolume);
        d_monitorPosition.setValue(sofa::type::Vec3d(position[0], position[1], position[2]));
    }
}

sofa::core::behavior::LinearSolver* GpuTissueSolver::cpuReplicaLinearSolver() const
{
    return m_cpu ? m_cpu->linearSolver : nullptr;
}

SReal GpuTissueSolver::getIntegrationFactor(int inputDerivative, int outputDerivative) const
{
    const SReal dt = this->getContext()->getDt();
    const SReal matrix[3][3] = { { 1, dt, 0 }, { 0, 1, 0 }, { 0, 0, 0 } };
    if (inputDerivative >= 3 || outputDerivative >= 3) return 0;
    return matrix[outputDerivative][inputDerivative];
}

SReal GpuTissueSolver::getSolutionIntegrationFactor(int outputDerivative) const
{
    const SReal dt = this->getContext()->getDt();
    const SReal vect[3] = { dt, 1, 1 / dt };
    if (outputDerivative >= 3) return 0;
    return vect[outputDerivative];
}

} // namespace SofaGpuCollision
