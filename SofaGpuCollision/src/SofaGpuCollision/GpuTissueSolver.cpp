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
#include <map>
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

// An edge's key, whichever way round its ends are given.
std::uint64_t edgeKey(int a, int b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
}

// A tetrahedron's key: its vertices, sorted.
std::array<int, 4> tetrahedronKey(std::array<int, 4> v)
{
    std::sort(v.begin(), v.end());
    return v;
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

// A SOFA core material name -> backend::TissueCoreMaterial value (0: none, -1: unknown)
// and its parameter count.
int parseCoreMaterial(const std::string& name, std::size_t& count)
{
    struct Entry { const char* name; backend::TissueCoreMaterial id; std::size_t count; };
    static const Entry entries[] = {
        { "Ogden", backend::TissueCoreMaterial::Ogden, 3 },
        { "NeoHookean", backend::TissueCoreMaterial::NeoHookean, 2 },
        { "StableNeoHookean", backend::TissueCoreMaterial::StableNeoHookean, 2 },
        { "StVenantKirchhoff", backend::TissueCoreMaterial::StVenantKirchhoff, 2 },
        { "MooneyRivlin", backend::TissueCoreMaterial::MooneyRivlin, 3 },
    };
    count = 0;
    if (name.empty()) return 0;
    for (const auto& e : entries)
    {
        if (name == e.name)
        {
            count = e.count;
            return static_cast<int>(e.id);
        }
    }
    return -1;
}

} // namespace

// SOFA's own CPU tissue (the scene's CPU set-up), outside the scene graph, for compareWithCpu.
struct GpuTissueSolver::CpuReplica
{
    sofa::simulation::NodeSPtr node;
    sofa::core::behavior::MechanicalState<Vec3dTypes>* state { nullptr };
    sofa::core::behavior::OdeSolver* odeSolver { nullptr };
    sofa::core::behavior::LinearSolver* linearSolver { nullptr };
    sofa::core::objectmodel::BaseObject* load { nullptr };   // ConstantForceField: the GPU's external forces
};

GpuTissueSolver::GpuTissueSolver()
    : d_hyperelasticMaterial(initData(&d_hyperelasticMaterial, std::string(), "hyperelasticMaterial",
        "A SOFA core hyperelastic material, as TetrahedronHyperelasticityFEMForceField's materialName: Ogden, "
        "NeoHookean, StableNeoHookean, StVenantKirchhoff or MooneyRivlin (stress and stiffness as SOFA computes "
        "them). Empty: none."))
    , d_hyperelasticParameters(initData(&d_hyperelasticParameters, "hyperelasticParameters",
        "Its ParameterSet: Ogden mu1 alpha1 k0; NeoHookean, StableNeoHookean, StVenantKirchhoff mu lambda; "
        "MooneyRivlin c1 c2 k0."))
    , d_ogdenParameters(initData(&d_ogdenParameters, "ogdenParameters",
        "SofaViscoElastic's SLSOgdenFirstOrder ParameterSet: mu1 alpha1 G1 tau k0 (as "
        "TetrahedronViscoHyperelasticityFEMForceField). Empty: none."))
    , d_maxwellParameters(initData(&d_maxwellParameters, "maxwellParameters",
        "MaxwellFirstOrder's ParameterSet: G1 tau lambda (as TetrahedronViscoelasticityFEMForceField). Empty: no Maxwell branch."))
    , d_ogdenEigenvectors(initData(&d_ogdenEigenvectors, std::string("sofa"), "ogdenEigenvectors",
        "sofa: C^p built as SofaViscoElastic v25.12 builds it (its Eigen call computes no eigenvectors and pairs the "
        "sorted eigenvalues with C's scaled lower triangle), so results match SOFA's CPU components. "
        "exact: the true eigenvectors (the Ogden material as written)."))
    , d_ogdenTangent(initData(&d_ogdenTangent, std::string("robust"), "ogdenTangent",
        "hyperelasticMaterial Ogden's stiffness. robust: the divided differences between principal stretches "
        "computed without cancellation. sofa: as SOFA v25.12 computes them (a plain quotient, garbage when two "
        "stretches differ only by rounding, as in uniaxial states; SOFA's own run then fails). The two agree "
        "whenever the stretches are distinct."))
    , d_partialFixedIndices(initData(&d_partialFixedIndices, "partialFixedIndices",
        "Vertices with some directions fixed (as PartialFixedProjectiveConstraint, which has no GPU version)."))
    , d_partialFixedMasks(initData(&d_partialFixedMasks, "partialFixedMasks",
        "For each of partialFixedIndices, the fixed directions: 1 = x, 2 = y, 4 = z (sums for several)."))
    , d_massDensity(initData(&d_massDensity, 1.0_sreal, "massDensity", "Mass density (MeshMatrixMass, not lumped)."))
    , d_rayleighStiffness(initData(&d_rayleighStiffness, 0.0_sreal, "rayleighStiffness", "Rayleigh damping, stiffness part (EulerImplicitSolver)."))
    , d_rayleighMass(initData(&d_rayleighMass, 0.0_sreal, "rayleighMass", "Rayleigh damping, mass part (EulerImplicitSolver)."))
    , d_refinementSteps(initData(&d_refinementSteps, 1, "refinementSteps",
        "Iterative refinement steps (in double) after the single-precision solve."))
    , d_factorization(initData(&d_factorization, std::string("auto"), "factorization",
        "How A is factorised each step. band: the vertices renumbered by reverse Cuthill-McKee so A is a band "
        "matrix, then a blocked Cholesky inside the band (work about n b^2 instead of n^3/3). dense: cuSOLVER's "
        "Cholesky of the whole matrix. auto: band when the band is under a third of the matrix."))
    , d_bandPanel(initData(&d_bandPanel, 128, "bandPanel",
        "Band factorisation: its blocks are the bandwidth rounded up to a multiple of this."))
    , d_restPositions(initData(&d_restPositions, "restPositions",
        "Rest positions in double precision (the state keeps them in single). Empty: the state's rest positions."))
    , d_monitorVertex(initData(&d_monitorVertex, -1, "monitorVertex",
        "Vertex whose position is reported in monitorPosition, with minVolumeRatio (-1: off)."))
    , d_measureTimes(initData(&d_measureTimes, false, "measureTimes", "Time each GPU stage with CUDA events."))
    , d_compareWithCpu(initData(&d_compareWithCpu, false, "compareWithCpu",
        "Run SOFA's CPU components (EulerImplicitSolver, SparseLDLSolver, the material's force fields, "
        "MeshMatrixMass, FixedProjectiveConstraint) on the same state every step and write the differences to "
        "compareFile. Slow."))
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
    , d_luFallbackSteps(initData(&d_luFallbackSteps, 0, "luFallbackSteps",
        "OUTPUT: steps whose system matrix was not positive definite (a strongly compressed state), "
        "solved with LU instead of Cholesky, as SOFA's LDL solver would go on."))
    , d_bandwidth(initData(&d_bandwidth, 0, "bandwidth",
        "OUTPUT: the band factorisation's half-bandwidth in DOFs (0: dense factorisation)."))
    , l_topology(initLink("topology", "The tetrahedral topology (default: the node's)."))
{
    d_minVolumeRatio.setReadOnly(true);
    d_monitorPosition.setReadOnly(true);
    d_stepGpuMilliseconds.setReadOnly(true);
    d_stageMilliseconds.setReadOnly(true);
    d_luFallbackSteps.setReadOnly(true);
    d_bandwidth.setReadOnly(true);
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
    if ((!ogden.empty() && ogden.size() != 5) || (!maxwell.empty() && maxwell.size() != 3))
    {
        msg_error() << "ogdenParameters needs 5 values (mu1 alpha1 G1 tau k0) or none, maxwellParameters 3 (G1 tau lambda) or none.";
        return;
    }
    if (d_ogdenEigenvectors.getValue() != "sofa" && d_ogdenEigenvectors.getValue() != "exact")
    {
        msg_error() << "ogdenEigenvectors must be 'sofa' or 'exact'.";
        return;
    }
    if (d_ogdenTangent.getValue() != "robust" && d_ogdenTangent.getValue() != "sofa")
    {
        msg_error() << "ogdenTangent must be 'robust' or 'sofa'.";
        return;
    }
    if (d_factorization.getValue() != "auto" && d_factorization.getValue() != "band" && d_factorization.getValue() != "dense")
    {
        msg_error() << "factorization must be 'auto', 'band' or 'dense'.";
        return;
    }
    std::size_t coreCount = 0;
    m_core = parseCoreMaterial(d_hyperelasticMaterial.getValue(), coreCount);
    if (m_core < 0)
    {
        msg_error() << "hyperelasticMaterial '" << d_hyperelasticMaterial.getValue()
                    << "' is not one of Ogden, NeoHookean, StableNeoHookean, StVenantKirchhoff, MooneyRivlin.";
        return;
    }
    if (m_core > 0 && d_hyperelasticParameters.getValue().size() != coreCount)
    {
        msg_error() << "hyperelasticParameters needs " << coreCount << " values for " << d_hyperelasticMaterial.getValue() << ".";
        return;
    }
    if (m_core == 0 && ogden.empty())
    {
        msg_error() << "No elastic material: set hyperelasticMaterial (a SOFA core material) or ogdenParameters.";
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
    // Per-direction fixed DOFs (SofaCUDA has no GPU PartialFixedProjectiveConstraint).
    const auto& partialIndices = d_partialFixedIndices.getValue();
    const auto& partialMasks = d_partialFixedMasks.getValue();
    if (partialIndices.size() != partialMasks.size())
    {
        diagnostic = "partialFixedIndices and partialFixedMasks must have the same length";
        return false;
    }
    for (std::size_t k = 0; k < partialIndices.size(); ++k)
    {
        const int i = partialIndices[k];
        if (i < 0 || static_cast<std::size_t>(i) >= n || partialMasks[k] < 0 || partialMasks[k] > 7)
        {
            diagnostic = "partialFixedIndices/partialFixedMasks: index " + std::to_string(i) + " or mask out of range";
            return false;
        }
        m_fixedDofs[i] |= static_cast<unsigned char>(partialMasks[k]);
    }
    diagnostic.clear();
    return true;
}

bool GpuTissueSolver::collectLoads(std::string& diagnostic)
{
    // The loads on the tissue: the node's ConstantForceFields. This solver computes
    // the material, the mass and gravity itself and would silently ignore any other
    // force field or mass in the node, so those are refused.
    m_loads.clear();
    for (auto* forceField : this->getContext()->getObjects<sofa::core::behavior::BaseForceField>(BaseContext::Local))
    {
        if (forceField->getClassName() != "ConstantForceField")
        {
            diagnostic = forceField->getClassName() + " '" + forceField->getName() + "' is in the tissue's node: "
                         "GpuTissueSolver computes the material, the mass (massDensity) and gravity itself and "
                         "takes only ConstantForceField loads";
            return false;
        }
        auto* load = dynamic_cast<sofa::core::behavior::ForceField<StateTypes>*>(forceField);
        if (load == nullptr || load->getMState() != m_state)
        {
            diagnostic = "ConstantForceField '" + forceField->getName() + "' must be a CudaVec3f one on the tissue's state";
            return false;
        }
        m_loads.push_back(load);
    }
    diagnostic.clear();
    return true;
}

bool GpuTissueSolver::updateExternalForces(std::string& diagnostic)
{
    // SOFA's own ConstantForceField::addForce gives the nodal forces (indices,
    // forces, totalForce, indexFromEnd all as SOFA reads them); evaluated again only
    // when one of the loads' data changed (a controller may change a load).
    diagnostic.clear();
    if (m_loads.empty()) return true;
    std::size_t signature = 17;
    for (auto* load : m_loads)
        for (const auto* data : load->getDataFields()) signature = signature * 1000003u + static_cast<std::size_t>(data->getCounter());
    if (m_loadsApplied && signature == m_loadSignature) return true;

    const std::size_t n = m_state->getSize();
    sofa::core::objectmodel::Data<StateTypes::VecDeriv> f;
    {
        auto& vec = *f.beginEdit();
        vec.resize(n);
        for (std::size_t i = 0; i < n; ++i) vec[i] = StateTypes::Deriv(0.0f, 0.0f, 0.0f);
        f.endEdit();
    }
    const sofa::core::MechanicalParams* mparams = sofa::core::mechanicalparams::defaultInstance();
    for (auto* load : m_loads)
        load->addForce(mparams, f, *m_state->read(sofa::core::vec_id::read_access::position),
                       *m_state->read(sofa::core::vec_id::read_access::velocity));
    std::vector<double> forces(3 * n, 0.0);
    bool any = false;
    const auto& values = f.getValue();
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
        {
            forces[3 * i + c] = values[i][c];
            any = any || values[i][c] != 0.0f;
        }
    if (!backend::setTissueExternalForces(m_workspace, any ? forces : std::vector<double>(), diagnostic)) return false;
    if (m_cpu && m_cpu->load)
    {
        // The replica's ConstantForceField (every vertex) gets the same nodal forces.
        auto* data = m_cpu->load->findData("forces");
        if (data == nullptr || !data->read(joinNumbers(forces)))
        {
            diagnostic = "the CPU replica's ConstantForceField could not be updated";
            return false;
        }
    }
    m_loadSignature = signature;
    m_loadsApplied = true;
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
    m_edgeIndex.clear();
    for (std::size_t e = 0; e < edges.size(); ++e)
    {
        setup.edges[2 * e] = static_cast<int>(edges[e][0]);
        setup.edges[2 * e + 1] = static_cast<int>(edges[e][1]);
        m_edgeIndex[edgeKey(setup.edges[2 * e], setup.edges[2 * e + 1])] = static_cast<int>(e);
    }
    m_edges = setup.edges;
    m_tetrahedra.resize(tets.size());
    for (std::size_t t = 0; t < tets.size(); ++t)
        for (int j = 0; j < 4; ++j) m_tetrahedra[t][j] = static_cast<int>(tets[t][j]);
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
    setup.material.core = static_cast<backend::TissueCoreMaterial>(m_core);
    setup.material.ogdenRobustTangent = d_ogdenTangent.getValue() == "robust";
    const auto& coreParameters = d_hyperelasticParameters.getValue();
    for (std::size_t k = 0; k < coreParameters.size() && k < 4; ++k) setup.material.coreParameters[k] = coreParameters[k];
    setup.material.hasOgden = ogden.size() == 5;
    if (setup.material.hasOgden)
    {
        setup.material.ogdenMu1 = ogden[0];
        setup.material.ogdenAlpha1 = ogden[1];
        setup.material.ogdenG1 = ogden[2];
        setup.material.ogdenTau = ogden[3];
        setup.material.ogdenK0 = ogden[4];
    }
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
    const std::string& factorization = d_factorization.getValue();
    setup.factorization = factorization == "band" ? backend::TissueFactorization::Band
                        : factorization == "dense" ? backend::TissueFactorization::Dense
                                                   : backend::TissueFactorization::Automatic;
    setup.bandPanel = d_bandPanel.getValue();
    m_workspace = backend::createTissueWorkspace(setup, diagnostic);
    d_bandwidth.setValue(backend::tissueBandwidth(m_workspace));
    if (m_workspace != nullptr)
        msg_info() << "factorisation: "
                   << (d_bandwidth.getValue() > 0 ? "band, half-bandwidth " + std::to_string(d_bandwidth.getValue()) + " of "
                                                  : std::string("dense, "))
                   << backend::tissueDofCount(m_workspace) << " DOFs";
    return m_workspace != nullptr;
}

bool GpuTissueSolver::followTopologyChanges(std::string& diagnostic)
{
    // Cutting removes tetrahedra (SOFA moves the last one into each hole). Rebuild
    // the remaining elements in the topology's new order, with the edges numbered as
    // at creation, the mass recomputed as MeshMatrixMass does for them, and each
    // tetrahedron's viscous state carried to its new position.
    diagnostic.clear();
    const auto& tets = m_topology->getTetrahedra();
    if (tets.size() == m_tetrahedra.size()) return true;
    const std::size_t n = m_state->getSize();
    if (n * 3 != m_restPositions.size())
    {
        diagnostic = "the topology change removed vertices (" + std::to_string(m_restPositions.size() / 3) + " -> " +
                     std::to_string(n) + "); only removing tetrahedra is supported: cut whole layers of cells, so that "
                     "every vertex keeps a tetrahedron";
        return false;
    }
    if (tets.empty())
    {
        diagnostic = "every tetrahedron was removed";
        return false;
    }
    std::map<std::array<int, 4>, int> previous;
    for (std::size_t t = 0; t < m_tetrahedra.size(); ++t) previous[tetrahedronKey(m_tetrahedra[t])] = static_cast<int>(t);

    backend::TissueElementUpdate update;
    const std::size_t nt = tets.size();
    update.tetrahedra.resize(4 * nt);
    update.tetrahedronEdges.resize(6 * nt);
    update.tetrahedronEdgeSides.resize(6 * nt);
    update.previousIndex.assign(nt, -1);
    update.restPositions = m_restPositions;
    update.vertexMass.assign(n, 0.0);
    update.edgeMass.assign(m_edges.size() / 2, 0.0);
    const double density = d_massDensity.getValue();
    std::vector<std::array<int, 4>> current(nt);
    for (std::size_t t = 0; t < nt; ++t)
    {
        for (int j = 0; j < 4; ++j)
        {
            current[t][j] = static_cast<int>(tets[t][j]);
            update.tetrahedra[4 * t + j] = current[t][j];
        }
        const auto found = previous.find(tetrahedronKey(current[t]));
        if (found != previous.end()) update.previousIndex[t] = found->second;
        sofa::type::Vec3d p[4];
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c) p[j][c] = m_restPositions[3 * current[t][j] + c];
        const double volume = sofa::geometry::Tetrahedron::volume(p[0], p[1], p[2], p[3]);
        for (int j = 0; j < 4; ++j) update.vertexMass[current[t][j]] += density * volume / 10.0;
        for (int j = 0; j < 6; ++j)
        {
            // As createWorkspace: k is the local vertex at the edge's first end.
            const auto local = m_topology->getLocalEdgesInTetrahedron(static_cast<sofa::Index>(j));
            int k = static_cast<int>(local[0]);
            int l = static_cast<int>(local[1]);
            const auto edge = m_edgeIndex.find(edgeKey(current[t][k], current[t][l]));
            if (edge == m_edgeIndex.end())
            {
                diagnostic = "a tetrahedron after the topology change has an edge the tissue did not have (only "
                             "removing tetrahedra is supported)";
                return false;
            }
            const int e = edge->second;
            if (m_edges[2 * e] != current[t][k]) std::swap(k, l);
            update.tetrahedronEdges[6 * t + j] = e;
            update.tetrahedronEdgeSides[6 * t + j] = static_cast<unsigned char>(k | (l << 2));
            update.edgeMass[e] += density * volume / 20.0;
        }
    }
    update.gravityMass.resize(n);
    update.fixedDofs = m_fixedDofs;
    int isolated = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        update.gravityMass[i] = update.vertexMass[i] * 2.5;   // MeshMatrixMass's lumping coefficient (tetrahedra)
        if (update.vertexMass[i] == 0.0)
        {
            update.fixedDofs[i] = 7;   // no tetrahedron left: no mass and no stiffness; held where it is
            ++isolated;
        }
    }
    if (isolated > m_isolatedVertices)
        msg_warning() << isolated << " vertices have no tetrahedron left after cutting; they are held in place.";
    m_isolatedVertices = isolated;
    if (!backend::updateTissueElements(m_workspace, update, diagnostic)) return false;
    msg_info() << "cutting: " << m_tetrahedra.size() - nt << " tetrahedra removed, " << nt << " left.";
    m_tetrahedra = std::move(current);
    if (m_cpu)
    {
        msg_warning() << "compareWithCpu stops: its CPU replica does not follow topology changes.";
        m_compareStream.close();
        m_cpu.reset();
    }
    return true;
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
    // Fully fixed vertices: FixedProjectiveConstraint; partly fixed ones: one
    // PartialFixedProjectiveConstraint per direction mask.
    std::ostringstream fixed;
    std::ostringstream partialByMask[8];
    for (std::size_t i = 0; i < m_fixedDofs.size(); ++i)
    {
        const unsigned mask = m_fixedDofs[i];
        if (mask == 7) fixed << (fixed.tellp() > 0 ? " " : "") << i;
        else if (mask != 0) partialByMask[mask] << (partialByMask[mask].tellp() > 0 ? " " : "") << i;
    }
    const auto& ogden = d_ogdenParameters.getValue();
    const auto& maxwell = d_maxwellParameters.getValue();
    std::ostringstream ogdenSet, maxwellSet;
    if (ogden.size() == 5)
        ogdenSet << std::setprecision(17) << ogden[0] << ' ' << ogden[1] << ' ' << ogden[2] << ' ' << ogden[3] << ' ' << ogden[4];
    if (maxwell.size() == 3) maxwellSet << std::setprecision(17) << maxwell[0] << ' ' << maxwell[1] << ' ' << maxwell[2];

    create("odeSolver", "EulerImplicitSolver", { { "rayleighStiffness", std::to_string(d_rayleighStiffness.getValue()) },
                                                  { "rayleighMass", std::to_string(d_rayleighMass.getValue()) } });
    create("linearSolver", "SparseLDLSolver", { { "template", "CompressedRowSparseMatrixMat3x3d" } });
    create("dofs", "MechanicalObject", { { "template", "Vec3d" }, { "position", joinNumbers(m_restPositions) } });
    create("topology", "TetrahedronSetTopologyContainer", { { "tetrahedra", tetrahedra.str() } });
    create("mass", "MeshMatrixMass", { { "massDensity", std::to_string(d_massDensity.getValue()) } });
    bool materialMissing = false;
    if (m_core > 0)
    {
        std::vector<double> coreValues(d_hyperelasticParameters.getValue().begin(), d_hyperelasticParameters.getValue().end());
        materialMissing |= !create("hyperelastic", "TetrahedronHyperelasticityFEMForceField",
                                   { { "template", "Vec3d" }, { "materialName", d_hyperelasticMaterial.getValue() },
                                     { "ParameterSet", joinNumbers(coreValues) } });
    }
    if (ogden.size() == 5)
        materialMissing |= !create("elastic", "TetrahedronViscoHyperelasticityFEMForceField",
                                   { { "template", "Vec3d" }, { "materialName", "SLSOgdenFirstOrder" }, { "ParameterSet", ogdenSet.str() } });
    if (maxwell.size() == 3)
        materialMissing |= !create("viscous", "TetrahedronViscoelasticityFEMForceField",
                                   { { "template", "Vec3d" }, { "materialName", "MaxwellFirstOrder" }, { "ParameterSet", maxwellSet.str() } });
    if (materialMissing)
    {
        diagnostic = "a material force field could not be created for the CPU replica (load "
                     "Sofa.Component.SolidMechanics.FEM.HyperElastic and SofaViscoElastic)";
        return false;
    }
    if (!m_loads.empty())
    {
        // Every vertex; the forces are set by updateExternalForces.
        std::ostringstream all, zeros;
        for (std::size_t i = 0; i < m_fixedDofs.size(); ++i)
        {
            all << (i ? " " : "") << i;
            zeros << (i ? " " : "") << "0 0 0";
        }
        m_cpu->load = create("externalLoad", "ConstantForceField",
                             { { "template", "Vec3d" }, { "indices", all.str() }, { "forces", zeros.str() } }).get();
        if (m_cpu->load == nullptr)
        {
            diagnostic = "ConstantForceField could not be created for the CPU replica";
            return false;
        }
    }
    if (fixed.tellp() > 0) create("fixed", "FixedProjectiveConstraint", { { "indices", fixed.str() } });
    for (unsigned mask = 1; mask < 7; ++mask)
    {
        if (partialByMask[mask].tellp() <= 0) continue;
        const std::string name = "partialFixed" + std::to_string(mask);
        const std::string directions = std::string((mask & 1u) ? "1" : "0") + ((mask & 2u) ? " 1" : " 0") + ((mask & 4u) ? " 1" : " 0");
        if (!create(name.c_str(), "PartialFixedProjectiveConstraint",
                    { { "template", "Vec3d" }, { "indices", partialByMask[mask].str() }, { "fixedDirections", directions } }))
        {
            diagnostic = "PartialFixedProjectiveConstraint could not be created for the CPU replica";
            return false;
        }
    }
    sofa::simulation::node::init(node);

    m_cpu->state = dynamic_cast<sofa::core::behavior::MechanicalState<Vec3dTypes>*>(node->getMechanicalState());
    m_cpu->odeSolver = node->get<sofa::core::behavior::OdeSolver>();
    m_cpu->linearSolver = node->get<sofa::core::behavior::LinearSolver>();
    if (m_cpu->state == nullptr || m_cpu->odeSolver == nullptr || m_cpu->linearSolver == nullptr)
    {
        diagnostic = "the CPU replica could not be built (are the material's plugins "
                     "(Sofa.Component.SolidMechanics.FEM.HyperElastic, SofaViscoElastic) and SOFA's direct solvers loaded?)";
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
        if (!computeFixedDofs(diagnostic) || !collectLoads(diagnostic) || !createWorkspace(diagnostic))
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
    if (!followTopologyChanges(diagnostic))
    {
        msg_error() << "GPU tissue: " << diagnostic;
        m_failed = true;
        return;
    }
    if (!updateExternalForces(diagnostic)) reportFailure("external forces", diagnostic);

    auto* state = m_state;
    if (m_cpu)
    {
        // The comparison's start state, read before the step: with DefaultAnimationLoop
        // xResult/vResult are the position and velocity themselves, updated in place.
        const std::size_t n = state->getSize();
        const auto& xs = state->read(sofa::core::vec_id::read_access::position)->getValue();
        const auto& vs = state->read(sofa::core::vec_id::read_access::velocity)->getValue();
        m_startX.resize(3 * n);
        m_startV.resize(3 * n);
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c)
            {
                m_startX[3 * i + c] = xs[i][c];
                m_startV[3 * i + c] = vs[i][c];
            }
    }
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
    const int luSteps = backend::tissueLuFallbackSteps(m_workspace);
    if (luSteps != d_luFallbackSteps.getValue())
    {
        if (d_luFallbackSteps.getValue() == 0)
            msg_warning() << "t = " << this->getContext()->getTime() << ": the system matrix is not positive definite "
                          << "(a strongly compressed state); this step and any like it are solved with LU instead of "
                          << "Cholesky (luFallbackSteps counts them).";
        d_luFallbackSteps.setValue(luSteps);
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
    // The replica starts from the GPU's state at the step's start (read back in
    // solve() before the step) and runs SOFA's own free motion every step, which
    // also keeps its viscous strains in step with the GPU's.
    const std::size_t n = m_state->getSize();
    const std::vector<double>& v0 = m_startV;
    {
        auto xr = sofa::helper::getWriteOnlyAccessor(*m_cpu->state->write(sofa::core::vec_id::write_access::position));
        auto vr = sofa::helper::getWriteOnlyAccessor(*m_cpu->state->write(sofa::core::vec_id::write_access::velocity));
        xr.resize(n);
        vr.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c)
            {
                xr[i][c] = m_startX[3 * i + c];
                vr[i][c] = m_startV[3 * i + c];
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
