// Checks for the GPU contact constraints against SOFA's CPU implementation, on
// identical inputs.
//
//   C1  friction solver. Our GPU block Gauss-Seidel against SOFA's own
//       BlockGaussSeidelConstraintSolver (the real SOFA code, run through
//       GenericConstraintProblem::solveTimed), for the same W, dfree, mu,
//       tolerance and iteration limit. W is rounded to float first and both
//       sides get the rounded values, so the inputs are bit-identical; what is
//       left is summation order (and float accumulation, in the fast mode).
//   C2  compliance. The block of A^-1 on chosen DOFs from our GPU dense Cholesky
//       (float) against a double-precision CPU factorisation (Eigen
//       SimplicialLDLT) of the same sparse matrix: an FEM-like 3D grid matrix the
//       size of the tissue-poke block.
//
// Prints one line per case, then C1/C2 PASS or FAIL. Exit code 0 = all passed.

#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <sofa/component/constraint/lagrangian/model/UnilateralConstraintResolution.h>
#include <sofa/component/constraint/lagrangian/solver/BlockGaussSeidelConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintProblem.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace backend = SofaGpuCollision::backend;
using sofa::component::constraint::lagrangian::model::UnilateralConstraintResolution;
using sofa::component::constraint::lagrangian::model::UnilateralConstraintResolutionWithFriction;
using sofa::component::constraint::lagrangian::solver::BlockGaussSeidelConstraintSolver;
using sofa::component::constraint::lagrangian::solver::GenericConstraintProblem;

namespace
{

double nowMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// C1: friction problems shaped like real contact problems.
// A deformable body's compliance on its surface vertices (a smooth SPD kernel:
// nearby vertices move together) plus a rigid body's 6x6 compliance, contacts on
// random vertices with SOFA's normal/tangent basis, W = J1 C1 J1^T + J2 C2 J2^T.
// ---------------------------------------------------------------------------
struct FrictionCase
{
    const char* name;
    int contacts;
    double mu;
    bool redundant;      // several contacts on the same vertex (rank-deficient W)
    unsigned seed;
};

struct Vec3
{
    double x, y, z;
};

Vec3 cross(const Vec3& a, const Vec3& b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

Vec3 normalized(const Vec3& v)
{
    const double l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return { v.x / l, v.y / l, v.z / l };
}

void buildFrictionProblem(const FrictionCase& fc, int& rows, std::vector<double>& W, std::vector<double>& dfree)
{
    std::mt19937 rng(fc.seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const int vertices = fc.redundant ? std::max(4, fc.contacts / 3) : fc.contacts;
    std::vector<Vec3> p(vertices);
    for (auto& v : p) v = { uni(rng) * 0.01, 0.0, uni(rng) * 0.01 };

    // Deformable compliance on the vertices: c0 exp(-d^2/s^2) I3 + c1 I3.
    const double c0 = 1.0, c1 = 0.25, s2 = 0.004 * 0.004;
    auto kernel = [&](int a, int b) {
        const double dx = p[a].x - p[b].x, dz = p[a].z - p[b].z;
        return c0 * std::exp(-(dx * dx + dz * dz) / s2) + (a == b ? c1 : 0.0);
    };
    // Rigid compliance: SPD 6x6.
    double C2[36] = {};
    for (int i = 0; i < 6; ++i) C2[i * 6 + i] = (i < 3) ? 0.5 : 40.0;
    C2[0 * 6 + 4] = C2[4 * 6 + 0] = 0.3;

    const int rowsPerContact = fc.mu > 0.0 ? 3 : 1;
    rows = fc.contacts * rowsPerContact;
    std::vector<int> vertexOf(fc.contacts);
    std::vector<Vec3> dir(rows);
    std::vector<std::array<double, 6>> j2(rows);
    dfree.assign(rows, 0.0);
    for (int c = 0; c < fc.contacts; ++c)
    {
        vertexOf[c] = fc.redundant ? static_cast<int>(rng() % vertices) : c;
        const Vec3 n = normalized({ uni(rng) * 0.3, 1.0, uni(rng) * 0.3 });
        // SOFA's basis (BaseContactLagrangianConstraint::addContact).
        Vec3 t = { n.z, n.x, n.y };
        const Vec3 s = normalized(cross(n, t));
        t = cross({ -n.x, -n.y, -n.z }, s);
        const Vec3 u[3] = { n, t, s };
        const Vec3 r = { p[vertexOf[c]].x, 0.02, p[vertexOf[c]].z };   // lever arm to the rigid centre
        for (int l = 0; l < rowsPerContact; ++l)
        {
            const int row = c * rowsPerContact + l;
            dir[row] = u[l];
            const Vec3 m = cross(r, u[l]);
            j2[row] = { u[l].x, u[l].y, u[l].z, m.x, m.y, m.z };
        }
        dfree[c * rowsPerContact] = -0.8 + 1.2 * (0.5 + 0.5 * uni(rng));    // mostly penetrating
        if (rowsPerContact == 3)
        {
            dfree[c * rowsPerContact + 1] = 0.3 * uni(rng);
            dfree[c * rowsPerContact + 2] = 0.3 * uni(rng);
        }
    }
    W.assign(static_cast<std::size_t>(rows) * rows, 0.0);
    for (int i = 0; i < rows; ++i)
    {
        const int ci = i / rowsPerContact;
        for (int j = 0; j < rows; ++j)
        {
            const int cj = j / rowsPerContact;
            // J1 row = -u on the vertex; C1 block = kernel * I3.
            const double w1 = kernel(vertexOf[ci], vertexOf[cj]) *
                (dir[i].x * dir[j].x + dir[i].y * dir[j].y + dir[i].z * dir[j].z);
            double w2 = 0.0;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) w2 += j2[i][a] * C2[a * 6 + b] * j2[j][b];
            W[static_cast<std::size_t>(i) * rows + j] = w1 + w2;
        }
    }
    // Round to float so both solvers see bit-identical inputs.
    for (double& w : W) w = static_cast<double>(static_cast<float>(w));
    for (double& d : dfree) d = static_cast<double>(static_cast<float>(d));
}

struct SofaSolveResult
{
    std::vector<double> lambda;
    int iterations { 0 };
    double error { 0.0 };
    double ms { 0.0 };
};

SofaSolveResult solveWithSofa(BlockGaussSeidelConstraintSolver* solver, int rows, int rowsPerContact, double mu,
                              const std::vector<double>& W, const std::vector<double>& dfree,
                              const backend::ConstraintSolveConfig& config)
{
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
    problem.scaleTolerance = config.scaleTolerance;
    problem.allVerified = config.allVerified;
    problem.sor = config.sor;
    const double t0 = nowMs();
    problem.solveTimed(config.tolerance, config.maxIterations, 0.0);
    SofaSolveResult result;
    result.ms = nowMs() - t0;
    result.lambda.resize(rows);
    for (int i = 0; i < rows; ++i) result.lambda[i] = problem.f[i];
    result.iterations = problem.currentIterations;   // SOFA reports sweeps + 1
    result.error = problem.currentError;
    return result;
}

double maxAbs(const std::vector<double>& v)
{
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Final violations d = dfree + W lambda: unique even when lambda is not.
std::vector<double> violations(int rows, const std::vector<double>& W, const std::vector<double>& dfree, const std::vector<double>& lambda)
{
    std::vector<double> d(dfree);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < rows; ++j) d[i] += W[static_cast<std::size_t>(i) * rows + j] * lambda[j];
    return d;
}

bool runC1()
{
    const FrictionCase cases[] = {
        { "frictionless",        60, 0.0, false, 11u },
        { "mu=0.1",              60, 0.1, false, 12u },
        { "mu=0.8",              60, 0.8, false, 13u },
        { "mu=0.1 redundant",    90, 0.1, true,  14u },
        { "mu=0.1 400 contacts", 400, 0.1, false, 15u },
        { "mu=0.1 800 contacts", 800, 0.1, false, 16u },
        // Above the shared-memory size: the multipliers live in global memory.
        { "mu=0.1 1600 contacts", 1600, 0.1, false, 17u },
    };
    auto sofaSolver = sofa::core::objectmodel::New<BlockGaussSeidelConstraintSolver>();

    // The first launch of each kernel pays a one-off JIT compile (the backend is
    // built for sm_52 and runs on sm_75); do it before timing anything.
    {
        int rows = 0;
        std::vector<double> W, dfree, lambda;
        buildFrictionProblem(cases[1], rows, W, dfree);
        std::string diagnostic;
        for (const bool exact : { true, false })
        {
            backend::ConstraintSolveConfig warm;
            warm.doubleAccumulation = exact;
            backend::solveFrictionProblemOnGpu(rows, 3, W, dfree, 0.1, warm, lambda, nullptr, diagnostic);
            buildFrictionProblem(cases[0], rows, W, dfree);
            backend::solveFrictionProblemOnGpu(rows, 1, W, dfree, 0.0, warm, lambda, nullptr, diagnostic);
            buildFrictionProblem(cases[4], rows, W, dfree);
            backend::solveFrictionProblemOnGpu(rows, 3, W, dfree, 0.1, warm, lambda, nullptr, diagnostic);
            buildFrictionProblem(cases[5], rows, W, dfree);
            backend::solveFrictionProblemOnGpu(rows, 3, W, dfree, 0.1, warm, lambda, nullptr, diagnostic);
            buildFrictionProblem(cases[1], rows, W, dfree);
        }
    }

    bool allPassed = true;
    for (const FrictionCase& fc : cases)
    {
        int rows = 0;
        std::vector<double> W, dfree;
        buildFrictionProblem(fc, rows, W, dfree);
        const int rowsPerContact = fc.mu > 0.0 ? 3 : 1;

        bool casePassed = true;
        for (const bool exact : { true, false })
        {
            // Exact mode: SOFA's arithmetic in double, deep tolerance: must match
            // SOFA to machine precision, sweep for sweep. Fast mode (float): at a
            // tolerance float can reach; must agree to float accuracy.
            backend::ConstraintSolveConfig config;
            config.maxIterations = 1000;
            config.tolerance = exact ? 1.0e-9 : 1.0e-6;
            config.scaleTolerance = true;
            config.doubleAccumulation = exact;
            const SofaSolveResult sofa = solveWithSofa(sofaSolver.get(), rows, rowsPerContact, fc.mu, W, dfree, config);

            std::vector<double> lambda;
            backend::ConstraintSolveStats stats;
            std::string diagnostic;
            const double t0 = nowMs();
            if (!backend::solveFrictionProblemOnGpu(rows, rowsPerContact, W, dfree, fc.mu, config, lambda, &stats, diagnostic))
            {
                std::cout << "C1 " << fc.name << ": GPU solve failed: " << diagnostic << "\n";
                return false;
            }
            const double wallMs = nowMs() - t0;
            const double gpuMs = stats.gpuMilliseconds;
            const double scale = std::max(maxAbs(sofa.lambda), 1e-30);
            const double lambdaRel = maxAbsDiff(lambda, sofa.lambda) / scale;
            const std::vector<double> dSofa = violations(rows, W, dfree, sofa.lambda);
            const std::vector<double> dGpu = violations(rows, W, dfree, lambda);
            const double violationDiff = maxAbsDiff(dSofa, dGpu) / std::max(maxAbs(dfree), 1e-30);
            const int sofaSweeps = sofa.iterations - 1;   // SOFA reports sweeps + 1
            const double lambdaLimit = exact ? (fc.redundant ? 1e-6 : 1e-10) : 1e-3;
            const double violationLimit = exact ? 1e-10 : 1e-4;
            const int sweepSlack = exact ? 0 : std::max(3, sofaSweeps / 5);
            const bool pass = violationDiff < violationLimit && lambdaRel < lambdaLimit &&
                              std::abs(stats.iterations - sofaSweeps) <= sweepSlack;
            casePassed = casePassed && pass;
            std::printf("C1 %-20s rows=%4d %s tol=%.0e  sweeps gpu=%4d sofa=%4d  lambda_rel_diff=%.2e  "
                        "violation_diff=%.2e  solve gpu=%7.2f ms sofa=%7.2f ms (gpu call incl. upload %.2f ms) %s\n",
                        fc.name, rows, exact ? "exact" : "fast ", config.tolerance, stats.iterations, sofaSweeps,
                        lambdaRel, violationDiff, gpuMs, sofa.ms, wallMs, pass ? "[ok]" : "[MISMATCH]");
        }
        allPassed = allPassed && casePassed;
    }
    std::printf("C1_FRICTION_SOLVER=%s\n", allPassed ? "PASS" : "FAIL");
    return allPassed;
}

// ---------------------------------------------------------------------------
// C2: compliance of an FEM-like grid matrix.
// Nodes on an nx*ny*nz grid (the poke block's size), 3 DOFs each; anisotropic
// springs to the 6 neighbours with random stiffness; lumped mass; bottom layer
// fixed the way SOFA's projective constraint leaves it (identity rows/cols).
// A = M + h^2 K, as the implicit solver builds it.
// ---------------------------------------------------------------------------
bool runC2(const bool cadence)
{
    const int nx = 15, ny = 8, nz = 15;
    const int nodes = nx * ny * nz;
    const int n = 3 * nodes;
    const double h = 0.01;
    std::mt19937 rng(21u);
    std::uniform_real_distribution<double> uni(0.5, 1.5);
    auto node = [&](int i, int j, int k) { return i + j * nx + k * nx * ny; };

    std::vector<Eigen::Triplet<double>> triplets;
    auto addBlock = [&](int a, int b, const double m[9]) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                if (m[r * 3 + c] != 0.0) triplets.emplace_back(3 * a + r, 3 * b + c, m[r * 3 + c]);
    };
    std::vector<bool> fixedNode(nodes, false);
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) fixedNode[node(i, 0, k)] = true;

    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
        const int a = node(i, j, k);
        const int di[3] = { 1, 0, 0 }, dj[3] = { 0, 1, 0 }, dk[3] = { 0, 0, 1 };
        for (int e = 0; e < 3; ++e)
        {
            const int i2 = i + di[e], j2 = j + dj[e], k2 = k + dk[e];
            if (i2 >= nx || j2 >= ny || k2 >= nz) continue;
            const int b = node(i2, j2, k2);
            const double kAxial = 5.0 * uni(rng), kShear = 1.0 * uni(rng);
            double block[9] = {};
            for (int d = 0; d < 3; ++d) block[d * 3 + d] = (d == e) ? kAxial : kShear;
            double scaled[9], neg[9];
            for (int q = 0; q < 9; ++q) { scaled[q] = h * h * block[q]; neg[q] = -scaled[q]; }
            if (!fixedNode[a]) addBlock(a, a, scaled);
            if (!fixedNode[b]) addBlock(b, b, scaled);
            if (!fixedNode[a] && !fixedNode[b]) { addBlock(a, b, neg); addBlock(b, a, neg); }
        }
        double mass[9] = {};
        for (int d = 0; d < 3; ++d) mass[d * 3 + d] = fixedNode[a] ? 1.0 : 1.0e-5 * uni(rng);
        addBlock(a, a, mass);
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // The touched vertices: a patch on the top, as under the probe.
    std::vector<int> vertices;
    for (int k = 4; k < 11; ++k)
        for (int i = 4; i < 11; ++i) vertices.push_back(node(i, ny - 1, k));
    for (int k = 5; k < 10; ++k)
        for (int i = 5; i < 10; ++i) vertices.push_back(node(i, ny - 2, k));
    const int m = 3 * static_cast<int>(vertices.size());

    std::vector<int> rowPtr(n + 1), columns(A.nonZeros());
    std::vector<double> values(A.nonZeros());
    for (int r = 0; r <= n; ++r) rowPtr[r] = A.outerIndexPtr()[r];
    for (int p = 0; p < A.nonZeros(); ++p) { columns[p] = A.innerIndexPtr()[p]; values[p] = A.valuePtr()[p]; }
    backend::HostCsrMatrix csr;
    csr.size = n;
    csr.nonZeros = static_cast<int>(A.nonZeros());
    csr.rowPtr = rowPtr.data();
    csr.columns = columns.data();
    csr.values = values.data();

    std::vector<double> gpu;
    backend::ConstraintTimings timings;
    std::string diagnostic;
    // First call warms up cuBLAS/cuSOLVER; time the second.
    if (!backend::computeDenseComplianceOnGpu(csr, vertices, gpu, &timings, diagnostic) ||
        !backend::computeDenseComplianceOnGpu(csr, vertices, gpu, &timings, diagnostic))
    {
        std::cout << "C2 GPU compliance failed: " << diagnostic << "\n";
        return false;
    }

    const double t0 = nowMs();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.compute(Eigen::SparseMatrix<double>(A));
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(n, m);
    for (std::size_t k = 0; k < vertices.size(); ++k)
        for (int c = 0; c < 3; ++c) E(3 * vertices[k] + c, 3 * static_cast<int>(k) + c) = 1.0;
    const Eigen::MatrixXd X = ldlt.solve(E);
    const Eigen::MatrixXd G = E.transpose() * X;
    const double cpuMs = nowMs() - t0;

    double maxDiff = 0.0, maxRef = 0.0, maxDiagRel = 0.0;
    for (int c = 0; c < m; ++c)
    {
        for (int r = 0; r < m; ++r)
        {
            const double ref = G(r, c);
            maxRef = std::max(maxRef, std::fabs(ref));
            maxDiff = std::max(maxDiff, std::fabs(gpu[static_cast<std::size_t>(c) * m + r] - ref));
        }
        maxDiagRel = std::max(maxDiagRel, std::fabs(gpu[static_cast<std::size_t>(c) * m + c] - G(c, c)) / std::fabs(G(c, c)));
    }
    const double rel = maxDiff / maxRef;
    const bool pass = rel < 1e-3 && maxDiagRel < 1e-3;
    std::printf("C2 compliance n=%d touched_dofs=%d  max_rel_diff=%.2e  max_diag_rel_diff=%.2e  "
                "gpu factorize=%.2f ms compliance=%.2f ms  cpu(Eigen LDLT, double)=%.2f ms %s\n",
                n, m, rel, maxDiagRel, timings.factorizeMs, timings.complianceMs, cpuMs, pass ? "[ok]" : "[MISMATCH]");
    std::printf("C2_COMPLIANCE=%s\n", pass ? "PASS" : "FAIL");

    if (cadence)
    {
        // Diagnostic (--cadence): the same factorization back to back, then with
        // the idle gap a CPU-bound scene leaves between steps (300 ms).
        for (const int pauseMs : { 0, 300 })
        {
            std::printf("C2 cadence pause=%3d ms  factorize ms:", pauseMs);
            for (int i = 0; i < 6; ++i)
            {
                if (pauseMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pauseMs));
                backend::computeDenseComplianceOnGpu(csr, vertices, gpu, &timings, diagnostic);
                std::printf(" %7.2f", timings.factorizeMs);
            }
            std::printf("\n");
        }
    }
    return pass;
}

} // namespace

int main(int argc, char** argv)
{
    // --cadence: also time the factorization at a CPU-bound scene's cadence.
    const bool cadence = argc > 1 && std::string(argv[1]) == "--cadence";
    std::cout << "GPU contact constraints vs SOFA CPU, identical inputs\n";
    const bool c1 = runC1();
    const bool c2 = runC2(cadence);
    std::cout << "CONSTRAINT_CHECKS=" << ((c1 && c2) ? "PASS" : "FAIL") << "\n"
              << "  C1 = GPU block Gauss-Seidel with friction vs SOFA's BlockGaussSeidelConstraintSolver\n"
              << "       (same W, dfree, mu, tolerance): multipliers and final violations\n"
              << "  C2 = GPU dense Cholesky compliance (float) vs CPU LDL^T (double) on an FEM-like matrix\n";
    return (c1 && c2) ? 0 : 1;
}
