// TissueSolver.cuh — part of the SINGLE GpuCollisionBackend.cu translation unit.
// Included after ContactConstraints.cuh: it reuses its helpers (ensureDeviceBuffer,
// launchFill, ConstraintEventTimer) and hands its Cholesky factor to the contact
// constraints.
//
// One implicit Euler step of a tetrahedral viscoelastic tissue, all on the GPU,
// following SOFA's CPU components stage for stage so the two can be compared:
//
//   material  SofaViscoElastic's TetrahedronViscoHyperelasticityFEMForceField with
//             SLSOgdenFirstOrder, plus (optionally) TetrahedronViscoelasticity-
//             FEMForceField with MaxwellFirstOrder: the same deformation gradient,
//             stresses, viscous-strain updates and per-edge stiffness blocks
//             (updateTangentMatrix), in double precision.
//   mass      MeshMatrixMass on tetrahedra, not lumped: vertex masses rho V/10,
//             edge masses rho V/20, gravity on the lumped mass (x 2.5).
//   fixed     FixedProjectiveConstraint through SOFA's linear system: the rows and
//             columns of fixed DOFs cleared and their diagonal set to 1
//             (MatrixLinearSystem::Dirichlet::discardRowCol), the RHS projected.
//   step      EulerImplicitSolver: b = h (f + (h + rS) K v - rM M v), projected;
//             A = (1 + h rM) M - h (h + rS) K; A dv = b; v_free = v + dv;
//             x_free = x + h v_free.
//   solve     A assembled in double (block rows: vertex blocks and edge blocks),
//             copied into a dense single-precision matrix and factorised with
//             cuSOLVER (Cholesky); iterative refinement in double brings the
//             solution back to double accuracy. The factor stays for the contact
//             constraints, which need A^-1 too.
//
// Every sum over elements is a gather in a fixed order (no atomics), so a step is
// reproducible bit for bit.

namespace
{

// SOFA MatSym storage: 00, 01, 11, 02, 12, 22.
__device__ __forceinline__ int symIndex(const int i, const int j)
{
    const int a = i < j ? i : j;
    const int b = i < j ? j : i;
    return (b == 0) ? 0 : (b == 1 ? (a == 0 ? 1 : 2) : (a == 0 ? 3 : (a == 1 ? 4 : 5)));
}

struct DSym3
{
    double s[6];
    __device__ __forceinline__ double at(const int i, const int j) const { return s[symIndex(i, j)]; }
};

__device__ __forceinline__ DSym3 dsymZero()
{
    DSym3 r;
    for (int k = 0; k < 6; ++k) r.s[k] = 0.0;
    return r;
}

__device__ __forceinline__ DSym3 dsymIdentity()
{
    DSym3 r = dsymZero();
    r.s[0] = r.s[2] = r.s[5] = 1.0;
    return r;
}

__device__ __forceinline__ DSym3 dsymAxpy(const DSym3& a, const double alpha, const DSym3& b)   // a + alpha b
{
    DSym3 r;
    for (int k = 0; k < 6; ++k) r.s[k] = a.s[k] + alpha * b.s[k];
    return r;
}

__device__ __forceinline__ DSym3 dsymScale(const DSym3& a, const double alpha)
{
    DSym3 r;
    for (int k = 0; k < 6; ++k) r.s[k] = alpha * a.s[k];
    return r;
}

// A : B with SOFA's convention (off-diagonal terms counted twice).
__device__ __forceinline__ double dsymContract(const DSym3& a, const DSym3& b)
{
    return a.s[0] * b.s[0] + a.s[2] * b.s[2] + a.s[5] * b.s[5] +
           2.0 * a.s[1] * b.s[1] + 2.0 * a.s[3] * b.s[3] + 2.0 * a.s[4] * b.s[4];
}

// P H P for symmetric P and H (the result is symmetric).
__device__ __forceinline__ DSym3 dsymSandwich(const DSym3& p, const DSym3& h)
{
    double t[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += h.at(i, k) * p.at(k, j);
            t[i][j] = acc;                        // H P
        }
    DSym3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += p.at(i, k) * t[k][j];
            r.s[symIndex(i, j)] = acc;            // P (H P)
        }
    return r;
}

__device__ __forceinline__ void dsymApply(const DSym3& a, const double v[3], double out[3])
{
    for (int i = 0; i < 3; ++i) out[i] = a.at(i, 0) * v[0] + a.at(i, 1) * v[1] + a.at(i, 2) * v[2];
}

// SOFA's invertMatrix for a symmetric 3x3 (adjugate / determinant).
__device__ __forceinline__ DSym3 dsymInverse(const DSym3& m)
{
    const double a00 = m.s[0], a01 = m.s[1], a11 = m.s[2], a02 = m.s[3], a12 = m.s[4], a22 = m.s[5];
    const double c00 = a11 * a22 - a12 * a12;
    const double c01 = a02 * a12 - a01 * a22;
    const double c02 = a01 * a12 - a02 * a11;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    const double inv = 1.0 / det;
    DSym3 r;
    r.s[0] = c00 * inv;
    r.s[1] = c01 * inv;
    r.s[3] = c02 * inv;
    r.s[2] = (a00 * a22 - a02 * a02) * inv;
    r.s[4] = (a01 * a02 - a00 * a12) * inv;
    r.s[5] = (a00 * a11 - a01 * a01) * inv;
    return r;
}

// Symmetric 3x3 eigen-decomposition (cyclic Jacobi, double). Columns of V are the
// eigenvectors. Only functions of C (C^p) are used, which do not depend on the
// eigenvectors' order or sign, so any accurate decomposition gives SOFA's result.
__device__ void dsymEigen(const DSym3& c, double lambda[3], double V[3][3])
{
    double a[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            a[i][j] = c.at(i, j);
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    for (int sweep = 0; sweep < 32; ++sweep)
    {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        const double diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (off <= 1e-34 * diag || off == 0.0) break;
        for (int p = 0; p < 2; ++p)
        {
            for (int q = p + 1; q < 3; ++q)
            {
                const double apq = a[p][q];
                if (apq == 0.0) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double cs = 1.0 / sqrt(t * t + 1.0);
                const double sn = t * cs;
                // A' = R^T A R with R = rotation in the (p, q) plane.
                for (int k = 0; k < 3; ++k)
                {
                    const double akp = a[k][p];
                    const double akq = a[k][q];
                    a[k][p] = cs * akp - sn * akq;
                    a[k][q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < 3; ++k)
                {
                    const double apk = a[p][k];
                    const double aqk = a[q][k];
                    a[p][k] = cs * apk - sn * aqk;
                    a[q][k] = sn * apk + cs * aqk;
                }
                for (int k = 0; k < 3; ++k)
                {
                    const double vkp = V[k][p];
                    const double vkq = V[k][q];
                    V[k][p] = cs * vkp - sn * vkq;
                    V[k][q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) lambda[i] = a[i][i];
}

// C^p from the eigen-decomposition: sum_i lambda_i^p v_i v_i^T.
__device__ __forceinline__ DSym3 dsymPower(const double lambda[3], const double V[3][3], const double p)
{
    const double w[3] = { pow(lambda[0], p), pow(lambda[1], p), pow(lambda[2], p) };
    DSym3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
            r.s[symIndex(i, j)] = w[0] * V[i][0] * V[j][0] + w[1] * V[i][1] * V[j][1] + w[2] * V[i][2] * V[j][2];
    return r;
}

// SofaViscoElastic's SLSOgdenFirstOrder (v25.12) calls Eigen::SelfAdjointEigenSolver(C, true).
// Eigen 3 reads 'true' as options = 1, which does not ask for eigenvectors, so eigenvectors()
// returns the solver's workspace: C's lower triangle divided by its largest entry. The
// eigenvalues are right (sorted ascending), and SOFA builds C^p from that pair. This gives the
// same pair, so SOFA's results can be reproduced. (SOFA's own Ogden material replaced the call
// in 11/2025 for this reason; SofaViscoElastic still has it.)
__device__ __forceinline__ void sofaOgdenBasis(const DSym3& c, double lambda[3], double V[3][3])
{
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2 - i; ++j)
            if (lambda[j] > lambda[j + 1])
            {
                const double t = lambda[j];
                lambda[j] = lambda[j + 1];
                lambda[j + 1] = t;
            }
    double scale = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) scale = fmax(scale, fabs(c.at(i, j)));
    if (scale == 0.0) scale = 1.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) V[i][j] = (j <= i) ? c.at(i, j) / scale : 0.0;
}

struct TissueMaterialDevice
{
    double mu1, alpha1, slsG1, slsTau, k0;
    double maxwellG1, maxwellTau, maxwellLambda;
    int hasOgden, hasMaxwell;
    int ogdenSofaBasis;   // 1: C^p as SofaViscoElastic computes it (sofaOgdenBasis), 0: exact
};

// Everything SLSOgdenFirstOrder::applyElasticityTensor needs, computed once per tetrahedron.
struct OgdenTangentState
{
    DSym3 cinv;      // C^-1
    DSym3 ca1;       // C^(alpha/2 - 1)
    DSym3 ca2;       // C^(alpha/4 - 1)
    double trCalpha; // tr C^(alpha/2)
    double jfac;     // mu1/alpha1 J^(-alpha1/3)
    double logJ;
};

// SLSOgdenFirstOrder::applyElasticityTensor, term for term.
__device__ __forceinline__ DSym3 ogdenTangent(const OgdenTangentState& o, const TissueMaterialDevice& m, const DSym3& h)
{
    const double alpha = m.alpha1;
    const double trHCalpha1 = dsymContract(h, o.ca1);
    const double trHC = dsymContract(h, o.cinv);
    const DSym3 first = dsymSandwich(o.cinv, h);
    const DSym3 second = dsymSandwich(o.ca2, h);
    // (trHC (-alpha/6) (-1/3 Cinv trCalpha + Ca1) + 1/3 First trCalpha
    //   - 1/3 Cinv trHCalpha1 alpha/2 + (alpha/2 - 1) Second) jfac
    //  + k0/2 trHC Cinv - k0 log J First
    DSym3 bracket = dsymAxpy(dsymScale(o.cinv, -o.trCalpha / 3.0), 1.0, o.ca1);
    bracket = dsymScale(bracket, trHC * (-alpha / 6.0));
    bracket = dsymAxpy(bracket, o.trCalpha / 3.0, first);
    bracket = dsymAxpy(bracket, -trHCalpha1 * alpha / 6.0, o.cinv);
    bracket = dsymAxpy(bracket, alpha / 2.0 - 1.0, second);
    DSym3 out = dsymScale(bracket, o.jfac);
    out = dsymAxpy(out, m.k0 / 2.0 * trHC, o.cinv);
    out = dsymAxpy(out, -(m.k0 * o.logJ), first);
    return out;
}

// One thread per tetrahedron: stresses (advancing the viscous strains once),
// nodal forces and the 6 per-edge stiffness blocks, written per tetrahedron for
// the gathers.
__global__ void tissueMaterialKernel(
    const int tetCount,
    const int4* __restrict__ tets,
    const double* __restrict__ shapeVectors,    // tet * 12
    const double* __restrict__ restVolume,      // tet
    const double* __restrict__ volScale,        // tet
    const unsigned char* __restrict__ edgeSides, // tet * 6: k | (l << 2)
    const float* __restrict__ x,                // vertices * 3
    const TissueMaterialDevice material,
    const double dt,
    double* __restrict__ slsViscous,            // tet * 6 (previous viscous strain), updated
    double* __restrict__ maxwellViscous,        // tet * 6, updated
    double* __restrict__ tetForce,              // tet * 12
    double* __restrict__ tetEdgeBlocks,         // tet * 54 (row-major 3x3 per edge)
    double* __restrict__ tetJ)                  // tet
{
    const int stride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < tetCount; t += stride)
    {
        const int4 ta = tets[t];
        const int vid[4] = { ta.x, ta.y, ta.z, ta.w };
        double sv[4][3];
        for (int a = 0; a < 4; ++a)
            for (int c = 0; c < 3; ++c) sv[a][c] = shapeVectors[static_cast<std::size_t>(t) * 12 + a * 3 + c];
        const double V = restVolume[t];

        // Deformation gradient, as TetrahedronViscoHyperelasticityFEMForceField::addForce.
        double x0[3], dp[3][3];
        for (int c = 0; c < 3; ++c) x0[c] = static_cast<double>(x[3 * vid[0] + c]);
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < 3; ++c) dp[j][c] = static_cast<double>(x[3 * vid[j + 1] + c]) - x0[c];
        double F[3][3];
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) F[k][l] = dp[0][k] * sv[1][l];
        for (int j = 1; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) F[k][l] += dp[j][k] * sv[j + 1][l];

        DSym3 C;
        for (int k = 0; k < 3; ++k)
            for (int l = k; l < 3; ++l)
                C.s[symIndex(k, l)] = F[0][k] * F[0][l] + F[1][k] * F[1][l] + F[2][k] * F[2][l];
        const DSym3 E = dsymScale(dsymAxpy(C, -1.0, dsymIdentity()), 0.5);
        const double areaVec[3] = { dp[1][1] * dp[2][2] - dp[1][2] * dp[2][1],
                                    dp[1][2] * dp[2][0] - dp[1][0] * dp[2][2],
                                    dp[1][0] * dp[2][1] - dp[1][1] * dp[2][0] };
        const double J = (areaVec[0] * dp[0][0] + areaVec[1] * dp[0][1] + areaVec[2] * dp[0][2]) * volScale[t];
        tetJ[t] = J;
        const double trE = E.s[0] + E.s[2] + E.s[5];

        DSym3 S = dsymZero();
        OgdenTangentState og {};
        if (material.hasOgden)
        {
            // SLSOgdenFirstOrder::deriveSPKTensor
            double lambda[3], Vec[3][3];
            dsymEigen(C, lambda, Vec);
            if (material.ogdenSofaBasis) sofaOgdenBasis(C, lambda, Vec);
            const double a = material.alpha1;
            og.trCalpha = pow(lambda[0], a / 2.0) + pow(lambda[1], a / 2.0) + pow(lambda[2], a / 2.0);
            og.ca1 = dsymPower(lambda, Vec, a / 2.0 - 1.0);
            og.ca2 = dsymPower(lambda, Vec, a / 4.0 - 1.0);
            og.cinv = dsymInverse(C);
            og.jfac = material.mu1 / a * pow(J, -a / 3.0);
            og.logJ = log(J);
            DSym3 sOgden = dsymAxpy(dsymScale(og.cinv, -og.trCalpha / 3.0), 1.0, og.ca1);
            sOgden = dsymScale(sOgden, og.jfac);
            sOgden = dsymAxpy(sOgden, material.k0 * og.logJ, og.cinv);
            // Its own viscous branch: Evisc1 = (tau/dt Evisc_prev + E) / (1 + tau/dt).
            const double r = material.slsTau / dt;
            DSym3 ev;
            for (int k = 0; k < 6; ++k)
            {
                ev.s[k] = (1.0 / (1.0 + r)) * (r * slsViscous[static_cast<std::size_t>(t) * 6 + k] + E.s[k]);
                slsViscous[static_cast<std::size_t>(t) * 6 + k] = ev.s[k];
            }
            sOgden = dsymAxpy(sOgden, 2.0 * material.slsG1, dsymAxpy(E, -1.0, ev));
            S = dsymAxpy(S, 1.0, sOgden);
        }
        if (material.hasMaxwell)
        {
            // MaxwellFirstOrder::deriveCauchyGreenStressTensor
            const double r = material.maxwellTau / dt;
            DSym3 ev;
            for (int k = 0; k < 6; ++k)
            {
                ev.s[k] = (1.0 / (1.0 + r)) * (r * maxwellViscous[static_cast<std::size_t>(t) * 6 + k] + E.s[k]);
                maxwellViscous[static_cast<std::size_t>(t) * 6 + k] = ev.s[k];
            }
            DSym3 sMaxwell = dsymScale(dsymAxpy(E, -1.0, ev), 2.0 * material.maxwellG1);
            sMaxwell = dsymAxpy(sMaxwell, material.maxwellLambda * trE, dsymIdentity());
            S = dsymAxpy(S, 1.0, sMaxwell);
        }

        // Nodal forces: f[ta[l]] -= F (S sv_l) V.
        for (int l = 0; l < 4; ++l)
        {
            double ssv[3];
            dsymApply(S, sv[l], ssv);
            for (int c = 0; c < 3; ++c)
            {
                const double fc = F[c][0] * ssv[0] + F[c][1] * ssv[1] + F[c][2] * ssv[2];
                tetForce[static_cast<std::size_t>(t) * 12 + l * 3 + c] = -fc * V;
            }
        }

        // Edge stiffness blocks: updateTangentMatrix, (M + N) V per edge.
        for (int j = 0; j < 6; ++j)
        {
            const unsigned char side = edgeSides[static_cast<std::size_t>(t) * 6 + j];
            const int k = side & 3;
            const int l = (side >> 2) & 3;
            const double* svk = sv[k];
            const double* svl = sv[l];
            double N[3][3];
            for (int m = 0; m < 3; ++m)
            {
                DSym3 h;
                for (int p = 0; p < 3; ++p)
                    for (int q = p; q < 3; ++q) h.s[symIndex(p, q)] = svl[p] * F[m][q] + F[m][p] * svl[q];
                DSym3 out = dsymZero();
                if (material.hasOgden) out = dsymAxpy(out, 1.0, ogdenTangent(og, material, h));
                if (material.hasMaxwell)
                {
                    // MaxwellFirstOrder::applyElasticityTensor: I trH lambda/2 + G1 H
                    const double trH = h.s[0] + h.s[2] + h.s[5];
                    out = dsymAxpy(out, material.maxwellG1, h);
                    out = dsymAxpy(out, trH * material.maxwellLambda / 2.0, dsymIdentity());
                }
                double osk[3];
                dsymApply(out, svk, osk);
                for (int u = 0; u < 3; ++u) N[m][u] = F[u][0] * osk[0] + F[u][1] * osk[1] + F[u][2] * osk[2];
            }
            double ssk[3];
            dsymApply(S, svk, ssk);
            const double productSD = ssk[0] * svl[0] + ssk[1] * svl[1] + ssk[2] * svl[2];
            double* block = tetEdgeBlocks + (static_cast<std::size_t>(t) * 6 + j) * 9;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) block[r * 3 + c] = (N[r][c] + (r == c ? productSD : 0.0)) * V;
        }
    }
}

// f_i = gravity mass_i g + sum of the tetrahedra's nodal forces (tetrahedron order).
__global__ void tissueGatherForcesKernel(
    const int vertexCount,
    const int* __restrict__ vertexTetStart,
    const int* __restrict__ vertexTetEntries,   // tet * 4 + local
    const double* __restrict__ tetForce,
    const double* __restrict__ gravityMass,
    const double g0, const double g1, const double g2,
    double* __restrict__ force)
{
    const int stride = gridDim.x * blockDim.x;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vertexCount; v += stride)
    {
        double f[3] = { g0 * gravityMass[v], g1 * gravityMass[v], g2 * gravityMass[v] };
        for (int p = vertexTetStart[v]; p < vertexTetStart[v + 1]; ++p)
        {
            const int e = vertexTetEntries[p];
            for (int c = 0; c < 3; ++c) f[c] += tetForce[static_cast<std::size_t>(e) * 3 + c];
        }
        for (int c = 0; c < 3; ++c) force[3 * v + c] = f[c];
    }
}

// DfDx of each edge: the sum of its tetrahedra's blocks (tetrahedron order).
__global__ void tissueGatherEdgesKernel(
    const int edgeCount,
    const int* __restrict__ edgeTetStart,
    const int* __restrict__ edgeTetEntries,     // tet * 6 + j
    const double* __restrict__ tetEdgeBlocks,
    double* __restrict__ edgeDfDx)              // edge * 9
{
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < edgeCount; e += stride)
    {
        double d[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        for (int p = edgeTetStart[e]; p < edgeTetStart[e + 1]; ++p)
        {
            const double* b = tetEdgeBlocks + static_cast<std::size_t>(edgeTetEntries[p]) * 9;
            for (int k = 0; k < 9; ++k) d[k] += b[k];
        }
        for (int k = 0; k < 9; ++k) edgeDfDx[static_cast<std::size_t>(e) * 9 + k] = d[k];
    }
}

// Per vertex: K's diagonal block (sum over incident edges: DfDx^T when the vertex
// is the edge's first end, DfDx when second), K v, M v, the projected RHS
// b = h (f + (h + rS) K v - rM M v), and A's diagonal block (Dirichlet applied).
__global__ void tissueVertexKernel(
    const int vertexCount,
    const int* __restrict__ vertexEdgeStart,
    const int* __restrict__ vertexEdgeEntries,  // edge * 2 + side (0: first end, 1: second end)
    const int* __restrict__ edges,              // edge * 2
    const double* __restrict__ edgeDfDx,
    const double* __restrict__ vertexMass,
    const double* __restrict__ edgeMass,
    const unsigned char* __restrict__ fixedDofs,
    const double* __restrict__ force,
    const float* __restrict__ vel,
    const double h, const double rayleighMass, const double rayleighStiffness,
    const double mFact, const double kFact,
    double* __restrict__ kv,                    // 3n (diagnostics)
    double* __restrict__ rhs,                   // 3n
    float* __restrict__ rhsFloat,               // 3n
    double* __restrict__ diagBlocks)            // vertex * 9: A's diagonal block
{
    const int stride = gridDim.x * blockDim.x;
    for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < vertexCount; a += stride)
    {
        double kdiag[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        double kvA[3] = { 0.0, 0.0, 0.0 };
        const double va[3] = { vel[3 * a], vel[3 * a + 1], vel[3 * a + 2] };
        double mv[3] = { vertexMass[a] * va[0], vertexMass[a] * va[1], vertexMass[a] * va[2] };
        for (int p = vertexEdgeStart[a]; p < vertexEdgeStart[a + 1]; ++p)
        {
            const int entry = vertexEdgeEntries[p];
            const int e = entry >> 1;
            const int side = entry & 1;
            const int other = edges[2 * e + (1 - side)];
            const double* d = edgeDfDx + static_cast<std::size_t>(e) * 9;
            // addDForce: df[v0] += DfDx^T (dx0 - dx1); df[v1] -= DfDx (dx0 - dx1).
            const double delta[3] = { va[0] - vel[3 * other], va[1] - vel[3 * other + 1], va[2] - vel[3 * other + 2] };
            for (int r = 0; r < 3; ++r)
            {
                double acc = 0.0;
                for (int c = 0; c < 3; ++c)
                {
                    const double drc = side == 0 ? d[c * 3 + r] : d[r * 3 + c];   // DfDx^T or DfDx
                    kdiag[r * 3 + c] += drc;
                    acc += drc * delta[c];
                }
                kvA[r] += acc;
            }
            const double em = edgeMass[e];
            for (int c = 0; c < 3; ++c) mv[c] += em * vel[3 * other + c];
        }
        const unsigned char fixedBits = fixedDofs[a];
        for (int c = 0; c < 3; ++c)
        {
            kv[3 * a + c] = kvA[c];
            double b = h * (force[3 * a + c] + (h + rayleighStiffness) * kvA[c] - rayleighMass * mv[c]);
            if ((fixedBits >> c) & 1u) b = 0.0;
            rhs[3 * a + c] = b;
            rhsFloat[3 * a + c] = static_cast<float>(b);
        }
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                double value = kFact * kdiag[r * 3 + c] + (r == c ? mFact * vertexMass[a] : 0.0);
                if (((fixedBits >> r) & 1u) || ((fixedBits >> c) & 1u)) value = (r == c && ((fixedBits >> r) & 1u)) ? 1.0 : 0.0;
                diagBlocks[static_cast<std::size_t>(a) * 9 + r * 3 + c] = value;
            }
        }
    }
}

// Per edge (p, q): A's off-diagonal blocks A_pq = mFact em I - kFact DfDx^T and
// A_qp = its transpose, with Dirichlet applied.
__global__ void tissueEdgeBlockKernel(
    const int edgeCount,
    const int* __restrict__ edges,
    const double* __restrict__ edgeDfDx,
    const double* __restrict__ edgeMass,
    const unsigned char* __restrict__ fixedDofs,
    const double mFact, const double kFact,
    double* __restrict__ edgeBlocks)            // edge * 9: A_pq
{
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < edgeCount; e += stride)
    {
        const int p = edges[2 * e];
        const int q = edges[2 * e + 1];
        const unsigned char fp = fixedDofs[p];
        const unsigned char fq = fixedDofs[q];
        const double* d = edgeDfDx + static_cast<std::size_t>(e) * 9;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                // K_pq = -DfDx^T  ->  A_pq = mFact em I + kFact K_pq
                double value = -kFact * d[c * 3 + r] + (r == c ? mFact * edgeMass[e] : 0.0);
                if (((fp >> r) & 1u) || ((fq >> c) & 1u)) value = 0.0;
                edgeBlocks[static_cast<std::size_t>(e) * 9 + r * 3 + c] = value;
            }
        }
    }
}

// Scatter the blocks into the dense single-precision matrix (column-major, both triangles).
__global__ void tissueDenseDiagonalKernel(const int vertexCount, const int n, const double* __restrict__ diagBlocks, float* __restrict__ dense)
{
    const int stride = gridDim.x * blockDim.x;
    for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < vertexCount; a += stride)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                dense[static_cast<std::size_t>(3 * a + c) * n + (3 * a + r)] = static_cast<float>(diagBlocks[static_cast<std::size_t>(a) * 9 + r * 3 + c]);
}

__global__ void tissueDenseEdgeKernel(const int edgeCount, const int n, const int* __restrict__ edges,
                                      const double* __restrict__ edgeBlocks, float* __restrict__ dense)
{
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < edgeCount; e += stride)
    {
        const int p = edges[2 * e];
        const int q = edges[2 * e + 1];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float value = static_cast<float>(edgeBlocks[static_cast<std::size_t>(e) * 9 + r * 3 + c]);
                dense[static_cast<std::size_t>(3 * q + c) * n + (3 * p + r)] = value;   // A_pq (r, c)
                dense[static_cast<std::size_t>(3 * p + r) * n + (3 * q + c)] = value;   // A_qp (c, r)
            }
        }
    }
}

// r = b - A dv (double), and a single-precision copy for the correction solve.
__global__ void tissueResidualKernel(
    const int vertexCount,
    const int* __restrict__ vertexEdgeStart,
    const int* __restrict__ vertexEdgeEntries,
    const int* __restrict__ edges,
    const double* __restrict__ diagBlocks,
    const double* __restrict__ edgeBlocks,
    const double* __restrict__ rhs,
    const double* __restrict__ dv,
    double* __restrict__ residual,
    float* __restrict__ residualFloat)
{
    const int stride = gridDim.x * blockDim.x;
    for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < vertexCount; a += stride)
    {
        double y[3];
        for (int r = 0; r < 3; ++r)
        {
            double acc = 0.0;
            for (int c = 0; c < 3; ++c) acc += diagBlocks[static_cast<std::size_t>(a) * 9 + r * 3 + c] * dv[3 * a + c];
            y[r] = acc;
        }
        for (int p = vertexEdgeStart[a]; p < vertexEdgeStart[a + 1]; ++p)
        {
            const int entry = vertexEdgeEntries[p];
            const int e = entry >> 1;
            const int side = entry & 1;
            const int other = edges[2 * e + (1 - side)];
            const double* blk = edgeBlocks + static_cast<std::size_t>(e) * 9;   // A_pq, p = first end
            for (int r = 0; r < 3; ++r)
            {
                double acc = 0.0;
                for (int c = 0; c < 3; ++c)
                {
                    const double arc = side == 0 ? blk[r * 3 + c] : blk[c * 3 + r];   // A_pq or A_qp = A_pq^T
                    acc += arc * dv[3 * other + c];
                }
                y[r] += acc;
            }
        }
        for (int r = 0; r < 3; ++r)
        {
            const double res = rhs[3 * a + r] - y[r];
            residual[3 * a + r] = res;
            residualFloat[3 * a + r] = static_cast<float>(res);
        }
    }
}

__global__ void tissueAccumulateKernel(const int n, const float* __restrict__ correction, double* __restrict__ dv)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dv[i] += static_cast<double>(correction[i]);
}

__global__ void tissueCopyToDoubleKernel(const int n, const float* __restrict__ in, double* __restrict__ out)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = static_cast<double>(in[i]);
}

// v_free = v + dv; x_free = x + h v_free (EulerImplicitSolver, v_multiop).
__global__ void tissueUpdateKernel(const int n, const float* __restrict__ x, const float* __restrict__ v,
                                   const double* __restrict__ dv, const double h,
                                   float* __restrict__ xFree, float* __restrict__ vFree)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        const float vNew = v[i] + static_cast<float>(dv[i]);
        vFree[i] = vNew;
        xFree[i] = x[i] + static_cast<float>(h) * vNew;
    }
}

// min over tetrahedra of det(F) (= volume / rest volume), and one vertex's position.
__global__ void tissueMonitorKernel(
    const int tetCount, const int4* __restrict__ tets, const double* __restrict__ shapeVectors,
    const double* __restrict__ volScale, const float* __restrict__ x, const int vertex, double* __restrict__ out)
{
    __shared__ double best[256];
    double local = 1e300;
    for (int t = threadIdx.x; t < tetCount; t += blockDim.x)
    {
        const int4 ta = tets[t];
        const int vid[4] = { ta.x, ta.y, ta.z, ta.w };
        double dp[3][3];
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < 3; ++c) dp[j][c] = static_cast<double>(x[3 * vid[j + 1] + c]) - static_cast<double>(x[3 * vid[0] + c]);
        const double area[3] = { dp[1][1] * dp[2][2] - dp[1][2] * dp[2][1],
                                 dp[1][2] * dp[2][0] - dp[1][0] * dp[2][2],
                                 dp[1][0] * dp[2][1] - dp[1][1] * dp[2][0] };
        const double J = (area[0] * dp[0][0] + area[1] * dp[0][1] + area[2] * dp[0][2]) * volScale[t];
        local = fmin(local, J);
    }
    best[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) best[threadIdx.x] = fmin(best[threadIdx.x], best[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[0] = best[0];
        for (int c = 0; c < 3; ++c) out[1 + c] = static_cast<double>(x[3 * vertex + c]);
    }
    (void)shapeVectors;
}

unsigned tissueBlocks(const int count)
{
    return static_cast<unsigned>(std::max(1, std::min((count + 255) / 256, 1024)));
}

} // namespace


namespace SofaGpuCollision::backend
{

struct TissueWorkspace
{
    cusolverDnHandle_t solver { nullptr };
    int vertexCount { 0 };
    int tetCount { 0 };
    int edgeCount { 0 };
    int n { 0 };
    TissueMaterialDevice material {};
    double gravity[3] { 0.0, 0.0, 0.0 };
    std::vector<unsigned char> fixedDofs;

    // Topology and rest state (device).
    int4* tets { nullptr };
    double* shapeVectors { nullptr };
    double* restVolume { nullptr };
    double* volScale { nullptr };
    unsigned char* edgeSides { nullptr };
    int* edges { nullptr };
    int* vertexTetStart { nullptr };
    int* vertexTetEntries { nullptr };
    int* edgeTetStart { nullptr };
    int* edgeTetEntries { nullptr };
    int* vertexEdgeStart { nullptr };
    int* vertexEdgeEntries { nullptr };
    double* vertexMass { nullptr };
    double* edgeMass { nullptr };
    double* gravityMass { nullptr };
    unsigned char* fixedDevice { nullptr };

    // Per-step state (device).
    double* slsViscous { nullptr };
    double* maxwellViscous { nullptr };
    double* tetForce { nullptr };
    double* tetEdgeBlocks { nullptr };
    double* tetJ { nullptr };
    double* force { nullptr };
    double* edgeDfDx { nullptr };
    double* kv { nullptr };
    double* rhs { nullptr };
    float* rhsFloat { nullptr };
    double* diagBlocks { nullptr };
    double* edgeBlocks { nullptr };
    double* dv { nullptr };
    double* residual { nullptr };
    float* correction { nullptr };
    float* dense { nullptr };
    float* factorWork { nullptr };  std::size_t factorWorkCapacity { 0 };
    int* info { nullptr };
    double* monitor { nullptr };
    bool factorized { false };
    bool stepped { false };

    ~TissueWorkspace()
    {
        if (solver) cusolverDnDestroy(solver);
        void* buffers[] = {
            tets, shapeVectors, restVolume, volScale, edgeSides, edges, vertexTetStart, vertexTetEntries,
            edgeTetStart, edgeTetEntries, vertexEdgeStart, vertexEdgeEntries, vertexMass, edgeMass, gravityMass,
            fixedDevice, slsViscous, maxwellViscous, tetForce, tetEdgeBlocks, tetJ, force, edgeDfDx, kv, rhs,
            rhsFloat, diagBlocks, edgeBlocks, dv, residual, correction, dense, factorWork, info, monitor };
        for (void* b : buffers)
        {
            if (b) cudaFree(b);
        }
    }
};

namespace
{
template <class T>
cudaError_t uploadVector(T*& device, const std::vector<T>& host)
{
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device), sizeof(T) * std::max<std::size_t>(host.size(), 1));
    if (err == cudaSuccess && !host.empty()) err = cudaMemcpy(device, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    return err;
}

template <class T>
cudaError_t allocateZeroed(T*& device, const std::size_t count)
{
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device), sizeof(T) * std::max<std::size_t>(count, 1));
    if (err == cudaSuccess) err = cudaMemset(device, 0, sizeof(T) * std::max<std::size_t>(count, 1));
    return err;
}
} // namespace

TissueWorkspace* createTissueWorkspace(const TissueSetup& setup, std::string& diagnostic)
{
    const int nv = setup.vertexCount;
    const int nt = static_cast<int>(setup.tetrahedra.size() / 4);
    const int ne = static_cast<int>(setup.edges.size() / 2);
    if (nv <= 0 || nt <= 0 || ne <= 0 ||
        setup.tetrahedronEdges.size() != static_cast<std::size_t>(nt) * 6 ||
        setup.tetrahedronEdgeSides.size() != static_cast<std::size_t>(nt) * 6 ||
        setup.restPositions.size() != static_cast<std::size_t>(nv) * 3 ||
        setup.vertexMass.size() != static_cast<std::size_t>(nv) || setup.edgeMass.size() != static_cast<std::size_t>(ne) ||
        setup.gravityMass.size() != static_cast<std::size_t>(nv) || setup.fixedDofs.size() != static_cast<std::size_t>(nv))
    {
        diagnostic = "Tissue setup has inconsistent sizes.";
        return nullptr;
    }

    // Rest information, as TetrahedronViscoHyperelasticityFEMForceField::createTetrahedronRestInformation.
    std::vector<double> shapeVectors(static_cast<std::size_t>(nt) * 12), restVolume(nt), volScale(nt);
    std::vector<int4> tets(nt);
    for (int t = 0; t < nt; ++t)
    {
        const int* ta = &setup.tetrahedra[static_cast<std::size_t>(t) * 4];
        tets[t] = make_int4(ta[0], ta[1], ta[2], ta[3]);
        double p[4][3];
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c) p[j][c] = setup.restPositions[static_cast<std::size_t>(ta[j]) * 3 + c];
        auto sub = [](const double* a, const double* b, double* out) { for (int c = 0; c < 3; ++c) out[c] = a[c] - b[c]; };
        auto cross = [](const double* a, const double* b, double* out) {
            out[0] = a[1] * b[2] - a[2] * b[1];
            out[1] = a[2] * b[0] - a[0] * b[2];
            out[2] = a[0] * b[1] - a[1] * b[0];
        };
        double e20[3], e30[3], e10[3], cr[3];
        sub(p[2], p[0], e20);
        sub(p[3], p[0], e30);
        sub(p[1], p[0], e10);
        cross(e20, e30, cr);
        const double volume = cr[0] * e10[0] + cr[1] * e10[1] + cr[2] * e10[2];   // 6 x signed volume
        volScale[t] = 1.0 / volume;
        restVolume[t] = std::fabs(volume / 6.0);
        for (int j = 0; j < 4; ++j)
        {
            double a[3], b[3], s[3];
            sub(p[(j + 2) % 4], p[(j + 1) % 4], a);
            sub(p[(j + 3) % 4], p[(j + 1) % 4], b);
            cross(a, b, s);
            const double sign = (j % 2) ? 1.0 : -1.0;
            for (int c = 0; c < 3; ++c) shapeVectors[static_cast<std::size_t>(t) * 12 + j * 3 + c] = sign * s[c] / volume;
        }
    }

    // Gather lists, each in tetrahedron / edge order.
    std::vector<int> vertexTetStart(nv + 1, 0), edgeTetStart(ne + 1, 0), vertexEdgeStart(nv + 1, 0);
    for (int t = 0; t < nt; ++t)
    {
        for (int j = 0; j < 4; ++j) ++vertexTetStart[setup.tetrahedra[static_cast<std::size_t>(t) * 4 + j] + 1];
        for (int j = 0; j < 6; ++j) ++edgeTetStart[setup.tetrahedronEdges[static_cast<std::size_t>(t) * 6 + j] + 1];
    }
    for (int e = 0; e < ne; ++e)
    {
        ++vertexEdgeStart[setup.edges[2 * e] + 1];
        ++vertexEdgeStart[setup.edges[2 * e + 1] + 1];
    }
    for (int v = 0; v < nv; ++v)
    {
        vertexTetStart[v + 1] += vertexTetStart[v];
        vertexEdgeStart[v + 1] += vertexEdgeStart[v];
    }
    for (int e = 0; e < ne; ++e) edgeTetStart[e + 1] += edgeTetStart[e];
    std::vector<int> vertexTetEntries(vertexTetStart[nv]), edgeTetEntries(edgeTetStart[ne]), vertexEdgeEntries(vertexEdgeStart[nv]);
    {
        std::vector<int> fillV(vertexTetStart.begin(), vertexTetStart.end() - 1);
        std::vector<int> fillE(edgeTetStart.begin(), edgeTetStart.end() - 1);
        for (int t = 0; t < nt; ++t)
        {
            for (int j = 0; j < 4; ++j) vertexTetEntries[fillV[setup.tetrahedra[static_cast<std::size_t>(t) * 4 + j]]++] = t * 4 + j;
            for (int j = 0; j < 6; ++j) edgeTetEntries[fillE[setup.tetrahedronEdges[static_cast<std::size_t>(t) * 6 + j]]++] = t * 6 + j;
        }
        std::vector<int> fillVE(vertexEdgeStart.begin(), vertexEdgeStart.end() - 1);
        for (int e = 0; e < ne; ++e)
        {
            vertexEdgeEntries[fillVE[setup.edges[2 * e]]++] = e * 2;
            vertexEdgeEntries[fillVE[setup.edges[2 * e + 1]]++] = e * 2 + 1;
        }
    }

    auto* ws = new TissueWorkspace();
    ws->vertexCount = nv;
    ws->tetCount = nt;
    ws->edgeCount = ne;
    ws->n = 3 * nv;
    ws->fixedDofs = setup.fixedDofs;
    ws->material.mu1 = setup.material.ogdenMu1;
    ws->material.alpha1 = setup.material.ogdenAlpha1;
    ws->material.slsG1 = setup.material.ogdenG1;
    ws->material.slsTau = setup.material.ogdenTau;
    ws->material.k0 = setup.material.ogdenK0;
    ws->material.maxwellG1 = setup.material.maxwellG1;
    ws->material.maxwellTau = setup.material.maxwellTau;
    ws->material.maxwellLambda = setup.material.maxwellLambda;
    ws->material.hasOgden = setup.material.hasOgden ? 1 : 0;
    ws->material.hasMaxwell = setup.material.hasMaxwell ? 1 : 0;
    ws->material.ogdenSofaBasis = setup.material.ogdenSofaEigenvectors ? 1 : 0;
    for (int c = 0; c < 3; ++c) ws->gravity[c] = setup.gravity[c];
    if (cusolverDnCreate(&ws->solver) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cuSOLVER could not be initialised.";
        delete ws;
        return nullptr;
    }

    const std::size_t n = static_cast<std::size_t>(ws->n);
    cudaError_t err = uploadVector(ws->tets, tets);
    if (err == cudaSuccess) err = uploadVector(ws->shapeVectors, shapeVectors);
    if (err == cudaSuccess) err = uploadVector(ws->restVolume, restVolume);
    if (err == cudaSuccess) err = uploadVector(ws->volScale, volScale);
    if (err == cudaSuccess) err = uploadVector(ws->edgeSides, setup.tetrahedronEdgeSides);
    if (err == cudaSuccess) err = uploadVector(ws->edges, setup.edges);
    if (err == cudaSuccess) err = uploadVector(ws->vertexTetStart, vertexTetStart);
    if (err == cudaSuccess) err = uploadVector(ws->vertexTetEntries, vertexTetEntries);
    if (err == cudaSuccess) err = uploadVector(ws->edgeTetStart, edgeTetStart);
    if (err == cudaSuccess) err = uploadVector(ws->edgeTetEntries, edgeTetEntries);
    if (err == cudaSuccess) err = uploadVector(ws->vertexEdgeStart, vertexEdgeStart);
    if (err == cudaSuccess) err = uploadVector(ws->vertexEdgeEntries, vertexEdgeEntries);
    if (err == cudaSuccess) err = uploadVector(ws->vertexMass, setup.vertexMass);
    if (err == cudaSuccess) err = uploadVector(ws->edgeMass, setup.edgeMass);
    if (err == cudaSuccess) err = uploadVector(ws->gravityMass, setup.gravityMass);
    if (err == cudaSuccess) err = uploadVector(ws->fixedDevice, setup.fixedDofs);
    if (err == cudaSuccess) err = allocateZeroed(ws->slsViscous, static_cast<std::size_t>(nt) * 6);
    if (err == cudaSuccess) err = allocateZeroed(ws->maxwellViscous, static_cast<std::size_t>(nt) * 6);
    if (err == cudaSuccess) err = allocateZeroed(ws->tetForce, static_cast<std::size_t>(nt) * 12);
    if (err == cudaSuccess) err = allocateZeroed(ws->tetEdgeBlocks, static_cast<std::size_t>(nt) * 54);
    if (err == cudaSuccess) err = allocateZeroed(ws->tetJ, static_cast<std::size_t>(nt));
    if (err == cudaSuccess) err = allocateZeroed(ws->force, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->edgeDfDx, static_cast<std::size_t>(ne) * 9);
    if (err == cudaSuccess) err = allocateZeroed(ws->kv, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->rhs, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->rhsFloat, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->diagBlocks, static_cast<std::size_t>(nv) * 9);
    if (err == cudaSuccess) err = allocateZeroed(ws->edgeBlocks, static_cast<std::size_t>(ne) * 9);
    if (err == cudaSuccess) err = allocateZeroed(ws->dv, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->residual, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->correction, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->dense, n * n);
    if (err == cudaSuccess) err = allocateZeroed(ws->info, 1);
    if (err == cudaSuccess) err = allocateZeroed(ws->monitor, 4);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue buffers: ") + cudaGetErrorString(err) +
                     " (the dense matrix alone needs " + std::to_string(n * n * 4 / (1024 * 1024)) + " MB)";
        delete ws;
        return nullptr;
    }
    diagnostic.clear();
    return ws;
}

void destroyTissueWorkspace(TissueWorkspace* workspace)
{
    delete workspace;
}

bool tissueFreeMotion(
    TissueWorkspace* ws,
    const void* x,
    const void* v,
    void* xFree,
    void* vFree,
    const TissueStepConfig& config,
    TissueTimings* timings,
    std::string& diagnostic)
{
    if (timings != nullptr) *timings = TissueTimings {};
    if (ws == nullptr) { diagnostic = "No tissue workspace."; return false; }
    if (x == nullptr || v == nullptr || xFree == nullptr || vFree == nullptr)
    {
        diagnostic = "Tissue step: missing state pointers.";
        return false;
    }
    ws->factorized = false;
    ws->stepped = false;
    const float* xf = static_cast<const float*>(x);
    const float* vf = static_cast<const float*>(v);
    const int nv = ws->vertexCount;
    const int n = ws->n;
    const double h = config.dt;
    const double tr = 1.0;
    const double mFact = 1.0 + tr * h * config.rayleighMass;
    const double kFact = -tr * h * (tr * h + config.rayleighStiffness);

    // 1. material: stresses, forces, edge stiffness.
    {
        ConstraintEventTimer timer(timings != nullptr);
        tissueMaterialKernel<<<tissueBlocks(ws->tetCount), 128>>>(
            ws->tetCount, ws->tets, ws->shapeVectors, ws->restVolume, ws->volScale, ws->edgeSides, xf, ws->material, h,
            ws->slsViscous, ws->maxwellViscous, ws->tetForce, ws->tetEdgeBlocks, ws->tetJ);
        tissueGatherForcesKernel<<<tissueBlocks(nv), 256>>>(
            nv, ws->vertexTetStart, ws->vertexTetEntries, ws->tetForce, ws->gravityMass,
            ws->gravity[0], ws->gravity[1], ws->gravity[2], ws->force);
        tissueGatherEdgesKernel<<<tissueBlocks(ws->edgeCount), 256>>>(
            ws->edgeCount, ws->edgeTetStart, ws->edgeTetEntries, ws->tetEdgeBlocks, ws->edgeDfDx);
        if (timings != nullptr) timings->materialMs = timer.finish();
    }

    // 2. RHS and the system matrix (blocks in double, dense copy in single precision).
    {
        ConstraintEventTimer timer(timings != nullptr);
        tissueVertexKernel<<<tissueBlocks(nv), 256>>>(
            nv, ws->vertexEdgeStart, ws->vertexEdgeEntries, ws->edges, ws->edgeDfDx, ws->vertexMass, ws->edgeMass,
            ws->fixedDevice, ws->force, vf, h, config.rayleighMass, config.rayleighStiffness, mFact, kFact,
            ws->kv, ws->rhs, ws->rhsFloat, ws->diagBlocks);
        tissueEdgeBlockKernel<<<tissueBlocks(ws->edgeCount), 256>>>(
            ws->edgeCount, ws->edges, ws->edgeDfDx, ws->edgeMass, ws->fixedDevice, mFact, kFact, ws->edgeBlocks);
        cudaMemsetAsync(ws->dense, 0, sizeof(float) * static_cast<std::size_t>(n) * n);
        tissueDenseDiagonalKernel<<<tissueBlocks(nv), 256>>>(nv, n, ws->diagBlocks, ws->dense);
        tissueDenseEdgeKernel<<<tissueBlocks(ws->edgeCount), 256>>>(ws->edgeCount, n, ws->edges, ws->edgeBlocks, ws->dense);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Tissue assembly: ") + cudaGetErrorString(err);
            return false;
        }
        if (timings != nullptr) timings->assembleMs = timer.finish();
    }

    // 3. Cholesky factorisation.
    {
        ConstraintEventTimer timer(timings != nullptr);
        int workSize = 0;
        if (cusolverDnSpotrf_bufferSize(ws->solver, CUBLAS_FILL_MODE_LOWER, n, ws->dense, n, &workSize) != CUSOLVER_STATUS_SUCCESS)
        {
            diagnostic = "cusolverDnSpotrf_bufferSize failed.";
            return false;
        }
        cudaError_t err = ensureDeviceBuffer(ws->factorWork, ws->factorWorkCapacity, static_cast<std::size_t>(std::max(workSize, 1)));
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Tissue factorisation workspace: ") + cudaGetErrorString(err);
            return false;
        }
        if (cusolverDnSpotrf(ws->solver, CUBLAS_FILL_MODE_LOWER, n, ws->dense, n, ws->factorWork, workSize, ws->info) != CUSOLVER_STATUS_SUCCESS)
        {
            diagnostic = "cusolverDnSpotrf failed to launch.";
            return false;
        }
        int info = 0;
        err = cudaMemcpy(&info, ws->info, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess || info != 0)
        {
            diagnostic = err != cudaSuccess ? std::string("Tissue factorisation: ") + cudaGetErrorString(err)
                                            : "The tissue system matrix is not positive definite (Cholesky stopped at row " +
                                                  std::to_string(info) + ").";
            return false;
        }
        ws->factorized = true;
        if (timings != nullptr) timings->factorizeMs = timer.finish();
    }

    // 4. Solve, refine in double, update.
    {
        ConstraintEventTimer timer(timings != nullptr);
        if (cusolverDnSpotrs(ws->solver, CUBLAS_FILL_MODE_LOWER, n, 1, ws->dense, n, ws->rhsFloat, n, ws->info) != CUSOLVER_STATUS_SUCCESS)
        {
            diagnostic = "cusolverDnSpotrs failed to launch.";
            return false;
        }
        tissueCopyToDoubleKernel<<<tissueBlocks(n), 256>>>(n, ws->rhsFloat, ws->dv);
        for (int step = 0; step < config.refinementSteps; ++step)
        {
            tissueResidualKernel<<<tissueBlocks(nv), 256>>>(
                nv, ws->vertexEdgeStart, ws->vertexEdgeEntries, ws->edges, ws->diagBlocks, ws->edgeBlocks,
                ws->rhs, ws->dv, ws->residual, ws->correction);
            if (cusolverDnSpotrs(ws->solver, CUBLAS_FILL_MODE_LOWER, n, 1, ws->dense, n, ws->correction, n, ws->info) != CUSOLVER_STATUS_SUCCESS)
            {
                diagnostic = "cusolverDnSpotrs (refinement) failed to launch.";
                return false;
            }
            tissueAccumulateKernel<<<tissueBlocks(n), 256>>>(n, ws->correction, ws->dv);
        }
        tissueUpdateKernel<<<tissueBlocks(n), 256>>>(n, xf, vf, ws->dv, h, static_cast<float*>(xFree), static_cast<float*>(vFree));
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Tissue solve: ") + cudaGetErrorString(err);
            return false;
        }
        if (timings != nullptr) timings->solveMs = timer.finish();
    }
    ws->stepped = true;
    diagnostic.clear();
    return true;
}

const float* tissueFactor(const TissueWorkspace* ws, int& n)
{
    if (ws == nullptr || !ws->factorized)
    {
        n = 0;
        return nullptr;
    }
    n = ws->n;
    return ws->dense;
}

bool tissueMonitor(TissueWorkspace* ws, const void* x, const int vertex, double& minVolumeRatio, double position[3],
                   std::string& diagnostic)
{
    if (ws == nullptr || x == nullptr || vertex < 0 || vertex >= ws->vertexCount)
    {
        diagnostic = "Tissue monitor: bad input.";
        return false;
    }
    tissueMonitorKernel<<<1, 256>>>(ws->tetCount, ws->tets, ws->shapeVectors, ws->volScale,
                                    static_cast<const float*>(x), vertex, ws->monitor);
    double host[4];
    const cudaError_t err = cudaMemcpy(host, ws->monitor, sizeof(host), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue monitor: ") + cudaGetErrorString(err);
        return false;
    }
    minVolumeRatio = host[0];
    for (int c = 0; c < 3; ++c) position[c] = host[1 + c];
    diagnostic.clear();
    return true;
}

bool downloadTissueStep(TissueWorkspace* ws, const bool withMatrix, TissueStepSnapshot& snapshot, std::string& diagnostic)
{
    snapshot = TissueStepSnapshot {};
    if (ws == nullptr || !ws->stepped) { diagnostic = "No tissue step to download."; return false; }
    const std::size_t n = static_cast<std::size_t>(ws->n);
    snapshot.force.resize(n);
    snapshot.stiffnessTimesVelocity.resize(n);
    snapshot.rhs.resize(n);
    snapshot.dv.resize(n);
    cudaError_t err = cudaMemcpy(snapshot.force.data(), ws->force, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(snapshot.stiffnessTimesVelocity.data(), ws->kv, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(snapshot.rhs.data(), ws->rhs, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(snapshot.dv.data(), ws->dv, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && withMatrix)
    {
        // A in CSR: per DOF row, the vertex's own block then its edge blocks (edge order).
        const int nv = ws->vertexCount;
        const int ne = ws->edgeCount;
        std::vector<double> diag(static_cast<std::size_t>(nv) * 9), blocks(static_cast<std::size_t>(ne) * 9);
        std::vector<int> edges(static_cast<std::size_t>(ne) * 2);
        err = cudaMemcpy(diag.data(), ws->diagBlocks, sizeof(double) * diag.size(), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) err = cudaMemcpy(blocks.data(), ws->edgeBlocks, sizeof(double) * blocks.size(), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) err = cudaMemcpy(edges.data(), ws->edges, sizeof(int) * edges.size(), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess)
        {
            std::vector<std::vector<std::pair<int, int>>> neighbours(nv);   // (other vertex, edge*2+side)
            for (int e = 0; e < ne; ++e)
            {
                neighbours[edges[2 * e]].push_back({ edges[2 * e + 1], e * 2 });
                neighbours[edges[2 * e + 1]].push_back({ edges[2 * e], e * 2 + 1 });
            }
            snapshot.rowPtr.assign(n + 1, 0);
            for (int a = 0; a < nv; ++a)
            {
                std::sort(neighbours[a].begin(), neighbours[a].end());
                for (int r = 0; r < 3; ++r) snapshot.rowPtr[3 * a + r + 1] = 3 * static_cast<int>(1 + neighbours[a].size());
            }
            for (std::size_t i = 0; i < n; ++i) snapshot.rowPtr[i + 1] += snapshot.rowPtr[i];
            snapshot.columns.resize(static_cast<std::size_t>(snapshot.rowPtr[n]));
            snapshot.values.resize(static_cast<std::size_t>(snapshot.rowPtr[n]));
            for (int a = 0; a < nv; ++a)
            {
                for (int r = 0; r < 3; ++r)
                {
                    int p = snapshot.rowPtr[3 * a + r];
                    // Columns in increasing order: the diagonal block among the sorted neighbours.
                    std::vector<std::pair<int, int>> order;           // (vertex, edge * 2 + side); -1 = diagonal
                    order.push_back({ a, -1 });
                    for (const auto& nb : neighbours[a]) order.push_back({ nb.first, nb.second });
                    std::sort(order.begin(), order.end());
                    for (const auto& o : order)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            double value;
                            if (o.second < 0) value = diag[static_cast<std::size_t>(a) * 9 + r * 3 + c];
                            else
                            {
                                const int e = o.second >> 1;
                                const int side = o.second & 1;
                                value = side == 0 ? blocks[static_cast<std::size_t>(e) * 9 + r * 3 + c]
                                                  : blocks[static_cast<std::size_t>(e) * 9 + c * 3 + r];
                            }
                            snapshot.columns[p] = 3 * o.first + c;
                            snapshot.values[p] = value;
                            ++p;
                        }
                    }
                }
            }
        }
    }
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue step download: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

} // namespace SofaGpuCollision::backend
