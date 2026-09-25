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
// eigenvectors, orthonormal even where eigenvalues coincide. Only functions of C
// (C^p) are used, which do not depend on the eigenvectors' order or sign. (SOFA
// v25.12's core Ogden takes them from Eigen's general EigenSolver, which does not
// keep them orthogonal where two eigenvalues coincide; its V D V^T is then wrong:
// README 7.9, patches/SOFA-Ogden-orthonormal-eigenvectors.patch.)
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

// SOFA core hyperelastic materials (TetrahedronHyperelasticityFEMForceField).
enum : int
{
    CoreNone = 0,
    CoreOgden = 1,              // mu1 alpha1 k0
    CoreNeoHookean = 2,         // mu lambda
    CoreStableNeoHookean = 3,   // mu lambda
    CoreStVenantKirchhoff = 4,  // mu lambda
    CoreMooneyRivlin = 5,       // c1 c2 k0
};

struct TissueMaterialDevice
{
    double mu1, alpha1, slsG1, slsTau, k0;          // SofaViscoElastic SLSOgdenFirstOrder
    double maxwellG1, maxwellTau, maxwellLambda;    // SofaViscoElastic MaxwellFirstOrder
    int hasOgden, hasMaxwell;
    int ogdenSofaBasis;   // 1: C^p as SofaViscoElastic computes it (sofaOgdenBasis), 0: exact
    int core;             // CoreNone or a SOFA core material
    double coreParams[4];
    int ogdenRobustTangent;   // core Ogden: 1 divided differences without cancellation, 0 as SOFA
};

// Storage index (SOFA MatSym order, also its "Voigt" order) -> (i, j).
__device__ __forceinline__ void symPair(const int m, int& i, int& j)
{
    i = (m == 0 || m == 1 || m == 3) ? 0 : (m == 2 || m == 4) ? 1 : 2;
    j = (m == 0) ? 0 : (m == 1 || m == 2) ? 1 : 2;
}

// Everything a SOFA core material's stiffness needs, computed once per tetrahedron.
struct CoreTangentState
{
    DSym3 C;
    DSym3 cinv;
    double J, logJ, I1, I2;
    double T[36];   // Ogden: its elasticity tensor dS/dC, rows and columns in MatSym storage order
};

// A SOFA core material's second Piola-Kirchhoff stress, term for term as its
// deriveSPKTensor (and, for Ogden, the elasticity tensor of its ElasticityTensor).
// (a^p - b^p) / (a - b) for a, b > 0, without the cancellation of the plain quotient
// when a and b are close; tends to p b^(p-1) as a -> b.
__device__ double powerDividedDifference(const double a, const double b, const double p)
{
    const double d = (a - b) / b;
    const double base = pow(b, p - 1.0);
    if (d == 0.0) return p * base;
    return base * expm1(p * log1p(d)) / d;
}

__device__ DSym3 coreStress(const TissueMaterialDevice& m, const DSym3& C, const double J, CoreTangentState& st)
{
    st.C = C;
    st.cinv = dsymInverse(C);
    st.J = J;
    st.logJ = log(J);
    st.I1 = C.s[0] + C.s[2] + C.s[5];
    const double I1square = C.s[0] * C.s[0] + C.s[2] * C.s[2] + C.s[5] * C.s[5] +
                            2.0 * (C.s[1] * C.s[1] + C.s[3] * C.s[3] + C.s[4] * C.s[4]);
    st.I2 = (st.I1 * st.I1 - I1square) / 2.0;
    const DSym3 I = dsymIdentity();
    const double* p = m.coreParams;
    switch (m.core)
    {
    case CoreNeoHookean:   // mu I + (lambda lnJ - mu) C^-1
        return dsymAxpy(dsymScale(I, p[0]), p[1] * st.logJ - p[0], st.cinv);
    case CoreStableNeoHookean:
    {
        const double alpha = 1.0 + p[0] / (p[1] + p[0]);
        return dsymAxpy(dsymScale(I, p[0]), (p[1] + p[0]) * J * (J - alpha), st.cinv);
    }
    case CoreStVenantKirchhoff:
    {
        const double trE = 0.5 * (st.I1 - 3.0);
        return dsymAxpy(dsymScale(I, p[1] * trE - p[0]), p[0], C);
    }
    case CoreMooneyRivlin:
    {
        const double a = 2.0 * p[0] * pow(J, -2.0 / 3.0);
        const double b = 2.0 * p[1] * pow(J, -4.0 / 3.0);
        DSym3 S = dsymScale(dsymAxpy(I, -st.I1 / 3.0, st.cinv), a);
        DSym3 second = dsymAxpy(dsymAxpy(dsymScale(st.cinv, -2.0 * st.I2 / 3.0), st.I1, I), -1.0, C);
        S = dsymAxpy(S, b, second);
        return dsymAxpy(S, p[2] * st.logJ, st.cinv);
    }
    case CoreOgden:
    {
        const double mu1 = p[0], alpha1 = p[1], k0 = p[2];
        double lambda[3], V[3][3];
        dsymEigen(C, lambda, V);
        const double aBy2 = alpha1 / 2.0;
        const double aBy2Minus1 = aBy2 - 1.0;
        const double aBy2Minus2 = aBy2 - 2.0;
        const double FJ = pow(J, -alpha1 / 3.0);
        const double trCa = pow(lambda[0], aBy2) + pow(lambda[1], aBy2) + pow(lambda[2], aBy2);
        const DSym3 ca1 = dsymPower(lambda, V, aBy2Minus1);
        DSym3 S = dsymScale(ca1, FJ * mu1 / alpha1);
        S = dsymAxpy(S, -FJ * mu1 / (3.0 * alpha1) * trCa, st.cinv);
        S = dsymAxpy(S, k0 * st.logJ, st.cinv);
        // Ogden::ElasticityTensor, term for term. coef[I][I] = (alpha/2-1) lambda_I^(alpha/2-2);
        // coef[I][J] (I != J) = SOFA's coefRot, the divided difference of lambda^(alpha/2-1)
        // between two eigenvalues of C. SOFA takes the plain quotient unless the two are
        // exactly equal (within 2.2e-16): when two principal stretches are equal up to
        // rounding (uniaxial states, e.g. confined compression), its numerator is rounding
        // noise and the stiffness garbage (SOFA's CPU run went NaN in that test).
        // ogdenRobustTangent computes it without cancellation; else as SOFA does.
        const double c = FJ * mu1 / alpha1;
        double coef[3][3];
        for (int eI = 0; eI < 3; ++eI)
        {
            coef[eI][eI] = aBy2Minus1 * pow(lambda[eI], aBy2Minus2);
            for (int eJ = 0; eJ < 3; ++eJ)
            {
                if (eJ == eI) continue;
                if (m.ogdenRobustTangent)
                {
                    coef[eI][eJ] = powerDividedDifference(lambda[eI], lambda[eJ], aBy2Minus1);
                }
                else
                {
                    const bool degenerate = fabs(lambda[eI] - lambda[eJ]) < 2.220446049250313e-16;
                    coef[eI][eJ] = degenerate ? coef[eI][eI]
                                              : (pow(lambda[eI], aBy2Minus1) - pow(lambda[eJ], aBy2Minus1)) / (lambda[eI] - lambda[eJ]);
                }
            }
        }
        for (int mm = 0; mm < 6; ++mm)
        {
            int i, j;
            symPair(mm, i, j);
            for (int nn = 0; nn < 6; ++nn)
            {
                int k, l;
                symPair(nn, k, l);
                double t = 0.0;
                for (int eI = 0; eI < 3; ++eI)
                {
                    t += c * coef[eI][eI] * V[i][eI] * V[j][eI] * V[k][eI] * V[l][eI];
                    for (int eJ = 0; eJ < 3; ++eJ)
                    {
                        if (eJ == eI) continue;
                        t += 0.5 * c * coef[eI][eJ] *
                             (V[i][eI] * V[j][eJ] * V[k][eJ] * V[l][eI] + V[i][eI] * V[j][eJ] * V[k][eI] * V[l][eJ]);
                    }
                }
                t += -FJ * mu1 / 6.0 * (ca1.at(i, j) * st.cinv.at(l, k) + ca1.at(k, l) * st.cinv.at(j, i))
                     + FJ * mu1 * trCa / 18.0 * st.cinv.at(j, i) * st.cinv.at(l, k)
                     + FJ * mu1 / (6.0 * alpha1) * trCa * (st.cinv.at(k, i) * st.cinv.at(l, j) + st.cinv.at(l, i) * st.cinv.at(k, j));
                t += 0.5 * k0 * st.cinv.at(j, i) * st.cinv.at(l, k)
                     - 0.5 * k0 * st.logJ * (st.cinv.at(j, k) * st.cinv.at(l, i) + st.cinv.at(j, l) * st.cinv.at(k, i));
                st.T[mm * 6 + nn] = t;
            }
        }
        return S;
    }
    default:
        return dsymZero();
    }
}

// A SOFA core material's applyElasticityTensor: dS/dC : H.
__device__ DSym3 coreTangent(const TissueMaterialDevice& m, const CoreTangentState& st, const DSym3& h)
{
    const double* p = m.coreParams;
    const DSym3 I = dsymIdentity();
    switch (m.core)
    {
    case CoreNeoHookean:
    {
        const double trHC = dsymContract(h, st.cinv);
        const DSym3 first = dsymSandwich(st.cinv, h);
        return dsymAxpy(dsymScale(first, p[0] - p[1] * st.logJ), p[1] * trHC / 2.0, st.cinv);
    }
    case CoreStableNeoHookean:
    {
        const double alpha = 1.0 + p[0] / (p[1] + p[0]);
        const double trHC = dsymContract(h, st.cinv);
        const DSym3 first = dsymSandwich(st.cinv, h);
        const DSym3 inner = dsymAxpy(dsymScale(first, -2.0 * st.J * (st.J - alpha)), st.J * (2.0 * st.J - alpha) * trHC, st.cinv);
        return dsymScale(inner, 0.5 * (p[1] + p[0]));
    }
    case CoreStVenantKirchhoff:
    {
        const double trH = h.s[0] + h.s[2] + h.s[5];
        return dsymAxpy(dsymScale(I, trH * p[1] / 2.0), p[0], h);
    }
    case CoreMooneyRivlin:
    {
        const double a = 2.0 * p[0] * pow(st.J, -2.0 / 3.0);
        const double b = 2.0 * p[1] * pow(st.J, -4.0 / 3.0);
        const double trHCinv = dsymContract(h, st.cinv);   // SOFA's _trHC
        const double trHC = dsymContract(h, st.C);
        const double trH = h.s[0] + h.s[2] + h.s[5];
        const DSym3 first = dsymSandwich(st.cinv, h);
        DSym3 t1 = dsymScale(dsymAxpy(I, -st.I1 / 3.0, st.cinv), -trHCinv / 3.0);
        t1 = dsymAxpy(t1, st.I1 / 3.0, first);
        t1 = dsymAxpy(t1, -trH / 3.0, st.cinv);
        DSym3 inner = dsymAxpy(dsymAxpy(dsymScale(st.cinv, -2.0 * st.I2 / 3.0), st.I1, I), -1.0, st.C);
        DSym3 t2 = dsymScale(inner, -2.0 * trHCinv / 3.0);
        t2 = dsymAxpy(t2, 2.0 * st.I2 / 3.0, first);
        t2 = dsymAxpy(t2, -2.0 * (st.I1 * trH - trHC) / 3.0, st.cinv);
        t2 = dsymAxpy(t2, trH, I);
        t2 = dsymAxpy(t2, -1.0, h);
        DSym3 out = dsymAxpy(dsymScale(t1, a), b, t2);
        out = dsymAxpy(out, trHCinv * p[2] / 2.0, st.cinv);
        return dsymAxpy(out, -p[2] * st.logJ, first);
    }
    case CoreOgden:
    {
        // Ogden::applyElasticityTensor: (2T with off-diagonal columns doubled) h / 2.
        DSym3 out;
        for (int mm = 0; mm < 6; ++mm)
        {
            double acc = 0.0;
            for (int nn = 0; nn < 6; ++nn)
                acc += st.T[mm * 6 + nn] * h.s[nn] * ((nn == 1 || nn == 3 || nn == 4) ? 2.0 : 1.0);
            out.s[mm] = acc;
        }
        return out;
    }
    default:
        return dsymZero();
    }
}

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
        // A SOFA core hyperelastic material (TetrahedronHyperelasticityFEMForceField).
        CoreTangentState core;
        if (material.core != CoreNone) S = dsymAxpy(S, 1.0, coreStress(material, C, J, core));
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
                if (material.core != CoreNone) out = dsymAxpy(out, 1.0, coreTangent(material, core, h));
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

// f_i = gravity mass_i g + sum of the tetrahedra's nodal forces (tetrahedron order)
// + the external force (ConstantForceField; null: none).
__global__ void tissueGatherForcesKernel(
    const int vertexCount,
    const int* __restrict__ vertexTetStart,
    const int* __restrict__ vertexTetEntries,   // tet * 4 + local
    const double* __restrict__ tetForce,
    const double* __restrict__ gravityMass,
    const double g0, const double g1, const double g2,
    const double* __restrict__ externalForce,
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
        if (externalForce != nullptr)
            for (int c = 0; c < 3; ++c) f[c] += externalForce[3 * v + c];
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
// The dense single-precision A in the solver's order: vertex a's block at rows and
// columns 3 perm[a] .. 3 perm[a] + 2 (perm null: the natural order).
__global__ void tissueDenseDiagonalKernel(const int vertexCount, const int n, const int* __restrict__ perm,
                                          const double* __restrict__ diagBlocks, float* __restrict__ dense)
{
    const int stride = gridDim.x * blockDim.x;
    for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < vertexCount; a += stride)
    {
        const int pa = perm != nullptr ? perm[a] : a;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                dense[static_cast<std::size_t>(3 * pa + c) * n + (3 * pa + r)] = static_cast<float>(diagBlocks[static_cast<std::size_t>(a) * 9 + r * 3 + c]);
    }
}

__global__ void tissueDenseEdgeKernel(const int edgeCount, const int n, const int* __restrict__ perm, const int* __restrict__ edges,
                                      const double* __restrict__ edgeBlocks, float* __restrict__ dense)
{
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < edgeCount; e += stride)
    {
        const int p = perm != nullptr ? perm[edges[2 * e]] : edges[2 * e];
        const int q = perm != nullptr ? perm[edges[2 * e + 1]] : edges[2 * e + 1];
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

// A(i, j) = A(j, i) = value into the block-tridiagonal band storage (solver order):
// the lower entry (r >= c) goes into D_k (both triangles, for readability of the
// block) or into S_k, the block below D_k. With blockWidth >= bandwidth every entry
// is in one of the two.
__device__ __forceinline__ void tissueBandStore(float* __restrict__ blocks, const int w, const int i, const int j,
                                                const float value)
{
    const int r = i > j ? i : j;
    const int c = i > j ? j : i;
    const int kc = c / w;
    const int kr = r / w;
    const std::size_t w2 = static_cast<std::size_t>(w) * w;
    if (kr == kc)
    {
        float* d = blocks + 2 * kc * w2;
        d[static_cast<std::size_t>(c - kc * w) * w + (r - kc * w)] = value;
        d[static_cast<std::size_t>(r - kc * w) * w + (c - kc * w)] = value;
    }
    else
    {
        float* s = blocks + (2 * kc + 1) * w2;
        s[static_cast<std::size_t>(c - kc * w) * w + (r - kr * w)] = value;
    }
}

__global__ void tissueBandDiagonalKernel(const int vertexCount, const int w, const int* __restrict__ perm,
                                         const double* __restrict__ diagBlocks, float* __restrict__ blocks)
{
    const int stride = gridDim.x * blockDim.x;
    for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < vertexCount; a += stride)
    {
        const int pa = perm[a];
        for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c)
                tissueBandStore(blocks, w, 3 * pa + r, 3 * pa + c, static_cast<float>(diagBlocks[static_cast<std::size_t>(a) * 9 + r * 3 + c]));
    }
}

__global__ void tissueBandEdgeKernel(const int edgeCount, const int w, const int* __restrict__ perm, const int* __restrict__ edges,
                                     const double* __restrict__ edgeBlocks, float* __restrict__ blocks)
{
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < edgeCount; e += stride)
    {
        const int p = perm[edges[2 * e]];
        const int q = perm[edges[2 * e + 1]];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)   // A_pq (r, c)
                tissueBandStore(blocks, w, 3 * p + r, 3 * q + c, static_cast<float>(edgeBlocks[static_cast<std::size_t>(e) * 9 + r * 3 + c]));
    }
}

// nrhs vectors of 3 values per vertex between the natural order and the solver's
// (row 3 perm[v] + c holds row 3 v + c); column stride n.
__global__ void tissuePermuteKernel(const int vertexCount, const int n, const int nrhs, const int* __restrict__ perm,
                                    const bool toSolverOrder, const float* __restrict__ in, float* __restrict__ out)
{
    const long long total = static_cast<long long>(vertexCount) * 3 * nrhs;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const long long column = i / (3LL * vertexCount);
        const int rem = static_cast<int>(i - column * 3LL * vertexCount);
        const int v = rem / 3;
        const int c = rem - 3 * v;
        const std::size_t natural = static_cast<std::size_t>(column) * n + 3 * v + c;
        const std::size_t solver = static_cast<std::size_t>(column) * n + 3 * perm[v] + c;
        if (toSolverOrder) out[solver] = in[natural];
        else out[natural] = in[solver];
    }
}

// The touched vertices' DOF columns of the identity, in the solver's order
// (n x 3 touched, column-major, zeroed beforehand).
__global__ void tissueSelectorKernel(const int* __restrict__ touchedVertices, const int touched, const int n,
                                     const int* __restrict__ perm, float* __restrict__ selector)
{
    const int stride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < touched; k += stride)
    {
        const int v = touchedVertices[k];
        const int pv = perm != nullptr ? perm[v] : v;
        for (int c = 0; c < 3; ++c) selector[static_cast<std::size_t>(3 * k + c) * n + (3 * pv + c)] = 1.0f;
    }
}

// G = E^T X for that selector E: G's row r is X's row at touched DOF r (solver order).
__global__ void tissueGatherSelectedRowsKernel(const int* __restrict__ touchedVertices, const int touched, const int n,
                                               const int* __restrict__ perm, const float* __restrict__ X, float* __restrict__ G)
{
    const int m = 3 * touched;
    const int total = m * m;
    const int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    {
        const int r = e % m;
        const int column = e / m;
        const int v = touchedVertices[r / 3];
        const int row = 3 * (perm != nullptr ? perm[v] : v) + r % 3;
        G[static_cast<std::size_t>(column) * m + r] = X[static_cast<std::size_t>(column) * n + row];
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
    cublasHandle_t blas { nullptr };
    // Factorisation. band: the vertices renumbered by reverse Cuthill-McKee (perm:
    // vertex -> position) so A is a band matrix of half-width `bandwidth` DOFs. Cut
    // into blocks of `blockWidth` >= bandwidth rows and columns, A is block
    // tridiagonal, and only its diagonal blocks D_k and the blocks S_k below them are
    // stored (bandBlocks: D_k at 2k w^2, S_k at (2k + 1) w^2, column-major, ld w):
    // 2 n w floats instead of n^2. The block Cholesky works on them in place.
    // dense: cuSOLVER's Cholesky on the whole matrix, in the natural order (perm null);
    // the band mode allocates the dense matrix only for an LU step (not positive definite).
    bool band { false };
    int* perm { nullptr };
    std::vector<int> permHost;
    int bandwidth { 0 };
    int panel { 128 };              // band: the block width is the bandwidth rounded up to a multiple of this
    int blockWidth { 0 };           // band: w
    int blockCount { 0 };           // band: K = ceil(n / w)
    float* bandBlocks { nullptr };
    int* panelInfo { nullptr };     std::size_t panelInfoCapacity { 0 };
    float* scratch { nullptr };     std::size_t scratchCapacity { 0 };   // permuted right-hand sides, selectors
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
    double* externalForce { nullptr };   // 3 per vertex; null: no external force

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
    int* pivots { nullptr };        // LU fallback (n)
    int* info { nullptr };
    double* monitor { nullptr };
    bool factorized { false };
    bool luFactor { false };        // this step's factor is LU (A was not positive definite), not Cholesky
    int luSteps { 0 };              // steps that needed the LU fallback so far
    bool stepped { false };

    ~TissueWorkspace()
    {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
        void* buffers[] = {
            tets, shapeVectors, restVolume, volScale, edgeSides, edges, vertexTetStart, vertexTetEntries,
            edgeTetStart, edgeTetEntries, vertexEdgeStart, vertexEdgeEntries, vertexMass, edgeMass, gravityMass,
            fixedDevice, externalForce, slsViscous, maxwellViscous, tetForce, tetEdgeBlocks, tetJ, force, edgeDfDx, kv, rhs,
            rhsFloat, diagBlocks, edgeBlocks, dv, residual, correction, dense, factorWork, pivots, info, monitor,
            perm, panelInfo, scratch, bandBlocks };
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

// Per-tetrahedron rest information (as TetrahedronViscoHyperelasticityFEMForceField::
// createTetrahedronRestInformation) and the gather lists (each vertex's and each
// edge's tetrahedron slots, in tetrahedron order), for a list of tetrahedra.
struct TissueElementArrays
{
    std::vector<int4> tets;
    std::vector<double> shapeVectors, restVolume, volScale;
    std::vector<int> vertexTetStart, vertexTetEntries, edgeTetStart, edgeTetEntries;
};

void tissueElementArrays(const int nv, const int ne, const std::vector<int>& tetrahedra, const std::vector<int>& tetrahedronEdges,
                         const std::vector<double>& restPositions, TissueElementArrays& out)
{
    const int nt = static_cast<int>(tetrahedra.size() / 4);
    out.shapeVectors.assign(static_cast<std::size_t>(nt) * 12, 0.0);
    out.restVolume.assign(nt, 0.0);
    out.volScale.assign(nt, 0.0);
    out.tets.resize(nt);
    for (int t = 0; t < nt; ++t)
    {
        const int* ta = &tetrahedra[static_cast<std::size_t>(t) * 4];
        out.tets[t] = make_int4(ta[0], ta[1], ta[2], ta[3]);
        double p[4][3];
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c) p[j][c] = restPositions[static_cast<std::size_t>(ta[j]) * 3 + c];
        auto sub = [](const double* a, const double* b, double* r) { for (int c = 0; c < 3; ++c) r[c] = a[c] - b[c]; };
        auto cross = [](const double* a, const double* b, double* r) {
            r[0] = a[1] * b[2] - a[2] * b[1];
            r[1] = a[2] * b[0] - a[0] * b[2];
            r[2] = a[0] * b[1] - a[1] * b[0];
        };
        double e20[3], e30[3], e10[3], cr[3];
        sub(p[2], p[0], e20);
        sub(p[3], p[0], e30);
        sub(p[1], p[0], e10);
        cross(e20, e30, cr);
        const double volume = cr[0] * e10[0] + cr[1] * e10[1] + cr[2] * e10[2];   // 6 x signed volume
        out.volScale[t] = 1.0 / volume;
        out.restVolume[t] = std::fabs(volume / 6.0);
        for (int j = 0; j < 4; ++j)
        {
            double a[3], b[3], s[3];
            sub(p[(j + 2) % 4], p[(j + 1) % 4], a);
            sub(p[(j + 3) % 4], p[(j + 1) % 4], b);
            cross(a, b, s);
            const double sign = (j % 2) ? 1.0 : -1.0;
            for (int c = 0; c < 3; ++c) out.shapeVectors[static_cast<std::size_t>(t) * 12 + j * 3 + c] = sign * s[c] / volume;
        }
    }
    out.vertexTetStart.assign(nv + 1, 0);
    out.edgeTetStart.assign(ne + 1, 0);
    for (int t = 0; t < nt; ++t)
    {
        for (int j = 0; j < 4; ++j) ++out.vertexTetStart[tetrahedra[static_cast<std::size_t>(t) * 4 + j] + 1];
        for (int j = 0; j < 6; ++j) ++out.edgeTetStart[tetrahedronEdges[static_cast<std::size_t>(t) * 6 + j] + 1];
    }
    for (int v = 0; v < nv; ++v) out.vertexTetStart[v + 1] += out.vertexTetStart[v];
    for (int e = 0; e < ne; ++e) out.edgeTetStart[e + 1] += out.edgeTetStart[e];
    out.vertexTetEntries.assign(out.vertexTetStart[nv], 0);
    out.edgeTetEntries.assign(out.edgeTetStart[ne], 0);
    std::vector<int> fillV(out.vertexTetStart.begin(), out.vertexTetStart.end() - 1);
    std::vector<int> fillE(out.edgeTetStart.begin(), out.edgeTetStart.end() - 1);
    for (int t = 0; t < nt; ++t)
    {
        for (int j = 0; j < 4; ++j) out.vertexTetEntries[fillV[tetrahedra[static_cast<std::size_t>(t) * 4 + j]]++] = t * 4 + j;
        for (int j = 0; j < 6; ++j) out.edgeTetEntries[fillE[tetrahedronEdges[static_cast<std::size_t>(t) * 6 + j]]++] = t * 6 + j;
    }
}

// Reverse Cuthill-McKee on the vertex graph: perm[vertex] = its new position, so
// that neighbours get close positions; nodeBandwidth = the largest |perm[a] -
// perm[b]| over the edges. Each connected part starts from a pseudo-peripheral
// vertex (repeated breadth-first searches from the smallest degree), neighbours
// are visited in order of increasing degree, and the order is reversed at the end.
std::vector<int> reverseCuthillMcKee(const int vertexCount, const std::vector<int>& edges, int& nodeBandwidth)
{
    std::vector<std::vector<int>> adjacency(vertexCount);
    for (std::size_t e = 0; e + 1 < edges.size(); e += 2)
    {
        adjacency[edges[e]].push_back(edges[e + 1]);
        adjacency[edges[e + 1]].push_back(edges[e]);
    }
    std::vector<int> degree(vertexCount);
    for (int v = 0; v < vertexCount; ++v) degree[v] = static_cast<int>(adjacency[v].size());
    for (auto& list : adjacency)
        std::sort(list.begin(), list.end(), [&](const int a, const int b) {
            return degree[a] != degree[b] ? degree[a] < degree[b] : a < b;
        });
    std::vector<char> placed(vertexCount, 0);
    std::vector<int> order;
    order.reserve(vertexCount);
    std::vector<int> level(vertexCount, -1);
    std::vector<int> queue;
    while (static_cast<int>(order.size()) < vertexCount)
    {
        int start = -1;
        for (int v = 0; v < vertexCount; ++v)
            if (!placed[v] && (start < 0 || degree[v] < degree[start])) start = v;
        int eccentricity = -1;
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            // Breadth-first levels from `start` over the unplaced vertices.
            queue.assign(1, start);
            std::vector<int> touched(1, start);
            level[start] = 0;
            for (std::size_t head = 0; head < queue.size(); ++head)
            {
                const int u = queue[head];
                for (const int w : adjacency[u])
                    if (!placed[w] && level[w] < 0)
                    {
                        level[w] = level[u] + 1;
                        queue.push_back(w);
                        touched.push_back(w);
                    }
            }
            const int depth = level[queue.back()];
            int candidate = queue.back();
            for (const int u : queue)
                if (level[u] == depth && degree[u] < degree[candidate]) candidate = u;
            for (const int u : touched) level[u] = -1;
            if (depth <= eccentricity) break;
            eccentricity = depth;
            start = candidate;
        }
        std::size_t head = order.size();
        order.push_back(start);
        placed[start] = 1;
        while (head < order.size())
        {
            const int u = order[head++];
            for (const int w : adjacency[u])
                if (!placed[w])
                {
                    placed[w] = 1;
                    order.push_back(w);
                }
        }
    }
    std::reverse(order.begin(), order.end());
    std::vector<int> perm(vertexCount);
    for (int i = 0; i < vertexCount; ++i) perm[order[i]] = i;
    nodeBandwidth = 0;
    for (std::size_t e = 0; e + 1 < edges.size(); e += 2)
        nodeBandwidth = std::max(nodeBandwidth, std::abs(perm[edges[e]] - perm[edges[e + 1]]));
    return perm;
}
} // namespace

namespace
{
// ---- The factor of A and solves with it (solver order) -----------------------

// A (from its double blocks) into the dense single-precision matrix, in the solver's order.
void tissueFillDense(TissueWorkspace* ws)
{
    const int n = ws->n;
    cudaMemsetAsync(ws->dense, 0, sizeof(float) * static_cast<std::size_t>(n) * n);
    tissueDenseDiagonalKernel<<<tissueBlocks(ws->vertexCount), 256>>>(ws->vertexCount, n, ws->perm, ws->diagBlocks, ws->dense);
    tissueDenseEdgeKernel<<<tissueBlocks(ws->edgeCount), 256>>>(ws->edgeCount, n, ws->perm, ws->edges, ws->edgeBlocks, ws->dense);
}

// A into the band's block storage (solver order): the diagonal blocks D_k and the
// blocks S_k below them, from the double blocks.
void tissueFillBand(TissueWorkspace* ws)
{
    const std::size_t w2 = static_cast<std::size_t>(ws->blockWidth) * ws->blockWidth;
    cudaMemsetAsync(ws->bandBlocks, 0, sizeof(float) * 2 * ws->blockCount * w2);
    tissueBandDiagonalKernel<<<tissueBlocks(ws->vertexCount), 256>>>(ws->vertexCount, ws->blockWidth, ws->perm, ws->diagBlocks,
                                                                      ws->bandBlocks);
    tissueBandEdgeKernel<<<tissueBlocks(ws->edgeCount), 256>>>(ws->edgeCount, ws->blockWidth, ws->perm, ws->edges,
                                                                ws->edgeBlocks, ws->bandBlocks);
}

// Rows (and columns) of band block k.
inline int tissueBandBlockSize(const TissueWorkspace* ws, const int k)
{
    return std::min(ws->blockWidth, ws->n - k * ws->blockWidth);
}

// Block Cholesky of the block-tridiagonal A in place: for each block k,
// D_k = L_k L_k^T, S_k <- S_k L_k^-T, D_{k+1} <- D_{k+1} - S_k S_k^T. Every call
// works on whole w x w blocks. Returns false with notPositive set if a pivot fails.
bool tissueBandCholesky(TissueWorkspace* ws, bool& notPositive, std::string& diagnostic)
{
    notPositive = false;
    const int w = ws->blockWidth;
    const int blocks = ws->blockCount;
    const std::size_t w2 = static_cast<std::size_t>(w) * w;
    int workSize = 0;
    if (cusolverDnSpotrf_bufferSize(ws->solver, CUBLAS_FILL_MODE_LOWER, w, ws->bandBlocks, w, &workSize) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cusolverDnSpotrf_bufferSize (band block) failed.";
        return false;
    }
    cudaError_t err = ensureDeviceBuffer(ws->factorWork, ws->factorWorkCapacity, static_cast<std::size_t>(std::max(workSize, 1)));
    if (err == cudaSuccess) err = ensureDeviceBuffer(ws->panelInfo, ws->panelInfoCapacity, static_cast<std::size_t>(blocks));
    if (err == cudaSuccess) err = cudaMemsetAsync(ws->panelInfo, 0, sizeof(int) * blocks);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Band factorisation buffers: ") + cudaGetErrorString(err);
        return false;
    }
    const float one = 1.0f;
    const float minusOne = -1.0f;
    for (int k = 0; k < blocks; ++k)
    {
        const int wk = tissueBandBlockSize(ws, k);
        float* d = ws->bandBlocks + 2 * k * w2;
        if (cusolverDnSpotrf(ws->solver, CUBLAS_FILL_MODE_LOWER, wk, d, w, ws->factorWork, workSize, ws->panelInfo + k) !=
            CUSOLVER_STATUS_SUCCESS)
        {
            diagnostic = "cusolverDnSpotrf (band block) failed to launch.";
            return false;
        }
        if (k + 1 < blocks)
        {
            const int wn = tissueBandBlockSize(ws, k + 1);
            float* s = ws->bandBlocks + (2 * k + 1) * w2;
            float* dn = ws->bandBlocks + 2 * (k + 1) * w2;
            cublasStatus_t status = cublasStrsm(ws->blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                                CUBLAS_DIAG_NON_UNIT, wn, wk, &one, d, w, s, w);
            if (status == CUBLAS_STATUS_SUCCESS)
                status = cublasSsyrk(ws->blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, wn, wk, &minusOne, s, w, &one, dn, w);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                diagnostic = std::string("Band factorisation: cuBLAS ") + cublasStatusName(status);
                return false;
            }
        }
    }
    std::vector<int> info(blocks);
    err = cudaMemcpy(info.data(), ws->panelInfo, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Band factorisation: ") + cudaGetErrorString(err);
        return false;
    }
    for (const int value : info) notPositive = notPositive || value > 0;
    diagnostic.clear();
    return true;
}

// b <- L^-1 b (forward) and, unless forwardOnly, then L^-T b: nrhs columns of n
// values (solver order, column stride n), with the block factor: one triangular
// solve per block, and one product with the block below it.
bool tissueBandSolve(TissueWorkspace* ws, float* b, const int nrhs, const bool forwardOnly, std::string& diagnostic)
{
    const int n = ws->n;
    const int w = ws->blockWidth;
    const int blocks = ws->blockCount;
    const std::size_t w2 = static_cast<std::size_t>(w) * w;
    const float one = 1.0f;
    const float minusOne = -1.0f;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int k = 0; k < blocks && status == CUBLAS_STATUS_SUCCESS; ++k)
    {
        const int wk = tissueBandBlockSize(ws, k);
        float* bk = b + static_cast<std::size_t>(k) * w;
        if (k > 0)   // b_k -= S_{k-1} y_{k-1}
            status = cublasSgemm(ws->blas, CUBLAS_OP_N, CUBLAS_OP_N, wk, nrhs, w, &minusOne,
                                 ws->bandBlocks + (2 * (k - 1) + 1) * w2, w, bk - w, n, &one, bk, n);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                 wk, nrhs, &one, ws->bandBlocks + 2 * k * w2, w, bk, n);
    }
    for (int k = blocks - 1; !forwardOnly && k >= 0 && status == CUBLAS_STATUS_SUCCESS; --k)
    {
        const int wk = tissueBandBlockSize(ws, k);
        float* bk = b + static_cast<std::size_t>(k) * w;
        if (k + 1 < blocks)   // b_k -= S_k^T x_{k+1}
            status = cublasSgemm(ws->blas, CUBLAS_OP_T, CUBLAS_OP_N, wk, nrhs, tissueBandBlockSize(ws, k + 1), &minusOne,
                                 ws->bandBlocks + (2 * k + 1) * w2, w, bk + w, n, &one, bk, n);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                 wk, nrhs, &one, ws->bandBlocks + 2 * k * w2, w, bk, n);
    }
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        diagnostic = std::string("Band solve: cuBLAS ") + cublasStatusName(status);
        return false;
    }
    diagnostic.clear();
    return true;
}

// A^-1 b in place, b in the solver's order (nrhs columns of n): LU when this step
// fell back to it, else the band or dense Cholesky.
bool tissueSolveSolverOrder(TissueWorkspace* ws, float* b, const int nrhs, std::string& diagnostic)
{
    const int n = ws->n;
    if (ws->luFactor)
    {
        if (cusolverDnSgetrs(ws->solver, CUBLAS_OP_N, n, nrhs, ws->dense, n, ws->pivots, b, n, ws->info) != CUSOLVER_STATUS_SUCCESS)
        {
            diagnostic = "cusolverDnSgetrs failed to launch.";
            return false;
        }
        diagnostic.clear();
        return true;
    }
    if (ws->band) return tissueBandSolve(ws, b, nrhs, false, diagnostic);
    if (cusolverDnSpotrs(ws->solver, CUBLAS_FILL_MODE_LOWER, n, nrhs, ws->dense, n, b, n, ws->info) != CUSOLVER_STATUS_SUCCESS)
    {
        diagnostic = "cusolverDnSpotrs failed to launch.";
        return false;
    }
    diagnostic.clear();
    return true;
}

// A^-1 b in place, b in the natural vertex order.
bool tissueSolveNaturalOrder(TissueWorkspace* ws, float* b, const int nrhs, std::string& diagnostic)
{
    if (ws->perm == nullptr) return tissueSolveSolverOrder(ws, b, nrhs, diagnostic);
    const std::size_t count = static_cast<std::size_t>(ws->n) * nrhs;
    cudaError_t err = ensureDeviceBuffer(ws->scratch, ws->scratchCapacity, count);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue solve buffer: ") + cudaGetErrorString(err);
        return false;
    }
    const int blocks = static_cast<int>(std::min<std::size_t>((count + 255) / 256, 4096));
    tissuePermuteKernel<<<blocks, 256>>>(ws->vertexCount, ws->n, nrhs, ws->perm, true, b, ws->scratch);
    if (!tissueSolveSolverOrder(ws, ws->scratch, nrhs, diagnostic)) return false;
    tissuePermuteKernel<<<blocks, 256>>>(ws->vertexCount, ws->n, nrhs, ws->perm, false, ws->scratch, b);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue solve permutation: ") + cudaGetErrorString(err);
        return false;
    }
    return true;
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

    TissueElementArrays elements;
    tissueElementArrays(nv, ne, setup.tetrahedra, setup.tetrahedronEdges, setup.restPositions, elements);
    const auto& tets = elements.tets;
    // Each vertex's edges (the edges never change, cutting only removes tetrahedra).
    std::vector<int> vertexEdgeStart(nv + 1, 0);
    for (int e = 0; e < ne; ++e)
    {
        ++vertexEdgeStart[setup.edges[2 * e] + 1];
        ++vertexEdgeStart[setup.edges[2 * e + 1] + 1];
    }
    for (int v = 0; v < nv; ++v) vertexEdgeStart[v + 1] += vertexEdgeStart[v];
    std::vector<int> vertexEdgeEntries(vertexEdgeStart[nv]);
    {
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
    ws->material.ogdenRobustTangent = setup.material.ogdenRobustTangent ? 1 : 0;
    ws->material.core = static_cast<int>(setup.material.core);
    for (int k = 0; k < 4; ++k) ws->material.coreParams[k] = setup.material.coreParameters[k];
    for (int c = 0; c < 3; ++c) ws->gravity[c] = setup.gravity[c];
    if (cusolverDnCreate(&ws->solver) != CUSOLVER_STATUS_SUCCESS || cublasCreate(&ws->blas) != CUBLAS_STATUS_SUCCESS)
    {
        diagnostic = "cuSOLVER or cuBLAS could not be initialised.";
        delete ws;
        return nullptr;
    }

    // The factorisation: a band Cholesky in reverse Cuthill-McKee order when the
    // band is narrow enough to pay (its work is about n b^2 against n^3 / 3).
    {
        int nodeBandwidth = 0;
        std::vector<int> perm = reverseCuthillMcKee(nv, setup.edges, nodeBandwidth);
        const int bandwidth = 3 * nodeBandwidth + 2;
        const bool band = setup.factorization == TissueFactorization::Band ||
                          (setup.factorization == TissueFactorization::Automatic && 3 * bandwidth < ws->n);
        ws->band = band;
        ws->panel = std::max(16, setup.bandPanel);
        if (band)
        {
            ws->bandwidth = bandwidth;
            ws->permHost = perm;
            ws->blockWidth = std::min(ws->n, (bandwidth + ws->panel - 1) / ws->panel * ws->panel);
            ws->blockCount = (ws->n + ws->blockWidth - 1) / ws->blockWidth;
        }
        else
        {
            ws->bandwidth = ws->n;
        }
    }

    const std::size_t n = static_cast<std::size_t>(ws->n);
    cudaError_t err = ws->band ? uploadVector(ws->perm, ws->permHost) : cudaSuccess;
    if (err == cudaSuccess && ws->band)
        err = allocateZeroed(ws->bandBlocks, 2 * static_cast<std::size_t>(ws->blockCount) * ws->blockWidth * ws->blockWidth);
    if (err == cudaSuccess) err = uploadVector(ws->tets, tets);
    if (err == cudaSuccess) err = uploadVector(ws->shapeVectors, elements.shapeVectors);
    if (err == cudaSuccess) err = uploadVector(ws->restVolume, elements.restVolume);
    if (err == cudaSuccess) err = uploadVector(ws->volScale, elements.volScale);
    if (err == cudaSuccess) err = uploadVector(ws->edgeSides, setup.tetrahedronEdgeSides);
    if (err == cudaSuccess) err = uploadVector(ws->edges, setup.edges);
    if (err == cudaSuccess) err = uploadVector(ws->vertexTetStart, elements.vertexTetStart);
    if (err == cudaSuccess) err = uploadVector(ws->vertexTetEntries, elements.vertexTetEntries);
    if (err == cudaSuccess) err = uploadVector(ws->edgeTetStart, elements.edgeTetStart);
    if (err == cudaSuccess) err = uploadVector(ws->edgeTetEntries, elements.edgeTetEntries);
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
    if (err == cudaSuccess && !ws->band) err = allocateZeroed(ws->dense, n * n);   // band: only for an LU step
    if (err == cudaSuccess) err = allocateZeroed(ws->pivots, n);
    if (err == cudaSuccess) err = allocateZeroed(ws->info, 1);
    if (err == cudaSuccess) err = allocateZeroed(ws->monitor, 4);
    if (err != cudaSuccess)
    {
        const std::size_t matrixBytes = ws->band ? 8 * static_cast<std::size_t>(ws->blockCount) * ws->blockWidth * ws->blockWidth
                                                 : n * n * 4;
        diagnostic = std::string("Tissue buffers: ") + cudaGetErrorString(err) + " (the " + (ws->band ? "band" : "dense") +
                     " matrix alone needs " + std::to_string(matrixBytes / (1024 * 1024)) + " MB)";
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
    ws->luFactor = false;
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
            ws->gravity[0], ws->gravity[1], ws->gravity[2], ws->externalForce, ws->force);
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
        if (ws->band) tissueFillBand(ws);
        else tissueFillDense(ws);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            diagnostic = std::string("Tissue assembly: ") + cudaGetErrorString(err);
            return false;
        }
        if (timings != nullptr) timings->assembleMs = timer.finish();
    }

    // 3. Cholesky factorisation (band or dense).
    {
        ConstraintEventTimer timer(timings != nullptr);
        bool notPositive = false;
        cudaError_t err = cudaSuccess;
        int info = 0;
        if (ws->band)
        {
            if (!tissueBandCholesky(ws, notPositive, diagnostic)) return false;
        }
        else
        {
            int workSize = 0;
            if (cusolverDnSpotrf_bufferSize(ws->solver, CUBLAS_FILL_MODE_LOWER, n, ws->dense, n, &workSize) != CUSOLVER_STATUS_SUCCESS)
            {
                diagnostic = "cusolverDnSpotrf_bufferSize failed.";
                return false;
            }
            err = ensureDeviceBuffer(ws->factorWork, ws->factorWorkCapacity, static_cast<std::size_t>(std::max(workSize, 1)));
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
            err = cudaMemcpy(&info, ws->info, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                diagnostic = std::string("Tissue factorisation: ") + cudaGetErrorString(err);
                return false;
            }
            notPositive = info > 0;
        }
        if (notPositive)
        {
            // A is not positive definite (a strongly compressed state: the geometric
            // stiffness outweighs M / h^2). SOFA's SparseLDLSolver factors such a matrix
            // (LDL^T, pivots of both signs) and goes on; here LU with partial pivoting
            // (on the whole matrix, in the solver's order) does the same job for this step.
            // The band mode keeps no dense matrix: it is allocated for the first such step.
            if (ws->dense == nullptr)
            {
                err = allocateZeroed(ws->dense, static_cast<std::size_t>(n) * n);
                if (err != cudaSuccess)
                {
                    ws->dense = nullptr;
                    diagnostic = "The tissue system matrix is not positive definite, and its dense copy for LU (" +
                                 std::to_string(static_cast<std::size_t>(n) * n * 4 / (1024 * 1024)) + " MB) does not fit: " +
                                 cudaGetErrorString(err);
                    return false;
                }
            }
            tissueFillDense(ws);
            int luWork = 0;
            if (cusolverDnSgetrf_bufferSize(ws->solver, n, n, ws->dense, n, &luWork) != CUSOLVER_STATUS_SUCCESS)
            {
                diagnostic = "cusolverDnSgetrf_bufferSize failed.";
                return false;
            }
            err = ensureDeviceBuffer(ws->factorWork, ws->factorWorkCapacity, static_cast<std::size_t>(std::max(luWork, 1)));
            if (err != cudaSuccess)
            {
                diagnostic = std::string("Tissue LU workspace: ") + cudaGetErrorString(err);
                return false;
            }
            if (cusolverDnSgetrf(ws->solver, n, n, ws->dense, n, ws->factorWork, ws->pivots, ws->info) != CUSOLVER_STATUS_SUCCESS)
            {
                diagnostic = "cusolverDnSgetrf failed to launch.";
                return false;
            }
            err = cudaMemcpy(&info, ws->info, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess || info != 0)
            {
                diagnostic = err != cudaSuccess ? std::string("Tissue LU factorisation: ") + cudaGetErrorString(err)
                                                : "The tissue system matrix is singular (LU pivot " + std::to_string(info) + " is zero).";
                return false;
            }
            ws->luFactor = true;
            ++ws->luSteps;
        }
        ws->factorized = true;
        if (timings != nullptr) timings->factorizeMs = timer.finish();
    }

    // 4. Solve, refine in double, update.
    {
        ConstraintEventTimer timer(timings != nullptr);
        if (!tissueSolveNaturalOrder(ws, ws->rhsFloat, 1, diagnostic)) return false;
        tissueCopyToDoubleKernel<<<tissueBlocks(n), 256>>>(n, ws->rhsFloat, ws->dv);
        for (int step = 0; step < config.refinementSteps; ++step)
        {
            tissueResidualKernel<<<tissueBlocks(nv), 256>>>(
                nv, ws->vertexEdgeStart, ws->vertexEdgeEntries, ws->edges, ws->diagBlocks, ws->edgeBlocks,
                ws->rhs, ws->dv, ws->residual, ws->correction);
            if (!tissueSolveNaturalOrder(ws, ws->correction, 1, diagnostic)) return false;
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

bool setTissueExternalForces(TissueWorkspace* ws, const std::vector<double>& forces, std::string& diagnostic)
{
    if (ws == nullptr) { diagnostic = "No tissue workspace."; return false; }
    if (forces.empty())
    {
        if (ws->externalForce) cudaFree(ws->externalForce);
        ws->externalForce = nullptr;
        diagnostic.clear();
        return true;
    }
    if (forces.size() != static_cast<std::size_t>(ws->n))
    {
        diagnostic = "External forces: " + std::to_string(forces.size()) + " values for " + std::to_string(ws->n) + " DOFs.";
        return false;
    }
    cudaError_t err = cudaSuccess;
    if (ws->externalForce == nullptr)
        err = cudaMalloc(reinterpret_cast<void**>(&ws->externalForce), sizeof(double) * forces.size());
    if (err == cudaSuccess)
        err = cudaMemcpy(ws->externalForce, forces.data(), sizeof(double) * forces.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("External forces: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

bool updateTissueElements(TissueWorkspace* ws, const TissueElementUpdate& update, std::string& diagnostic)
{
    // Cutting removed tetrahedra: the element arrays, the gather lists and the mass
    // are rebuilt for the remaining ones, and each keeps its viscous state. The
    // vertices, the edges (so the matrix pattern and the band renumbering) and the
    // factorisation's storage stay as they are: an edge no tetrahedron uses any more
    // just adds nothing.
    if (ws == nullptr) { diagnostic = "No tissue workspace."; return false; }
    const int nv = ws->vertexCount;
    const int ne = ws->edgeCount;
    const int nt = static_cast<int>(update.tetrahedra.size() / 4);
    if (nt <= 0 || update.tetrahedronEdges.size() != static_cast<std::size_t>(nt) * 6 ||
        update.tetrahedronEdgeSides.size() != static_cast<std::size_t>(nt) * 6 ||
        update.previousIndex.size() != static_cast<std::size_t>(nt) ||
        update.restPositions.size() != static_cast<std::size_t>(nv) * 3 ||
        update.vertexMass.size() != static_cast<std::size_t>(nv) || update.edgeMass.size() != static_cast<std::size_t>(ne) ||
        update.gravityMass.size() != static_cast<std::size_t>(nv) || update.fixedDofs.size() != static_cast<std::size_t>(nv))
    {
        diagnostic = "Tissue element update has inconsistent sizes.";
        return false;
    }
    for (const int e : update.tetrahedronEdges)
        if (e < 0 || e >= ne)
        {
            diagnostic = "Tissue element update: an edge index is out of range.";
            return false;
        }
    TissueElementArrays elements;
    tissueElementArrays(nv, ne, update.tetrahedra, update.tetrahedronEdges, update.restPositions, elements);

    // The viscous states move with their tetrahedra.
    const int oldCount = ws->tetCount;
    std::vector<double> sls(static_cast<std::size_t>(oldCount) * 6), maxwell(static_cast<std::size_t>(oldCount) * 6);
    cudaError_t err = cudaMemcpy(sls.data(), ws->slsViscous, sizeof(double) * sls.size(), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(maxwell.data(), ws->maxwellViscous, sizeof(double) * maxwell.size(), cudaMemcpyDeviceToHost);
    std::vector<double> slsNew(static_cast<std::size_t>(nt) * 6, 0.0), maxwellNew(static_cast<std::size_t>(nt) * 6, 0.0);
    for (int t = 0; t < nt && err == cudaSuccess; ++t)
    {
        const int old = update.previousIndex[t];
        if (old < 0 || old >= oldCount) continue;   // a new tetrahedron starts relaxed
        for (int k = 0; k < 6; ++k)
        {
            slsNew[static_cast<std::size_t>(t) * 6 + k] = sls[static_cast<std::size_t>(old) * 6 + k];
            maxwellNew[static_cast<std::size_t>(t) * 6 + k] = maxwell[static_cast<std::size_t>(old) * 6 + k];
        }
    }
    auto replace = [&](auto*& device, const auto& host) {
        if (err != cudaSuccess) return;
        if (device) cudaFree(device);
        device = nullptr;
        err = uploadVector(device, host);
    };
    replace(ws->tets, elements.tets);
    replace(ws->shapeVectors, elements.shapeVectors);
    replace(ws->restVolume, elements.restVolume);
    replace(ws->volScale, elements.volScale);
    replace(ws->edgeSides, update.tetrahedronEdgeSides);
    replace(ws->vertexTetStart, elements.vertexTetStart);
    replace(ws->vertexTetEntries, elements.vertexTetEntries);
    replace(ws->edgeTetStart, elements.edgeTetStart);
    replace(ws->edgeTetEntries, elements.edgeTetEntries);
    replace(ws->vertexMass, update.vertexMass);
    replace(ws->edgeMass, update.edgeMass);
    replace(ws->gravityMass, update.gravityMass);
    replace(ws->fixedDevice, update.fixedDofs);
    replace(ws->slsViscous, slsNew);
    replace(ws->maxwellViscous, maxwellNew);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue element update: ") + cudaGetErrorString(err);
        return false;
    }
    ws->tetCount = nt;   // tetForce, tetEdgeBlocks and tetJ keep their (larger) size
    ws->fixedDofs = update.fixedDofs;
    ws->factorized = false;
    diagnostic.clear();
    return true;
}

int tissueDofCount(const TissueWorkspace* ws)
{
    return ws != nullptr ? ws->n : 0;
}

bool tissueFactorReady(const TissueWorkspace* ws)
{
    return ws != nullptr && ws->factorized;
}

int tissueBandwidth(const TissueWorkspace* ws)
{
    return ws != nullptr && ws->band ? ws->bandwidth : 0;
}

bool tissueSolveInPlace(TissueWorkspace* ws, float* b, const int nrhs, std::string& diagnostic)
{
    if (!tissueFactorReady(ws))
    {
        diagnostic = "The tissue step has not factorised its matrix.";
        return false;
    }
    return tissueSolveNaturalOrder(ws, b, nrhs, diagnostic);
}

bool tissueComplianceBlock(TissueWorkspace* ws, const int* touchedVertices, const int touched, float* G, std::string& diagnostic)
{
    if (!tissueFactorReady(ws))
    {
        diagnostic = "The tissue step has not factorised its matrix.";
        return false;
    }
    const int n = ws->n;
    const int m = 3 * touched;
    if (m == 0) { diagnostic.clear(); return true; }
    const std::size_t count = static_cast<std::size_t>(n) * m;
    cudaError_t err = ensureDeviceBuffer(ws->scratch, ws->scratchCapacity, count);
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue compliance buffer: ") + cudaGetErrorString(err);
        return false;
    }
    cudaMemsetAsync(ws->scratch, 0, sizeof(float) * count);
    tissueSelectorKernel<<<std::max(1, std::min((touched + 255) / 256, 1024)), 256>>>(touchedVertices, touched, n, ws->perm, ws->scratch);
    if (ws->luFactor)
    {
        // X = A^-1 E, G = E^T X.
        if (!tissueSolveSolverOrder(ws, ws->scratch, m, diagnostic)) return false;
        tissueGatherSelectedRowsKernel<<<std::max(1, std::min((m * m + 255) / 256, 4096)), 256>>>(
            touchedVertices, touched, n, ws->perm, ws->scratch, G);
    }
    else
    {
        // Y = L^-1 E, G = Y^T Y.
        const float one = 1.0f;
        const float zero = 0.0f;
        if (ws->band)
        {
            if (!tissueBandSolve(ws, ws->scratch, m, true, diagnostic)) return false;
        }
        else if (cublasStrsm(ws->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                             n, m, &one, ws->dense, n, ws->scratch, n) != CUBLAS_STATUS_SUCCESS)
        {
            diagnostic = "Tissue compliance: cuBLAS trsm failed.";
            return false;
        }
        if (cublasSgemm(ws->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, ws->scratch, n, ws->scratch, n, &zero, G, m) !=
            CUBLAS_STATUS_SUCCESS)
        {
            diagnostic = "Tissue compliance: cuBLAS gemm failed.";
            return false;
        }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        diagnostic = std::string("Tissue compliance: ") + cudaGetErrorString(err);
        return false;
    }
    diagnostic.clear();
    return true;
}

int tissueLuFallbackSteps(const TissueWorkspace* ws)
{
    return ws != nullptr ? ws->luSteps : 0;
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
