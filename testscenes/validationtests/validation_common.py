"""Shared setup for the CPU-versus-GPU validation tests.

Each test scene builds the same physical set-up twice, selected by
SOFA_VALIDATION_SIDE:
  cpu  SOFA's own, widely used CPU components: TetrahedronHyperelasticityFEMForceField
       (SOFA core materials), SofaViscoElastic's Maxwell branch, MeshMatrixMass,
       EulerImplicitSolver + SparseLDLSolver, Fixed/PartialFixedProjectiveConstraint,
       and for contact SOFA's collision pipeline with FrictionContactConstraint and
       LinearSolverConstraintCorrection.
  gpu  this project's GPU components: GpuTissueSolver (the same materials), the GPU
       collision detection and GpuContactConstraintSolver.
Both write the same CSV files, so scripts/compare_validation.py can put the two
side by side and against the test's known answer.

Units are SI throughout (m, kg, s, Pa, N).
"""

import math
import os
import time

import numpy as np
import Sofa.Core

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

SIDE = os.environ.get("SOFA_VALIDATION_SIDE", "gpu")
MATERIAL = os.environ.get("SOFA_VALIDATION_MATERIAL", "ogden_maxwell")
COMPARE = os.environ.get("SOFA_VALIDATION_COMPARE", "0") == "1"   # GPU: stage-by-stage against SOFA's CPU components
COMPARE_EVERY = int(os.environ.get("SOFA_VALIDATION_COMPARE_EVERY", "1"))
MEASURE_TIMES = os.environ.get("SOFA_VALIDATION_MEASURE_TIMES", "0") == "1"
LOG_POSITIONS = os.environ.get("SOFA_VALIDATION_LOG_POSITIONS", "1") != "0"   # off for timing runs
TRACE = os.environ.get("SOFA_VALIDATION_TRACE", "0") == "1"
# GPU core Ogden stiffness: robust (default) or sofa (SOFA v25.12's formula, see GpuTissueSolver.ogdenTangent).
OGDEN_TANGENT = os.environ.get("SOFA_VALIDATION_OGDEN_TANGENT", "robust")
# GPU factorisation of the tissue matrix: auto (default), band or dense (GpuTissueSolver.factorization).
FACTORIZATION = os.environ.get("SOFA_VALIDATION_FACTORIZATION", "auto")
GPU_COLLISION_LIB = os.environ.get(
    "SOFA_GPU_COLLISION_LIB", os.path.join(REPO_ROOT, "SofaGpuCollision", "build-profile", "libSofaGpuCollision.so"))


def log_dir():
    return os.environ.get("SOFA_BENCHMARK_LOG_DIR", os.path.join(REPO_ROOT, "output", "benchmark_logs", "validation"))


def env_float(name, default):
    return float(os.environ.get(name, default))


def run_label(test):
    """<test>_<material>_<side>[_compare]: the stem of every file a run writes."""
    return (f"{test}_{MATERIAL}_{SIDE}" + ("_compare" if SIDE == "gpu" and COMPARE else "")
            + ("_sofatangent" if SIDE == "gpu" and OGDEN_TANGENT != "robust" else "")
            + ("_plt" if SIDE == "cpu" and CPU_CONTACT_MODELS != "point_triangle" else ""))


# ---------------------------------------------------------------------------
# Materials: the same stress on both sides.
#   cpu: SOFA force fields (class, attributes)
#   gpu: GpuTissueSolver settings
# ---------------------------------------------------------------------------
DENSITY = 1060.0                 # kg/m^3, soft tissue

# Liver-like: Ogden long-term spring (shear modulus mu1/2 = 1 kPa, stiffening with
# stretch, bulk 20 kPa) plus a Maxwell branch that adds 1 kPa and relaxes with 0.58 s.
OGDEN = (2000.0, 6.0, 20000.0)          # mu1 alpha1 k0
MAXWELL = (1000.0, 0.58, 0.0)           # G1 tau lambda

# Simpler rubber-like materials with the same small-strain moduli as the Ogden
# spring: shear modulus 1 kPa, Poisson ratio 0.45.
SHEAR_MODULUS = 1000.0
POISSON = 0.45
LAME_LAMBDA = 2.0 * SHEAR_MODULUS * POISSON / (1.0 - 2.0 * POISSON)   # 9 kPa


def make_material(name, params, maxwell=None):
    """A SOFA core hyperelastic material (TetrahedronHyperelasticityFEMForceField's
    materialName and ParameterSet), optionally with SofaViscoElastic's Maxwell branch
    (MaxwellFirstOrder: G1 tau lambda) in parallel. The entry also keeps the
    parameters, for the analytic answers below."""
    params = tuple(float(p) for p in params)
    entry = {"core": name, "params": params, "maxwell": tuple(float(p) for p in maxwell) if maxwell else None,
             "cpu": [("TetrahedronHyperelasticityFEMForceField",
                      {"materialName": name, "ParameterSet": " ".join(repr(p) for p in params)})],
             "gpu": {"hyperelasticMaterial": name, "hyperelasticParameters": list(params)}}
    if maxwell:
        entry["cpu"].append(("TetrahedronViscoelasticityFEMForceField",
                             {"materialName": "MaxwellFirstOrder", "ParameterSet": " ".join(repr(float(p)) for p in maxwell)}))
        entry["gpu"]["maxwellParameters"] = [float(p) for p in maxwell]
    return entry


MATERIALS = {
    # SOFA core materials (TetrahedronHyperelasticityFEMForceField)
    "neohookean": make_material("NeoHookean", (SHEAR_MODULUS, LAME_LAMBDA)),
    "stable_neohookean": make_material("StableNeoHookean", (SHEAR_MODULUS, LAME_LAMBDA)),
    "stvk": make_material("StVenantKirchhoff", (SHEAR_MODULUS, LAME_LAMBDA)),
    # c1 + c2 = mu/2, k0 = bulk modulus
    "mooney_rivlin": make_material("MooneyRivlin", (0.3 * SHEAR_MODULUS, 0.2 * SHEAR_MODULUS,
                                                    LAME_LAMBDA + 2.0 * SHEAR_MODULUS / 3.0)),
    "ogden": make_material("Ogden", OGDEN),
    # the realistic default: SOFA core Ogden + SofaViscoElastic Maxwell branch
    "ogden_maxwell": make_material("Ogden", OGDEN, MAXWELL),
    # SofaViscoElastic's SLSOgdenFirstOrder (G1 = 0) + Maxwell, as it runs in SOFA v25.12
    # (its Eigen call computes no eigenvectors; the GPU reproduces that)
    "sls_ogden_sofa": {
        "core": None, "params": (), "maxwell": MAXWELL,   # no analytic answer: the material is not the Ogden law
        "cpu": [("TetrahedronViscoHyperelasticityFEMForceField",
                 {"materialName": "SLSOgdenFirstOrder",
                  "ParameterSet": f"{OGDEN[0]} {OGDEN[1]} 0 {MAXWELL[1]} {OGDEN[2]}"}),
                ("TetrahedronViscoelasticityFEMForceField",
                 {"materialName": "MaxwellFirstOrder", "ParameterSet": " ".join(repr(p) for p in MAXWELL)})],
        "gpu": {"ogdenParameters": [OGDEN[0], OGDEN[1], 0.0, MAXWELL[1], OGDEN[2]],
                "maxwellParameters": list(MAXWELL), "ogdenEigenvectors": "sofa"},
    },
}


def material(name=None):
    name = name or MATERIAL
    if name not in MATERIALS:
        raise ValueError(f"unknown material '{name}'; one of {sorted(MATERIALS)}")
    return MATERIALS[name]


# ---------------------------------------------------------------------------
# Known answers (continuum mechanics, independent of any mesh or code)
# ---------------------------------------------------------------------------
def small_strain_moduli(mat):
    """(shear modulus mu, Lame lambda) of the elastic (long-term) part at small strain."""
    name, p = mat["core"], mat["params"]
    if name in ("NeoHookean", "StableNeoHookean", "StVenantKirchhoff"):
        return p[0], p[1]
    if name == "MooneyRivlin":          # W = c1 (I1bar - 3) + c2 (I2bar - 3) + k0/2 ln(J)^2
        mu = 2.0 * (p[0] + p[1])
        return mu, p[2] - 2.0 * mu / 3.0
    if name == "Ogden":                 # W = mu1/alpha1^2 sum(lambdabar^alpha1 - 1) + k0/2 ln(J)^2
        mu = p[0] / 2.0
        return mu, p[2] - 2.0 * mu / 3.0
    raise ValueError(f"no small-strain moduli for material {name}")


def youngs_modulus(mat):
    mu, lam = small_strain_moduli(mat)
    return mu * (3.0 * lam + 2.0 * mu) / (lam + mu), lam / (2.0 * (lam + mu))   # E, Poisson ratio


def uniaxial_strain_stress(mat, s):
    """Nominal (first Piola-Kirchhoff) stress P_yy of the elastic part for the
    homogeneous deformation F = diag(1, s, 1) (confined compression, s < 1), from
    each material's second Piola-Kirchhoff stress as SOFA defines it:
      NeoHookean        S = mu I + (lambda ln J - mu) C^-1
      StableNeoHookean  S = mu I + (lambda + mu) J (J - a) C^-1,  a = 1 + mu / (lambda + mu)
      StVenantKirchhoff S = lambda tr(E) I + 2 mu E
      MooneyRivlin      S = 2 c1 J^(-2/3) (I - I1/3 C^-1) + 2 c2 J^(-4/3) (I1 I - C - 2 I2/3 C^-1) + k0 ln J C^-1
      Ogden             S = mu1/alpha1 J^(-alpha1/3) (C^(alpha1/2 - 1) - tr(C^(alpha1/2))/3 C^-1) + k0 ln J C^-1
    P_yy = s S_yy (C = diag(1, s^2, 1), J = s)."""
    name, p = mat["core"], mat["params"]
    lnj = math.log(s)
    if name == "NeoHookean":
        mu, lam = p
        syy = mu + (lam * lnj - mu) / s**2
    elif name == "StableNeoHookean":
        mu, lam = p
        a = 1.0 + mu / (lam + mu)
        syy = mu + (lam + mu) * s * (s - a) / s**2
    elif name == "StVenantKirchhoff":
        mu, lam = p
        eyy = 0.5 * (s * s - 1.0)
        syy = (lam + 2.0 * mu) * eyy
    elif name == "MooneyRivlin":
        c1, c2, k0 = p
        i1, i2 = 2.0 + s * s, 1.0 + 2.0 * s * s
        a, b = 2.0 * c1 * s ** (-2.0 / 3.0), 2.0 * c2 * s ** (-4.0 / 3.0)
        syy = a * (1.0 - i1 / (3.0 * s * s)) + b * (i1 - s * s - 2.0 * i2 / (3.0 * s * s)) + k0 * lnj / s**2
    elif name == "Ogden":
        mu1, alpha, k0 = p
        syy = (mu1 / alpha) * s ** (-alpha / 3.0) * (s ** (alpha - 2.0) - (2.0 + s**alpha) / (3.0 * s * s)) + k0 * lnj / s**2
    else:
        raise ValueError(f"no analytic stress for material {name}")
    return s * syy


def principal_nominal_stress(mat, stretches):
    """(P_1, P_2, P_3): nominal stresses of the elastic part for F = diag(stretches),
    from the same second Piola-Kirchhoff stresses (uniaxial_strain_stress), with
    C = diag(l_i^2) so every term is diagonal: P_i = l_i S_i."""
    name, p = mat["core"], mat["params"]
    l = [float(v) for v in stretches]
    c = [v * v for v in l]
    j = l[0] * l[1] * l[2]
    lnj = math.log(j)
    i1 = c[0] + c[1] + c[2]
    i2 = c[0] * c[1] + c[1] * c[2] + c[0] * c[2]
    out = []
    for i in range(3):
        cinv = 1.0 / c[i]
        if name == "NeoHookean":
            mu, lam = p
            s = mu + (lam * lnj - mu) * cinv
        elif name == "StableNeoHookean":
            mu, lam = p
            a = 1.0 + mu / (lam + mu)
            s = mu + (lam + mu) * j * (j - a) * cinv
        elif name == "StVenantKirchhoff":
            mu, lam = p
            tr_e = 0.5 * (i1 - 3.0)
            s = lam * tr_e + mu * (c[i] - 1.0)
        elif name == "MooneyRivlin":
            c1, c2, k0 = p
            a, b = 2.0 * c1 * j ** (-2.0 / 3.0), 2.0 * c2 * j ** (-4.0 / 3.0)
            s = a * (1.0 - i1 / 3.0 * cinv) + b * (i1 - c[i] - 2.0 * i2 / 3.0 * cinv) + k0 * lnj * cinv
        elif name == "Ogden":
            mu1, alpha, k0 = p
            tr_ca = sum(v ** (alpha / 2.0) for v in c)
            s = (mu1 / alpha) * j ** (-alpha / 3.0) * (c[i] ** (alpha / 2.0 - 1.0) - tr_ca / 3.0 * cinv) + k0 * lnj * cinv
        else:
            raise ValueError(f"no analytic stress for material {name}")
        out.append(l[i] * s)
    return out


def _bisect(f, lo, hi, iterations=200):
    fa = f(lo)
    for _ in range(iterations):
        m = 0.5 * (lo + hi)
        fm = f(m)
        if (fm < 0.0) == (fa < 0.0):
            lo, fa = m, fm
        else:
            hi = m
    return 0.5 * (lo + hi)


def uniaxial_stress_state(mat, s):
    """Uniaxial stress along y with free sides (a frictionless plate on a block whose
    base may slide): the lateral stretch a with P_x(a, s, a) = 0, and the axial
    nominal stress P_y. Returns (a, P_y)."""
    a = _bisect(lambda a: principal_nominal_stress(mat, (a, s, a))[0], 0.5, 2.0)
    return a, principal_nominal_stress(mat, (a, s, a))[1]


def solve_uniaxial_strain(mat, nominal_stress, lo=0.2, hi=3.0):
    """The stretch s with uniaxial_strain_stress(mat, s) = nominal_stress (bisection)."""
    f = lambda s: uniaxial_strain_stress(mat, s) - nominal_stress
    a, b = lo, hi
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm < 0.0) == (fa < 0.0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def face_traction_loads(mesh, axis, side, traction):
    """Nodal forces of a uniform traction (force per area, a 3-vector) on one face of
    a BoxMesh (axis 0/1/2, side 0 = min, 1 = max): each surface triangle on the face
    gives a third of its area times the traction to each corner, which is exact for
    linear elements. Returns (indices, forces)."""
    coord = (mesh.xs, mesh.ys, mesh.zs)[axis]
    value = coord[0] if side == 0 else coord[-1]
    nodal = {}
    for tri in mesh.surface_triangles:
        pts = [mesh.positions[n] for n in tri]
        if any(abs(pt[axis] - value) > 1e-12 for pt in pts):
            continue
        area = 0.5 * math.sqrt(sum(c * c for c in _cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))))
        for n in tri:
            nodal[n] = nodal.get(n, 0.0) + area / 3.0
    indices = sorted(nodal)
    return indices, [[traction[c] * nodal[i] for c in range(3)] for i in indices]


def face_pressure_loads(mesh, axis, side, pressure):
    """A uniform pressure pushing one face of a BoxMesh inward (face_traction_loads)."""
    inward = [0.0, 0.0, 0.0]
    inward[axis] = pressure if side == 0 else -pressure
    return face_traction_loads(mesh, axis, side, inward)


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def edges_of(triangles):
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


class BoxMesh:
    """A box of tetrahedra: xs, ys, zs are the grid coordinates along each axis.
    Each grid cube is split into 6 tetrahedra around its main diagonal, all with
    positive volume; surface triangles are wound with outward normals."""

    def __init__(self, xs, ys, zs):
        nx, ny, nz = len(xs), len(ys), len(zs)
        self.xs, self.ys, self.zs = list(xs), list(ys), list(zs)
        self.shape = (nx, ny, nz)
        self.positions = [[xs[i], ys[j], zs[k]] for k in range(nz) for j in range(ny) for i in range(nx)]
        tets = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    v = [self.node(i, j, k), self.node(i + 1, j, k), self.node(i + 1, j + 1, k), self.node(i, j + 1, k),
                         self.node(i, j, k + 1), self.node(i + 1, j, k + 1), self.node(i + 1, j + 1, k + 1),
                         self.node(i, j + 1, k + 1)]
                    for a, b, c, d in ((0, 1, 2, 6), (0, 5, 1, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 4, 5, 6), (0, 7, 4, 6)):
                        tet = [v[a], v[b], v[c], v[d]]
                        p = [self.positions[n] for n in tet]
                        if _dot(_sub(p[1], p[0]), _cross(_sub(p[2], p[0]), _sub(p[3], p[0]))) < 0.0:
                            tet[1], tet[2] = tet[2], tet[1]
                        tets.append(tet)
        self.tetrahedra = tets
        seen = {}
        for tet in tets:
            for f, opposite in (((0, 1, 2), 3), ((0, 1, 3), 2), ((0, 2, 3), 1), ((1, 2, 3), 0)):
                face = [tet[f[0]], tet[f[1]], tet[f[2]]]
                key = tuple(sorted(face))
                seen[key] = None if key in seen else (face, tet[opposite])
        surface = []
        for entry in seen.values():
            if entry is None:
                continue
            (a, b, c), opposite = entry
            pa, pb, pc = (self.positions[n] for n in (a, b, c))
            if _dot(_cross(_sub(pb, pa), _sub(pc, pa)), _sub(self.positions[opposite], pa)) > 0.0:
                b, c = c, b
            surface.append([a, b, c])
        self.surface_triangles = surface
        self.surface_nodes = sorted({n for tri in surface for n in tri})
        local = {n: i for i, n in enumerate(self.surface_nodes)}
        self.surface_local_triangles = [[local[a], local[b], local[c]] for a, b, c in surface]
        self.surface_local_edges = edges_of(self.surface_local_triangles)
        self.surface_local_positions = [self.positions[n] for n in self.surface_nodes]

    def node(self, i, j, k):
        nx, ny, _ = self.shape
        return i + j * nx + k * nx * ny

    def face(self, axis, side):
        """Indices of the nodes on one face: axis 0/1/2, side 0 (min) or 1 (max)."""
        nx, ny, nz = self.shape
        out = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = (i, j, k)[axis]
                    last = (nx, ny, nz)[axis] - 1
                    if idx == (0 if side == 0 else last):
                        out.append(self.node(i, j, k))
        return out

    def jitter_interior(self, fraction, seed=1):
        """Moves every node not on the box's faces by a random offset of up to
        `fraction` of the local cell size per axis. A homogeneous deformation stays
        exact on the jittered mesh (linear elements), but no two elements are alike."""
        if fraction <= 0.0:
            return
        rng = np.random.default_rng(seed)
        nx, ny, nz = self.shape
        axes = (self.xs, self.ys, self.zs)
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    n = self.node(i, j, k)
                    for a, idx in enumerate((i, j, k)):
                        h = min(axes[a][idx] - axes[a][idx - 1], axes[a][idx + 1] - axes[a][idx])
                        self.positions[n][a] += fraction * h * float(rng.uniform(-1.0, 1.0))
        self.surface_local_positions = [self.positions[n] for n in self.surface_nodes]

    def nearest(self, point):
        p = np.asarray(self.positions)
        return int(np.argmin(np.sum((p - np.asarray(point)) ** 2, axis=1)))


def uniform(lo, hi, cells):
    return [lo + (hi - lo) * i / cells for i in range(cells + 1)]


class ClosedBoxShape:
    """A closed triangulated box (a rigid tool) in its own frame, outward normals."""

    def __init__(self, sx, sy, sz, subdivisions=4):
        mesh = BoxMesh(uniform(-sx / 2, sx / 2, subdivisions), uniform(-sy / 2, sy / 2, subdivisions),
                       uniform(-sz / 2, sz / 2, subdivisions))
        self.positions = mesh.surface_local_positions
        self.triangles = mesh.surface_local_triangles
        self.edges = mesh.surface_local_edges


# ---------------------------------------------------------------------------
# Scene pieces
# ---------------------------------------------------------------------------
CPU_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mapping.NonLinear",
    "Sofa.Component.Mass",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Visual",
    "SofaViscoElastic",
]


def add_plugins(root):
    plugins = list(CPU_PLUGINS)
    if SIDE == "gpu":
        plugins += ["SofaCUDA", GPU_COLLISION_LIB if os.path.isfile(GPU_COLLISION_LIB) else "SofaGpuCollision"]
    root.addObject("RequiredPlugin", pluginName=plugins)


def add_tissue(parent, mesh, mat, fixed=(), partial=(), name="Tissue", label="tissue", contact=None,
               rest_positions=None, monitor_vertex=-1, loads=None, rayleigh_mass=0.0, rayleigh_stiffness=0.0,
               cuttable=False):
    """The deformable body. fixed: fully fixed node indices. partial: (index, mask)
    pairs, mask bits 1 = x, 2 = y, 4 = z. contact: None, or 'cpu'/'gpu' to add the
    pieces constraint contact needs (the CPU collision surface / constraint correction).
    loads: None or (indices, forces), constant nodal forces (a ConstantForceField named
    'load' on both sides; GpuTissueSolver reads it from its node). rayleigh_mass/stiffness:
    EulerImplicitSolver's Rayleigh damping (the same on both sides). cuttable: add a
    TetrahedronSetTopologyModifier next to the topology (tetrahedra may be removed).
    Returns (node, surface node or None)."""
    tissue = parent.addChild(name)
    partial = list(partial)
    if SIDE == "cpu":
        tissue.addObject("EulerImplicitSolver", name="odeSolver", rayleighStiffness=rayleigh_stiffness,
                         rayleighMass=rayleigh_mass)
        tissue.addObject("SparseLDLSolver", name="linearSolver", template="CompressedRowSparseMatrixMat3x3d")
        tissue.addObject("MechanicalObject", name="dofs", template="Vec3d", position=mesh.positions)
        tissue.addObject("TetrahedronSetTopologyContainer", name="topo", tetrahedra=mesh.tetrahedra)
        if cuttable:
            tissue.addObject("TetrahedronSetTopologyModifier", name="modifier")
        tissue.addObject("MeshMatrixMass", name="mass", massDensity=DENSITY)
        for k, (cls, attrs) in enumerate(mat["cpu"]):
            tissue.addObject(cls, name=f"material{k}", template="Vec3d", **attrs)
        if loads:
            tissue.addObject("ConstantForceField", name="load", template="Vec3d", indices=list(loads[0]),
                             forces=[list(f) for f in loads[1]])
        if fixed:
            tissue.addObject("FixedProjectiveConstraint", indices=list(fixed))
        for mask in range(1, 7):
            idx = [i for i, m in partial if m == mask]
            if idx:
                tissue.addObject("PartialFixedProjectiveConstraint", template="Vec3d", indices=idx,
                                 fixedDirections=[int(bool(mask & 1)), int(bool(mask & 2)), int(bool(mask & 4))])
        surface = None
        if contact:
            tissue.addObject("LinearSolverConstraintCorrection")
            surface = tissue.addChild("Surface")
            surface.addObject("MechanicalObject", name="dofs", template="Vec3d", position=mesh.surface_local_positions)
            surface.addObject("MeshTopology", triangles=mesh.surface_local_triangles, edges=mesh.surface_local_edges)
            _add_cpu_collision_models(surface)
            surface.addObject("SubsetMapping", indices=mesh.surface_nodes)
        return tissue, surface

    tissue.addObject("MechanicalObject", name="dofs", template="CudaVec3f", position=mesh.positions)
    tissue.addObject("TetrahedronSetTopologyContainer", name="topo", tetrahedra=mesh.tetrahedra)
    if cuttable:
        tissue.addObject("TetrahedronSetTopologyModifier", name="modifier")
    if fixed:
        tissue.addObject("FixedProjectiveConstraint", template="CudaVec3f", indices=list(fixed))
    if loads:
        tissue.addObject("ConstantForceField", name="load", template="CudaVec3f", indices=list(loads[0]),
                         forces=[list(f) for f in loads[1]])
    gpu = dict(mat["gpu"])
    gpu["factorization"] = FACTORIZATION
    if os.environ.get("SOFA_VALIDATION_BAND_PANEL"):
        gpu["bandPanel"] = int(os.environ["SOFA_VALIDATION_BAND_PANEL"])
    if gpu.get("hyperelasticMaterial") == "Ogden":
        gpu["ogdenTangent"] = OGDEN_TANGENT
    if partial:
        gpu["partialFixedIndices"] = [i for i, _ in partial]
        gpu["partialFixedMasks"] = [m for _, m in partial]
    tissue.addObject("GpuTissueSolver", name="odeSolver", massDensity=DENSITY,
                     rayleighMass=rayleigh_mass, rayleighStiffness=rayleigh_stiffness,
                     restPositions=rest_positions or mesh.positions, monitorVertex=monitor_vertex,
                     measureTimes=MEASURE_TIMES, compareWithCpu=COMPARE, compareEvery=COMPARE_EVERY,
                     compareFile=os.path.join(log_dir(), f"{label}_gpu_tissue_compare.csv"), **gpu)
    surface = None
    if contact:
        surface = tissue.addChild("GpuSurface")
        surface.addObject("MechanicalObject", name="dofs", template="CudaVec3f", position=mesh.positions)
        surface.addObject("MeshTopology", triangles=mesh.surface_triangles)
        surface.addObject("TriangleCollisionModel", selfCollision=False)
        surface.addObject("IdentityMapping", template="CudaVec3f,CudaVec3f", mapForces=False)
    return tissue, surface


# ---------------------------------------------------------------------------
# Contact: a deformable body against a rigid one, constraint contact with friction
# ---------------------------------------------------------------------------
CONTACT_DISTANCE = 0.0005             # m, the gap the constraint keeps (as in the poke)
ALARM_DISTANCE = 3.0 * CONTACT_DISTANCE
# Gauss-Seidel stopping rule, the same on both sides (SOFA_VALIDATION_TOLERANCE, _MAX_ITERATIONS).
CONSTRAINT_TOLERANCE = env_float("SOFA_VALIDATION_TOLERANCE", 1e-7)
CONSTRAINT_MAX_ITERATIONS = int(os.environ.get("SOFA_VALIDATION_MAX_ITERATIONS", "1000"))
# CPU collision models on each surface: "point_triangle" (default: vertex-face contacts only,
# like the GPU's contactFilter=vertexFace) or "point_line_triangle" (as the poke's CPU scene).
# With lines, SOFA's LocalMinDistance also makes edge-edge contacts between flat faces lying
# on each other; in these tests the block then drifted sideways on a level floor and fell
# through it, and the plate pushed through the block.
CPU_CONTACT_MODELS = os.environ.get("SOFA_VALIDATION_CPU_MODELS", "point_triangle")


def _add_cpu_collision_models(surface, group=None):
    extra = {} if group is None else {"group": group}
    surface.addObject("PointCollisionModel", **extra)
    if CPU_CONTACT_MODELS == "point_line_triangle":
        surface.addObject("LineCollisionModel", **extra)
    surface.addObject("TriangleCollisionModel", **extra)


def add_contact_pipeline(root, friction, bounds, cell=0.005):
    """The animation loop and collision pipeline for constraint contact.
    cpu: FreeMotionAnimationLoop, BlockGaussSeidelConstraintSolver, SOFA's collision
         pipeline (BruteForce + BVH + LocalMinDistance), FrictionContactConstraint.
    gpu: FreeMotionAnimationLoop with GpuContactConstraintSolver (add_gpu_contact_solver),
         GpuCollisionPipeline + GpuCollisionBroadPhase/NarrowPhase over `bounds`
         ((xmin, ymin, zmin), (xmax, ymax, zmax)) with cells of about `cell` metres."""
    if SIDE == "cpu":
        root.addObject("FreeMotionAnimationLoop")
        # SOFA_VALIDATION_CPU_SOLVER: BlockGaussSeidel (default, what the GPU solver follows),
        # or another of SOFA's constraint solvers (NNCG, ProjectedGaussSeidel, ...).
        solver = os.environ.get("SOFA_VALIDATION_CPU_SOLVER", "BlockGaussSeidel") + "ConstraintSolver"
        extra = {}
        if os.environ.get("SOFA_VALIDATION_SOR"):
            extra["sor"] = env_float("SOFA_VALIDATION_SOR", 1.0)
        if os.environ.get("SOFA_VALIDATION_REGULARIZATION"):
            extra["regularizationTerm"] = env_float("SOFA_VALIDATION_REGULARIZATION", 0.0)
        root.addObject(solver, name="constraintSolver",
                       tolerance=CONSTRAINT_TOLERANCE, maxIterations=CONSTRAINT_MAX_ITERATIONS, **extra)
        root.addObject("CollisionPipeline")
        root.addObject("BruteForceBroadPhase")
        root.addObject("BVHNarrowPhase")
        intersection = os.environ.get("SOFA_VALIDATION_CPU_INTERSECTION", "LocalMinDistance")
        if intersection == "LocalMinDistance":
            root.addObject("LocalMinDistance", alarmDistance=ALARM_DISTANCE, contactDistance=CONTACT_DISTANCE, angleCone=0.0)
        else:   # MinProximityIntersection, NewProximityIntersection
            root.addObject(intersection, alarmDistance=ALARM_DISTANCE, contactDistance=CONTACT_DISTANCE)
        root.addObject("CollisionResponse", response="FrictionContactConstraint", responseParams=f"mu={friction}")
        return
    (x0, y0, z0), (x1, y1, z1) = bounds
    res = [max(4, int(math.ceil((hi - lo) / cell))) for lo, hi in ((x0, x1), (y0, y1), (z0, z1))]
    root.addObject("FreeMotionAnimationLoop", constraintSolver="@gpuConstraints", computeBoundingBox=False)
    root.addObject("GpuCollisionPipeline")
    root.addObject("GpuCollisionBroadPhase", enableGPU=True, allowCPUFallback=False, logBackendStatus=False,
                   useObjectAabbCulling=False, testGpuModelBoxes=False)
    root.addObject("GpuCollisionNarrowPhase",
                   enableGPU=True, allowCPUFallback=False, logBackendStatus=False,
                   useDenseGrid=True, useIndexedDenseGridInput=True, useDirectDevicePositions=True,
                   cacheTriangleTopology=True, copyContactsToHost=False,
                   useFeatureBasedProximity=True, useBigCellFusedGeneration=True,
                   proximityComputeBarycentrics=True, proximityKeepContactsOnDevice=True,
                   proximityReadContactCounter=False, proximityMaxContacts=200000,
                   minGPUPairCount=1, contactDistance=ALARM_DISTANCE,
                   gridMinX=x0, gridMaxX=x1, gridMinY=y0, gridMaxY=y1, gridMinZ=z0, gridMaxZ=z1,
                   gridResolutionX=res[0], gridResolutionY=res[1], gridResolutionZ=res[2],
                   maxTissueTrianglesPerCell=128, maxToolTrianglesPerCell=256, maxCandidatePairs=400000)
    root.addObject("LocalMinDistance", alarmDistance=ALARM_DISTANCE, contactDistance=CONTACT_DISTANCE, angleCone=0.0)


def add_rigid_body(parent, name, shape, pose, mass, fixed=False, spring_target=None, stiffness=0.0,
                   angular_stiffness=0.0, cancel_gravity=None, group=None):
    """A rigid body with a triangulated surface (shape: positions, triangles, edges in
    its own frame) at pose [x, y, z, qx, qy, qz, qw]: its own implicit solver and direct
    solver (the constraint response reads them), fixed in place, or pulled by a spring
    toward `spring_target` (a Rigid3d state's link path). cancel_gravity: the gravity
    vector whose weight a ConstantForceField cancels (the body is held, as by a robot).
    group: SOFA collision group of its surface; bodies in one group do not collide with
    each other (both sides: the GPU broad phase keeps SOFA's rule), as the GPU contact
    solver has only tissue-tool contacts. Returns (node, surface node)."""
    body = parent.addChild(name)
    body.addObject("EulerImplicitSolver", name="odeSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    body.addObject("SparseLDLSolver", name="linearSolver", template="CompressedRowSparseMatrixd")
    body.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=[list(pose)])
    body.addObject("UniformMass", totalMass=mass)
    if cancel_gravity is not None:
        body.addObject("ConstantForceField", totalForce=[-mass * g for g in cancel_gravity] + [0.0, 0.0, 0.0])
    if fixed:
        body.addObject("FixedProjectiveConstraint", template="Rigid3d", indices=[0])
    if spring_target is not None:
        body.addObject("RestShapeSpringsForceField", stiffness=stiffness, angularStiffness=angular_stiffness,
                       external_rest_shape=spring_target)
    if SIDE == "cpu":
        body.addObject("LinearSolverConstraintCorrection")
        surface = body.addChild("Surface")
        surface.addObject("MechanicalObject", name="dofs", template="Vec3d", position=shape.positions)
        surface.addObject("MeshTopology", triangles=shape.triangles, edges=shape.edges)
        _add_cpu_collision_models(surface, group)
        surface.addObject("RigidMapping")
        return body, surface
    surface = body.addChild("GpuSurface")
    surface.addObject("MechanicalObject", name="dofs", template="CudaVec3f", position=shape.positions)
    surface.addObject("MeshTopology", triangles=shape.triangles)
    surface.addObject("TriangleCollisionModel", selfCollision=False, **({} if group is None else {"group": group}))
    surface.addObject("GpuRigidMapping", mapForces=False)
    return body, surface


def add_gpu_contact_solver(root, tissue, tissue_surface, rigid, rigid_surface, friction, label, more=()):
    """GpuContactConstraintSolver between the GPU tissue (body 1) and a rigid body
    (body 2), plus more rigid bodies: `more` = [(body node, surface node), ...]; with
    SOFA_VALIDATION_COMPARE=1 (one rigid body) it also runs SOFA's CPU constraint
    pipeline on the same contacts, stage by stage (<label>_contact_compare.csv)."""
    if SIDE == "cpu":
        return None
    extra = {}
    if os.environ.get("SOFA_VALIDATION_SOR"):
        extra["sor"] = env_float("SOFA_VALIDATION_SOR", 1.0)
    if more:
        extra.update({"additionalRigidStates": " ".join(b.dofs.getLinkPath() for b, _ in more),
                      "additionalRigidSurfaces": " ".join(s.dofs.getLinkPath() for _, s in more),
                      "additionalRigidLinearSolvers": " ".join(b.linearSolver.getLinkPath() for b, _ in more),
                      "additionalRigidOdeSolvers": " ".join(b.odeSolver.getLinkPath() for b, _ in more)})
    return root.addObject(
        "GpuContactConstraintSolver", name="gpuConstraints", friction=friction, contactDistance=CONTACT_DISTANCE,
        tolerance=CONSTRAINT_TOLERANCE, maxIterations=CONSTRAINT_MAX_ITERATIONS,
        deformableGpuSolver=tissue.odeSolver.getLinkPath(), deformableSurface=tissue_surface.dofs.getLinkPath(),
        rigidState=rigid.dofs.getLinkPath(), rigidSurface=rigid_surface.dofs.getLinkPath(),
        rigidLinearSolver=rigid.linearSolver.getLinkPath(), rigidOdeSolver=rigid.odeSolver.getLinkPath(),
        compareWithCpu=COMPARE and not more, compareEvery=COMPARE_EVERY, measureTimes=MEASURE_TIMES,
        compareFile=os.path.join(log_dir(), f"{label}_contact_compare.csv"),
        dumpContactsFile=(os.path.join(log_dir(), f"{label}_contacts.csv")
                          if os.environ.get("SOFA_VALIDATION_DUMP_CONTACTS", "0") == "1" else ""),
        **extra)


class LoadRamp(Sofa.Core.Controller):
    """Scales a ConstantForceField's forces from 0 to their full value over ramp_time
    (linearly, set before each step), as a load is applied to tissue in practice; a
    sudden large load throws a soft body far from equilibrium in one step. On the
    GPU this also exercises GpuTissueSolver's re-reading of a changed load."""

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.load = kwargs["root"], kwargs["load"]
        self.full = np.asarray(kwargs["forces"], dtype=float)
        self.ramp_time = float(kwargs["ramp_time"])
        self.last_factor = None

    def onAnimateBeginEvent(self, event):
        t = self.root.time.value + self.root.dt.value   # the load at the end of the coming step
        factor = 1.0 if self.ramp_time <= 0.0 else min(1.0, t / self.ramp_time)
        if factor != self.last_factor:
            self.load.forces.value = (factor * self.full).tolist()
            self.last_factor = factor


class PositionLogger(Sofa.Core.Controller):
    """One CSV row per step: time, the step's wall time, and the positions of the
    watched vertices (reading a GPU state copies it to the CPU: validation runs only;
    SOFA_VALIDATION_LOG_POSITIONS=0 leaves them out, for timing). With `solver` (a
    GpuTissueSolver measuring its times) also its GPU time per stage."""

    STAGES = ["gpu_ms", "gpu_material_ms", "gpu_assembly_ms", "gpu_factorize_ms", "gpu_solve_ms"]

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.dofs = kwargs["dofs"]
        self.solver = kwargs.get("solver")
        self.vertices = list(kwargs["vertices"])
        self.names = list(kwargs.get("names", [f"v{v}" for v in self.vertices]))
        path = kwargs["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv = open(path, "w")
        cols = ["time", "wall_ms"]
        if self.solver is not None:
            cols += self.STAGES
        if LOG_POSITIONS:
            cols += [f"{n}_{c}" for n in self.names for c in "xyz"]
        self.csv.write(",".join(cols) + "\n")
        self.last = time.perf_counter()

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 1000.0 * (now - self.last)
        self.last = now
        row = [self.root.time.value, wall]
        if self.solver is not None:
            if not getattr(self, "_reported", False):
                self._reported = True
                band = int(self.solver.bandwidth.value)
                print(f"GpuTissueSolver factorisation: {'band, half-bandwidth ' + str(band) if band > 0 else 'dense'}, "
                      f"{3 * len(self.dofs.position.value)} DOFs", flush=True)
            stages = [float(v) for v in self.solver.stageMilliseconds.value] or [0.0] * 4
            row += [float(self.solver.stepGpuMilliseconds.value)] + stages
        if LOG_POSITIONS:
            x = np.asarray(self.dofs.position.value, dtype=float)
            for v in self.vertices:
                row += [float(x[v][0]), float(x[v][1]), float(x[v][2])]
        self.csv.write(",".join(f"{float(v):.12g}" for v in row) + "\n")
        self.csv.flush()
        if TRACE:
            print(f"VALIDATION_T t={self.root.time.value:.3f} wall={wall:.1f}ms", flush=True)


def summary_line(label, **values):
    """Appends a key=value summary line for compare_validation.py."""
    path = os.path.join(log_dir(), f"{label}_summary.txt")
    with open(path, "a") as fh:
        fh.write(" ".join(f"{k}={v}" for k, v in values.items()) + "\n")
