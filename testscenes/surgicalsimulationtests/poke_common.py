"""Shared setup for the tissue-poke scenes (CPU and GPU versions).

A liver-like block of tissue rests on a table. A rigid round-tip probe moves
down, pokes it once, holds, and pulls back out. Everything is in SI units:
metres, kilograms, seconds, pascals.

Both scenes build the identical mesh, material, probe, trajectory and logger
from this module, so their force curves can be compared directly; they differ
only in where the work runs.

The probe is not moved directly. Like a haptic device, it is pulled along its
path by a stiff spring (a "virtual coupling"), so the force the tissue pushes
back with is simply spring stiffness x (probe position - target position). That
gives both scenes the same, direct force measurement.
"""

import math
import os
import time

import numpy as np
import Sofa.Core


def _env_float(name, default):
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DT = _env_float("SOFA_POKE_DT", 0.01)                     # s
GRAVITY = [0.0, -9.81, 0.0]

# Tissue block: top surface at y = 0, resting on a table at y = -BLOCK_HEIGHT.
BLOCK_HALF_WIDTH = 0.030                                  # 6 cm x 6 cm footprint
BLOCK_HEIGHT = 0.030                                      # 3 cm thick
FINE_HALF_WIDTH = _env_float("SOFA_POKE_FINE_HALF", 0.006)  # finely meshed zone under the probe
FINE_STEP = _env_float("SOFA_POKE_FINE_STEP", 0.002)      # element size there
GROWTH = _env_float("SOFA_POKE_GROWTH", 1.5)              # element growth away from it

# Tissue material: liver-like viscoelastic Ogden, i.e. a standard linear solid
# whose long-term spring is Ogden hyperelastic. Second Piola-Kirchhoff stress:
#   long-term spring  S = mu1/alpha1 J^(-alpha1/3) (C^(alpha1/2-1) - tr(C^(alpha1/2))/3 C^-1) + k0 ln(J) C^-1
#   viscous branch    S = 2 G1 (E - E_viscous),  E_viscous relaxing toward E with time constant tau
# The long-term shear modulus is mu1 / 2; the viscous branch adds G1 at the
# instant of loading and then relaxes away. SOFA builds it from its core Ogden and
# SofaViscoElastic's Maxwell element (MATERIAL_MODE below; add_tissue_material()).
DENSITY = 1060.0          # kg/m^3, soft tissue
OGDEN_MU1 = 2000.0        # Pa  -> long-term shear modulus 1 kPa
OGDEN_ALPHA1 = 6.0        # stiffens under large stretch
VISCOUS_G1 = 1000.0       # Pa  -> instant shear modulus 2 kPa, half of it relaxes
RELAXATION_TAU = 0.58     # s   (liver value used by Taylor et al.; also in SofaCUDA's TLED)
BULK_K0 = 20000.0         # Pa  -> nearly incompressible (Poisson ratio about 0.45)

# Probe: stainless-steel rod with a round tip.
PROBE_RADIUS = 0.0025     # 5 mm diameter
PROBE_LENGTH = 0.040
PROBE_SEGMENTS = 24       # around the axis
PROBE_TIP_RINGS = 8       # rings on the round tip

# Motion of the tip, measured from the undeformed tissue top (y = 0).
START_HEIGHT = 0.005      # starts 5 mm above
POKE_DEPTH = _env_float("SOFA_POKE_DEPTH", 0.008)     # goes 8 mm below
SPEED = _env_float("SOFA_POKE_SPEED", 0.005)          # 5 mm/s
SETTLE_TIME = 1.0         # s: let gravity settle the tissue first
HOLD_TIME = 1.0           # s at full depth: the force relaxes here
REST_TIME = 1.0           # s after pulling out: the tissue recovers here

# Virtual coupling between the scripted target and the probe.
TOOL_MASS = 0.1           # kg (its weight is carried by the "robot": gravity off)
COUPLING_STIFFNESS = _env_float("SOFA_POKE_COUPLING_STIFFNESS", 2000.0)  # N/m
COUPLING_ANGULAR_STIFFNESS = 10.0                                         # N m / rad

# Contact
CONTACT_DISTANCE = _env_float("SOFA_POKE_CONTACT_DISTANCE", 0.0005)  # 0.5 mm
ALARM_DISTANCE = 3.0 * CONTACT_DISTANCE
FRICTION = _env_float("SOFA_POKE_FRICTION", 0.1)                     # wet tissue on steel
# GPU scene only (penalty contact): stiffness PER CONTACT. The probe tip makes
# hundreds of contacts (vertex-face, face-vertex, edge-edge), so the total is a
# few thousand N/m, about 100x the tissue's own push-back (20-50 N/m): enough to
# keep the probe on the surface while the implicit solve stays well conditioned.
# (200 N/m per contact made ~40,000 N/m in total and the solve blew up at first touch.)
PENALTY_STIFFNESS = _env_float("SOFA_POKE_PENALTY_STIFFNESS", 10.0)  # N/m per contact

# Diagnostics: SOFA_POKE_TRACE=1 prints one line per frame (time, tip, force).
TRACE = os.environ.get("SOFA_POKE_TRACE", "0") == "1"


# Plugins the tissue material needs (both scenes list them in RequiredPlugin).
MATERIAL_PLUGINS = ["Sofa.Component.SolidMechanics.FEM.HyperElastic", "SofaViscoElastic"]

# SOFA_POKE_MATERIAL picks how the material is built (the same on both scenes):
#   core   (default) SOFA's own Ogden (TetrahedronHyperelasticityFEMForceField, fixed in
#          SOFA in November 2025: exact eigenvectors, exact stiffness) for the long-term
#          spring, plus SofaViscoElastic's Maxwell element for the viscous branch.
#          The material as written. Slow on the CPU (SOFA's Ogden: about 6 s per step here).
#   split  SofaViscoElastic's SLSOgdenFirstOrder (G1 = 0) + Maxwell element: the earlier
#          default. SLSOgdenFirstOrder's Eigen call computes no eigenvectors in SOFA
#          v25.12, so its stress is wrong once the tissue deforms (GpuTissueSolver
#          reproduces it for comparisons).
#   single the one-piece SLSOgdenFirstOrder (diagnostics: its stiffness leaves out the
#          viscous branch, see add_tissue_material).
MATERIAL_MODE = os.environ.get("SOFA_POKE_MATERIAL", "core")
if MATERIAL_MODE not in ("core", "split", "single"):
    raise ValueError(f"SOFA_POKE_MATERIAL must be core, split or single, not {MATERIAL_MODE!r}")


def tissue_material_parameters():
    """SofaViscoElastic modes ('split', 'single'): the material's two parameter sets, as
    add_tissue_material() gives them to the CPU force fields: SLSOgdenFirstOrder's
    [mu1, alpha1, G1, tau, k0] and MaxwellFirstOrder's [G1, tau, lambda] (empty in
    'single' mode)."""
    if MATERIAL_MODE == "single":
        return [OGDEN_MU1, OGDEN_ALPHA1, VISCOUS_G1, RELAXATION_TAU, BULK_K0], []
    return [OGDEN_MU1, OGDEN_ALPHA1, 0.0, RELAXATION_TAU, BULK_K0], [VISCOUS_G1, RELAXATION_TAU, 0.0]


def gpu_tissue_material(ogden_eigenvectors="sofa", ogden_tangent="robust"):
    """GpuTissueSolver's material settings for MATERIAL_MODE: the same material as
    add_tissue_material() builds from SOFA's CPU components."""
    maxwell = [VISCOUS_G1, RELAXATION_TAU, 0.0]
    if MATERIAL_MODE == "core":
        return {"hyperelasticMaterial": "Ogden", "hyperelasticParameters": [OGDEN_MU1, OGDEN_ALPHA1, BULK_K0],
                "maxwellParameters": maxwell, "ogdenTangent": ogden_tangent}
    ogden, maxwell = tissue_material_parameters()
    settings = {"ogdenParameters": ogden, "ogdenEigenvectors": ogden_eigenvectors}
    if maxwell:
        settings["maxwellParameters"] = maxwell
    return settings


def material_note():
    return {"core": "viscoelastic Ogden (SOFA core Ogden + SofaViscoElastic Maxwell)",
            "split": "viscoelastic Ogden (SofaViscoElastic SLSOgdenFirstOrder + Maxwell)",
            "single": "viscoelastic Ogden (SofaViscoElastic SLSOgdenFirstOrder, one piece)"}[MATERIAL_MODE]


def add_tissue_material(node):
    """Adds the viscoelastic Ogden material to the tetrahedral mesh in `node`.

    Both branches act in parallel on the same mesh:
      * long-term spring: Ogden. 'core': SOFA's own Ogden, whose stiffness matrix is
        exact. 'split': SofaViscoElastic's SLSOgdenFirstOrder with G1 = 0.
      * viscous branch: SofaViscoElastic's Maxwell element, parameters [G1, tau,
        lambda = 0], whose stiffness matrix is its instant stiffness G1.

    Why two components ('single' shows it): SLSOgdenFirstOrder computes the whole
    viscoelastic stress in one component, but the stiffness matrix it gives the
    implicit solver leaves out the viscous branch (every viscoelastic material in
    that plugin does). The solver then sees the tissue about half as stiff as it
    is right after loading, overshoots every step, the solution rings from one
    step to the next, and under the probe an element turns inside out: the poke
    blew up 2.5 mm in.
    """
    if MATERIAL_MODE == "single":
        node.addObject("TetrahedronViscoHyperelasticityFEMForceField", name="material",
                       materialName="SLSOgdenFirstOrder",
                       ParameterSet=f"{OGDEN_MU1} {OGDEN_ALPHA1} {VISCOUS_G1} {RELAXATION_TAU} {BULK_K0}")
        return
    if MATERIAL_MODE == "core":
        node.addObject("TetrahedronHyperelasticityFEMForceField", name="elastic", template="Vec3d",
                       materialName="Ogden", ParameterSet=f"{OGDEN_MU1} {OGDEN_ALPHA1} {BULK_K0}")
    else:
        node.addObject("TetrahedronViscoHyperelasticityFEMForceField", name="elastic",
                       materialName="SLSOgdenFirstOrder",
                       ParameterSet=f"{OGDEN_MU1} {OGDEN_ALPHA1} 0 {RELAXATION_TAU} {BULK_K0}")
    node.addObject("TetrahedronViscoelasticityFEMForceField", name="viscous", materialName="MaxwellFirstOrder",
                   ParameterSet=f"{VISCOUS_G1} {RELAXATION_TAU} 0")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
TRAVEL = START_HEIGHT + POKE_DEPTH
T_DOWN_START = SETTLE_TIME
T_DOWN_END = T_DOWN_START + TRAVEL / SPEED
T_HOLD_END = T_DOWN_END + HOLD_TIME
T_UP_END = T_HOLD_END + TRAVEL / SPEED
TOTAL_TIME = T_UP_END + REST_TIME


def target_tip_height(t):
    """Scripted height of the probe tip at time t."""
    if t <= T_DOWN_START:
        return START_HEIGHT
    if t <= T_DOWN_END:
        return START_HEIGHT - SPEED * (t - T_DOWN_START)
    if t <= T_HOLD_END:
        return -POKE_DEPTH
    if t <= T_UP_END:
        return -POKE_DEPTH + SPEED * (t - T_HOLD_END)
    return START_HEIGHT


def total_steps():
    return int(round(TOTAL_TIME / DT))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _graded_half_axis(half, fine_half, fine_step, growth):
    """Coordinates from 0 to `half`: fine_step steps up to fine_half, then growing."""
    coords = [0.0]
    x = 0.0
    while x + fine_step <= fine_half + 1e-12:
        x += fine_step
        coords.append(x)
    step = fine_step
    while True:
        step *= growth
        if x + step >= half - 0.4 * step:
            break
        x += step
        coords.append(x)
    if half - coords[-1] > 1e-12:
        coords.append(half)
    return coords


def _edges_of(triangles):
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


class TissueMesh:
    """Tetrahedral block: fine under the probe, coarser away from it."""

    def __init__(self):
        half = _graded_half_axis(BLOCK_HALF_WIDTH, FINE_HALF_WIDTH, FINE_STEP, GROWTH)
        xs = [-c for c in reversed(half[1:])] + half
        down = _graded_half_axis(BLOCK_HEIGHT, FINE_HALF_WIDTH, FINE_STEP, GROWTH)
        ys = [-c for c in reversed(down)]           # bottom (-height) ... top (0)
        nx, ny, nz = len(xs), len(ys), len(xs)
        self.shape = (nx, ny, nz)

        def node(i, j, k):
            return i + j * nx + k * nx * ny

        self.positions = [[xs[i], ys[j], xs[k]] for k in range(nz) for j in range(ny) for i in range(nx)]

        tetrahedra = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    v = [node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                         node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)]
                    for a, b, c, d in ((0, 1, 2, 6), (0, 5, 1, 6), (0, 2, 3, 6),
                                       (0, 3, 7, 6), (0, 4, 5, 6), (0, 7, 4, 6)):
                        tet = [v[a], v[b], v[c], v[d]]
                        p = [self.positions[n] for n in tet]
                        if _dot(_sub(p[1], p[0]), _cross(_sub(p[2], p[0]), _sub(p[3], p[0]))) < 0.0:
                            tet[1], tet[2] = tet[2], tet[1]      # positive volume
                        tetrahedra.append(tet)
        self.tetrahedra = tetrahedra

        # Boundary faces, wound so their normals point OUT of the block (the GPU
        # contact uses the winding to tell inside from outside).
        seen = {}
        for tet in tetrahedra:
            for f, opposite in (((0, 1, 2), 3), ((0, 1, 3), 2), ((0, 2, 3), 1), ((1, 2, 3), 0)):
                face = [tet[f[0]], tet[f[1]], tet[f[2]]]
                key = tuple(sorted(face))
                if key in seen:
                    seen[key] = None
                else:
                    seen[key] = (face, tet[opposite])
        surface = []
        for entry in seen.values():
            if entry is None:
                continue
            (a, b, c), opposite = entry
            pa, pb, pc = (self.positions[n] for n in (a, b, c))
            normal = _cross(_sub(pb, pa), _sub(pc, pa))
            if _dot(normal, _sub(self.positions[opposite], pa)) > 0.0:
                b, c = c, b
            surface.append([a, b, c])
        self.surface_triangles = surface

        self.bottom_indices = [node(i, 0, k) for k in range(nz) for i in range(nx)]
        self.top_center_index = node(xs.index(0.0), ny - 1, xs.index(0.0))

        # Surface-only point set (for the CPU collision models).
        self.surface_nodes = sorted({n for tri in surface for n in tri})
        local = {n: i for i, n in enumerate(self.surface_nodes)}
        self.surface_local_triangles = [[local[a], local[b], local[c]] for a, b, c in surface]
        self.surface_local_edges = _edges_of(self.surface_local_triangles)
        self.surface_local_positions = [self.positions[n] for n in self.surface_nodes]


class ProbeMesh:
    """Closed round-tip rod in its own frame: tip at the origin, shaft along +y."""

    def __init__(self):
        r, length = PROBE_RADIUS, PROBE_LENGTH
        n, rings = PROBE_SEGMENTS, PROBE_TIP_RINGS
        positions = [[0.0, 0.0, 0.0]]                         # tip pole
        ring_start = []
        for ring in range(1, rings + 1):                      # round tip, up to its equator
            phi = 0.5 * math.pi * ring / rings
            rho, y = r * math.sin(phi), r - r * math.cos(phi)
            ring_start.append(len(positions))
            positions += [[rho * math.cos(2 * math.pi * s / n), y, rho * math.sin(2 * math.pi * s / n)]
                          for s in range(n)]
        ring_start.append(len(positions))                     # top of the shaft
        positions += [[r * math.cos(2 * math.pi * s / n), length, r * math.sin(2 * math.pi * s / n)]
                      for s in range(n)]
        top = len(positions)
        positions.append([0.0, length, 0.0])

        triangles = []
        first = ring_start[0]
        for s in range(n):
            triangles.append([0, first + (s + 1) % n, first + s])
        for lower, upper in zip(ring_start[:-1], ring_start[1:]):
            for s in range(n):
                a, b = lower + s, lower + (s + 1) % n
                c, d = upper + (s + 1) % n, upper + s
                triangles += [[a, b, c], [a, c, d]]
        last = ring_start[-1]
        for s in range(n):
            triangles.append([top, last + s, last + (s + 1) % n])

        # Wind every triangle outward (the probe is convex: outward = away from the axis).
        for tri in triangles:
            p = [positions[v] for v in tri]
            centroid = [(p[0][q] + p[1][q] + p[2][q]) / 3.0 for q in range(3)]
            if tri[0] == top:
                outward = (0.0, 1.0, 0.0)
            else:
                outward = _sub(centroid, (0.0, min(max(centroid[1], r), length), 0.0))
            if _dot(_cross(_sub(p[1], p[0]), _sub(p[2], p[0])), outward) < 0.0:
                tri[1], tri[2] = tri[2], tri[1]

        self.positions = positions
        self.triangles = triangles
        self.edges = _edges_of(triangles)


def rigid_pose(tip_height):
    return [[0.0, tip_height, 0.0, 0.0, 0.0, 0.0, 1.0]]


def add_probe_body(node, external_rest_shape):
    """Rigid probe body: mass, its weight carried (as by a robot arm), and the
    coupling spring that pulls it toward the scripted target."""
    node.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=rigid_pose(START_HEIGHT))
    node.addObject("UniformMass", totalMass=TOOL_MASS)
    # Setting gravity to zero on a child node is not honoured here, so the weight
    # is cancelled explicitly; otherwise the spring would carry 0.98 N at rest.
    node.addObject("ConstantForceField", totalForce=[0.0, -TOOL_MASS * GRAVITY[1], 0.0, 0.0, 0.0, 0.0])
    node.addObject("RestShapeSpringsForceField", stiffness=COUPLING_STIFFNESS,
                   angularStiffness=COUPLING_ANGULAR_STIFFNESS, external_rest_shape=external_rest_shape)


def add_camera(root):
    """The view in SOFA's window: 11 cm away, 27 degrees above the tissue top, on the spot
    the probe presses. Without a camera in the scene, runSofa 25.12 leaves the one it
    creates at the origin, inside the tissue: BaseViewer::load() calls only bwdInit() on
    it, and BaseCamera::setDefaultView() moves a camera only after its init()."""
    root.addObject("InteractiveCamera", name="camera", position=[0.0, 0.045, 0.10], lookAt=[0.0, -0.005, 0.0])


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class ProbeDriver(Sofa.Core.Controller):
    """Moves the coupling target along the scripted path, one step ahead."""

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.target = kwargs["target"]

    def onAnimateBeginEvent(self, event):
        t = self.root.time.value + self.root.dt.value
        self.target.position.value = rigid_pose(target_tip_height(t))


def _tet_volumes(positions, tets):
    p = positions[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0


class PokeLogger(Sofa.Core.Controller):
    """Writes time, depth and the tissue's push on the probe to a CSV, plus a summary.

    gap = probe tip height - tissue surface height under it (negative = the tip is
    below the surface point, i.e. overlap). min_volume_ratio = the most squashed
    tetrahedron's volume over its rest volume (0 = crushed flat, < 0 = inside out).
    """

    COLUMNS = ["time", "phase", "target_tip_y", "probe_tip_y", "surface_center_y",
               "indentation", "gap", "force_x", "force_y", "force_z", "min_volume_ratio", "wall_ms"]

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.probe = kwargs["probe"]
        self.target = kwargs["target"]
        self.tissue = kwargs["tissue"]
        # A tissue on the GPU: its solver reports the surface point and the smallest
        # volume ratio itself, so the logger never copies the tissue back to the CPU.
        self.tissue_monitor = kwargs.get("tissue_monitor")
        self.top_center = kwargs["top_center_index"]
        self.tets = np.asarray(kwargs["tetrahedra"], dtype=np.int64)
        self.rest_volumes = _tet_volumes(np.asarray(kwargs["rest_positions"], dtype=float), self.tets)
        self.label = kwargs["label"]
        self.notes = kwargs.get("notes", "")
        log_dir = kwargs["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, self.label + ".csv")
        self.summary_path = os.path.join(log_dir, self.label + "_summary.txt")
        self.csv = open(self.csv_path, "w")
        self.csv.write(",".join(self.COLUMNS) + "\n")
        self.rows = []
        self.settled_top = None
        self.last_wall = time.perf_counter()
        self.done = False

    @staticmethod
    def phase(t):
        if t <= T_DOWN_START:
            return "settle"
        if t <= T_DOWN_END:
            return "press"
        if t <= T_HOLD_END:
            return "hold"
        if t <= T_UP_END:
            return "retract"
        return "rest"

    def onAnimateEndEvent(self, event):
        if self.done:
            return
        now = time.perf_counter()
        wall_ms = 1000.0 * (now - self.last_wall)
        self.last_wall = now

        t = self.root.time.value
        probe = self.probe.position.value[0]
        target = self.target.position.value[0]
        if self.tissue_monitor is not None:
            surface_y = float(self.tissue_monitor.monitorPosition.value[1])
            min_volume_ratio = float(self.tissue_monitor.minVolumeRatio.value)
        else:
            tissue_positions = np.asarray(self.tissue.position.value, dtype=float)
            surface_y = float(tissue_positions[self.top_center][1])
            min_volume_ratio = float(np.min(_tet_volumes(tissue_positions, self.tets) / self.rest_volumes))
        if self.settled_top is None and t >= T_DOWN_START - 0.5 * DT:
            self.settled_top = surface_y
        reference_top = self.settled_top if self.settled_top is not None else 0.0

        # Force the tissue exerts on the probe = what the coupling spring must hold back.
        force = [COUPLING_STIFFNESS * (float(probe[q]) - float(target[q])) for q in range(3)]
        row = [t, self.phase(t), float(target[1]), float(probe[1]), surface_y,
               reference_top - float(probe[1]), float(probe[1]) - surface_y,
               force[0], force[1], force[2], min_volume_ratio, wall_ms]
        self.rows.append(row)
        self.csv.write(",".join(f"{v:.9g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        # Every row: a run stopped early (fixed frame count, crash) must keep its tail.
        self.csv.flush()
        if TRACE:
            # One line per frame on stdout, so it interleaves with the scene's own
            # messages (e.g. the GPU contact counts) in the run log.
            print(f"POKE_T t={t:.3f} tip={1000.0 * float(probe[1]):.4f}mm "
                  f"surface={1000.0 * surface_y:.4f}mm force_y={force[1]:.5f}N "
                  f"min_vol={min_volume_ratio:.4f}", flush=True)
        if t >= TOTAL_TIME - 0.5 * DT:
            self.finish()

    def finish(self):
        self.done = True
        self.csv.close()
        rows = self.rows
        col = {name: i for i, name in enumerate(self.COLUMNS)}

        def at(name, r):
            return r[col[name]]

        pressing = [r for r in rows if at("phase", r) in ("press", "hold")]
        peak = max(pressing, key=lambda r: at("force_y", r)) if pressing else rows[-1]
        # Contact starts when the tissue pushes AND the tip is near the surface; the
        # spring's first pull at 1 s (accelerating the probe's mass) is not contact.
        onset = next((r for r in pressing
                      if at("force_y", r) > 1e-3 and at("gap", r) < 2.0 * ALARM_DISTANCE), None)
        hold = [r for r in rows if at("phase", r) == "hold"]
        rest = [r for r in rows if at("phase", r) == "rest"]
        hold_start = hold[0] if hold else peak
        hold_end = hold[-1] if hold else peak
        relaxation = (1.0 - at("force_y", hold_end) / at("force_y", hold_start)
                      if hold and at("force_y", hold_start) > 0 else float("nan"))
        min_gap = min((at("gap", r) for r in pressing), default=float("nan"))
        min_volume = min(at("min_volume_ratio", r) for r in rows)
        steps = [at("wall_ms", r) for r in rows[1:]]
        lines = [
            f"label={self.label}",
            f"notes={self.notes}",
            f"steps={len(rows)}",
            f"dt_s={DT}",
            f"settled_surface_y_mm={1000.0 * (self.settled_top or 0.0):.4f}",
            f"contact_onset_time_s={at('time', onset) if onset else float('nan'):.3f}",
            f"peak_force_N={at('force_y', peak):.6f}",
            f"indentation_at_peak_mm={1000.0 * at('indentation', peak):.4f}",
            f"force_hold_start_N={at('force_y', hold_start):.6f}",
            f"force_hold_end_N={at('force_y', hold_end):.6f}",
            f"relaxation_during_hold_percent={100.0 * relaxation:.2f}",
            f"min_gap_probe_to_surface_center_mm={1000.0 * min_gap:.4f}",
            f"min_tet_volume_ratio={min_volume:.4f}",
            f"final_surface_offset_mm={1000.0 * (at('surface_center_y', rows[-1]) - (self.settled_top or 0.0)):.4f}",
            f"final_force_N={at('force_y', rest[-1] if rest else rows[-1]):.6f}",
            f"avg_step_wall_ms={sum(steps) / max(1, len(steps)):.3f}",
            f"csv={self.csv_path}",
        ]
        with open(self.summary_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("POKE_SUMMARY " + " ".join(lines[2:16]), flush=True)


class ConstraintSolverStatsLogger(Sofa.Core.Controller):
    """One CSV row per step from SOFA's constraint solver outputs (CPU scene):
    contacts (constraint groups), constraint rows, Gauss-Seidel iterations, error."""

    COLUMNS = ["time", "contacts", "rows", "iterations", "gs_error"]

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.solver = kwargs["solver"]
        path = kwargs["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv = open(path, "w")
        self.csv.write(",".join(self.COLUMNS) + "\n")

    def onAnimateEndEvent(self, event):
        s = self.solver
        row = [self.root.time.value, s.currentNumConstraintGroups.value, s.currentNumConstraints.value,
               s.currentIterations.value, s.currentError.value]
        self.csv.write(",".join(f"{v:.9g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        self.csv.flush()


def default_log_dir(scene_dir):
    """<repo>/output/benchmark_logs unless SOFA_BENCHMARK_LOG_DIR is set."""
    repo_root = os.path.abspath(os.path.join(scene_dir, os.pardir, os.pardir))
    return os.environ.get("SOFA_BENCHMARK_LOG_DIR", os.path.join(repo_root, "output", "benchmark_logs"))
