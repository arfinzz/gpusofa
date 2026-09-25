"""Block on an incline: contact with Coulomb friction, CPU (SOFA) against GPU and
against the sliding law.

A soft 2 x 1 x 2 cm block rests on a fixed rigid floor. Gravity is tilted by
SOFA_VALIDATION_ANGLE degrees (default 25) along x, so the floor acts as an
incline, with friction coefficient SOFA_VALIDATION_FRICTION (default 0.3,
i.e. sticking below atan(0.3) = 16.7 degrees). Coulomb's law gives the block's
centre of mass
  sticking  (tan(angle) <= mu): no motion (only the block's own small shear)
  sliding   (tan(angle) >  mu): x(t) = a t^2 / 2 with a = g (sin(angle) - mu cos(angle))
whatever the block's stiffness, since the floor's total normal force is its
weight's normal part.

cpu: SOFA's collision pipeline (LocalMinDistance), FrictionContactConstraint,
     BlockGaussSeidelConstraintSolver, LinearSolverConstraintCorrection.
gpu: GPU collision detection and GpuContactConstraintSolver (the GPU tissue on
     one side, the floor as a fixed rigid body on the other).

The floor's top face is one quad by default. SOFA_VALIDATION_FLOOR_CELLS=n splits
it into n x n cells: then floor vertices lie in the block's path, and SOFA's CPU
contact (LocalMinDistance, MinProximity or NewProximityIntersection) either lets
the block fall through the floor or slows its slide by half or more: a floor
vertex just ahead of the block makes an oblique contact with the block's leading
edge. The GPU contact keeps only contacts that leave their vertex's surface
(GpuContactConstraintSolver.vertexConeFilter) and follows Coulomb's law on any
floor mesh.

Output (log dir): incline_friction_a<angle>_mu<mu>_<material>_<side>.csv, _summary.txt.
"""

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import numpy as np  # noqa: E402
import Sofa.Core  # noqa: E402

import validation_common as vc  # noqa: E402

ANGLE = vc.env_float("SOFA_VALIDATION_ANGLE", 25.0)
MU = vc.env_float("SOFA_VALIDATION_FRICTION", 0.3)
G = 9.81
DT = 0.01
STEPS = int(os.environ.get("SOFA_VALIDATION_STEPS", "50"))
BLOCK = (0.02, 0.01, 0.02)
FLOOR = (0.60, 0.01, 0.06)
FLOOR_X0 = -0.02            # the floor's left end


def expected_x(t):
    """Coulomb sliding, as implicit Euler integrates it with this time step: v_n = a n dt,
    x_n = sum of v_k dt = a t (t + dt) / 2 (the continuous a t^2 / 2 plus one step's lag)."""
    theta = math.radians(ANGLE)
    if math.tan(theta) <= MU:
        return 0.0
    return 0.5 * G * (math.sin(theta) - MU * math.cos(theta)) * t * (t + DT)


class SlideLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs, self.label = kwargs["root"], kwargs["dofs"], kwargs["label"]
        self.solver = kwargs.get("solver")
        self.x0 = None
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,com_dx,com_y,expected_dx,bottom_min_y\n")
        self.last = None
        self.rows = 0

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        x = np.asarray(self.dofs.position.value, dtype=float)
        com = x.mean(axis=0)
        if self.x0 is None:
            self.x0 = float(com[0])
        t = self.root.time.value
        dx = float(com[0]) - self.x0 if self.rows else 0.0
        self.csv.write(f"{t:.6g},{wall:.4f},{dx:.12g},{float(com[1]):.12g},{expected_x(t):.12g},{float(x[:, 1].min()):.12g}\n")
        self.csv.flush()
        if vc.TRACE and self.solver is not None:
            s = self.solver
            print(f"INCLINE_T t={t:.3f} contacts={s.currentContacts.value} rows={s.currentConstraints.value} "
                  f"sweeps={s.currentIterations.value} normal_impulse={s.normalImpulse.value:.6g} com_y={float(com[1]):.6g}",
                  flush=True)
        elif vc.TRACE:
            print(f"INCLINE_T t={t:.3f} com_y={float(com[1]):.6g} com_dx={dx:.6g}", flush=True)
        self.rows += 1
        if self.rows == STEPS:
            exp = expected_x(t)
            vc.summary_line(self.label, test="incline_friction", angle_deg=ANGLE, mu=MU, material=vc.MATERIAL,
                            side=vc.SIDE, time_s=f"{t:.4g}", com_dx_m=f"{dx:.8g}", expected_dx_m=f"{exp:.8g}",
                            error_m=f"{abs(dx - exp):.3e}", bottom_min_y_m=f"{float(x[:, 1].min()):.6g}")


def createScene(root):
    mat = vc.material()
    label = vc.run_label(f"incline_friction_a{ANGLE:g}_mu{MU:g}")
    root.name = "InclineFriction"
    root.dt = DT
    theta = math.radians(ANGLE)
    gravity = [G * math.sin(theta), -G * math.cos(theta), 0.0]
    root.gravity = gravity
    vc.add_plugins(root)
    bounds = ((FLOOR_X0 - 0.01, -FLOOR[1] - 0.005, -FLOOR[2] / 2 - 0.005),
              (FLOOR_X0 + FLOOR[0] + 0.01, BLOCK[1] + 0.01, FLOOR[2] / 2 + 0.005))
    vc.add_contact_pipeline(root, MU, bounds, cell=vc.env_float("SOFA_VALIDATION_GRID_CELL", 0.005))

    bx, by, bz = BLOCK
    d0 = vc.CONTACT_DISTANCE
    mesh = vc.BoxMesh(vc.uniform(-bx / 2, bx / 2, 4), vc.uniform(d0, d0 + by, 2), vc.uniform(-bz / 2, bz / 2, 4))
    tissue, surface = vc.add_tissue(root, mesh, mat, label=label, contact=vc.SIDE)

    floor_shape = vc.ClosedBoxShape(*FLOOR, subdivisions=int(os.environ.get("SOFA_VALIDATION_FLOOR_CELLS", "1")))
    # SOFA_VALIDATION_FLOOR (diagnostics): fixed (default) = a rigid body held by
    # FixedProjectiveConstraint; heavy = a free rigid body 1000x heavier than the block, its
    # weight cancelled.
    floor_mode = os.environ.get("SOFA_VALIDATION_FLOOR", "fixed")
    floor, floor_surface = vc.add_rigid_body(
        root, "Floor", floor_shape, [FLOOR_X0 + FLOOR[0] / 2, -FLOOR[1] / 2, 0.0, 0.0, 0.0, 0.0, 1.0],
        mass=1.0 if floor_mode == "fixed" else 5.0, fixed=floor_mode == "fixed",
        cancel_gravity=gravity if floor_mode == "heavy" else None)
    solver = vc.add_gpu_contact_solver(root, tissue, surface, floor, floor_surface, MU, label)
    root.addObject(SlideLogger(name="logger", root=root, dofs=tissue.dofs, label=label, solver=solver))
    print(f"InclineFriction: side={vc.SIDE} material={vc.MATERIAL} angle={ANGLE} deg mu={MU} "
          f"({'slides' if math.tan(theta) > MU else 'sticks'}; a={G * (math.sin(theta) - MU * math.cos(theta)):.4f} m/s^2) "
          f"nodes={len(mesh.positions)} steps={STEPS}", flush=True)
    return root
