"""Grasp and lift: two rigid jaws squeeze a soft block and lift it by friction.
CPU (SOFA) against GPU, and both against Coulomb's law.

A soft 2 cm cube rests on a fixed pedestal (1.4 cm wide, so the closed jaws pass
beside it). Two rigid jaws, each pulled by a spring toward a target (a haptic
coupling, as the poke's probe), close on its x sides, squeeze it by 2 x SQUEEZE,
hold, then lift by 4 mm. Friction coefficient SOFA_VALIDATION_FRICTION (default
0.5) for every contact (jaws and pedestal). The jaws and the pedestal are one
collision group: tool-tool contacts are not computed on either side.

No tool edge or tool vertex touches the block: each jaw face is one quad, taller
and deeper than the block and reaching 5 mm below it, and the lift (4 mm) keeps
the jaws' bottom edges below the block even when it slips; the pedestal is one quad
on top. The contacts are the block's vertices on the tools' faces. (With 1.6 cm jaws
whose bottom edges pressed into the block's side, and a jaw vertex in the middle of
each face, both SOFA's CPU pipeline and the GPU lost the block within 0.4 s: a sharp
edge between the vertices of a coarse mesh is what vertex-face contacts cannot hold.)

Time step 0.005 s (SOFA_VALIDATION_DT). With 0.01 s the implicit step's matrix is
dominated by the stiffness (dt^2 K against M), and the contact problem of this
pinched, nearly incompressible block became so badly conditioned near full squeeze
that the Gauss-Seidel stopped converging (1,000 sweeps and more) for mu >= 0.3 and
the block blew up: with SOFA's BlockGaussSeidel, NNCG, under-relaxation and
regularisation alike, and on the GPU. With 0.005 s it converges in about 30 sweeps.

Coulomb's law: the jaws carry the block when the friction they can give exceeds
its weight, 2 mu N >= m g, with N the squeezing force of each jaw while lifting;
otherwise the block slips out and stays on the pedestal. (N drops by about 15% when
the lift starts: until then the squeezed block, bulging, presses down on the pedestal
through the jaws' friction. The squeezing force before the lift overestimates what
the jaws can carry.) The test reports, from the start of the
lift, the jaws' rise, the block's rise and the difference (slip), the measured N,
and 2 mu N / (m g).

cpu: SOFA's collision pipeline (LocalMinDistance), FrictionContactConstraint,
     BlockGaussSeidelConstraintSolver; three rigid bodies (two jaws, the pedestal).
gpu: GPU collision detection and one GpuContactConstraintSolver for all three
     rigid bodies (additionalRigidStates, ...), solved together.

Output (log dir): grasp_lift_mu<mu>_<material>_<side>.csv (per step: block rise,
squeeze force, constraint rows, sweeps, error, most squashed element) and _summary.txt.
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

MU = vc.env_float("SOFA_VALIDATION_FRICTION", 0.5)
G = 9.81
DT = vc.env_float("SOFA_VALIDATION_DT", 0.005)   # see the docstring: 0.01 is too long a step here
BLOCK = 0.02
JAW = (0.006, 0.03, 0.03)          # thickness (x), height, depth: covers the block's side, 5 mm beyond it all round
SQUEEZE = vc.env_float("SOFA_VALIDATION_SQUEEZE", 0.002)   # each jaw's target goes this far into the block
LIFT = 0.004                       # less than the jaws reach below the block: their edges never meet it
T_CLOSE, T_HOLD, T_LIFT, T_END = 0.4, 0.6, 1.4, 1.8
STEPS = int(round(T_END / DT))
JAW_MASS = 0.05
SPRING = 200.0                     # N/m
ANGULAR_SPRING = vc.env_float("SOFA_VALIDATION_ANGULAR_SPRING", 5.0)
PEDESTAL = (0.014, 0.01, 0.03)     # narrower than the closed jaws' gap
SUPPORT = os.environ.get("SOFA_VALIDATION_GRASP_SUPPORT", "1") != "0"   # 0: no pedestal, no gravity (diagnostics)
TOOLS = 1                          # collision group of the jaws and the pedestal: no tool-tool contacts


def jaw_targets(t):
    """x of the right jaw's inner face target (the left one mirrors it) and the lift."""
    d0 = vc.CONTACT_DISTANCE
    x_open = BLOCK / 2 + d0 + 0.003
    x_closed = BLOCK / 2 + d0 - SQUEEZE
    closing = min(1.0, t / T_CLOSE)
    x = x_open + (x_closed - x_open) * closing
    lift = 0.0 if t <= T_HOLD else LIFT * min(1.0, (t - T_HOLD) / (T_LIFT - T_HOLD))
    return x, lift


def jaw_pose(side, t):
    x, lift = jaw_targets(t)
    cx = side * (x + JAW[0] / 2)
    cy = vc.CONTACT_DISTANCE + BLOCK / 2 + lift
    return [cx, cy, 0.0, 0.0, 0.0, 0.0, 1.0]


def _tet_volumes(x, tets):
    a, b, c, d = (x[tets[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


class JawDriver(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.targets = kwargs["root"], kwargs["targets"]

    def onAnimateBeginEvent(self, event):
        t = self.root.time.value + self.root.dt.value
        for side, target in self.targets:
            target.position.value = [jaw_pose(side, t)]


class GraspLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs, self.label = kwargs["root"], kwargs["dofs"], kwargs["label"]
        self.jaws, self.targets = kwargs["jaws"], kwargs["targets"]
        self.mass = kwargs["mass"]
        self.tets = np.asarray(kwargs["tets"], dtype=int)
        self.rest_volumes = _tet_volumes(np.asarray(kwargs["rest"], dtype=float), self.tets)
        self.solver = kwargs.get("solver")
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,block_rise,jaw_target_rise,jaw_rise,squeeze_force_n,block_com_x,constraints,"
                       "iterations,error,min_volume_ratio,right_jaw_contact_fy\n")
        self.y0 = None
        self.jaw_y0 = None
        self.at_lift = None   # (block y, jaw y) when the lift starts
        self.last = None
        self.rows = 0
        self.forces = []
        self.lift_forces = []

    def solver_stats(self):
        """Constraint rows, sweeps and error of this step (SOFA's solver or the GPU one)."""
        s = self.solver
        if s is None:
            return 0, 0, 0.0
        rows = s.currentConstraints.value if vc.SIDE == "gpu" else s.currentNumConstraints.value
        return int(rows), int(s.currentIterations.value), float(s.currentError.value)

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        t = self.root.time.value
        x = np.asarray(self.dofs.position.value, dtype=float)
        com = x.mean(axis=0)
        jaw_y = float(np.mean([float(jaw.position.value[0][1]) for _, jaw in self.jaws]))
        if self.y0 is None:
            self.y0 = float(com[1])
            self.jaw_y0 = jaw_y
        if self.at_lift is None and t >= T_HOLD - 0.5 * DT:
            self.at_lift = (float(com[1]), jaw_y)
        # Each jaw's squeezing force: its spring's pull, k (target - jaw) along x, toward the block.
        f = []
        for (side, jaw), (_, target) in zip(self.jaws, self.targets):
            jx = float(jaw.position.value[0][0])
            tx = float(target.position.value[0][0])
            f.append(SPRING * side * (jx - tx))
        squeeze = 0.5 * (f[0] + f[1])
        target_rise = jaw_targets(t)[1]
        rise = float(com[1]) - self.y0
        constraints, iterations, error = self.solver_stats()
        # GPU: the vertical contact force on the right jaw (J^T lambda / dt): while the block
        # slips it is mu times the jaw's normal force (Coulomb, at the contacts).
        jaw_fy = ""
        if vc.SIDE == "gpu" and self.solver is not None:
            jaw_fy = f"{float(self.solver.rigidContactForce.value[1]):.6g}"
        volume = float(np.min(_tet_volumes(x, self.tets) / self.rest_volumes))
        self.csv.write(f"{t:.6g},{wall:.4f},{rise:.10g},{target_rise:.10g},{jaw_y - self.jaw_y0:.10g},{squeeze:.10g},"
                       f"{float(com[0]):.10g},{constraints},{iterations},{error:.4g},{volume:.6g},{jaw_fy}\n")
        self.csv.flush()
        # The squeezing force before the lift, and while lifting: it drops by about 15% once
        # the lift starts (the squeezed block had been pressing down on the pedestal through
        # the jaws' friction), and Coulomb's balance holds with the force at that time.
        if T_HOLD - 0.1 <= t <= T_HOLD:
            self.forces.append(squeeze)
        if T_LIFT - 0.2 <= t <= T_LIFT:
            self.lift_forces.append(squeeze)
        self.rows += 1
        if self.rows == STEPS:
            n_hold = float(np.mean(self.forces)) if self.forces else float("nan")
            n = float(np.mean(self.lift_forces)) if self.lift_forces else float("nan")
            capacity = 2.0 * MU * n / (self.mass * G)
            # From the start of the lift: the jaws' rise, the block's, and the difference (slip).
            block_lift = float(com[1]) - self.at_lift[0]
            jaw_lift = jaw_y - self.at_lift[1]
            vc.summary_line(self.label, test="grasp_lift", material=vc.MATERIAL, side=vc.SIDE, mu=MU, dt=DT,
                            squeeze_hold_n=f"{n_hold:.6g}", squeeze_lift_n=f"{n:.6g}", weight_n=f"{self.mass * G:.6g}",
                            friction_capacity_ratio=f"{capacity:.4f}",
                            expected="lifted" if capacity >= 1.0 else "slips",
                            jaw_lift_m=f"{jaw_lift:.6g}", block_lift_m=f"{block_lift:.6g}",
                            slip_m=f"{jaw_lift - block_lift:.6g}",
                            result="lifted" if block_lift > 0.5 * jaw_lift else "slipped")


def createScene(root):
    mat = vc.material()
    label = vc.run_label(f"grasp_lift_mu{MU:g}")
    root.name = "GraspLift"
    root.dt = DT
    root.gravity = [0.0, -G if SUPPORT else 0.0, 0.0]
    vc.add_plugins(root)
    reach = BLOCK / 2 + vc.CONTACT_DISTANCE + 0.003 + JAW[0] + 0.005   # the open jaws' outer faces, and a margin
    bounds = ((-reach, -PEDESTAL[1] - 0.005, -JAW[2] / 2 - 0.005), (reach, BLOCK + LIFT + JAW[1], JAW[2] / 2 + 0.005))
    vc.add_contact_pipeline(root, MU, bounds)

    half = BLOCK / 2
    d0 = vc.CONTACT_DISTANCE
    cells = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "4"))
    mesh = vc.BoxMesh(vc.uniform(-half, half, cells), vc.uniform(d0, d0 + BLOCK, cells), vc.uniform(-half, half, cells))
    tissue, surface = vc.add_tissue(root, mesh, mat, label=label, contact=vc.SIDE)

    pedestal, pedestal_surface = None, None
    if SUPPORT:
        pedestal, pedestal_surface = vc.add_rigid_body(root, "Pedestal", vc.ClosedBoxShape(*PEDESTAL, subdivisions=1),
                                                       [0.0, -PEDESTAL[1] / 2, 0.0, 0.0, 0.0, 0.0, 1.0], mass=1.0,
                                                       fixed=True, group=TOOLS)
    jaws, targets, bodies = [], [], []
    for side, name in ((1, "RightJaw"), (-1, "LeftJaw")):
        target = root.addChild(name + "Target")
        target.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=[jaw_pose(side, 0.0)])
        jaw, jaw_surface = vc.add_rigid_body(root, name, vc.ClosedBoxShape(*JAW, subdivisions=1), jaw_pose(side, 0.0),
                                             JAW_MASS, spring_target=target.dofs.getLinkPath(), stiffness=SPRING,
                                             angular_stiffness=ANGULAR_SPRING, cancel_gravity=root.gravity.value, group=TOOLS)
        jaws.append((side, jaw.dofs))
        targets.append((side, target.dofs))
        bodies.append((jaw, jaw_surface))
    (right, right_surface), (left, left_surface) = bodies
    gpu_solver = vc.add_gpu_contact_solver(root, tissue, surface, right, right_surface, MU, label,
                                           more=[(left, left_surface)] + ([(pedestal, pedestal_surface)] if SUPPORT else []))
    solver = gpu_solver if vc.SIDE == "gpu" else root.constraintSolver
    root.addObject(JawDriver(name="driver", root=root, targets=targets))
    mass = vc.DENSITY * BLOCK ** 3
    root.addObject(GraspLogger(name="logger", root=root, dofs=tissue.dofs, label=label, jaws=jaws, targets=targets,
                               mass=mass, solver=solver, tets=mesh.tetrahedra, rest=mesh.positions))
    print(f"GraspLift: side={vc.SIDE} material={vc.MATERIAL} mu={MU} squeeze={SQUEEZE} m weight={mass * G:.4g} N "
          f"nodes={len(mesh.positions)} steps={STEPS}", flush=True)
    return root
