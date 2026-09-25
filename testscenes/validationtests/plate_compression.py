"""Plate compression: the contact force, CPU (SOFA) against GPU and against the
exact uniaxial-stress answer.

A soft 2 cm cube stands on a frictionless base (its bottom nodes may slide
sideways; one node is pinned and one more held in z so it cannot drift or spin).
A rigid plate, pulled by a stiff spring toward a target that moves down at
2 mm/s, presses the top face through frictionless contact (gravity off), to 15%
compression, then holds for 0.5 s.

With a frictionless plate and base the deformation is uniform, F = diag(a, s, a)
with free sides (uniaxial stress), whatever the mesh: the force the block pushes
back with is its reference area times the nominal stress, A0 |P_y(s)|, with the
lateral stretch a from P_x = 0 (validation_common.uniaxial_stress_state). The
plate's force is read from its spring (k times the plate's lag behind the
target), the same way on both sides.

This checks the contact response as a force: detection, the constraint rows,
the compliance of both bodies, the Gauss-Seidel solve and the correction.

Output (log dir): plate_compression_<material>_<side>.csv, _summary.txt.
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

SIZE = 0.02
DIVISIONS = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "4"))
DT = 0.01
SPEED = 0.002             # m/s
DEPTH = 0.15 * SIZE       # 15% compression
HOLD = 0.5
STEPS = int(round((DEPTH / SPEED + HOLD) / DT))
PLATE = (0.03, 0.005, 0.03)
PLATE_MASS = 0.05
SPRING = 5000.0           # N/m
ANGULAR_SPRING = 10.0     # N m / rad


def target_bottom(t):
    """Height of the plate's lower face on the target: touching at 0, then down."""
    return SIZE + vc.CONTACT_DISTANCE - min(DEPTH, SPEED * t)


def pose_for_bottom(y):
    return [0.0, y + PLATE[1] / 2, 0.0, 0.0, 0.0, 0.0, 1.0]


class PlateDriver(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.target = kwargs["root"], kwargs["target"]

    def onAnimateBeginEvent(self, event):
        t = self.root.time.value + self.root.dt.value
        self.target.position.value = [pose_for_bottom(target_bottom(t))]


class PlateLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs, self.plate, self.target = kwargs["root"], kwargs["dofs"], kwargs["plate"], kwargs["target"]
        self.top, self.side, self.mat, self.label = kwargs["top"], kwargs["side"], kwargs["mat"], kwargs["label"]
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,stretch,lateral_stretch,force_n,expected_force_n,expected_lateral\n")
        self.last = None
        self.rows = 0
        self.elastic = mat_is_elastic(self.mat)

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        x = np.asarray(self.dofs.position.value, dtype=float)
        s = float(np.mean(x[self.top, 1])) / SIZE
        a = float(np.mean(x[self.side, 0])) / (SIZE / 2)
        force = SPRING * (float(self.plate.position.value[0][1]) - float(self.target.position.value[0][1]))
        if self.elastic and s < 1.0:
            a_exp, p_y = vc.uniaxial_stress_state(self.mat, s)
            f_exp = -SIZE * SIZE * p_y
        else:
            a_exp, f_exp = float("nan"), float("nan")
        self.csv.write(f"{self.root.time.value:.6g},{wall:.4f},{s:.12g},{a:.12g},{force:.12g},{f_exp:.12g},{a_exp:.12g}\n")
        self.csv.flush()
        self.rows += 1
        if self.rows == STEPS:
            vc.summary_line(self.label, test="plate_compression", material=vc.MATERIAL, side=vc.SIDE,
                            stretch=f"{s:.8g}", lateral_stretch=f"{a:.8g}", expected_lateral=f"{a_exp:.8g}",
                            force_n=f"{force:.8g}", expected_force_n=f"{f_exp:.8g}",
                            force_error_percent=f"{100.0 * (force - f_exp) / f_exp:.4f}" if f_exp == f_exp else "nan")


def mat_is_elastic(mat):
    return mat.get("core") is not None and not mat.get("maxwell")


def createScene(root):
    mat = vc.material()
    label = vc.run_label("plate_compression")
    root.name = "PlateCompression"
    root.dt = DT
    root.gravity = [0.0, 0.0, 0.0]
    vc.add_plugins(root)
    half = SIZE / 2
    bounds = ((-PLATE[0] / 2 - 0.006, -0.005, -PLATE[2] / 2 - 0.006), (PLATE[0] / 2 + 0.006, SIZE + 0.012, PLATE[2] / 2 + 0.006))
    vc.add_contact_pipeline(root, 0.0, bounds)

    mesh = vc.BoxMesh(vc.uniform(-half, half, DIVISIONS), vc.uniform(0.0, SIZE, DIVISIONS), vc.uniform(-half, half, DIVISIONS))
    bottom = mesh.face(1, 0)
    pin = mesh.nearest([0.0, 0.0, 0.0])
    guide = mesh.nearest([half, 0.0, 0.0])
    partial = [(n, 7 if n == pin else (2 | 4) if n == guide else 2) for n in bottom]
    tissue, surface = vc.add_tissue(root, mesh, mat, fixed=[pin], partial=[p for p in partial if p[0] != pin],
                                    label=label, contact=vc.SIDE)

    target = root.addChild("Target")
    target.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=[pose_for_bottom(target_bottom(0.0))])
    plate_shape = vc.ClosedBoxShape(*PLATE, subdivisions=4)
    plate, plate_surface = vc.add_rigid_body(root, "Plate", plate_shape, pose_for_bottom(target_bottom(0.0)), PLATE_MASS,
                                             spring_target=target.dofs.getLinkPath(), stiffness=SPRING,
                                             angular_stiffness=ANGULAR_SPRING)
    vc.add_gpu_contact_solver(root, tissue, surface, plate, plate_surface, 0.0, label)
    root.addObject(PlateDriver(name="driver", root=root, target=target.dofs))
    top = mesh.face(1, 1)
    side = [n for n in mesh.face(0, 1)]
    root.addObject(PlateLogger(name="logger", root=root, dofs=tissue.dofs, plate=plate.dofs, target=target.dofs,
                               top=top, side=side, mat=mat, label=label))
    print(f"PlateCompression: side={vc.SIDE} material={vc.MATERIAL} nodes={len(mesh.positions)} steps={STEPS} "
          f"depth={DEPTH} m", flush=True)
    return root
