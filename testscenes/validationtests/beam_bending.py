"""Cantilever beam: CPU (SOFA) against GPU, and against beam theory as the mesh is refined.

A 10 cm x 2 cm x 2 cm beam of tissue is clamped at x = 0 and loaded by a uniform
downward shear traction on its free end (gravity off). Rayleigh mass damping
(critical for the first bending mode) brings it to rest.

SOFA_VALIDATION_LOAD:
  small (default)  tip deflection about 1% of the length (linear range). Checked:
                   the static tip deflection against Timoshenko beam theory,
                   d = P L^3 / (3 E I) + P L / (k G A), with E, G the material's
                   small-strain moduli and k = 10 (1 + nu) / (12 + 11 nu). Linear
                   tetrahedra are too stiff in bending, and more so the closer the
                   material is to incompressible (locking), so the ratio d_FE / d
                   approaches 1 only as SOFA_VALIDATION_DIVISIONS (cells across the
                   thickness; 5x as many along the length) grows: run several.
  large            tip deflection about 30% of the length (geometric nonlinearity),
                   ramped up over 2 s: CPU against GPU only.

Output (log dir): beam_bending_<load>_d<divisions>_<material>_<side>.csv, _summary.txt.
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

LENGTH, HEIGHT, WIDTH = 0.10, 0.02, 0.02
DIVISIONS = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "4"))
LOAD = os.environ.get("SOFA_VALIDATION_LOAD", "small")
DT = 0.02
STEPS = int(os.environ.get("SOFA_VALIDATION_STEPS", "500"))   # 10 s


def timoshenko_tip_deflection(mat, force):
    e, nu = vc.youngs_modulus(mat)
    g = e / (2.0 * (1.0 + nu))
    inertia = WIDTH * HEIGHT**3 / 12.0
    area = WIDTH * HEIGHT
    kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    return force * LENGTH**3 / (3.0 * e * inertia) + force * LENGTH / (kappa * g * area)


def first_bending_frequency(mat):
    """rad/s, Euler-Bernoulli."""
    e, _ = vc.youngs_modulus(mat)
    inertia = WIDTH * HEIGHT**3 / 12.0
    return 1.8751**2 * math.sqrt(e * inertia / (vc.DENSITY * WIDTH * HEIGHT * LENGTH**4))


class TipLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs, self.tip = kwargs["root"], kwargs["dofs"], kwargs["tip"]
        self.label, self.force, self.theory = kwargs["label"], kwargs["force"], kwargs["theory"]
        self.nodes = kwargs["nodes"]
        self.y0 = kwargs["y0"]
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,tip_dx,tip_dy,tip_dz\n")
        self.last = None
        self.rows = 0

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        x = np.asarray(self.dofs.position.value, dtype=float)[self.tip]
        d = x - self.y0
        self.csv.write(f"{self.root.time.value:.6g},{wall:.4f},{d[0]:.12g},{d[1]:.12g},{d[2]:.12g}\n")
        self.csv.flush()
        self.rows += 1
        if self.rows == STEPS:
            deflection = -d[1]
            vc.summary_line(self.label, test="beam_bending", load=LOAD, material=vc.MATERIAL, side=vc.SIDE,
                            divisions=DIVISIONS, nodes=self.nodes, tip_deflection_m=f"{deflection:.10g}",
                            timoshenko_m=f"{self.theory:.10g}", ratio=f"{deflection / self.theory:.6f}",
                            tip_force_n=f"{self.force:.6g}")


def createScene(root):
    mat = vc.material()
    label = vc.run_label(f"beam_bending_{LOAD}_d{DIVISIONS}")
    root.name = "BeamBending"
    root.dt = DT
    root.gravity = [0.0, 0.0, 0.0]
    vc.add_plugins(root)
    root.addObject("DefaultAnimationLoop")

    n = DIVISIONS
    mesh = vc.BoxMesh(vc.uniform(0.0, LENGTH, 5 * n), vc.uniform(-HEIGHT / 2, HEIGHT / 2, n),
                      vc.uniform(-WIDTH / 2, WIDTH / 2, n))
    clamped = mesh.face(0, 0)
    # Tip force for ~1% (small) or ~30% (large) deflection by beam theory.
    unit = timoshenko_tip_deflection(mat, 1.0)
    target = 0.01 if LOAD == "small" else 0.30
    force = target * LENGTH / unit
    loads = vc.face_traction_loads(mesh, axis=0, side=1, traction=[0.0, -force / (HEIGHT * WIDTH), 0.0])
    omega = first_bending_frequency(mat)
    tissue, _ = vc.add_tissue(root, mesh, mat, fixed=clamped, label=label, loads=loads,
                              rayleigh_mass=2.0 * omega)
    if LOAD != "small":
        # A large load is ramped up over 2 s, as it would be applied in practice.
        root.addObject(vc.LoadRamp(name="ramp", root=root, load=tissue.load, forces=loads[1], ramp_time=2.0))
    tip = mesh.nearest([LENGTH, 0.0, 0.0])
    root.addObject(TipLogger(name="logger", root=root, dofs=tissue.dofs, tip=tip, label=label, force=force,
                             theory=timoshenko_tip_deflection(mat, force), nodes=len(mesh.positions),
                             y0=np.asarray(mesh.positions[tip], dtype=float)))
    print(f"BeamBending: side={vc.SIDE} material={vc.MATERIAL} load={LOAD} divisions={n} nodes={len(mesh.positions)} "
          f"tetrahedra={len(mesh.tetrahedra)} tip_force={force:.6g} N theory={timoshenko_tip_deflection(mat, force):.6g} m "
          f"rayleighMass={2.0 * omega:.4g} steps={STEPS}", flush=True)
    return root
