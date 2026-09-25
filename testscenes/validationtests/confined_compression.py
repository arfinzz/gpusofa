"""Confined compression: CPU (SOFA) against GPU, and both against the exact answer.

A 2 cm cube of tissue sits in a rigid, frictionless box: the bottom may slide
only sideways, the x sides only in y and z, the z sides only in x and y. A uniform
pressure pushes the top face down (gravity off). The exact deformation is then the
same at every point, F = diag(1, s, 1) (uniaxial strain), whatever the mesh, and
linear tetrahedra represent it exactly: the computed stretch must equal the one the
material law gives, P_yy(s) = -pressure (validation_common.uniaxial_strain_stress),
to the solver's precision. This checks the material law, the assembly, the partly
fixed DOFs, the load and the solve together.

SOFA_VALIDATION_LOAD:
  large (default)  the pressure that gives s = 0.9 exactly (10% compression, in the
                   nonlinear range), ramped up over 1 s; the run lasts long enough for
                   a Maxwell branch to relax. Checked: the final stretch, 0.9.
                   Not more: under a dead load the compressed block's free top face can
                   wrinkle, and on this mesh the uniform state of the Ogden tissue stops
                   being stable near s = 0.83 (the model's stiffness gets a negative
                   eigenvalue there; NeoHookean holds to below 0.8). Pushed to 0.8, SOFA's
                   CPU run and the GPU run both leave the uniform state and collapse, as
                   they should. SOFA_VALIDATION_RAMP=0 applies the load at once (a stress
                   test: the first step throws the top layer far from equilibrium).
  creep            a small pressure (0.1% strain, linear range) held from t = 0: the
                   Maxwell branch's creep. Checked against the standard linear solid:
                   e(t) = p/M_inf - (p/M_inf - p/M_0) exp(-t/tau_r), M_inf = lambda + 2 mu,
                   M_0 = M_inf + 2 G1, tau_r = tau M_0 / M_inf.

Output (log dir): confined_compression_<load>_<material>_<side>.csv and _summary.txt.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import numpy as np  # noqa: E402
import Sofa.Core  # noqa: E402

import validation_common as vc  # noqa: E402

SIZE = 0.02
DIVISIONS = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "4"))
LOAD = os.environ.get("SOFA_VALIDATION_LOAD", "large")
DT = 0.01
TARGET_STRETCH = 0.9
CREEP_STRAIN = 0.001


RAMP_TIME = 0.0 if LOAD == "creep" else vc.env_float("SOFA_VALIDATION_RAMP", 1.0)
# SOFA_VALIDATION_JITTER=f moves the interior nodes by up to f of a cell (the answer stays exact).
JITTER = vc.env_float("SOFA_VALIDATION_JITTER", 0.0)


def steps():
    return 1500 if (LOAD == "creep" or vc.material()["maxwell"]) else 300


def expected(mat, pressure, t):
    """The exact stretch at time t (large: the static answer; creep: the linear
    standard-linear-solid curve)."""
    if LOAD != "creep":
        return vc.solve_uniaxial_strain(mat, -pressure)
    mu, lam = vc.small_strain_moduli(mat)
    m_inf = lam + 2.0 * mu
    if not mat["maxwell"]:
        return 1.0 - pressure / m_inf
    g1, tau, lam_v = mat["maxwell"]
    m_0 = m_inf + 2.0 * g1 + lam_v
    tau_r = tau * m_0 / m_inf
    strain = pressure / m_inf - (pressure / m_inf - pressure / m_0) * math.exp(-t / tau_r)
    return 1.0 - strain


class StretchLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs = kwargs["root"], kwargs["dofs"]
        self.top, self.bottom_y = kwargs["top"], kwargs["bottom_y"]
        self.watch = kwargs["watch"]
        self.mat, self.pressure, self.label = kwargs["mat"], kwargs["pressure"], kwargs["label"]
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,stretch,stretch_min,stretch_max,expected,side_x_drift,top_center_y\n")
        self.last = None
        self.rows = 0
        import time as _time
        self._time = _time

    def onAnimateEndEvent(self, event):
        now = self._time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        t = self.root.time.value
        x = np.asarray(self.dofs.position.value, dtype=float)
        s = (x[self.top, 1] - self.bottom_y) / SIZE
        drift = float(np.max(np.abs(x[self.watch["side"], 0] - self.watch["side_x0"])))
        exp = expected(self.mat, self.pressure, t)
        self.csv.write(f"{t:.6g},{wall:.4f},{float(np.mean(s)):.12g},{float(np.min(s)):.12g},{float(np.max(s)):.12g},"
                       f"{exp:.12g},{drift:.6g},{float(x[self.watch['center'], 1]):.12g}\n")
        self.csv.flush()
        self.rows += 1
        if self.rows == steps():
            vc.summary_line(self.label, test="confined_compression", load=LOAD, material=vc.MATERIAL, side=vc.SIDE,
                            nodes=len(x), stretch=f"{float(np.mean(s)):.10g}", expected=f"{exp:.10g}",
                            error=f"{float(np.max(np.abs(s - exp))):.3e}", pressure_pa=f"{self.pressure:.6g}")


def createScene(root):
    mat = vc.material()
    label = vc.run_label(f"confined_compression_{LOAD}" + ("_jitter" if JITTER > 0.0 else ""))
    root.name = "ConfinedCompression"
    root.dt = DT
    root.gravity = [0.0, 0.0, 0.0]
    vc.add_plugins(root)
    root.addObject("DefaultAnimationLoop")

    half = SIZE / 2
    mesh = vc.BoxMesh(vc.uniform(-half, half, DIVISIONS), vc.uniform(0.0, SIZE, DIVISIONS),
                      vc.uniform(-half, half, DIVISIONS))
    mesh.jitter_interior(JITTER)
    # Rollers: bottom y, x sides x, z sides z (bits 1 = x, 2 = y, 4 = z; edges and corners combine).
    masks = {}
    for axis, bit in ((1, 2), (0, 1), (2, 4)):
        for side in (0, 1):
            if axis == 1 and side == 1:
                continue  # the top is loaded, not held
            for n in mesh.face(axis, side):
                masks[n] = masks.get(n, 0) | bit
    fixed = [n for n, m in masks.items() if m == 7]
    partial = [(n, m) for n, m in sorted(masks.items()) if m != 7]

    if LOAD == "creep":
        mu, lam = vc.small_strain_moduli(mat)
        pressure = CREEP_STRAIN * (lam + 2.0 * mu)
    else:
        # StVenantKirchhoff loses strong ellipticity before s = 0.9 in this state (the
        # smallest acoustic-tensor eigenvalue is -56 Pa there, +461 Pa at 0.95): it
        # collapses, identically on the CPU and the GPU. It is tested at 5%.
        target = 0.95 if mat.get("core") == "StVenantKirchhoff" else TARGET_STRETCH
        pressure = -vc.uniaxial_strain_stress(mat, target)
    loads = vc.face_pressure_loads(mesh, axis=1, side=1, pressure=pressure)

    tissue, _ = vc.add_tissue(root, mesh, mat, fixed=fixed, partial=partial, label=label, loads=loads)
    if RAMP_TIME > 0.0:
        root.addObject(vc.LoadRamp(name="ramp", root=root, load=tissue.load, forces=loads[1], ramp_time=RAMP_TIME))
    top = mesh.face(1, 1)
    side = mesh.face(0, 1)
    watch = {"side": side, "side_x0": np.asarray([mesh.positions[n][0] for n in side]),
             "center": mesh.nearest([0.0, SIZE, 0.0])}
    root.addObject(StretchLogger(name="logger", root=root, dofs=tissue.dofs, top=top, bottom_y=0.0, watch=watch,
                                 mat=mat, pressure=pressure, label=label))
    print(f"ConfinedCompression: side={vc.SIDE} material={vc.MATERIAL} load={LOAD} pressure={pressure:.6g} Pa "
          f"nodes={len(mesh.positions)} tetrahedra={len(mesh.tetrahedra)} steps={steps()} "
          f"expected_final_stretch={expected(mat, pressure, 1e9):.8f}", flush=True)
    return root
