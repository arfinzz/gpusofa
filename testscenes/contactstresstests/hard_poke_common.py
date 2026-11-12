"""Hard-poke stress tests: a large round-tip tool pressed deep into tissue, so that the
contact patch holds hundreds of contacts, with every timer of the step recorded.

Two scenes share this module:
  diaphragm_poke.py  a thin flat sheet (diaphragm-like muscle), clamped on its rim and
                     poked from above: it deflects like a drum skin around the tool.
  cuboid_poke.py     a block of liver-like tissue standing on a table, poked from above.

The tool is a rigid rod with a hemispherical tip, 8 mm in radius, finely triangulated
(48 segments around, 16 rings on the tip) so that both surfaces bring many vertices
into contact. As in the poke (poke_common.py), it is pulled along a scripted path by a
stiff spring (a virtual coupling): the force is the spring's stiffness times its stretch.

Each scene runs on the GPU (SOFA_VALIDATION_SIDE=gpu, the default) or on SOFA's CPU
components (SOFA_VALIDATION_SIDE=cpu); the bodies, contact and solvers come from
validation_common.py.

What a run records (scripts/run_hard_poke_wsl.sh runs each kind):
  always                               <label>.csv, one row per step: the frame's wall time
                                       (Python clock, frame to frame and inside SOFA's step),
                                       the tool, the force, contacts, rows, sweeps
  SOFA_VALIDATION_MEASURE_TIMES=1      + the GPU stages, timed with CUDA events: tissue
                                       material, assembly, factorisation, solve; contact
                                       rows, compliance, Gauss-Seidel, correction
  SOFA_HARD_POKE_COLLISION_TIMES=1     + the collision's GPU time and its raw contact counts
                                       (<label>_collision_timings.csv, from
                                       GpuPipelineBenchmarkController); adds one read-back
  runSofa --computationTimeSampling 1  SOFA's own timer: every stage of every step
<label>_parameters.json lists every setting of the run; <label>_summary.txt its results.
"""

import json
import math
import os
import sys
import time

import numpy as np
import Sofa.Core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, os.pardir, "validationtests"))
sys.path.append(os.path.join(HERE, os.pardir, "surgicalsimulationtests"))

import poke_common as pc  # noqa: E402
import validation_common as vc  # noqa: E402

env = vc.env_float

# ---------------------------------------------------------------------------
# Parameters (SI units). SOFA_HARD_POKE_* overrides the ones marked env.
# ---------------------------------------------------------------------------
DT = env("SOFA_HARD_POKE_DT", 0.01)                       # s, env
GRAVITY = [0.0, -9.81, 0.0]

# Tool: rigid rod with a hemispherical tip, held by a "robot" (its weight cancelled).
TOOL_RADIUS = env("SOFA_HARD_POKE_TOOL_RADIUS", 0.008)    # m, env
TOOL_LENGTH = 0.030                                       # m
TOOL_SEGMENTS = int(env("SOFA_HARD_POKE_TOOL_SEGMENTS", 48))   # around the axis, env
TOOL_RINGS = int(env("SOFA_HARD_POKE_TOOL_RINGS", 16))         # rings on the tip, env
TOOL_MASS = 0.1                                           # kg
COUPLING_STIFFNESS = env("SOFA_HARD_POKE_COUPLING_STIFFNESS", 10000.0)   # N/m, env
COUPLING_ANGULAR_STIFFNESS = 10.0                         # N m / rad

# Motion of the tip, measured from the undeformed tissue top (y = 0).
START_HEIGHT = 0.005                                      # starts 5 mm above
SPEED = env("SOFA_HARD_POKE_SPEED", 0.020)                # m/s, env
SETTLE_TIME = 0.5                                         # s: gravity settles the tissue
HOLD_TIME = 0.5                                           # s at full depth
REST_TIME = 0.5                                           # s after pulling out

# Contact: the poke's and the validation tests' values.
FRICTION = env("SOFA_HARD_POKE_FRICTION", 0.1)            # env
MAX_RAW_CONTACTS = int(env("SOFA_HARD_POKE_MAX_RAW_CONTACTS", 2000000))   # GPU contact buffer, env
COLLISION_CELL = 0.004                                    # m, collision grid cell

# The two tissues. depth: how far below the undeformed top the tip goes. The
# diaphragm sags under its own weight first, so part of its depth is the sag.
SCENES = {
    "diaphragm": {
        "description": "thin flat sheet (diaphragm-like muscle), 8 x 8 cm, 4.5 mm thick, clamped on its rim",
        "half_width": 0.040, "thickness": 0.0045, "layers": 3,
        "fine_half": 0.016, "fine_step": 0.0015, "growth": 1.5,
        "depth": 0.020,
        # muscle-like: long-term shear modulus 10 kPa (mu1 / 2), nearly incompressible,
        # plus a Maxwell branch adding 10 kPa that relaxes with 0.58 s
        "material": vc.make_material("Ogden", (20000.0, 6.0, 200000.0), (10000.0, 0.58, 0.0)),
        "material_note": "muscle-like Ogden (mu1 20 kPa, alpha 6, k0 200 kPa) + Maxwell (G1 10 kPa, tau 0.58 s)",
    },
    "cuboid": {
        "description": "block of liver-like tissue, 6 x 6 cm footprint, 3 cm tall, bottom fixed to a table",
        "half_width": 0.030, "height": 0.030,
        "fine_half": 0.014, "fine_step": 0.002, "growth": 1.5,
        "depth": 0.012,
        # the poke's liver: long-term shear modulus 1 kPa + a 1 kPa Maxwell branch
        "material": vc.make_material("Ogden", vc.OGDEN, vc.MAXWELL),
        "material_note": "liver-like Ogden (mu1 2 kPa, alpha 6, k0 20 kPa) + Maxwell (G1 1 kPa, tau 0.58 s)",
    },
}

COLLISION_TIMES = os.environ.get("SOFA_HARD_POKE_COLLISION_TIMES", "0") == "1"
VISUAL = os.environ.get("SOFA_HARD_POKE_VISUAL", "1") != "0"


def scene_settings(name):
    """The scene's settings with the SOFA_HARD_POKE_* overrides applied."""
    s = dict(SCENES[name])
    s["fine_step"] = env("SOFA_HARD_POKE_FINE_STEP", s["fine_step"])
    s["depth"] = env("SOFA_HARD_POKE_DEPTH", s["depth"])
    return s


class Trajectory:
    """Scripted tip height: settle, press down at SPEED, hold, pull back, rest."""

    def __init__(self, depth):
        self.travel = START_HEIGHT + depth
        self.depth = depth
        self.t_down_start = SETTLE_TIME
        self.t_down_end = self.t_down_start + self.travel / SPEED
        self.t_hold_end = self.t_down_end + HOLD_TIME
        self.t_up_end = self.t_hold_end + self.travel / SPEED
        self.total_time = self.t_up_end + REST_TIME

    def height(self, t):
        if t <= self.t_down_start:
            return START_HEIGHT
        if t <= self.t_down_end:
            return START_HEIGHT - SPEED * (t - self.t_down_start)
        if t <= self.t_hold_end:
            return -self.depth
        if t <= self.t_up_end:
            return -self.depth + SPEED * (t - self.t_hold_end)
        return START_HEIGHT

    def phase(self, t):
        if t <= self.t_down_start:
            return "settle"
        if t <= self.t_down_end:
            return "press"
        if t <= self.t_hold_end:
            return "hold"
        if t <= self.t_up_end:
            return "retract"
        return "rest"

    def steps(self):
        return int(round(self.total_time / DT))


def total_steps(name):
    return Trajectory(scene_settings(name)["depth"]).steps()


def build_mesh(name, s):
    """The tissue's tetrahedral mesh (fine under the tool, growing away from it), its
    fixed nodes and the vertex at the middle of its top."""
    half = pc._graded_half_axis(s["half_width"], s["fine_half"], s["fine_step"], s["growth"])
    xs = [-c for c in reversed(half[1:])] + half
    if name == "diaphragm":
        ys = vc.uniform(-s["thickness"], 0.0, s["layers"])
        mesh = vc.BoxMesh(xs, ys, xs)
        fixed = sorted(set(mesh.face(0, 0) + mesh.face(0, 1) + mesh.face(2, 0) + mesh.face(2, 1)))   # the rim
        bottom = -s["thickness"]
    else:
        down = pc._graded_half_axis(s["height"], s["fine_half"], s["fine_step"], s["growth"])
        ys = [-c for c in reversed(down)]
        mesh = vc.BoxMesh(xs, ys, xs)
        fixed = mesh.face(1, 0)                                # the base, on the table
        bottom = -s["height"]
    return mesh, fixed, mesh.nearest((0.0, 0.0, 0.0)), bottom


def tool_pose(tip_height):
    return [0.0, tip_height, 0.0, 0.0, 0.0, 0.0, 1.0]


def _tet_volumes(positions, tets):
    p = positions[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0


class HardPokeController(Sofa.Core.Controller):
    """Drives the coupling target and logs every step, in one controller (each Python
    controller costs about 0.8 ms per step in SofaPython3's event dispatch).

    frame_ms: wall time from the end of the previous step to the end of this one (what a
    user waits per frame). step_ms: from AnimateBeginEvent to AnimateEndEvent (inside
    SOFA's step: free motion, collision, contact). The GPU columns are the solvers'
    outputs: their counts always, their stage times only with measureTimes."""

    CONTACT_STAGES = ["rows_ms", "contact_factorize_ms", "compliance_ms", "gauss_seidel_ms", "correction_ms"]
    TISSUE_STAGES = ["tissue_material_ms", "tissue_assembly_ms", "tissue_factorize_ms", "tissue_solve_ms"]
    COLUMNS = (["step", "time", "phase", "target_tip_y", "tool_tip_y", "monitor_y", "indentation",
                "spring_force_y", "contact_force_y", "min_volume_ratio", "frame_ms", "step_ms",
                "contacts", "rows", "sweeps", "gs_error", "warm_started", "contacts_on_tissue_vertices",
                "contacts_on_tool_vertices", "min_separation_tissue_vertices_m", "min_separation_tool_vertices_m",
                "contact_gpu_ms"] + CONTACT_STAGES
               + ["tissue_gpu_ms"] + TISSUE_STAGES)

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.target = kwargs["target"]
        self.tool = kwargs["tool"]
        self.tissue = kwargs["tissue"]
        self.contact = kwargs.get("contact")              # GPU: GpuContactConstraintSolver
        self.cpu_solver = kwargs.get("cpu_solver")        # CPU: SOFA's constraint solver
        self.tissue_solver = kwargs.get("tissue_solver")  # GPU: GpuTissueSolver
        self.monitor = kwargs["monitor"]
        self.trajectory = kwargs["trajectory"]
        self.label = kwargs["label"]
        self.tets = np.asarray(kwargs["tetrahedra"], dtype=np.int64)
        self.rest_volumes = _tet_volumes(np.asarray(kwargs["rest_positions"], dtype=float), self.tets)
        out = vc.log_dir()
        os.makedirs(out, exist_ok=True)
        self.csv_path = os.path.join(out, self.label + ".csv")
        self.summary_path = os.path.join(out, self.label + "_summary.txt")
        self.csv = open(self.csv_path, "w")
        self.csv.write(",".join(self.COLUMNS) + "\n")
        self.rows = []
        self.step = 0
        self.settled_top = None
        self.run_start = time.perf_counter()
        self.last_end = self.run_start
        self.step_begin = self.run_start
        self.done = False

    def onAnimateBeginEvent(self, event):
        self.step_begin = time.perf_counter()
        t = self.root.time.value + self.root.dt.value
        self.target.position.value = [tool_pose(self.trajectory.height(t))]

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        frame_ms, step_ms = 1000.0 * (now - self.last_end), 1000.0 * (now - self.step_begin)
        self.last_end = now
        if self.done:
            return
        t = self.root.time.value
        tool = self.tool.position.value[0]
        target = self.target.position.value[0]
        if self.tissue_solver is not None:
            monitor_y = float(self.tissue_solver.monitorPosition.value[1])
            min_volume = float(self.tissue_solver.minVolumeRatio.value)
        else:
            x = np.asarray(self.tissue.position.value, dtype=float)
            monitor_y = float(x[self.monitor][1])
            min_volume = float(np.min(_tet_volumes(x, self.tets) / self.rest_volumes))
        if self.settled_top is None and t >= self.trajectory.t_down_start - 0.5 * DT:
            self.settled_top = monitor_y
        reference = self.settled_top if self.settled_top is not None else 0.0
        spring_y = COUPLING_STIFFNESS * (float(tool[1]) - float(target[1]))

        if self.contact is not None:
            c = self.contact
            contact_y = float(c.rigidContactForce.value[1])
            sides = list(c.contactSideCounts.value) or [-1, -1]
            # nan: not measured (only with measureTimes or verifyEvery), or no contact on that side
            separations = [float(v) if abs(float(v)) < 1.0 else float("nan")
                           for v in (list(c.contactSideMinSeparation.value) or [float("nan")] * 2)]
            counts = [int(c.currentContacts.value), int(c.currentConstraints.value),
                      int(c.currentIterations.value), float(c.currentError.value),
                      int(c.warmStartedContacts.value), int(sides[0]), int(sides[1])] + separations
            contact_times = [float(c.stepGpuMilliseconds.value)] + (
                [float(v) for v in c.stageMilliseconds.value] or [0.0] * len(self.CONTACT_STAGES))
        else:
            s = self.cpu_solver
            contact_y = float("nan")
            counts = [int(s.currentNumConstraintGroups.value), int(s.currentNumConstraints.value),
                      int(s.currentIterations.value), float(s.currentError.value), 0, -1, -1,
                      float("nan"), float("nan")]
            contact_times = [0.0] * (1 + len(self.CONTACT_STAGES))
        if self.tissue_solver is not None:
            ts = self.tissue_solver
            tissue_times = [float(ts.stepGpuMilliseconds.value)] + (
                [float(v) for v in ts.stageMilliseconds.value] or [0.0] * len(self.TISSUE_STAGES))
        else:
            tissue_times = [0.0] * (1 + len(self.TISSUE_STAGES))

        row = ([self.step, t, self.trajectory.phase(t), float(target[1]), float(tool[1]), monitor_y,
                reference - float(tool[1]), spring_y, contact_y, min_volume, frame_ms, step_ms]
               + counts + contact_times + tissue_times)
        self.step += 1
        self.rows.append(row)
        self.csv.write(",".join(f"{v:.9g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        self.csv.flush()
        if t >= self.trajectory.total_time - 0.5 * DT:
            self.finish()

    def finish(self):
        self.done = True
        self.csv.close()
        col = {name: i for i, name in enumerate(self.COLUMNS)}
        rows = self.rows

        def mean(values):
            return sum(values) / len(values) if values else float("nan")

        lines = [f"label={self.label}", f"steps={len(rows)}",
                 f"run_wall_s={time.perf_counter() - self.run_start:.3f}",
                 f"settled_top_y_mm={1000.0 * (self.settled_top or 0.0):.4f}"]
        for phase in ("settle", "press", "hold", "retract", "rest"):
            # the first frame carries SOFA's start-up work: left out
            frames = [r[col["frame_ms"]] for r in rows[1:] if r[col["phase"]] == phase]
            lines.append(f"frame_ms_{phase}={mean(frames):.3f}")
        lines.append(f"frame_ms_all={mean([r[col['frame_ms']] for r in rows[1:]]):.3f}")
        peak = max(rows, key=lambda r: r[col["spring_force_y"]])
        most = max(rows, key=lambda r: r[col["rows"]])
        lines += [f"peak_force_N={peak[col['spring_force_y']]:.6f}",
                  f"peak_force_time_s={peak[col['time']]:.3f}",
                  f"max_indentation_mm={1000.0 * max(r[col['indentation']] for r in rows):.4f}",
                  f"max_contacts={most[col['contacts']]}", f"max_rows={most[col['rows']]}",
                  f"max_rows_time_s={most[col['time']]:.3f}",
                  f"max_sweeps={max(r[col['sweeps']] for r in rows)}",
                  f"min_volume_ratio={min(r[col['min_volume_ratio']] for r in rows):.4f}",
                  f"csv={self.csv_path}"]
        with open(self.summary_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("HARD_POKE_SUMMARY " + " ".join(lines[1:-1]), flush=True)


def build_scene(root, name):
    s = scene_settings(name)
    side = vc.SIDE
    label = f"{name}_poke_{side}"
    trajectory = Trajectory(s["depth"])
    mesh, fixed, monitor, bottom = build_mesh(name, s)
    tool_mesh = pc.ProbeMesh(TOOL_RADIUS, TOOL_LENGTH, TOOL_SEGMENTS, TOOL_RINGS)

    root.name = f"{name.capitalize()}Poke"
    root.dt = DT
    root.gravity = GRAVITY
    vc.add_plugins(root)
    root.addObject("RequiredPlugin", pluginName=["Sofa.GL.Component.Rendering3D"])
    root.addObject("VisualStyle", displayFlags="showVisualModels")
    root.addObject("InteractiveCamera", name="camera", position=[0.0, 0.06, 0.14], lookAt=[0.0, -0.008, 0.0])

    # The collision grid holds the tissue, the tool's whole path, and (diaphragm) the
    # sheet pushed down as far as the tool goes.
    margin = 0.005
    w = s["half_width"] + margin
    bounds = ((-w, min(bottom, -s["depth"] - s.get("thickness", 0.0)) - 2.0 * margin, -w),
              (w, START_HEIGHT + TOOL_LENGTH + margin, w))
    vc.add_contact_pipeline(root, FRICTION, bounds, cell=COLLISION_CELL, max_contacts=MAX_RAW_CONTACTS)

    tissue, tissue_surface = vc.add_tissue(root, mesh, s["material"], fixed=fixed, label=label, contact=side,
                                           monitor_vertex=monitor)
    target = root.addChild("Target")
    target.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=[tool_pose(START_HEIGHT)])
    tool, tool_surface = vc.add_rigid_body(root, "Tool", tool_mesh, tool_pose(START_HEIGHT), TOOL_MASS,
                                           spring_target=target.dofs.getLinkPath(), stiffness=COUPLING_STIFFNESS,
                                           angular_stiffness=COUPLING_ANGULAR_STIFFNESS, cancel_gravity=GRAVITY)
    contact = vc.add_gpu_contact_solver(root, tissue, tissue_surface, tool, tool_surface, FRICTION, label)
    gpu = side == "gpu"

    if VISUAL:
        if gpu:
            # Draws the GPU surface; reads it back only when the view is drawn, never in a batch run.
            tissue_surface.addObject("CudaVisualModel", template="CudaVec3f", computeNormals=True,
                                     diffuse=[0.78, 0.36, 0.33, 1.0], ambient=[0.2, 0.09, 0.08, 1.0])
        else:
            visual = tissue.addChild("Visual")
            visual.addObject("OglModel", name="model", position=mesh.positions,
                             triangles=mesh.surface_triangles, color=[0.78, 0.36, 0.33, 1.0])
            visual.addObject("IdentityMapping")
        tool_visual = tool.addChild("Visual")
        tool_visual.addObject("OglModel", name="model", position=tool_mesh.positions,
                              triangles=tool_mesh.triangles, color=[0.75, 0.75, 0.8, 1.0])
        tool_visual.addObject("RigidMapping")

    if COLLISION_TIMES and gpu:
        # The collision's own GPU time (CUDA events around its CUDA graph) and its raw
        # contact counts: needs the contact count read back every frame.
        root.GpuCollisionNarrowPhase.proximityReadContactCounter.value = True
        root.addObject("GpuPipelineBenchmarkController", name="collisionTimer", label=label + "_collision",
                       outputDir=vc.log_dir(), warmupSteps=0, printProgress=False)

    parameters = {
        "scene": name, "description": s["description"], "side": side,
        "mesh": {"nodes": len(mesh.positions), "tetrahedra": len(mesh.tetrahedra),
                 "surface_triangles": len(mesh.surface_triangles), "surface_nodes": len(mesh.surface_nodes),
                 "fixed_nodes": len(fixed), "monitor_vertex": monitor,
                 "grid": {"x": len(mesh.xs), "y": len(mesh.ys), "z": len(mesh.zs)},
                 "fine_step_m": s["fine_step"], "fine_half_width_m": s["fine_half"], "growth": s["growth"],
                 "half_width_m": s["half_width"], "height_m": s.get("height"), "thickness_m": s.get("thickness"),
                 "layers": s.get("layers")},
        "material": {"note": s["material_note"], "core": s["material"]["core"], "params": s["material"]["params"],
                     "maxwell": s["material"]["maxwell"], "density_kg_m3": vc.DENSITY},
        "tool": {"radius_m": TOOL_RADIUS, "length_m": TOOL_LENGTH, "segments": TOOL_SEGMENTS, "rings": TOOL_RINGS,
                 "vertices": len(tool_mesh.positions), "triangles": len(tool_mesh.triangles), "mass_kg": TOOL_MASS,
                 "coupling_stiffness_N_m": COUPLING_STIFFNESS, "coupling_angular_stiffness": COUPLING_ANGULAR_STIFFNESS},
        "motion": {"dt_s": DT, "gravity": GRAVITY, "start_height_m": START_HEIGHT, "depth_m": s["depth"],
                   "speed_m_s": SPEED, "settle_s": SETTLE_TIME, "hold_s": HOLD_TIME, "rest_s": REST_TIME,
                   "press_s": trajectory.travel / SPEED, "total_time_s": trajectory.total_time,
                   "steps": trajectory.steps()},
        "contact": {"friction": FRICTION, "contact_distance_m": vc.CONTACT_DISTANCE,
                    "alarm_distance_m": vc.ALARM_DISTANCE, "tolerance": vc.CONSTRAINT_TOLERANCE,
                    "max_iterations": vc.CONSTRAINT_MAX_ITERATIONS, "filter": "vertexFace"},
        "collision": {"grid_min": list(bounds[0]), "grid_max": list(bounds[1]), "cell_m": COLLISION_CELL,
                      "resolution": [max(4, int(math.ceil((hi - lo) / COLLISION_CELL)))
                                     for lo, hi in zip(bounds[0], bounds[1])],
                      "max_raw_contacts": MAX_RAW_CONTACTS, "way": "6 (big-cell fused)"},
        "timers": {"measure_times": vc.MEASURE_TIMES, "collision_times": COLLISION_TIMES,
                   "cuda_timer_sync": os.environ.get("CUDA_TIMER_SYNC", "1")},
        "factorization": vc.FACTORIZATION,
        "contact_solver_variant": vc.CONTACT_SOLVER_VARIANT,
    }
    os.makedirs(vc.log_dir(), exist_ok=True)
    with open(os.path.join(vc.log_dir(), label + "_parameters.json"), "w") as fh:
        json.dump(parameters, fh, indent=2)

    root.addObject(HardPokeController(
        name="hardPoke", root=root, target=target.dofs, tool=tool.dofs, tissue=tissue.dofs,
        contact=contact, cpu_solver=None if gpu else root.constraintSolver,
        tissue_solver=tissue.odeSolver if gpu else None, monitor=monitor, trajectory=trajectory, label=label,
        tetrahedra=mesh.tetrahedra, rest_positions=mesh.positions))
    print(f"HardPoke {name} ({side}): {len(mesh.positions)} nodes, {len(mesh.tetrahedra)} tetrahedra, "
          f"{len(mesh.surface_triangles)} surface triangles; tool {len(tool_mesh.positions)} vertices, "
          f"{len(tool_mesh.triangles)} triangles; {trajectory.steps()} steps of {DT} s", flush=True)
    return root
