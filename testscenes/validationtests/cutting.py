"""Cutting: a blade cuts a slot into a loaded cantilever beam. CPU (SOFA) against GPU.

The beam of beam_bending.py (10 x 2 x 2 cm, clamped at x = 0, a tip load for about
1% deflection, Rayleigh damping) settles for 2 s. Then a blade in the plane
x = CUT_X, its edge moving down from the top at 2 cm/s, cuts a slot one cell wide
into the top half of the beam, near the clamp where the top is stretched most
(0.5 s), and the beam settles again, sagging further (4 s).

TetrahedronCutter (this plugin) removes the tetrahedra the edge has passed, through
SOFA's TetrahedronSetTopologyModifier, identically on both sides (centroids in the
rest configuration), the slot a whole layer of cells so that no vertex loses all
its tetrahedra.
  cpu: SOFA follows the removal with its own components: MeshMatrixMass and the
       material's force field update themselves, SparseLDLSolver refactors.
  gpu: GpuTissueSolver rebuilds its elements and mass for the remaining tetrahedra.
Both also keep a boundary surface (Tetra2TriangleTopologicalMapping) that grows
the slot's faces, as a collision surface would.

Checked: the tip deflection along the whole run, CPU against GPU; the number of
tetrahedra removed and of surface triangles, which must be equal; and the stiffness
lost: the tip deflection after the cut against before.

Output (log dir): cutting_<material>_<side>.csv and _summary.txt.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import numpy as np  # noqa: E402
import Sofa.Core  # noqa: E402

import validation_common as vc  # noqa: E402
import beam_bending as beam  # noqa: E402

DIVISIONS = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "4"))
DT = 0.02
T_CUT, CUT_SPEED = 2.0, 0.02
STEPS = int(os.environ.get("SOFA_VALIDATION_STEPS", "300"))   # 6 s
CUT_CELL = 3                                                     # the slot: the 4th cell layer from the clamp


class CutLogger(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.dofs, self.tip, self.y0 = kwargs["root"], kwargs["dofs"], kwargs["tip"], kwargs["y0"]
        self.topology, self.surface, self.cutter = kwargs["topology"], kwargs["surface"], kwargs["cutter"]
        self.label = kwargs["label"]
        self.csv = open(os.path.join(vc.log_dir(), self.label + ".csv"), "w")
        self.csv.write("time,wall_ms,tip_dx,tip_dy,tip_dz,tetrahedra,surface_triangles,removed\n")
        self.last = None
        self.rows = 0
        self.before = None

    def onAnimateEndEvent(self, event):
        now = time.perf_counter()
        wall = 0.0 if self.last is None else 1000.0 * (now - self.last)
        self.last = now
        t = self.root.time.value
        d = np.asarray(self.dofs.position.value, dtype=float)[self.tip] - self.y0
        tets = len(self.topology.tetrahedra.value)
        triangles = len(self.surface.triangles.value)
        removed = int(self.cutter.removedCount.value)
        self.csv.write(f"{t:.6g},{wall:.4f},{d[0]:.12g},{d[1]:.12g},{d[2]:.12g},{tets},{triangles},{removed}\n")
        self.csv.flush()
        if self.before is None and t >= T_CUT - 0.5 * DT:
            self.before = -float(d[1])
        self.rows += 1
        if self.rows == STEPS:
            after = -float(d[1])
            vc.summary_line(self.label, test="cutting", material=vc.MATERIAL, side=vc.SIDE, divisions=DIVISIONS,
                            removed_tetrahedra=removed, tetrahedra_left=tets, surface_triangles=triangles,
                            deflection_before_m=f"{self.before:.10g}", deflection_after_m=f"{after:.10g}",
                            softening_ratio=f"{after / self.before:.6f}")


def createScene(root):
    mat = vc.material()
    label = vc.run_label("cutting")
    root.name = "Cutting"
    root.dt = DT
    root.gravity = [0.0, 0.0, 0.0]
    vc.add_plugins(root)
    if vc.SIDE == "cpu":   # TetrahedronCutter is this plugin's (a CPU component: it only edits the topology)
        root.addObject("RequiredPlugin", name="cutterPlugin",
                       pluginName=[vc.GPU_COLLISION_LIB if os.path.isfile(vc.GPU_COLLISION_LIB) else "SofaGpuCollision"])
    root.addObject("RequiredPlugin", name="topologyPlugins",
                   pluginName=["Sofa.Component.Topology.Mapping", "Sofa.Component.Topology.Container.Dynamic"])
    root.addObject("DefaultAnimationLoop")

    n = DIVISIONS
    h = beam.HEIGHT / n
    mesh = vc.BoxMesh(vc.uniform(0.0, beam.LENGTH, 5 * n), vc.uniform(-beam.HEIGHT / 2, beam.HEIGHT / 2, n),
                      vc.uniform(-beam.WIDTH / 2, beam.WIDTH / 2, n))
    clamped = mesh.face(0, 0)
    unit = beam.timoshenko_tip_deflection(mat, 1.0)
    force = 0.01 * beam.LENGTH / unit
    loads = vc.face_traction_loads(mesh, axis=0, side=1, traction=[0.0, -force / (beam.HEIGHT * beam.WIDTH), 0.0])
    omega = beam.first_bending_frequency(mat)
    tissue, _ = vc.add_tissue(root, mesh, mat, fixed=clamped, label=label, loads=loads, rayleigh_mass=2.0 * omega,
                              cuttable=True)
    surface = tissue.addChild("Surface")
    surface.addObject("TriangleSetTopologyContainer", name="topo")
    surface.addObject("TriangleSetTopologyModifier", name="modifier")
    surface.addObject("Tetra2TriangleTopologicalMapping", input=tissue.topo.getLinkPath(), output=surface.topo.getLinkPath())
    # The slot: one cell layer (x in [CUT_CELL h, (CUT_CELL + 1) h]), from the top down to mid-height.
    cut_x = (CUT_CELL + 0.5) * h
    cutter = tissue.addObject("TetrahedronCutter", name="cutter", restPositions=mesh.positions,
                              planePoint=[cut_x, beam.HEIGHT / 2, 0.0], planeNormal=[1.0, 0.0, 0.0],
                              cutDirection=[0.0, -1.0, 0.0], kerf=0.49 * h, edgeStart=0.0,
                              edgeStop=beam.HEIGHT / 2, speed=CUT_SPEED, startTime=T_CUT)
    tip = mesh.nearest([beam.LENGTH, 0.0, 0.0])
    root.addObject(CutLogger(name="logger", root=root, dofs=tissue.dofs, tip=tip,
                             y0=np.asarray(mesh.positions[tip], dtype=float), topology=tissue.topo,
                             surface=surface.topo, cutter=cutter, label=label))
    print(f"Cutting: side={vc.SIDE} material={vc.MATERIAL} divisions={n} nodes={len(mesh.positions)} "
          f"tetrahedra={len(mesh.tetrahedra)} cut_x={cut_x:.4g} tip_force={force:.6g} N steps={STEPS}", flush=True)
    return root
