"""Tissue poke, CPU version: the most realistic setup SOFA offers, all on the CPU.

A 6 x 6 x 3 cm block of liver-like tissue rests on a table. A 5 mm round-tip
probe comes down at 5 mm/s, pokes 8 mm into it, holds for 1 s, and pulls back
out. Geometry, material, motion and logging are shared with the GPU version
(poke_common.py), so the two can be compared directly.

Realism choices:
  * tissue: viscoelastic Ogden - stiffens under large stretch and relaxes over
    time; SOFA's core Ogden spring plus SofaViscoElastic's Maxwell branch, so the
    solver gets the exact stiffness (poke_common.add_tissue_material;
    SOFA_POKE_MATERIAL=split for the earlier SofaViscoElastic Ogden);
  * mass from density (MeshMatrixMass), gravity on, bottom fixed to the table;
  * contact: Lagrange-multiplier constraints with friction
    (FreeMotionAnimationLoop + FrictionContactConstraint): no overlap, mu = 0.1;
  * exact constraint response: a direct sparse solver + LinearSolverConstraintCorrection
    on both tissue and probe;
  * probe: a rigid body pulled along its path by a stiff spring (a virtual
    coupling, like a haptic device), which also measures the push-back force.

Output: <log dir>/tissue_poke_cpu.csv and tissue_poke_cpu_summary.txt.
Run: bash scripts/run_tissue_poke_wsl.sh cpu
The GPU version also uses build_scene() as its fallback when no GPU contact is available.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import poke_common as pc  # noqa: E402

LABEL = "tissue_poke_cpu" + os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")
NOTES = f"CPU: {pc.material_note()} tissue, constraint contact with friction, direct solvers"


def createScene(root):
    return build_scene(root, name="TissuePokeCPU", label=LABEL, notes=NOTES)


def build_scene(root, name, label, notes):
    root.name = name
    root.dt = pc.DT
    root.gravity = pc.GRAVITY

    root.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ] + pc.MATERIAL_PLUGINS)
    root.addObject("VisualStyle", displayFlags="showVisualModels")
    pc.add_camera(root)

    root.addObject("FreeMotionAnimationLoop")
    constraint_solver = root.addObject("BlockGaussSeidelConstraintSolver", name="constraintSolver",
                                       tolerance=1e-7, maxIterations=1000)
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("LocalMinDistance", alarmDistance=pc.ALARM_DISTANCE,
                   contactDistance=pc.CONTACT_DISTANCE, angleCone=0.0)
    root.addObject("CollisionResponse", response="FrictionContactConstraint",
                   responseParams=f"mu={pc.FRICTION}")

    mesh = pc.TissueMesh()
    probe_mesh = pc.ProbeMesh()

    # ---- tissue -------------------------------------------------------------
    tissue = root.addChild("Tissue")
    tissue.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    tissue.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    tissue.addObject("MechanicalObject", name="dofs", template="Vec3d", position=mesh.positions)
    tissue.addObject("TetrahedronSetTopologyContainer", name="topo", tetrahedra=mesh.tetrahedra)
    tissue.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3d")
    tissue.addObject("MeshMatrixMass", massDensity=pc.DENSITY)
    pc.add_tissue_material(tissue)
    tissue.addObject("FixedProjectiveConstraint", indices=mesh.bottom_indices)
    tissue.addObject("LinearSolverConstraintCorrection")

    surface = tissue.addChild("Surface")
    surface.addObject("MechanicalObject", name="dofs", template="Vec3d", position=mesh.surface_local_positions)
    surface.addObject("MeshTopology", triangles=mesh.surface_local_triangles, edges=mesh.surface_local_edges)
    surface.addObject("PointCollisionModel")
    surface.addObject("LineCollisionModel")
    surface.addObject("TriangleCollisionModel")
    surface.addObject("SubsetMapping", indices=mesh.surface_nodes)

    visual = tissue.addChild("Visual")
    visual.addObject("OglModel", name="model", position=mesh.positions,
                     triangles=mesh.surface_triangles, color=[0.78, 0.36, 0.33, 1.0])
    visual.addObject("IdentityMapping")

    # ---- probe (rigid, pulled by the coupling spring) -------------------------
    target = root.addChild("Target")
    target.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                     position=pc.rigid_pose(pc.START_HEIGHT))

    probe = root.addChild("Probe")
    probe.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    probe.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    pc.add_probe_body(probe, external_rest_shape="@../Target/dofs")
    # The exact compliance, coupling spring included. UncoupledConstraintCorrection
    # would use the mass alone (dt^2 / m): the constraint solver would then move
    # the probe (1 + k dt^2 / m) = 3x too far for the force the tissue receives,
    # and the coupling spring would read 3x the real contact force.
    probe.addObject("LinearSolverConstraintCorrection")

    probe_collision = probe.addChild("Collision")
    probe_collision.addObject("MechanicalObject", name="dofs", template="Vec3d", position=probe_mesh.positions)
    probe_collision.addObject("MeshTopology", triangles=probe_mesh.triangles, edges=probe_mesh.edges)
    probe_collision.addObject("PointCollisionModel")
    probe_collision.addObject("LineCollisionModel")
    probe_collision.addObject("TriangleCollisionModel")
    probe_collision.addObject("RigidMapping")

    probe_visual = probe.addChild("Visual")
    probe_visual.addObject("OglModel", name="model", position=probe_mesh.positions,
                           triangles=probe_mesh.triangles, color=[0.75, 0.75, 0.8, 1.0])
    probe_visual.addObject("RigidMapping")

    # ---- drive + measure ------------------------------------------------------
    root.addObject(pc.ProbeDriver(name="driver", root=root, target=target.dofs))
    root.addObject(pc.PokeLogger(
        name="logger", root=root, probe=probe.dofs, target=target.dofs, tissue=tissue.dofs,
        top_center_index=mesh.top_center_index, tetrahedra=mesh.tetrahedra,
        rest_positions=mesh.positions, label=label,
        log_dir=pc.default_log_dir(current_dir), notes=notes))
    root.addObject(pc.ConstraintSolverStatsLogger(
        name="constraintStats", root=root, solver=constraint_solver,
        path=os.path.join(pc.default_log_dir(current_dir), label + "_constraints.csv")))
    print(f"{name}: {len(mesh.positions)} nodes, {len(mesh.tetrahedra)} tetrahedra, "
          f"{len(mesh.surface_triangles)} surface triangles, probe {len(probe_mesh.triangles)} triangles, "
          f"{pc.total_steps()} steps of {pc.DT} s", flush=True)
    return root
