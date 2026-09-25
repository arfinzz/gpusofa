"""Tissue poke, GPU version: GPU wherever a GPU component exists, CPU otherwise.

The same block, probe, material, motion and logging as tissue_poke_cpu.py
(poke_common.py). Each piece is checked at load time and placed on the GPU when
its GPU component is available:

  piece                 on the GPU                          today
  --------------------  ----------------------------------  ---------------------------------
  tissue (material,     GpuTissueSolver: the viscoelastic   GPU (SOFA_POKE_GPU_TISSUE=cpu:
    mass, fixed base,     Ogden, the consistent mass, the     SOFA's material components +
    implicit solve)       fixed DOFs, implicit Euler with     MeshMatrixMass + EulerImplicit/
                          a direct solve                      SparseLDL, as in the CPU scene)
  tissue surface        CudaVec3f via IdentityMapping       GPU
  collision detection   GpuCollisionPipeline + Broad/       GPU (way 6); no per-frame CPU
                          NarrowPhase                         bounding boxes
  contact response      GpuContactConstraintSolver          GPU (constraints with friction)
  probe surface         CudaVec3f via GpuRigidMapping       GPU
  probe body            a 6-DOF rigid body on a spring      CPU (7 numbers; its GPU surface is
                                                              mapped from it every step)

With the tissue on the GPU, no GPU state is copied between the CPU and the GPU
during a step (SofaCUDA's copy trace shows none after the first step): the
logger reads the tissue solver's GPU-computed surface point and smallest volume
ratio instead of the tissue's positions, the tissue is drawn by CudaVisualModel,
and the surfaces' mappings don't map forces (see _add_gpu_surface).

Contact (SOFA_POKE_GPU_CONTACT):
  constraint (default)  the same contact model as the CPU scene: Lagrange
                        multipliers with Coulomb friction (mu = 0.1), no overlap,
                        exact compliance from each body's direct solver. The
                        rows, the compliance, the Gauss-Seidel solve and the
                        correction all run on the GPU (GpuContactConstraintSolver).
  penalty               the older GPU contact: a side-aware penalty force inside
                        one implicit solve (CudaContactPenaltyForceField), no friction.

SOFA_POKE_COMPARE=1 (constraint mode) also runs SOFA's CPU constraint pipeline
on the very same contacts every step and writes the stage-by-stage differences
and both sides' times to <log dir>/tissue_poke_gpu_compare.csv
(SOFA_POKE_COMPARE_EVERY=N compares every N-th contact step).
SOFA_POKE_RESPONSE=cpu lets that CPU pipeline move the bodies instead of the
GPU's (same GPU-detected contacts): a whole run whose only difference from the
default is who computed the contact response.
SOFA_POKE_COMPARE_TISSUE=1 (GPU tissue) runs SOFA's own CPU tissue components on
the same state every step and writes the differences of the tissue step to
<log dir>/tissue_poke_gpu_tissue_compare.csv. SOFA_POKE_COMPARE and
SOFA_POKE_RESPONSE=cpu turn it on too (their CPU pipeline needs its linear solver).
The material (SOFA_POKE_MATERIAL, poke_common.py) is the same in both scenes: by
default SOFA's core Ogden + SofaViscoElastic's Maxwell element. With
SOFA_POKE_MATERIAL=split (SofaViscoElastic's SLSOgdenFirstOrder) the GPU
reproduces that component as it runs in SOFA v25.12, whose Eigen call computes no
eigenvectors (GpuTissueSolver explains); SOFA_POKE_OGDEN=exact gives it the true
eigenvectors instead.

If any piece of the GPU contact chain is missing (SofaCUDA, this plugin, or one
of its components), the scene falls back to the CPU scene's setup: constraint
contact with friction. SOFA's CPU penalty contact is not a usable fallback: it
cannot tell inside from outside, and the probe passes straight through the
tissue at first touch.

Output: <log dir>/tissue_poke_gpu.csv and tissue_poke_gpu_summary.txt.
Run: bash scripts/run_tissue_poke_wsl.sh gpu
"""

import os
import sys

import Sofa.Core
import SofaRuntime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import poke_common as pc  # noqa: E402
import tissue_poke_cpu  # noqa: E402  (the fallback)

LABEL = "tissue_poke_gpu" + os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")
CONTACT_MODE = os.environ.get("SOFA_POKE_GPU_CONTACT", "constraint")
# Constraint mode: who computes the contact response (gpu, or SOFA's CPU pipeline
# on the same GPU-detected contacts), the stage-by-stage comparison with SOFA's
# CPU pipeline (always on with the cpu response), and GPU stage timings.
RESPONSE = os.environ.get("SOFA_POKE_RESPONSE", "gpu")
COMPARE = os.environ.get("SOFA_POKE_COMPARE", "0") == "1" or RESPONSE == "cpu"
COMPARE_EVERY = int(os.environ.get("SOFA_POKE_COMPARE_EVERY", "1"))
MEASURE_TIMES = os.environ.get("SOFA_POKE_MEASURE_TIMES", "0") == "1"
# Where the tissue runs (constraint mode): gpu (GpuTissueSolver) or cpu.
TISSUE_MODE = os.environ.get("SOFA_POKE_GPU_TISSUE", "gpu")
COMPARE_TISSUE = os.environ.get("SOFA_POKE_COMPARE_TISSUE", "0") == "1" or COMPARE
# SOFA_POKE_MATERIAL=split (SofaViscoElastic's Ogden): "sofa" reproduces it as it runs
# (so the CPU and GPU scenes match), "exact" is the Ogden material as written.
OGDEN_EIGENVECTORS = os.environ.get("SOFA_POKE_OGDEN", "sofa")
# Default material (SOFA core Ogden): its stiffness "robust" (default) or as SOFA computes it
# ("sofa"; they differ only when two principal stretches are equal up to rounding).
OGDEN_TANGENT = os.environ.get("SOFA_POKE_OGDEN_TANGENT", "robust")
# The GPU tissue matrix's factorisation: auto (default), band or dense (GpuTissueSolver.factorization).
FACTORIZATION = os.environ.get("SOFA_POKE_FACTORIZATION", "auto")
# Visual models. The GPU tissue is drawn by CudaVisualModel, which reads it back
# only when the view is drawn; SOFA_POKE_VISUAL=0 removes the visual models.
VISUAL = os.environ.get("SOFA_POKE_VISUAL", "1") != "0"
# Constraint solver settings: the CPU scene's (BlockGaussSeidelConstraintSolver).
CONSTRAINT_TOLERANCE = 1e-7
CONSTRAINT_MAX_ITERATIONS = 1000
# Diagnostic switches: the older unsigned contact law (penalty mode), the
# dense-grid way instead of way 6, per-frame contact counts (one read-back per
# frame), and running the CPU fallback even when the GPU plugins load.
SIDE_AWARE = os.environ.get("SOFA_POKE_SIDE_AWARE", "1") != "0"
USE_BIGCELL = os.environ.get("SOFA_POKE_GPU_WAY", "bigcell") == "bigcell"
REPORT_STATS = os.environ.get("SOFA_CONTACT_REPORT_STATS", "0") == "1"
FORCE_CPU = os.environ.get("SOFA_POKE_FORCE_CPU", "0") == "1"
REPO_ROOT = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
GPU_COLLISION_LIB = os.environ.get(
    "SOFA_GPU_COLLISION_LIB",
    os.path.join(REPO_ROOT, "SofaGpuCollision", "build-profile", "libSofaGpuCollision.so"))


def _available(component, template=None):
    try:
        entry = Sofa.Core.ObjectFactory.getComponent(component)
    except Exception:  # noqa: BLE001 - not registered at all
        return False
    return template is None or template in entry.templates


class ConstraintStatsLogger(Sofa.Core.Controller):
    """One CSV row per step from GpuContactConstraintSolver's outputs: contacts,
    rows, Gauss-Seidel sweeps, the step's GPU time (with measureTimes) and the
    contact force on the probe; with the GPU tissue, also its step's GPU times
    (material, assembly, factorisation, solve). Reading them costs no GPU read-back."""

    COLUMNS = ["time", "contacts", "rows", "sweeps", "gs_error", "gpu_ms", "normal_impulse",
               "contact_force_x", "contact_force_y", "contact_force_z",
               "rows_ms", "factorize_ms", "compliance_ms", "solve_ms", "correction_ms",
               "tissue_ms", "tissue_material_ms", "tissue_assembly_ms", "tissue_factorize_ms", "tissue_solve_ms"]

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.solver = kwargs["solver"]
        self.tissue_solver = kwargs.get("tissue_solver")
        path = kwargs["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv = open(path, "w")
        self.csv.write(",".join(self.COLUMNS) + "\n")

    def onAnimateEndEvent(self, event):
        s = self.solver
        force = s.rigidContactForce.value
        stages = list(s.stageMilliseconds.value) or [0.0] * 5
        tissue = [0.0] * 5
        if self.tissue_solver is not None:
            t = self.tissue_solver
            if not getattr(self, "_reported", False):
                self._reported = True
                band = int(t.bandwidth.value)
                print(f"TissuePokeGPU factorisation: {'band, half-bandwidth ' + str(band) if band > 0 else 'dense'}", flush=True)
            tissue = [float(t.stepGpuMilliseconds.value)] + ([float(v) for v in t.stageMilliseconds.value] or [0.0] * 4)
        row = [self.root.time.value, s.currentContacts.value, s.currentConstraints.value, s.currentIterations.value,
               s.currentError.value, s.stepGpuMilliseconds.value, s.normalImpulse.value,
               float(force[0]), float(force[1]), float(force[2])] + [float(v) for v in stages] + tissue
        self.csv.write(",".join(f"{v:.9g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        self.csv.flush()


def _choose_backends():
    """Load the GPU plugins if possible and decide where each piece runs."""
    cuda_loaded = SofaRuntime.importPlugin("SofaCUDA")
    collision_loaded = (SofaRuntime.importPlugin(GPU_COLLISION_LIB) if os.path.isfile(GPU_COLLISION_LIB)
                        else SofaRuntime.importPlugin("SofaGpuCollision"))
    contact_component = "GpuContactConstraintSolver" if CONTACT_MODE == "constraint" else "CudaContactPenaltyForceField"
    gpu_chain = (not FORCE_CPU and cuda_loaded and collision_loaded
                 and _available("MechanicalObject", "CudaVec3f")
                 and _available("IdentityMapping", "Vec3d,CudaVec3f")
                 and _available("RigidMapping", "Rigid3d,CudaVec3f")
                 and _available("GpuCollisionBroadPhase") and _available("GpuCollisionNarrowPhase")
                 and _available(contact_component))
    # The GPU contact reads the GPU collision's device buffer and needs GPU
    # surfaces, so surfaces, collision and contact move together.
    where = "GPU" if gpu_chain else "CPU"
    # The GPU tissue works with the GPU constraint contact (it hands it its factor).
    gpu_tissue = (gpu_chain and CONTACT_MODE == "constraint" and TISSUE_MODE == "gpu"
                  and _available("GpuTissueSolver") and _available("FixedProjectiveConstraint", "CudaVec3f")
                  and _available("IdentityMapping", "CudaVec3f,CudaVec3f") and _available("GpuCollisionPipeline"))
    return {
        "tissue": "GPU" if gpu_tissue else "CPU",
        "surfaces": where,
        "collision": where,
        "contact": where,
    }


def _add_gpu_surface(node, positions, triangles, mapping, template_in, map_forces=True):
    """map_forces=False: nothing acts on the surface through forces (constraint contact
    corrects the bodies itself), so SOFA's force visitors need not map its (zero) forces
    up, which copied the probe's surface to the CPU twice per step. Positions, velocities
    and corrections still reach the surface: FreeMotionAnimationLoop propagates them
    whatever this flag says."""
    surface = node.addChild("GpuSurface")
    surface.addObject("MechanicalObject", name="dofs", template="CudaVec3f", position=positions)
    surface.addObject("MeshTopology", triangles=triangles)
    surface.addObject("TriangleCollisionModel", selfCollision=False)
    if mapping == "RigidMapping" and _available("GpuRigidMapping"):
        # This plugin's GPU rigid mapping: SofaCUDA's RigidMapping<Rigid3d,CudaVec3f> maps
        # surface forces to a wrong torque (SOFA v25.12).
        surface.addObject("GpuRigidMapping", mapForces=map_forces)
    else:
        surface.addObject(mapping, template=f"{template_in},CudaVec3f", mapForces=map_forces)
    return surface


def _add_gpu_collision(root, contact_distance, gpu_pipeline=False):
    """GPU broad + narrow phase (way 6), contacts kept on the device. gpu_pipeline:
    GpuCollisionPipeline, which skips the per-frame CPU bounding boxes of GPU models."""
    root.addObject("GpuCollisionPipeline" if gpu_pipeline else "CollisionPipeline")
    margin = 0.005
    root.addObject("GpuCollisionBroadPhase", enableGPU=True, allowCPUFallback=False,
                   logBackendStatus=True, useObjectAabbCulling=False, testGpuModelBoxes=not gpu_pipeline)
    root.addObject(
        "GpuCollisionNarrowPhase",
        enableGPU=True, allowCPUFallback=False, logBackendStatus=True,
        useDenseGrid=True, useIndexedDenseGridInput=True, useDirectDevicePositions=True,
        cacheTriangleTopology=True, copyContactsToHost=False,
        useFeatureBasedProximity=True, useBigCellFusedGeneration=USE_BIGCELL,
        proximityComputeBarycentrics=True, proximityKeepContactsOnDevice=True,
        proximityReadContactCounter=False, proximityMaxContacts=200000,
        minGPUPairCount=1, contactDistance=contact_distance,
        gridMinX=-pc.BLOCK_HALF_WIDTH - margin, gridMaxX=pc.BLOCK_HALF_WIDTH + margin,
        gridMinY=-pc.BLOCK_HEIGHT - margin, gridMaxY=pc.START_HEIGHT + pc.PROBE_LENGTH + margin,
        gridMinZ=-pc.BLOCK_HALF_WIDTH - margin, gridMaxZ=pc.BLOCK_HALF_WIDTH + margin,
        gridResolutionX=18, gridResolutionY=22, gridResolutionZ=18,
        maxTissueTrianglesPerCell=128, maxToolTrianglesPerCell=256, maxCandidatePairs=400000)
    # The broad phase uses its alarm distance to decide which object pairs to test.
    root.addObject("LocalMinDistance", alarmDistance=pc.ALARM_DISTANCE,
                   contactDistance=pc.CONTACT_DISTANCE, angleCone=0.0)


def _add_tissue_body(parent, mesh, map_forces=True):
    tissue = parent.addChild("Tissue")
    tissue.addObject("MechanicalObject", name="dofs", template="Vec3d", position=mesh.positions)
    tissue.addObject("TetrahedronSetTopologyContainer", name="topo", tetrahedra=mesh.tetrahedra)
    tissue.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3d")
    tissue.addObject("MeshMatrixMass", massDensity=pc.DENSITY)
    pc.add_tissue_material(tissue)
    tissue.addObject("FixedProjectiveConstraint", indices=mesh.bottom_indices)
    surface = _add_gpu_surface(tissue, mesh.positions, mesh.surface_triangles, "IdentityMapping", "Vec3d",
                               map_forces=map_forces)
    visual = tissue.addChild("Visual")
    visual.addObject("OglModel", name="model", position=mesh.positions,
                     triangles=mesh.surface_triangles, color=[0.78, 0.36, 0.33, 1.0])
    visual.addObject("IdentityMapping")
    return tissue, surface


def _add_gpu_tissue_body(parent, mesh):
    """The tissue on the GPU: a CudaVec3f state integrated by GpuTissueSolver, which
    replaces the material's force fields + MeshMatrixMass + EulerImplicitSolver + SparseLDLSolver."""
    tissue = parent.addChild("Tissue")
    tissue.addObject("MechanicalObject", name="dofs", template="CudaVec3f", position=mesh.positions)
    tissue.addObject("TetrahedronSetTopologyContainer", name="topo", tetrahedra=mesh.tetrahedra)
    tissue.addObject("FixedProjectiveConstraint", template="CudaVec3f", indices=mesh.bottom_indices)
    tissue.addObject("GpuTissueSolver", name="odeSolver",
                     **pc.gpu_tissue_material(ogden_eigenvectors=OGDEN_EIGENVECTORS, ogden_tangent=OGDEN_TANGENT),
                     factorization=FACTORIZATION,
                     massDensity=pc.DENSITY, restPositions=mesh.positions, monitorVertex=mesh.top_center_index,
                     measureTimes=MEASURE_TIMES, compareWithCpu=COMPARE_TISSUE, compareEvery=COMPARE_EVERY,
                     compareFile=os.path.join(pc.default_log_dir(current_dir), LABEL + "_tissue_compare.csv"))
    surface = _add_gpu_surface(tissue, mesh.positions, mesh.surface_triangles, "IdentityMapping", "CudaVec3f",
                               map_forces=False)
    if VISUAL and _available("CudaVisualModel", "CudaVec3f"):
        # Draws the GPU surface; it reads it back only when the view is drawn, never in a batch run.
        surface.addObject("CudaVisualModel", template="CudaVec3f", computeNormals=True,
                          diffuse=[0.78, 0.36, 0.33, 1.0], ambient=[0.2, 0.09, 0.08, 1.0])
    elif VISUAL:
        # A visual mapping is applied after every step: one copy of the tissue per step.
        visual = tissue.addChild("Visual")
        visual.addObject("OglModel", name="model", position=mesh.positions,
                         triangles=mesh.surface_triangles, color=[0.78, 0.36, 0.33, 1.0])
        visual.addObject("IdentityMapping", template="CudaVec3f,Vec3d")
    return tissue, surface


def _add_probe_visual(probe, probe_mesh):
    probe_visual = probe.addChild("Visual")
    probe_visual.addObject("OglModel", name="model", position=probe_mesh.positions,
                           triangles=probe_mesh.triangles, color=[0.75, 0.75, 0.8, 1.0])
    probe_visual.addObject("RigidMapping")


def _build_constraint_scene(root, mesh, probe_mesh, gpu_tissue):
    """Constraint contact with friction, as in the CPU scene, solved on the GPU."""
    # Bounding boxes are for drawing; with the tissue on the GPU computing them each
    # frame would copy it back to the CPU.
    root.addObject("FreeMotionAnimationLoop", constraintSolver="@gpuConstraints", computeBoundingBox=not gpu_tissue)
    # Contacts up to the alarm distance, as LocalMinDistance reports them to
    # SOFA's constraint response; the constraint keeps the contact distance.
    _add_gpu_collision(root, contact_distance=pc.ALARM_DISTANCE, gpu_pipeline=gpu_tissue)

    if gpu_tissue:
        tissue, tissue_surface = _add_gpu_tissue_body(root, mesh)
    else:
        # Each body has its own implicit solver and direct linear solver, as in the
        # CPU scene: the constraint response reads their matrices.
        tissue, tissue_surface = _add_tissue_body(root, mesh, map_forces=False)
        tissue.addObject("EulerImplicitSolver", name="odeSolver", rayleighStiffness=0.0, rayleighMass=0.0)
        tissue.addObject("SparseLDLSolver", name="linearSolver", template="CompressedRowSparseMatrixMat3x3d")

    target = root.addChild("Target")
    target.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                     position=pc.rigid_pose(pc.START_HEIGHT))

    probe = root.addChild("Probe")
    probe.addObject("EulerImplicitSolver", name="odeSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    probe.addObject("SparseLDLSolver", name="linearSolver", template="CompressedRowSparseMatrixd")
    pc.add_probe_body(probe, external_rest_shape="@../Target/dofs")
    probe_surface = _add_gpu_surface(probe, probe_mesh.positions, probe_mesh.triangles, "RigidMapping", "Rigid3d",
                                     map_forces=False)
    _add_probe_visual(probe, probe_mesh)

    body1 =({"deformableGpuSolver": tissue.odeSolver.getLinkPath()} if gpu_tissue else
             {"deformableState": tissue.dofs.getLinkPath(), "deformableLinearSolver": tissue.linearSolver.getLinkPath(),
              "deformableOdeSolver": tissue.odeSolver.getLinkPath()})
    solver = root.addObject(
        "GpuContactConstraintSolver", name="gpuConstraints",
        friction=pc.FRICTION, contactDistance=pc.CONTACT_DISTANCE,
        tolerance=CONSTRAINT_TOLERANCE, maxIterations=CONSTRAINT_MAX_ITERATIONS,
        deformableSurface=tissue_surface.dofs.getLinkPath(),
        rigidState=probe.dofs.getLinkPath(), rigidSurface=probe_surface.dofs.getLinkPath(),
        rigidLinearSolver=probe.linearSolver.getLinkPath(), rigidOdeSolver=probe.odeSolver.getLinkPath(),
        response=RESPONSE, compareWithCpu=COMPARE, compareEvery=COMPARE_EVERY,
        compareFile=os.path.join(pc.default_log_dir(current_dir), LABEL + "_compare.csv"),
        measureTimes=MEASURE_TIMES, **body1)
    root.addObject(ConstraintStatsLogger(
        name="constraintStats", root=root, solver=solver, tissue_solver=tissue.odeSolver if gpu_tissue else None,
        path=os.path.join(pc.default_log_dir(current_dir), LABEL + "_constraints.csv")))
    return tissue, probe, target


def _build_penalty_scene(root, mesh, probe_mesh):
    """The older GPU contact: side-aware penalty forces inside one implicit solve."""
    # Penalty contact acts inside the solve, so collision must run first.
    root.addObject("DefaultAnimationLoop")
    _add_gpu_collision(root, contact_distance=pc.CONTACT_DISTANCE)

    # One solver over tissue and probe, so the contact couples them implicitly.
    sim = root.addChild("Simulation")
    sim.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    sim.addObject("CGLinearSolver", iterations=400, tolerance=1e-12, threshold=1e-18)
    tissue, tissue_surface = _add_tissue_body(sim, mesh)

    probe = sim.addChild("Probe")
    pc.add_probe_body(probe, external_rest_shape="@../../Target/dofs")
    probe_surface = _add_gpu_surface(probe, probe_mesh.positions, probe_mesh.triangles, "RigidMapping", "Rigid3d")
    _add_probe_visual(probe, probe_mesh)

    sim.addObject("CudaContactPenaltyForceField", name="contact",
                  object1=tissue_surface.dofs.getLinkPath(), object2=probe_surface.dofs.getLinkPath(),
                  stiffness=pc.PENALTY_STIFFNESS, contactDistance=pc.CONTACT_DISTANCE,
                  useSurfaceNormals=SIDE_AWARE, reportStats=REPORT_STATS, printLog=REPORT_STATS)

    target = root.addChild("Target")
    target.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                     position=pc.rigid_pose(pc.START_HEIGHT))
    return tissue, probe, target


def createScene(root):
    choice = _choose_backends()
    placement = ", ".join(f"{piece}={where}" for piece, where in choice.items())
    if choice["contact"] == "CPU":
        print(f"TissuePokeGPU placement: {placement} -> GPU contact not available, "
              f"using the CPU scene's constraint contact with friction", flush=True)
        return tissue_poke_cpu.build_scene(
            root, name="TissuePokeGPU", label=LABEL,
            notes=f"CPU fallback ({placement}): constraint contact with friction, direct solvers")

    root.name = "TissuePokeGPU"
    root.dt = pc.DT
    root.gravity = pc.GRAVITY
    root.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
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
        "SofaCUDA",
        # By path, not by name: the SOFA install carries an older copy of this plugin.
        GPU_COLLISION_LIB if os.path.isfile(GPU_COLLISION_LIB) else "SofaGpuCollision",
    ] + pc.MATERIAL_PLUGINS)
    root.addObject("VisualStyle", displayFlags="showVisualModels")
    pc.add_camera(root)

    mesh = pc.TissueMesh()
    probe_mesh = pc.ProbeMesh()
    gpu_tissue = choice["tissue"] == "GPU"
    if CONTACT_MODE == "constraint":
        tissue, probe, target = _build_constraint_scene(root, mesh, probe_mesh, gpu_tissue)
        who = "the GPU" if RESPONSE == "gpu" else "SOFA's CPU pipeline, on the GPU's contacts"
        contact_note = (f"constraint contact with friction computed by {who} (GpuContactConstraintSolver "
                        f"response={RESPONSE}, mu={pc.FRICTION}, tolerance={CONSTRAINT_TOLERANCE})")
    else:
        tissue, probe, target = _build_penalty_scene(root, mesh, probe_mesh)
        contact_note = "side-aware penalty contact on the GPU, no friction"
    tissue_note = (f"tissue on the GPU (GpuTissueSolver, {pc.material_note()})" if gpu_tissue
                   else f"tissue on the CPU ({pc.material_note()}, MeshMatrixMass, EulerImplicit + SparseLDL)")

    # ---- drive + measure ------------------------------------------------------
    root.addObject(pc.ProbeDriver(name="driver", root=root, target=target.dofs))
    root.addObject(pc.PokeLogger(
        name="logger", root=root, probe=probe.dofs, target=target.dofs, tissue=tissue.dofs,
        tissue_monitor=tissue.odeSolver if gpu_tissue else None,
        top_center_index=mesh.top_center_index, tetrahedra=mesh.tetrahedra,
        rest_positions=mesh.positions, label=LABEL,
        log_dir=pc.default_log_dir(current_dir),
        notes=f"GPU where available ({placement}): {tissue_note}; {contact_note}"))
    print(f"TissuePokeGPU placement: {placement}, contact={CONTACT_MODE}", flush=True)
    print(f"TissuePokeGPU: {len(mesh.positions)} nodes, {len(mesh.tetrahedra)} tetrahedra, "
          f"{len(mesh.surface_triangles)} surface triangles, probe {len(probe_mesh.triangles)} triangles, "
          f"{pc.total_steps()} steps of {pc.DT} s", flush=True)
    return root
