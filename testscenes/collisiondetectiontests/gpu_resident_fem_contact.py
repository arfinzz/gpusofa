"""First fully GPU-resident simulation scene (Tier 1, 2026-07-15).

Every scene before this one was collision-detection-only: a MechanicalObject, a
topology and a collision model, with no solver, no mass and no force field —
contacts were computed and then discarded. This scene closes the loop:

    tet tissue (FEM, GPU)  --collision-->  device contact buffer
            ^                                      |
            +----- CudaContactPenaltyForceField <--+   (GPU, no host round trip)

Everything that touches simulation state is a Cuda* component, so the state
vectors never need a host copy. `GpuResidencyChecker` asserts exactly that every
frame and names any vector that gets pulled to the host.

Animation loop: DefaultAnimationLoop, whose step order is collision -> integrate.
The contact buffer produced during the collision step is therefore read while the
solver evaluates forces. (FreeMotionAnimationLoop is for the constraint path.)

The blade is a rigid body (2026-09-25): a Rigid3d pose with the box's mass and
inertia, and its GPU collision surface attached through GpuRigidMapping (computed
on the GPU from the pose; this plugin's own, because SofaCUDA's GPU RigidMapping
maps surface forces to a wrong torque). Before, it was 8 loose points with no force
holding them together, and it came apart on landing. GpuCollisionPipeline, the broad phase's testGpuModelBoxes=false and the
loop's computeBoundingBox=false keep SOFA's per-frame bounding boxes from copying
the GPU surfaces to the CPU. BladeLogger writes the blade's height, speed, tilt and
kinetic energy each frame (Gate 3).

Env toggles:
  SOFA_CONTACT_STIFFNESS      penalty stiffness           (default 2000)
  SOFA_CONTACT_DAMPING        penalty damping             (default 0, off)
  SOFA_TISSUE_YOUNG           tissue Young's modulus      (default 3000)
  SOFA_TISSUE_NX / NY / NZ    tet grid resolution         (default 21/4/21)
  SOFA_BLADE_MASS             tool mass, for the Gate-3 equilibrium check
  SOFA_RESIDENCY_FAIL_FAST    1 = msg_error on any transfer
  SOFA_USE_BIGCELL_FUSED_GENERATION / ... same broad-cull selectors as the other scenes
"""

import math
import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    create_blade_geometry,
    create_subdivided_blade_geometry,
    default_benchmark_log_dir,
    env_flag,
    generate_tissue_mesh,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
BENCHMARK_LABEL_SUFFIX = os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")

TISSUE_NX = int(os.environ.get("SOFA_TISSUE_NX", "21"))
TISSUE_NY = int(os.environ.get("SOFA_TISSUE_NY", "4"))
TISSUE_NZ = int(os.environ.get("SOFA_TISSUE_NZ", "21"))
TISSUE_YOUNG = float(os.environ.get("SOFA_TISSUE_YOUNG", "3000"))
TISSUE_POISSON = float(os.environ.get("SOFA_TISSUE_POISSON", "0.4"))
TISSUE_TOTAL_MASS = float(os.environ.get("SOFA_TISSUE_TOTAL_MASS", "1.0"))

CONTACT_STIFFNESS = float(os.environ.get("SOFA_CONTACT_STIFFNESS", "2000"))
# Diagnostics: hold every tissue node fixed; scale the blade's rotational inertia.
DIAG_FIX_ALL_TISSUE = env_flag("SOFA_DIAG_FIX_ALL_TISSUE", False)
DIAG_CPU_RIGID_MAPPING = env_flag("SOFA_DIAG_CPU_RIGID_MAPPING", False)   # SOFA's CPU RigidMapping, then onto the GPU
BLADE_INERTIA_SCALE = float(os.environ.get("SOFA_BLADE_INERTIA_SCALE", "1"))
CG_ITERATIONS = int(os.environ.get("SOFA_CG_ITERATIONS", "25"))
CG_TOLERANCE = float(os.environ.get("SOFA_CG_TOLERANCE", "1e-6"))
CG_THRESHOLD = float(os.environ.get("SOFA_CG_THRESHOLD", "1e-9"))
CONTACT_DAMPING = float(os.environ.get("SOFA_CONTACT_DAMPING", "0"))
CONTACT_DISTANCE = float(os.environ.get("SOFA_CONTACT_DISTANCE", "0.03"))
BLADE_MASS = float(os.environ.get("SOFA_BLADE_MASS", "0.05"))
BLADE_DROP_HEIGHT = float(os.environ.get("SOFA_BLADE_DROP_HEIGHT", "0.6"))

CONTACT_REPORT_STATS = env_flag("SOFA_CONTACT_REPORT_STATS", False)
# Diagnostic: drop the collision models + pipeline entirely. Used to attribute a
# device->host transfer of `position` seen at frame-begin: if it disappears with
# collision off, the culprit is SOFA's collision-model bookkeeping (namely
# TriangleCollisionModel::computeBoundingTree, for which SofaCUDA provides no
# GPU override) rather than the solver or the force fields.
NO_COLLISION = env_flag("SOFA_DIAG_NO_COLLISION", False)
# Diagnostic bisect: drop the benchmark controller / the FEM force field /
# the topology geometry algorithms, to attribute the frame-begin position
# transfer to a specific component.
NO_BENCH = env_flag("SOFA_DIAG_NO_BENCH", False)
NO_FEM = env_flag("SOFA_DIAG_NO_FEM", False)
NO_GEOMALGO = env_flag("SOFA_DIAG_NO_GEOMALGO", False)
RESIDENCY_FAIL_FAST = env_flag("SOFA_RESIDENCY_FAIL_FAST", False)
RESIDENCY_START_FRAME = int(os.environ.get("SOFA_RESIDENCY_START_FRAME", "5"))
DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)

USE_HASH_PREFIXSUM = env_flag("SOFA_USE_HASH_PREFIXSUM_GENERATION", False)
USE_SIMPLE_HASH = env_flag("SOFA_USE_SIMPLE_HASH_GENERATION", False)
USE_SORTED_GRID = env_flag("SOFA_USE_SORTED_GRID_GENERATION", False)
USE_BIGCELL_FUSED = env_flag("SOFA_USE_BIGCELL_FUSED_GENERATION", True)
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))

BLADE_LENGTH, BLADE_HEIGHT, BLADE_THICKNESS = 1.1, 0.28, 0.08


class BladeLogger(Sofa.Core.Controller):
    """Gate 3: the blade's height, speed, tilt and kinetic energy each frame (its
    Rigid3d state is on the CPU, so this reads nothing back from the GPU)."""

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.blade = kwargs["blade"]
        self.mass = kwargs["mass"]
        self.inertia = kwargs["inertia"]      # principal moments, kg m^2
        path = kwargs["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv = open(path, "w")
        self.csv.write("time,center_y,bottom_y,speed,tilt_deg,kinetic_energy,potential_energy\n")

    def onAnimateEndEvent(self, event):
        x = self.blade.position.value[0]
        v = self.blade.velocity.value[0]
        qx, qy, qz, qw = (float(q) for q in x[3:7])
        # tilt: angle between the blade's own y axis and the world y axis
        y_axis_y = 1.0 - 2.0 * (qx * qx + qz * qz)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, y_axis_y))))
        speed = math.sqrt(sum(float(c) ** 2 for c in v[0:3]))
        omega = [float(c) for c in v[3:6]]
        kinetic = 0.5 * self.mass * speed ** 2 + 0.5 * sum(i * w * w for i, w in zip(self.inertia, omega))
        potential = self.mass * 9.81 * float(x[1])
        row = [self.root.time.value, float(x[1]), float(x[1]) - 0.5 * BLADE_HEIGHT, speed, tilt, kinetic, potential]
        self.csv.write(",".join(f"{c:.9g}" for c in row) + "\n")
        self.csv.flush()


def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.005

    root.addObject('RequiredPlugin', pluginName=[
        'SofaCUDA',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Mass',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Mapping.NonLinear',
    ])

    tissue_positions, tissue_tets, fixed_indices, tissue_surface = generate_tissue_mesh(
        nx=TISSUE_NX, ny=TISSUE_NY, nz=TISSUE_NZ, sx=4.0, sy=0.5, sz=4.0, border=1)
    if DIAG_FIX_ALL_TISSUE:
        fixed_indices = list(range(len(tissue_positions)))

    # The blade's surface in its own frame (centred); its pose starts at the drop height.
    blade_verts, blade_tris = create_blade_geometry(length=BLADE_LENGTH, height=BLADE_HEIGHT,
                                                    thickness=BLADE_THICKNESS)

    # collision -> integrate ordering; the contact buffer is consumed during the solve.
    # No per-frame bounding box (it is for drawing, and would copy the GPU states back).
    root.addObject('DefaultAnimationLoop', computeBoundingBox=False)
    if not NO_COLLISION:
      # Builds the GPU surfaces' bounding trees once instead of every frame (each
      # rebuild copies the surface to the CPU); the broad phase then skips the box test
      # for pairs of GPU surfaces, whose boxes are no longer updated.
      root.addObject('GpuCollisionPipeline')
      root.addObject('GpuCollisionBroadPhase', enableGPU=True, allowCPUFallback=True,
                   logBackendStatus=True, useObjectAabbCulling=False, testGpuModelBoxes=False)
      root.addObject(
        'GpuCollisionNarrowPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        useDenseGrid=True,
        useIndexedDenseGridInput=True,
        useDirectDevicePositions=True,
        cacheTriangleTopology=True,
        copyContactsToHost=False,               # contacts stay on the device
        proximityKeepContactsOnDevice=True,     # ... and are consumed there
        useFeatureBasedProximity=True,
        useHashPrefixSumGeneration=USE_HASH_PREFIXSUM,
        useSimpleHashGeneration=USE_SIMPLE_HASH,
        useSortedGridGeneration=USE_SORTED_GRID,
        useBigCellFusedGeneration=USE_BIGCELL_FUSED,
        detailedProfiling=DETAILED_PROFILING,
        proximityComputeBarycentrics=True,      # the force scatter needs the weights
        proximityReadContactCounter=False,      # a readback here would break residency
        proximityMaxContacts=2000000,
        minGPUPairCount=1,
        gridMinX=-2.5, gridMinY=-1.0, gridMinZ=-2.5,
        gridMaxX=2.5, gridMaxY=1.5, gridMaxZ=2.5,
        gridResolutionX=48, gridResolutionY=16, gridResolutionZ=48,
        contactDistance=CONTACT_DISTANCE,
        maxTissueTrianglesPerCell=128,
        maxToolTrianglesPerCell=128,
        maxCandidatePairs=4000000,
    )
      root.addObject('LocalMinDistance', alarmDistance=CONTACT_DISTANCE * 2.0,
                     contactDistance=CONTACT_DISTANCE, angleCone=0.0)

    # ---- ONE solver over both bodies ----------------------------------------
    # Tissue and blade share a single ODE + linear solver so the contact force
    # field couples them IMPLICITLY: the solver assembles both bodies' DOFs into
    # one system and consults addDForce for the contact stiffness. Giving each
    # body its own solver would make the coupling explicit and unstable, and an
    # interaction force field spanning two independently-solved objects is not a
    # valid SOFA construction.
    sim = root.addChild('Simulation')
    sim.addObject('EulerImplicitSolver', name='odesolver',
                  rayleighStiffness=0.05, rayleighMass=0.05)
    # Matrix-free CG: no assembled system matrix, so the solve stays on the device.
    sim.addObject('CGLinearSolver', name='linearsolver',
                  iterations=CG_ITERATIONS, tolerance=CG_TOLERANCE, threshold=CG_THRESHOLD)

    # ---- deformable tissue: everything Cuda*, so state never leaves the GPU ----
    tissue = sim.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f',
                     position=tissue_positions)
    tissue.addObject('TetrahedronSetTopologyContainer', name='topo',
                     tetrahedra=tissue_tets)
    if not NO_GEOMALGO:
        tissue.addObject('TetrahedronSetGeometryAlgorithms', template='CudaVec3f')
    # NOTE: MeshMatrixMass<CudaVec3f,CudaVec3f> SEGFAULTS in copyVertexMass()
    # during init in this SOFA build (v25.12) — verified with a backtrace, and it
    # is a fault inside SOFA's own component, not in this scene. UniformMass is
    # used instead: GPU-resident, and its uniform distribution actually makes the
    # Gate-3 equilibrium prediction easier to reason about.
    tissue.addObject('UniformMass', template='CudaVec3f',
                     name='mass', totalMass=TISSUE_TOTAL_MASS)
    if not NO_FEM:
      tissue.addObject('TetrahedronFEMForceField', template='CudaVec3f', name='fem',
                     method='large', youngModulus=TISSUE_YOUNG, poissonRatio=TISSUE_POISSON)
    tissue.addObject('FixedProjectiveConstraint', template='CudaVec3f',
                     name='fixed', indices=fixed_indices)

    # Collision surface of the tissue, in the same node so it shares the DOFs
    # (the surface triangles index the same vertices — no mapping needed).
    tissue_surface_node = tissue.addChild('TissueSurface')
    tissue_surface_node.addObject('MeshTopology', name='surftopo', triangles=tissue_surface)
    if not NO_COLLISION:
        tissue_surface_node.addObject('TriangleCollisionModel', name='tissueCM', selfCollision=False)

    # ---- tool: a rigid blade falling under gravity ----
    # A box's mass and inertia: RigidMass takes the inertia per unit of mass.
    a, b, c = BLADE_LENGTH, BLADE_HEIGHT, BLADE_THICKNESS
    inertia_per_mass = [BLADE_INERTIA_SCALE * v for v in
                        ((b * b + c * c) / 12.0, (a * a + c * c) / 12.0, (a * a + b * b) / 12.0)]
    blade = sim.addChild('Blade')
    blade.addObject('MechanicalObject', name='dofs', template='Rigid3d',
                    position=[[0.0, BLADE_DROP_HEIGHT, 0.0, 0.0, 0.0, 0.0, 1.0]])
    blade_mass = blade.addObject('UniformMass', name='mass', template='Rigid3d',
                                 vertexMass=" ".join(repr(float(v)) for v in (
                                     BLADE_MASS, a * b * c, inertia_per_mass[0], 0, 0, 0,
                                     inertia_per_mass[1], 0, 0, 0, inertia_per_mass[2])))
    if DIAG_CPU_RIGID_MAPPING:
        cpu_surface = blade.addChild('CpuSurface')
        cpu_surface.addObject('MechanicalObject', name='dofs', template='Vec3d', position=blade_verts)
        cpu_surface.addObject('RigidMapping', template='Rigid3d,Vec3d')
        blade_surface = cpu_surface.addChild('Surface')
        blade_surface.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=blade_verts)
        blade_surface.addObject('MeshTopology', name='topo', triangles=blade_tris)
        if not NO_COLLISION:
            blade_surface.addObject('TriangleCollisionModel', name='bladeCM', selfCollision=False)
        blade_surface.addObject('IdentityMapping', template='Vec3d,CudaVec3f')
    else:
        blade_surface = blade.addChild('Surface')
        blade_surface.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=blade_verts)
        blade_surface.addObject('MeshTopology', name='topo', triangles=blade_tris)
        if not NO_COLLISION:
            blade_surface.addObject('TriangleCollisionModel', name='bladeCM', selfCollision=False)
        # Computed on the GPU from the pose; the penalty forces map back to the rigid body
        # as a force and a torque. (SofaCUDA's RigidMapping<Rigid3d,CudaVec3f> maps them to
        # a wrong torque in SOFA v25.12, and the blade spun up and shot through the tissue.)
        blade_surface.addObject('GpuRigidMapping')

    # ---- the contact consumer: device contacts -> device forces ----
    # Inside the solver node so the solver sees it; it resolves each surface id
    # from the CudaTriangleCollisionModel in that state's context, matching the
    # ids the narrow phase recorded alongside the contacts.
    if not NO_COLLISION:
      sim.addObject(
        'CudaContactPenaltyForceField',
        name='contactForces',
        object1='@Tissue/dofs',
        object2=blade_surface.dofs.getLinkPath(),
        stiffness=CONTACT_STIFFNESS,
        damping=CONTACT_DAMPING,
        useDamping=CONTACT_DAMPING > 0.0,
        contactDistance=CONTACT_DISTANCE,
        # Diagnostic only: reading the contact counters costs one sync per frame.
        # It reads OUR device buffer, not SOFA state, so Gate 5 still holds — but
        # it does serialise the frame, so leave it off for timing runs.
        reportStats=CONTACT_REPORT_STATS,
        printLog=CONTACT_REPORT_STATS,   # msg_info is suppressed without this
    )

    # ---- Gate 5: assert nothing pulled state to the host this frame ----
    root.addObject(
        'GpuResidencyChecker',
        name='residency',
        checkPosition=True,
        checkVelocity=True,
        checkForce=True,
        startFrame=RESIDENCY_START_FRAME,
        reportInterval=20,
        failFast=RESIDENCY_FAIL_FAST,
        printLog=True,   # so the clean/violation tally is visible in the log
    )

    print(f"Blade: rigid, mass={BLADE_MASS} kg, vertexMass='{blade_mass.vertexMass.getValueString()}'", flush=True)
    root.addObject(BladeLogger(
        name='bladeLogger', root=root, blade=blade.dofs, mass=BLADE_MASS,
        inertia=[BLADE_MASS * i for i in inertia_per_mass],
        path=os.path.join(BENCHMARK_LOG_DIR, 'gpu_resident_blade' + BENCHMARK_LABEL_SUFFIX + '.csv')))

    if not NO_BENCH:
      root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuResidentTiming',
        label='gpu_resident_fem_contact' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='gpu-resident-fem-contact',
        collisionStateTemplate='CudaVec3f',
        notes='First fully GPU-resident scene: Cuda FEM tissue + GPU collision + GPU penalty contact response. GpuResidencyChecker asserts zero device-to-host transfer.',
        collisionVertexCount=len(tissue_positions) + len(blade_verts),
        collisionElementCount=len(tissue_surface) + len(blade_tris),
        warmupSteps=WARMUP_STEPS,
        flushInterval=10,
        logInterval=10,
        printProgress=True,
    )

    return root
