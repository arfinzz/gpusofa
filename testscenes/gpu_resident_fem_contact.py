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

Env toggles:
  SOFA_CONTACT_STIFFNESS      penalty stiffness           (default 2000)
  SOFA_CONTACT_DAMPING        penalty damping             (default 0, off)
  SOFA_TISSUE_YOUNG           tissue Young's modulus      (default 3000)
  SOFA_TISSUE_NX / NY / NZ    tet grid resolution         (default 21/4/21)
  SOFA_BLADE_MASS             tool mass, for the Gate-3 equilibrium check
  SOFA_RESIDENCY_FAIL_FAST    1 = msg_error on any transfer
  SOFA_USE_BIGCELL_FUSED_GENERATION / ... same broad-cull selectors as the other scenes
"""

import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    create_blade_geometry,
    default_benchmark_log_dir,
    env_flag,
    generate_tissue_mesh,
    translate_vertices,
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
CONTACT_DAMPING = float(os.environ.get("SOFA_CONTACT_DAMPING", "0"))
CONTACT_DISTANCE = float(os.environ.get("SOFA_CONTACT_DISTANCE", "0.03"))
BLADE_MASS = float(os.environ.get("SOFA_BLADE_MASS", "0.05"))
BLADE_DROP_HEIGHT = float(os.environ.get("SOFA_BLADE_DROP_HEIGHT", "0.6"))

CONTACT_REPORT_STATS = env_flag("SOFA_CONTACT_REPORT_STATS", False)
RESIDENCY_FAIL_FAST = env_flag("SOFA_RESIDENCY_FAIL_FAST", False)
RESIDENCY_START_FRAME = int(os.environ.get("SOFA_RESIDENCY_START_FRAME", "5"))
DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)

USE_HASH_PREFIXSUM = env_flag("SOFA_USE_HASH_PREFIXSUM_GENERATION", False)
USE_SIMPLE_HASH = env_flag("SOFA_USE_SIMPLE_HASH_GENERATION", False)
USE_SORTED_GRID = env_flag("SOFA_USE_SORTED_GRID_GENERATION", False)
USE_BIGCELL_FUSED = env_flag("SOFA_USE_BIGCELL_FUSED_GENERATION", True)
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))


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
    ])

    tissue_positions, tissue_tets, fixed_indices, tissue_surface = generate_tissue_mesh(
        nx=TISSUE_NX, ny=TISSUE_NY, nz=TISSUE_NZ, sx=4.0, sy=0.5, sz=4.0, border=1)

    blade_verts, blade_tris = create_blade_geometry(length=1.1, height=0.28, thickness=0.08)
    blade_verts = translate_vertices(blade_verts, [0.0, BLADE_DROP_HEIGHT, 0.0])

    # collision -> integrate ordering; the contact buffer is consumed during the solve.
    root.addObject('DefaultAnimationLoop')
    root.addObject('CollisionPipeline')
    root.addObject('GpuCollisionBroadPhase', enableGPU=True, allowCPUFallback=True,
                   logBackendStatus=True, useObjectAabbCulling=False)
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
                  iterations=25, tolerance=1e-6, threshold=1e-9)

    # ---- deformable tissue: everything Cuda*, so state never leaves the GPU ----
    tissue = sim.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f',
                     position=tissue_positions)
    tissue.addObject('TetrahedronSetTopologyContainer', name='topo',
                     tetrahedra=tissue_tets)
    tissue.addObject('TetrahedronSetGeometryAlgorithms', template='CudaVec3f')
    # NOTE: MeshMatrixMass<CudaVec3f,CudaVec3f> SEGFAULTS in copyVertexMass()
    # during init in this SOFA build (v25.12) — verified with a backtrace, and it
    # is a fault inside SOFA's own component, not in this scene. UniformMass is
    # used instead: GPU-resident, and its uniform distribution actually makes the
    # Gate-3 equilibrium prediction easier to reason about.
    tissue.addObject('UniformMass', template='CudaVec3f',
                     name='mass', totalMass=TISSUE_TOTAL_MASS)
    tissue.addObject('TetrahedronFEMForceField', template='CudaVec3f', name='fem',
                     method='large', youngModulus=TISSUE_YOUNG, poissonRatio=TISSUE_POISSON)
    tissue.addObject('FixedProjectiveConstraint', template='CudaVec3f',
                     name='fixed', indices=fixed_indices)

    # Collision surface of the tissue, in the same node so it shares the DOFs
    # (the surface triangles index the same vertices — no mapping needed).
    tissue_surface_node = tissue.addChild('TissueSurface')
    tissue_surface_node.addObject('MeshTopology', name='surftopo', triangles=tissue_surface)
    tissue_surface_node.addObject('TriangleCollisionModel', name='tissueCM', selfCollision=False)

    # ---- tool: blade falling under gravity ----
    blade = sim.addChild('Blade')
    blade.addObject('MechanicalObject', name='dofs', template='CudaVec3f',
                    position=blade_verts)
    blade.addObject('MeshTopology', name='topo', triangles=blade_tris)
    blade.addObject('UniformMass', template='CudaVec3f', name='mass',
                    totalMass=BLADE_MASS)
    blade.addObject('TriangleCollisionModel', name='bladeCM', selfCollision=False)

    # ---- the contact consumer: device contacts -> device forces ----
    # Inside the solver node so the solver sees it; it resolves each surface id
    # from the CudaTriangleCollisionModel in that state's context, matching the
    # ids the narrow phase recorded alongside the contacts.
    sim.addObject(
        'CudaContactPenaltyForceField',
        name='contactForces',
        object1='@Tissue/dofs',
        object2='@Blade/dofs',
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
