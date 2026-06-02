import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import default_benchmark_log_dir


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.005

    root.addObject('RequiredPlugin', pluginName=[
        'SofaCUDA',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Mapping',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Lagrangian.Correction',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=50, tolerance=1e-6)
    root.addObject('CollisionPipeline')
    root.addObject(
        'GpuCollisionBroadPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        logBoxesOnce=_env_bool('SOFA_GPU_BROAD_LOG_BOXES_ONCE', False),
    )
    root.addObject(
        'GpuCollisionNarrowPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        detailedProfiling=_env_bool('SOFA_GPU_DETAILED_PROFILING', False),
        copyContactsToHost=_env_bool('SOFA_COPY_CONTACTS_TO_HOST', True),
        deduplicatePairs=_env_bool('SOFA_DEDUPLICATE_PAIRS', True),
        usePinnedHostStaging=_env_bool('SOFA_USE_PINNED_HOST_STAGING', True),
        useGpuHashDedupe=_env_bool('SOFA_USE_GPU_HASH_DEDUPE', _env_bool('SOFA_DEDUPLICATE_PAIRS', True)),
        canonicalPairEmission=_env_bool('SOFA_CANONICAL_PAIR_EMISSION', False),
        minGPUPairCount=1,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.2, contactDistance=0.05, angleCone=0.0)
    if _env_bool('SOFA_ENABLE_VALIDATION_RESPONSE', False):
        root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.4')

    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuPhase5OverlapValidationTiming',
        label='gpu_phase5_overlap_validation',
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='phase5-validation',
        tissueSolver='static-overlap-validation',
        tissueForceField='none',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='none (direct GPU collision surfaces)',
        visualMapping='none',
        notes='Phase 5 validation scene with two overlapping GPU triangle surfaces to guarantee GPU broad-phase pairs and exact GPU narrow-phase contacts. Collision response is disabled by default because response is a later phase.',
        simVertexCount=8,
        simElementCount=0,
        collisionVertexCount=8,
        collisionElementCount=4,
        visualVertexCount=0,
        visualElementCount=0,
        warmupSteps=0,
        flushInterval=10,
        logInterval=10,
        printProgress=True,
    )

    surface_a = root.addChild('SurfaceA')
    surface_a.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=[
        [-1.2, 0.0, -1.2],
        [1.2, 0.0, -1.2],
        [1.2, 0.0, 1.2],
        [-1.2, 0.0, 1.2],
    ])
    surface_a.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]])
    surface_a.addObject('TriangleCollisionModel', selfCollision=False)

    surface_b = root.addChild('SurfaceB')
    surface_b.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=[
        [-1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ])
    surface_b.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]])
    surface_b.addObject('TriangleCollisionModel', selfCollision=False)

    return root
