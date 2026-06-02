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
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.Constraint.Lagrangian.Solver',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=50, tolerance=1e-6)
    root.addObject('CollisionPipeline')
    root.addObject('GpuCollisionBroadPhase', enableGPU=True, allowCPUFallback=True, logBackendStatus=True)
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
    root.addObject('LocalMinDistance', alarmDistance=0.20, contactDistance=0.05, angleCone=0.0)
    if _env_bool('SOFA_ENABLE_VALIDATION_RESPONSE', False):
        root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.4')

    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuTissuePhase45ValidationTiming',
        label='gpu_tissue_phase45_compact_candidates',
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='phase45-validation',
        tissueSolver='validation-static-overlap',
        tissueForceField='none',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='none (direct GPU collision surfaces)',
        visualMapping='none',
        notes='Phase 4/5 validation scene with one tissue-like GPU triangle surface and one overlapping tool-like GPU triangle surface to guarantee GPU broad pairs and exact GPU contacts. Collision response is disabled by default because response is a later phase.',
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

    tissue = root.addChild('TissueSurface')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=[
        [-3.0, 0.0, -3.0],
        [3.0, 0.0, -3.0],
        [3.0, 0.0, 3.0],
        [-3.0, 0.0, 3.0],
    ])
    tissue.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]])
    tissue.addObject('TriangleCollisionModel', selfCollision=False)

    tool = root.addChild('ToolSurface')
    tool.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=[
        [-1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ])
    tool.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]])
    tool.addObject('TriangleCollisionModel', selfCollision=False)

    return root
