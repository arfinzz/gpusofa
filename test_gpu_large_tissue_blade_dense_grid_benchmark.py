import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    create_subdivided_blade_geometry,
    default_benchmark_log_dir,
    env_flag,
    generate_tissue_surface_grid,
    translate_vertices,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
BENCHMARK_LABEL_SUFFIX = os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")

TISSUE_NX = int(os.environ.get("SOFA_LARGE_TISSUE_NX", "181"))
TISSUE_NZ = int(os.environ.get("SOFA_LARGE_TISSUE_NZ", "181"))
BLADE_SEGMENTS_X = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_X", "128"))
BLADE_SEGMENTS_Y = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_Y", "24"))
BLADE_SEGMENTS_Z = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_Z", "4"))
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))
GRID_RESOLUTION_X = int(os.environ.get("SOFA_GRID_RESOLUTION_X", "96"))
GRID_RESOLUTION_Y = int(os.environ.get("SOFA_GRID_RESOLUTION_Y", "8"))
GRID_RESOLUTION_Z = int(os.environ.get("SOFA_GRID_RESOLUTION_Z", "96"))
MAX_TISSUE_TRIANGLES_PER_CELL = int(os.environ.get("SOFA_MAX_TISSUE_TRIANGLES_PER_CELL", "192"))
MAX_TOOL_TRIANGLES_PER_CELL = int(os.environ.get("SOFA_MAX_TOOL_TRIANGLES_PER_CELL", "192"))
MAX_CANDIDATE_PAIRS = int(os.environ.get("SOFA_MAX_CANDIDATE_PAIRS", "8000000"))
DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)
DEDUPLICATE_PAIRS = env_flag("SOFA_DEDUPLICATE_PAIRS", True)
COPY_CONTACTS_TO_HOST = env_flag("SOFA_COPY_CONTACTS_TO_HOST", False)
USE_PINNED_HOST_STAGING = env_flag("SOFA_USE_PINNED_HOST_STAGING", True)
USE_GPU_HASH_DEDUPE = env_flag("SOFA_USE_GPU_HASH_DEDUPE", DEDUPLICATE_PAIRS)
CANONICAL_PAIR_EMISSION = env_flag("SOFA_CANONICAL_PAIR_EMISSION", False)
USE_INDEXED_DENSE_GRID_INPUT = env_flag("SOFA_USE_INDEXED_DENSE_GRID_INPUT", True)
USE_DIRECT_DEVICE_POSITIONS = env_flag("SOFA_USE_DIRECT_DEVICE_POSITIONS", True)
READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE = env_flag("SOFA_READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE", False)
COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE = env_flag(
    "SOFA_COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE",
    False,
)
COMPACT_ACTIVE_CELLS = env_flag("SOFA_COMPACT_ACTIVE_CELLS", False)
BATCH_TRIANGLE_INSERT = env_flag("SOFA_BATCH_TRIANGLE_INSERT", False)


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
        'Sofa.Component.AnimationLoop',
    ])

    tissue_positions, tissue_triangles = generate_tissue_surface_grid(
        nx=TISSUE_NX,
        nz=TISSUE_NZ,
        sx=12.0,
        sz=12.0,
        y=0.0,
    )
    blade_vertices, blade_triangles = create_subdivided_blade_geometry(
        length=5.5,
        height=0.8,
        thickness=0.16,
        segments_x=BLADE_SEGMENTS_X,
        segments_y=BLADE_SEGMENTS_Y,
        segments_z=BLADE_SEGMENTS_Z,
    )
    blade_vertices = translate_vertices(blade_vertices, [0.0, 0.0, 0.0])

    root.addObject('DefaultAnimationLoop')
    root.addObject('CollisionPipeline')
    root.addObject(
        'GpuCollisionBroadPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        useObjectAabbCulling=False,
    )
    root.addObject(
        'GpuCollisionNarrowPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        useDenseGrid=True,
        deduplicatePairs=DEDUPLICATE_PAIRS,
        copyContactsToHost=COPY_CONTACTS_TO_HOST,
        detailedProfiling=DETAILED_PROFILING,
        usePinnedHostStaging=USE_PINNED_HOST_STAGING,
        useGpuHashDedupe=USE_GPU_HASH_DEDUPE,
        canonicalPairEmission=CANONICAL_PAIR_EMISSION,
        useIndexedDenseGridInput=USE_INDEXED_DENSE_GRID_INPUT,
        useDirectDevicePositions=USE_DIRECT_DEVICE_POSITIONS,
        readCountersWhenContactsStayOnDevice=READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE,
        computeDeviceContactsWhenContactsStayOnDevice=COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE,
        compactActiveCells=COMPACT_ACTIVE_CELLS,
        batchTriangleInsert=BATCH_TRIANGLE_INSERT,
        minGPUPairCount=1,
        gridMinX=-6.5,
        gridMinY=-0.7,
        gridMinZ=-6.5,
        gridMaxX=6.5,
        gridMaxY=0.7,
        gridMaxZ=6.5,
        gridResolutionX=GRID_RESOLUTION_X,
        gridResolutionY=GRID_RESOLUTION_Y,
        gridResolutionZ=GRID_RESOLUTION_Z,
        contactDistance=0.03,
        maxTissueTrianglesPerCell=MAX_TISSUE_TRIANGLES_PER_CELL,
        maxToolTrianglesPerCell=MAX_TOOL_TRIANGLES_PER_CELL,
        maxCandidatePairs=MAX_CANDIDATE_PAIRS,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuLargeTissueBladeTiming',
        label='gpu_large_tissue_blade_dense_grid_benchmark' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='large-plan-b-dense-grid-detection-only',
        tissueSolver='none',
        tissueForceField='none',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='none (direct GPU collision surfaces)',
        visualMapping='none',
        notes='Large static surgical collision benchmark: one high-resolution tissue surface and one subdivided blade slab. Dense-grid GPU narrow phase uses direct CUDA positions, indexed topology, GPU hash dedupe, optional experimental batched insertion/active-cell compaction, and keeps counters/contacts device-side when collision response is disabled.',
        simVertexCount=0,
        simElementCount=0,
        collisionVertexCount=len(tissue_positions) + len(blade_vertices),
        collisionElementCount=len(tissue_triangles) + len(blade_triangles),
        visualVertexCount=0,
        visualElementCount=0,
        warmupSteps=WARMUP_STEPS,
        flushInterval=10,
        logInterval=10,
        printProgress=True,
    )

    tissue = root.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
    tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
    tissue.addObject('TriangleCollisionModel', selfCollision=False)

    blade = root.addChild('Blade')
    blade.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=blade_vertices)
    blade.addObject('MeshTopology', name='topo', triangles=blade_triangles)
    blade.addObject('TriangleCollisionModel', selfCollision=False)

    return root
