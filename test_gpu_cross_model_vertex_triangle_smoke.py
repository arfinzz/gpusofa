"""Cross-model vertex-triangle smoke test (Phase 12 cross-model wiring).

Geometry: a tissue triangle mesh (a flat XZ grid) + a small separate point
cloud above it (e.g. tool-tip vertex set). Each tool point sits 0.04 above a
tissue triangle. With contactDistance=0.05 each tool point should emit one
or more proximity contacts against the tissue triangles.

The two sides are different SOFA components:
  - tissue uses `MechanicalObject template='CudaVec3f'` + `TriangleCollisionModel`
  - tool uses `MechanicalObject template='CudaVec3f'` + `CudaPointCollisionModel`

The narrow phase detects the cross-model pair and routes it through
`computeFeatureBasedVertexTriangleContacts`. Own-corner exclusion stays OFF
because the surfaceIds differ.
"""

import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    default_benchmark_log_dir,
    env_flag,
    generate_tissue_surface_grid,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
BENCHMARK_LABEL_SUFFIX = os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")
DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)
DEDUPLICATE_PAIRS = env_flag("SOFA_DEDUPLICATE_PAIRS", True)
COPY_CONTACTS_TO_HOST = env_flag("SOFA_COPY_CONTACTS_TO_HOST", False)
USE_PINNED_HOST_STAGING = env_flag("SOFA_USE_PINNED_HOST_STAGING", True)
USE_GPU_HASH_DEDUPE = env_flag("SOFA_USE_GPU_HASH_DEDUPE", DEDUPLICATE_PAIRS)
USE_INDEXED_DENSE_GRID_INPUT = env_flag("SOFA_USE_INDEXED_DENSE_GRID_INPUT", True)
USE_DIRECT_DEVICE_POSITIONS = env_flag("SOFA_USE_DIRECT_DEVICE_POSITIONS", True)
USE_FEATURE_BASED_PROXIMITY = env_flag("SOFA_USE_FEATURE_BASED_PROXIMITY", True)
USE_VERTEX_TRIANGLE_PROXIMITY = env_flag("SOFA_USE_VERTEX_TRIANGLE_PROXIMITY", True)
PROXIMITY_COMPUTE_BARYCENTRICS = env_flag("SOFA_PROXIMITY_COMPUTE_BARYCENTRICS", True)
PROXIMITY_READ_CONTACT_COUNTER = env_flag("SOFA_PROXIMITY_READ_CONTACT_COUNTER", True)
PROXIMITY_KEEP_CONTACTS_ON_DEVICE = env_flag("SOFA_PROXIMITY_KEEP_CONTACTS_ON_DEVICE", True)
PROXIMITY_MAX_CONTACTS = int(os.environ.get("SOFA_PROXIMITY_MAX_CONTACTS", "1000000"))
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))


def generate_tool_point_cloud(n=8, sx=2.0, sz=2.0, y=0.04):
    """A small grid of points sitting `y` above the XZ plane."""
    positions = []
    for k in range(n):
        for i in range(n):
            x = -sx / 2.0 + i * sx / (n - 1)
            z = -sz / 2.0 + k * sz / (n - 1)
            positions.append([x, y, z])
    return positions


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
        nx=41, nz=41, sx=4.0, sz=4.0, y=0.0)
    tool_positions = generate_tool_point_cloud(n=8, sx=2.0, sz=2.0, y=0.04)

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
        useIndexedDenseGridInput=USE_INDEXED_DENSE_GRID_INPUT,
        useDirectDevicePositions=USE_DIRECT_DEVICE_POSITIONS,
        useFeatureBasedProximity=USE_FEATURE_BASED_PROXIMITY,
        useVertexTriangleProximity=USE_VERTEX_TRIANGLE_PROXIMITY,
        proximityComputeBarycentrics=PROXIMITY_COMPUTE_BARYCENTRICS,
        proximityReadContactCounter=PROXIMITY_READ_CONTACT_COUNTER,
        proximityKeepContactsOnDevice=PROXIMITY_KEEP_CONTACTS_ON_DEVICE,
        proximityMaxContacts=PROXIMITY_MAX_CONTACTS,
        minGPUPairCount=1,
        gridMinX=-2.5,
        gridMinY=-0.2,
        gridMinZ=-2.5,
        gridMaxX=2.5,
        gridMaxY=0.3,
        gridMaxZ=2.5,
        gridResolutionX=32,
        gridResolutionY=4,
        gridResolutionZ=32,
        contactDistance=0.05,
        maxTissueTrianglesPerCell=64,
        maxToolTrianglesPerCell=64,
        maxCandidatePairs=2000000,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.05, angleCone=0.0)
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuCrossModelVtTiming',
        label='gpu_cross_model_vertex_triangle_smoke' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='cross-model-vertex-triangle-detection-only',
        tissueSolver='none',
        tissueForceField='none',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='none (direct GPU collision surfaces)',
        visualMapping='none',
        notes='Cross-model vertex-triangle smoke test. Tissue is a CudaTriangleCollisionModel; tool is a CudaPointCollisionModel hovering 0.04 above it. ContactDistance=0.05 emits one or more v-t contacts per tool point.',
        simVertexCount=0,
        simElementCount=0,
        collisionVertexCount=len(tissue_positions) + len(tool_positions),
        collisionElementCount=len(tissue_triangles),
        visualVertexCount=0,
        visualElementCount=0,
        warmupSteps=WARMUP_STEPS,
        flushInterval=20,
        logInterval=20,
        printProgress=True,
    )

    tissue = root.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
    tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
    tissue.addObject('TriangleCollisionModel', selfCollision=False)

    tool = root.addChild('Tool')
    tool.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tool_positions)
    tool.addObject('CudaPointCollisionModel')

    return root
