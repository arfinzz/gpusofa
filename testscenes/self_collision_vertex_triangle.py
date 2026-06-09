"""Vertex-triangle self-collision smoke test (Phase 12 SOFA wiring).

Geometry: a thin two-layer slab. Each top-layer vertex sits 0.05 above a
bottom-layer triangle. With contactDistance=0.06 every top vertex is in
proximity to its mirroring bottom triangle, so the kernel should emit one
contact per top vertex. Because both layers are part of the same
CudaTriangleCollisionModel the broad phase emits a (cm, cm) self-collision
pair and the narrow phase routes it through computeFeatureBasedVertexTriangleContacts.

Detection-only by default (no CPU response). Use the env vars listed below to
flip readback / barycentric output for validation runs.
"""

import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    default_benchmark_log_dir,
    env_flag,
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


def generate_two_layer_slab(nx=16, nz=16, sx=4.0, sz=4.0, layer_gap=0.05):
    """Build a two-layer flat mesh with `layer_gap` between layers.

    Returns (positions, triangles) where:
      - positions[0..nx*nz-1] are the bottom layer (y=0)
      - positions[nx*nz..] are the top layer (y=layer_gap)
      - triangles cover both layers separately (no connecting walls)
    """
    positions = []
    # Bottom layer y=0
    for k in range(nz):
        for i in range(nx):
            x = -sx / 2.0 + i * sx / (nx - 1)
            z = -sz / 2.0 + k * sz / (nz - 1)
            positions.append([x, 0.0, z])
    # Top layer y=layer_gap (same XZ grid)
    for k in range(nz):
        for i in range(nx):
            x = -sx / 2.0 + i * sx / (nx - 1)
            z = -sz / 2.0 + k * sz / (nz - 1)
            positions.append([x, layer_gap, z])

    triangles = []

    def add_layer_triangles(base):
        for k in range(nz - 1):
            for i in range(nx - 1):
                v0 = base + i + k * nx
                v1 = base + (i + 1) + k * nx
                v2 = base + (i + 1) + (k + 1) * nx
                v3 = base + i + (k + 1) * nx
                triangles.append([v0, v1, v2])
                triangles.append([v0, v2, v3])

    add_layer_triangles(0)              # bottom layer
    add_layer_triangles(nx * nz)        # top layer

    return positions, triangles


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

    positions, triangles = generate_two_layer_slab(nx=16, nz=16, sx=4.0, sz=4.0, layer_gap=0.05)

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
        # FBP + vertex-triangle wiring
        useFeatureBasedProximity=USE_FEATURE_BASED_PROXIMITY,
        useVertexTriangleProximity=USE_VERTEX_TRIANGLE_PROXIMITY,
        proximityComputeBarycentrics=PROXIMITY_COMPUTE_BARYCENTRICS,
        proximityReadContactCounter=PROXIMITY_READ_CONTACT_COUNTER,
        proximityKeepContactsOnDevice=PROXIMITY_KEEP_CONTACTS_ON_DEVICE,
        proximityMaxContacts=PROXIMITY_MAX_CONTACTS,
        minGPUPairCount=1,
        # Grid sized for a 4x4 unit slab centered at origin with thin Y
        gridMinX=-2.5,
        gridMinY=-0.2,
        gridMinZ=-2.5,
        gridMaxX=2.5,
        gridMaxY=0.3,
        gridMaxZ=2.5,
        gridResolutionX=32,
        gridResolutionY=4,
        gridResolutionZ=32,
        # contactDistance=0.06 > layer_gap=0.05 so every top vertex's nearest
        # bottom triangle is in proximity. Vertices of the same layer are not
        # near triangles of the OTHER layer's own corner exclusion (different
        # vertex ids) so they fire.
        contactDistance=0.06,
        maxTissueTrianglesPerCell=64,
        maxToolTrianglesPerCell=64,
        maxCandidatePairs=2000000,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.1, contactDistance=0.06, angleCone=0.0)
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuVertexTriangleSelfCollisionTiming',
        label='gpu_self_collision_vertex_triangle_smoke' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='vertex-triangle-self-collision-detection-only',
        tissueSolver='none',
        tissueForceField='none',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='none (direct GPU collision surfaces)',
        visualMapping='none',
        notes='Two-layer flat slab with selfCollision=True. Each top-layer vertex is 0.05 above a bottom-layer triangle; contactDistance=0.06 so every top vertex emits one v-t contact. Validates GpuCollisionNarrowPhase wiring of computeFeatureBasedVertexTriangleContacts.',
        simVertexCount=0,
        simElementCount=0,
        collisionVertexCount=len(positions),
        collisionElementCount=len(triangles),
        visualVertexCount=0,
        visualElementCount=0,
        warmupSteps=WARMUP_STEPS,
        flushInterval=20,
        logInterval=20,
        printProgress=True,
    )

    tissue = root.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=positions)
    tissue.addObject('MeshTopology', name='topo', triangles=triangles)
    tissue.addObject('TriangleCollisionModel', selfCollision=True)

    return root
