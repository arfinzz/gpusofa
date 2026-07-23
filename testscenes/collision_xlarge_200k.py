"""Extra-large collision benchmark — ~200k triangles, all six broad-cull ways.

Same construction as hash_prefixsum_large.py (static tissue surface grid +
subdivided blade lying in the tissue plane, no physics — the GPU collision
pipeline is what gets stressed), scaled to ~200,000 triangles:
315 x 315 quads x 2 = 198,450 tissue triangles + ~1,568 blade triangles.

Select the broad cull with the same envs as the other scenes:
  SOFA_USE_HASH_PREFIXSUM_GENERATION=1   -> way 3 (optimised hash)
  SOFA_USE_SIMPLE_HASH_GENERATION=1      -> way 4 (simple direct-bucket hash)
  SOFA_USE_SORTED_GRID_GENERATION=1      -> way 5 (sorted grid)
  SOFA_USE_BIGCELL_FUSED_GENERATION=1    -> way 6 (big-cell fused gen+narrow)
  (none)                                 -> dense grid (Phase 15 default)
All ways feed the same narrow-phase math, so contact counts MUST be identical.
"""

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

TISSUE_NX = int(os.environ.get("SOFA_XLARGE_TISSUE_NX", "316"))
TISSUE_NZ = int(os.environ.get("SOFA_XLARGE_TISSUE_NZ", "316"))
BLADE_SEGMENTS_X = int(os.environ.get("SOFA_XLARGE_BLADE_SEGMENTS_X", "30"))
BLADE_SEGMENTS_Y = int(os.environ.get("SOFA_XLARGE_BLADE_SEGMENTS_Y", "8"))
BLADE_SEGMENTS_Z = int(os.environ.get("SOFA_XLARGE_BLADE_SEGMENTS_Z", "4"))
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))

DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)
USE_HASH_PREFIXSUM = env_flag("SOFA_USE_HASH_PREFIXSUM_GENERATION", False)
USE_SIMPLE_HASH = env_flag("SOFA_USE_SIMPLE_HASH_GENERATION", False)
USE_SORTED_GRID = env_flag("SOFA_USE_SORTED_GRID_GENERATION", False)
SORTED_GRID_CUB_SORT = env_flag("SOFA_SORTED_GRID_CUB_SORT", False)
SORTED_GRID_PAIRHASH_DEDUP = env_flag("SOFA_SORTED_GRID_PAIRHASH_DEDUP", False)
USE_BIGCELL_FUSED = env_flag("SOFA_USE_BIGCELL_FUSED_GENERATION", False)
BIGCELL_FACTOR = int(os.environ.get("SOFA_BIGCELL_FACTOR", "2"))
BIGCELL_TOOL_TILE = int(os.environ.get("SOFA_BIGCELL_TOOL_TILE", "256"))
BIGCELL_HASH_BUILD = env_flag("SOFA_BIGCELL_HASH_BUILD", False)
BIGCELL_HASH_SLOTS = int(os.environ.get("SOFA_BIGCELL_HASH_SLOTS", "1024"))
BIGCELL_SHARED_BUILD = int(os.environ.get("SOFA_BIGCELL_SHARED_BUILD", "1"))
BIGCELL_PROFILE_INTERNALS = env_flag("SOFA_BIGCELL_PROFILE_INTERNALS", False)
HASH_TABLE_SIZE = int(os.environ.get("SOFA_HASH_TABLE_SIZE", "0"))
USE_TOOL_ACTIVE_CELL_GENERATION = env_flag("SOFA_USE_TOOL_ACTIVE_CELL_GENERATION", True)
PROXIMITY_READ_CONTACT_COUNTER = env_flag("SOFA_PROXIMITY_READ_CONTACT_COUNTER", True)

# At ~200k triangles the per-cell buckets and pair buffers need more headroom
# than the 14k scene; the grid itself is finer so per-cell occupancy stays sane
# (triangle edge ~0.025 at NX=316 vs ~0.10 at NX=81).
GRID_RESOLUTION_X = int(os.environ.get("SOFA_XLARGE_GRID_RESOLUTION_X", "128"))
GRID_RESOLUTION_Y = int(os.environ.get("SOFA_XLARGE_GRID_RESOLUTION_Y", "8"))
GRID_RESOLUTION_Z = int(os.environ.get("SOFA_XLARGE_GRID_RESOLUTION_Z", "128"))
MAX_TRIANGLES_PER_CELL = int(os.environ.get("SOFA_XLARGE_MAX_TRIANGLES_PER_CELL", "256"))


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
        nx=TISSUE_NX, nz=TISSUE_NZ, sx=8.0, sz=8.0, y=0.0)
    blade_vertices, blade_triangles = create_subdivided_blade_geometry(
        length=5.5, height=0.4, thickness=0.16,
        segments_x=BLADE_SEGMENTS_X, segments_y=BLADE_SEGMENTS_Y, segments_z=BLADE_SEGMENTS_Z)
    blade_vertices = translate_vertices(blade_vertices, [0.0, 0.0, 0.0])

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
        useGpuHashDedupe=True,
        copyContactsToHost=False,
        detailedProfiling=DETAILED_PROFILING,
        useFeatureBasedProximity=True,
        useToolActiveCellGeneration=USE_TOOL_ACTIVE_CELL_GENERATION,
        useHashPrefixSumGeneration=USE_HASH_PREFIXSUM,
        useSimpleHashGeneration=USE_SIMPLE_HASH,
        useSortedGridGeneration=USE_SORTED_GRID,
        sortedGridUseCubSort=SORTED_GRID_CUB_SORT,
        sortedGridUsePairHashDedup=SORTED_GRID_PAIRHASH_DEDUP,
        useBigCellFusedGeneration=USE_BIGCELL_FUSED,
        bigCellFactor=BIGCELL_FACTOR,
        bigCellToolTile=BIGCELL_TOOL_TILE,
        bigCellUseHashBuild=BIGCELL_HASH_BUILD,
        bigCellHashSlots=BIGCELL_HASH_SLOTS,
        bigCellSharedBuild=BIGCELL_SHARED_BUILD,
        bigCellProfileInternals=BIGCELL_PROFILE_INTERNALS,
        hashTableSize=HASH_TABLE_SIZE,
        proximityComputeBarycentrics=True,
        proximityKeepContactsOnDevice=True,
        proximityReadContactCounter=PROXIMITY_READ_CONTACT_COUNTER,
        proximityMaxContacts=4000000,
        minGPUPairCount=1,
        gridMinX=-4.5, gridMinY=-0.5, gridMinZ=-4.5,
        gridMaxX=4.5, gridMaxY=0.5, gridMaxZ=4.5,
        gridResolutionX=GRID_RESOLUTION_X,
        gridResolutionY=GRID_RESOLUTION_Y,
        gridResolutionZ=GRID_RESOLUTION_Z,
        contactDistance=0.03,
        maxTissueTrianglesPerCell=MAX_TRIANGLES_PER_CELL,
        maxToolTrianglesPerCell=MAX_TRIANGLES_PER_CELL,
        maxCandidatePairs=8000000,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuXLargeTiming',
        label='gpu_collision_xlarge_200k' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='xlarge-200k-tissue-blade',
        collisionStateTemplate='CudaVec3f',
        notes='Extra-large static benchmark: ~198k tissue tris + ~1.5k blade tris. Same env toggles as hash_prefixsum_large.py; all six ways must produce identical contact counts.',
        collisionVertexCount=len(tissue_positions) + len(blade_vertices),
        collisionElementCount=len(tissue_triangles) + len(blade_triangles),
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
