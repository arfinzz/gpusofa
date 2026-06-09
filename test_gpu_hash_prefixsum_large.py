"""Hash + prefix-sum broad-cull benchmark — large tissue + large tool.

EXPERIMENTAL (experiment/hash-prefixsum-broadphase branch).

Geometry: a large tissue surface (~12,800 triangles) and a large subdivided
blade tool (~1,500 triangles) sitting in the tissue plane so many cells are
mixed. This is the regime the spatial-hash + prefix-sum broad cull targets:
both sides large, so the tool/tissue asymmetry that Phase 15 exploits is weak.

Toggle the broad cull with SOFA_USE_HASH_PREFIXSUM_GENERATION:
  0 -> dense grid + tool-active-cell generation (the current default)
  1 -> spatial hash + prefix-sum work expansion (this experiment)
Both feed the SAME FBP narrow kernel, so contact counts MUST be identical.
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

TISSUE_NX = int(os.environ.get("SOFA_HASH_TISSUE_NX", "81"))
TISSUE_NZ = int(os.environ.get("SOFA_HASH_TISSUE_NZ", "81"))
# Subdivided blade sized to ~1,500 triangles (4*sx*sy + 4*sx*sz + 4*sy*sz).
BLADE_SEGMENTS_X = int(os.environ.get("SOFA_HASH_BLADE_SEGMENTS_X", "30"))
BLADE_SEGMENTS_Y = int(os.environ.get("SOFA_HASH_BLADE_SEGMENTS_Y", "8"))
BLADE_SEGMENTS_Z = int(os.environ.get("SOFA_HASH_BLADE_SEGMENTS_Z", "4"))
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))

DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)
USE_HASH_PREFIXSUM = env_flag("SOFA_USE_HASH_PREFIXSUM_GENERATION", False)
HASH_TABLE_SIZE = int(os.environ.get("SOFA_HASH_TABLE_SIZE", "0"))
USE_TOOL_ACTIVE_CELL_GENERATION = env_flag("SOFA_USE_TOOL_ACTIVE_CELL_GENERATION", True)
PROXIMITY_READ_CONTACT_COUNTER = env_flag("SOFA_PROXIMITY_READ_CONTACT_COUNTER", True)


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
    # Large subdivided blade lying in the tissue plane so it crosses many cells.
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
        hashTableSize=HASH_TABLE_SIZE,
        proximityComputeBarycentrics=True,
        proximityKeepContactsOnDevice=True,
        proximityReadContactCounter=PROXIMITY_READ_CONTACT_COUNTER,
        proximityMaxContacts=4000000,
        minGPUPairCount=1,
        gridMinX=-4.5, gridMinY=-0.5, gridMinZ=-4.5,
        gridMaxX=4.5, gridMaxY=0.5, gridMaxZ=4.5,
        gridResolutionX=64, gridResolutionY=8, gridResolutionZ=64,
        contactDistance=0.03,
        maxTissueTrianglesPerCell=128,
        maxToolTrianglesPerCell=128,
        maxCandidatePairs=8000000,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuHashPrefixSumTiming',
        label='gpu_hash_prefixsum_large' + BENCHMARK_LABEL_SUFFIX,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='hash-prefixsum-large-tissue-large-tool',
        collisionStateTemplate='CudaVec3f',
        notes='Large tissue + large tool (~1500 tris). A/B the dense-grid broad cull vs the spatial-hash + prefix-sum broad cull via SOFA_USE_HASH_PREFIXSUM_GENERATION. FBP narrow kernel is identical, so contact counts must match.',
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
