import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    create_subdivided_blade_geometry,
    default_benchmark_log_dir,
    generate_tissue_surface_grid,
    translate_vertices,
)
from deterministic_blade_controller import BenchmarkTimingController


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
BENCHMARK_LABEL_SUFFIX = os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")

TISSUE_NX = int(os.environ.get("SOFA_LARGE_TISSUE_NX", "181"))
TISSUE_NZ = int(os.environ.get("SOFA_LARGE_TISSUE_NZ", "181"))
BLADE_SEGMENTS_X = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_X", "128"))
BLADE_SEGMENTS_Y = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_Y", "24"))
BLADE_SEGMENTS_Z = int(os.environ.get("SOFA_LARGE_BLADE_SEGMENTS_Z", "4"))
WARMUP_STEPS = int(os.environ.get("SOFA_LARGE_WARMUP_STEPS", "10"))


def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.005

    root.addObject('RequiredPlugin', pluginName=[
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.AnimationLoop',
        'MultiThreading',
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
    root.addObject('ParallelBruteForceBroadPhase')
    root.addObject('ParallelBVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
    root.addObject(BenchmarkTimingController(
        name='CpuLargeTissueBladeTiming',
        label='cpu_large_tissue_blade_benchmark' + BENCHMARK_LABEL_SUFFIX,
        output_dir=BENCHMARK_LOG_DIR,
        warmup_steps=WARMUP_STEPS,
        flush_interval=10,
        log_interval=10,
    ))

    tissue = root.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='Vec3d', position=tissue_positions)
    tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
    tissue.addObject('TriangleCollisionModel', selfCollision=False)

    blade = root.addChild('Blade')
    blade.addObject('MechanicalObject', name='dofs', template='Vec3d', position=blade_vertices)
    blade.addObject('MeshTopology', name='topo', triangles=blade_triangles)
    blade.addObject('TriangleCollisionModel', selfCollision=False)

    return root
