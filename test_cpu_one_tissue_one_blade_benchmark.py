import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    create_blade_geometry,
    default_benchmark_log_dir,
    generate_tissue_surface_grid,
    translate_vertices,
)
from deterministic_blade_controller import BenchmarkTimingController


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
BENCHMARK_LABEL_SUFFIX = os.environ.get("SOFA_BENCHMARK_LABEL_SUFFIX", "")


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

    root.addObject('DefaultAnimationLoop')
    root.addObject('CollisionPipeline')
    root.addObject('ParallelBruteForceBroadPhase')
    root.addObject('ParallelBVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
    root.addObject(BenchmarkTimingController(
        name='CpuOneTissueOneBladeTiming',
        label='cpu_one_tissue_one_blade_benchmark' + BENCHMARK_LABEL_SUFFIX,
        output_dir=BENCHMARK_LOG_DIR,
        warmup_steps=10,
        flush_interval=20,
        log_interval=20,
    ))

    tissue_positions, tissue_triangles = generate_tissue_surface_grid()
    blade_vertices, blade_triangles = create_blade_geometry(length=1.8, height=0.6, thickness=0.12)
    blade_vertices = translate_vertices(blade_vertices, [0.0, 0.0, 0.0])

    tissue = root.addChild('Tissue')
    tissue.addObject('MechanicalObject', name='dofs', template='Vec3d', position=tissue_positions)
    tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
    tissue.addObject('TriangleCollisionModel', selfCollision=False)

    blade = root.addChild('Blade')
    blade.addObject('MechanicalObject', name='dofs', template='Vec3d', position=blade_vertices)
    blade.addObject('MeshTopology', name='topo', triangles=blade_triangles)
    blade.addObject('TriangleCollisionModel', selfCollision=False)

    return root
