import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import default_benchmark_log_dir
from gpu_phase2_tissue_common import add_phase2_gpu_tissue
from gpu_phase3_mesh_common import (
    PHASE3_EXPERIMENTAL_NOTES,
    add_gpu_collision_subset_surface,
    add_gpu_visual_surface,
    build_surface_mesh,
    build_surface_subset_mesh,
)
from test_gpu_tissue_deterministic_fixed import (
    DeterministicBladeForceController,
    USE_KINEMATIC_PATH,
    generate_tissue_mesh,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
PIPELINE_LABEL = os.environ.get('SOFA_PIPELINE_LABEL', 'gpu_tissue_phase3_dual_mesh_experimental')
PIPELINE_PHASE = os.environ.get('SOFA_PIPELINE_PHASE', 'phase3-experimental')
PIPELINE_NOTES = os.environ.get('SOFA_PIPELINE_NOTES', PHASE3_EXPERIMENTAL_NOTES)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.005

    root.addObject('RequiredPlugin', pluginName=[
        'SofaCUDA',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Mapping',
        'Sofa.Component.Topology.Utility',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
        'MultiThreading',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=1000, tolerance=1e-6)
    root.addObject('CollisionPipeline')
    root.addObject('ParallelBruteForceBroadPhase')
    root.addObject('ParallelBVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=0.10, contactDistance=0.03, angleCone=0.05)
    root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')

    positions, tetrahedra, fixed_indices, surface_tris = generate_tissue_mesh()
    visual_surface = build_surface_mesh(positions, surface_tris)
    collision_surface = build_surface_subset_mesh(positions, surface_tris, stride=2)

    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuTissuePhase3Timing',
        label=PIPELINE_LABEL,
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase=PIPELINE_PHASE,
        tissueSolver='CGLinearSolver(iterative, graph-scattered)',
        tissueForceField='TetrahedronFEMForceField(template=CudaVec3f, computeGlobalMatrix=false)',
        collisionStateTemplate='CudaVec3f',
        collisionMapping='SubsetMapping(CudaVec3f->CudaVec3f, reduced surface)',
        visualMapping='SubsetMapping(CudaVec3f->CudaVec3f full surface) + IdentityMapping to OglModel',
        notes=PIPELINE_NOTES,
        simVertexCount=len(positions),
        simElementCount=len(tetrahedra),
        collisionVertexCount=len(collision_surface['positions']),
        collisionElementCount=len(collision_surface['triangles']),
        visualVertexCount=len(visual_surface['positions']),
        visualElementCount=len(visual_surface['triangles']),
        warmupSteps=_env_int('SOFA_PIPELINE_WARMUP_STEPS', 50),
        flushInterval=50,
        logInterval=200,
        printProgress=True,
    )

    tissue = root.addChild('Tissue')
    add_phase2_gpu_tissue(
        tissue,
        positions,
        tetrahedra,
        fixed_indices,
        total_mass=0.5,
        young_modulus=500,
        poisson_ratio=0.45,
    )
    add_gpu_collision_subset_surface(tissue, collision_surface)
    add_gpu_visual_surface(tissue, visual_surface, '0.9 0.3 0.3 0.8')

    blade = root.addChild('Blade')
    blade.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.0, rayleighMass=0.1)
    blade.addObject('CGLinearSolver', name='LinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)
    blade.addObject('MechanicalObject', name='rigidDof', template='Rigid3d',
                    position=[[0.0, 3.0, 0.0, 0, 0, 0, 1]], showObject=True, showObjectScale=0.3)
    blade.addObject('UniformMass', name='mass', totalMass=2.0)
    blade_force = blade.addObject('ConstantForceField', name='externalForce', forces=[[0, 0, 0, 0, 0, 0]])
    blade.addObject('UncoupledConstraintCorrection', defaultCompliance=1.0e-6)

    if USE_KINEMATIC_PATH:
        blade.addObject(
            'GpuKinematicRigidController',
            name='BladePathController',
            startPosition=[0.0, 3.0, 0.0, 0, 0, 0, 1],
            settleSteps=100,
            descendSteps=350,
            sweepSteps=700,
            liftSteps=200,
            totalDown=2.6,
            totalSweepX=2.5,
            totalLift=2.0,
            sweepSign=1.0,
        )

    blade_col = blade.addChild('Collision')
    blade_verts = [
        [-1.0, -0.25, -0.04], [1.0, -0.25, -0.04],
        [1.0, 0.25, -0.04], [-1.0, 0.25, -0.04],
        [-1.0, -0.25, 0.04], [1.0, -0.25, 0.04],
        [1.0, 0.25, 0.04], [-1.0, 0.25, 0.04],
    ]
    blade_tris = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    blade_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=blade_verts)
    blade_col.addObject('MeshTopology', name='colTopo', triangles=blade_tris)
    blade_col.addObject('TriangleCollisionModel', selfCollision=False)
    blade_col.addObject('RigidMapping', input='@../rigidDof', output='@colDofs')

    blade_vis = blade.addChild('Visual')
    blade_vis.addObject('OglModel', name='visualModel', position=blade_verts, triangles=blade_tris,
                        color='0.75 0.75 0.85 1.0')
    blade_vis.addObject('RigidMapping', input='@../rigidDof', output='@visualModel')

    if not USE_KINEMATIC_PATH:
        root.addObject(DeterministicBladeForceController(
            name='BladeForceSchedule',
            force_field=blade_force,
            settle_steps=100,
            descend_steps=350,
            sweep_steps=700,
            lift_steps=200,
            down_force=25.0,
            sweep_force=10.0,
            lift_force=18.0,
        ))

    return root
