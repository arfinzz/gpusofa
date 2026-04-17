import os
import sys

import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from dense_collision_benchmark_common import (
    build_blade_grid,
    create_blade_geometry,
    default_benchmark_log_dir,
    generate_tissue_mesh,
)
from gpu_phase2_tissue_common import (
    PHASE2_EXPERIMENTAL_NOTES,
    add_cpu_collision_surface,
    add_phase2_gpu_tissue,
    add_tissue_visual,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)


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
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=1000, tolerance=1e-6)
    root.addObject('CollisionPipeline')
    root.addObject('GpuCollisionBroadPhase', enableGPU=True, allowCPUFallback=True, logBackendStatus=True)
    root.addObject(
        'GpuCollisionNarrowPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        minGPUPairCount=8,
    )
    root.addObject('LocalMinDistance', alarmDistance=0.12, contactDistance=0.04, angleCone=0.05)
    root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')
    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuDenseCollisionPhase2Timing',
        label='gpu_dense_collision_phase2_experimental',
        outputDir=BENCHMARK_LOG_DIR,
        pipelinePhase='phase2-experimental',
        tissueSolver='CGLinearSolver(iterative, graph-scattered)',
        tissueForceField='TetrahedronFEMForceField(template=CudaVec3f, computeGlobalMatrix=false)',
        notes=PHASE2_EXPERIMENTAL_NOTES,
        warmupSteps=_env_int('SOFA_PIPELINE_WARMUP_STEPS', 50),
        flushInterval=50,
        logInterval=200,
        printProgress=True,
    )

    positions, tetrahedra, fixed_indices, surface_tris = generate_tissue_mesh()

    tissue = root.addChild('Tissue')
    add_phase2_gpu_tissue(
        tissue,
        positions,
        tetrahedra,
        fixed_indices,
        total_mass=1.0,
        young_modulus=800,
        poisson_ratio=0.45,
    )
    add_cpu_collision_surface(tissue, positions, surface_tris)
    add_tissue_visual(tissue, positions, surface_tris, '0.88 0.34 0.34 0.8')

    blade_verts, blade_tris = create_blade_geometry()
    start_positions, sweep_signs = build_blade_grid(rows=4, cols=4, spacing_x=3.1, spacing_z=3.1, start_y=3.6)

    for blade_index, start_pos in enumerate(start_positions):
        blade = root.addChild(f'Blade{blade_index:02d}')
        blade.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.0, rayleighMass=0.1)
        blade.addObject('CGLinearSolver', name='LinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)
        blade.addObject(
            'MechanicalObject',
            name='rigidDof',
            template='Rigid3d',
            position=[start_pos],
            showObject=True,
            showObjectScale=0.12,
        )
        blade.addObject('UniformMass', name='mass', totalMass=0.25)
        blade.addObject('UncoupledConstraintCorrection', defaultCompliance=1.0e-6)
        blade.addObject(
            'GpuKinematicRigidController',
            name='RigidPathController',
            startPosition=start_pos,
            sweepSign=sweep_signs[blade_index],
            settleSteps=80,
            descendSteps=260,
            sweepSteps=420,
            liftSteps=140,
            totalDown=3.0,
            totalSweepX=0.9,
            totalLift=2.5,
        )

        blade_col = blade.addChild('Collision')
        blade_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=blade_verts)
        blade_col.addObject('MeshTopology', name='colTopo', triangles=blade_tris)
        blade_col.addObject('TriangleCollisionModel', selfCollision=False)
        blade_col.addObject('LineCollisionModel')
        blade_col.addObject('PointCollisionModel')
        blade_col.addObject('RigidMapping', input='@../rigidDof', output='@colDofs')

        blade_vis = blade.addChild('Visual')
        blade_vis.addObject('OglModel', name='visualModel', position=blade_verts, triangles=blade_tris,
                            color='0.72 0.76 0.88 1.0')
        blade_vis.addObject('RigidMapping', input='@../rigidDof', output='@visualModel')

    return root
