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
from gpu_phase2_tissue_common import add_phase2_gpu_tissue
from gpu_phase3_mesh_common import (
    PHASE3_EXPERIMENTAL_NOTES,
    add_gpu_collision_subset_surface,
    add_gpu_visual_surface,
    build_surface_mesh,
    build_surface_subset_mesh,
)


BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)
PIPELINE_LABEL = os.environ.get('SOFA_PIPELINE_LABEL', 'gpu_dense_collision_phase3_dual_mesh_experimental')
PIPELINE_PHASE = os.environ.get('SOFA_PIPELINE_PHASE', 'phase3-experimental')
PIPELINE_NOTES = os.environ.get('SOFA_PIPELINE_NOTES', PHASE3_EXPERIMENTAL_NOTES)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


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
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=1000, tolerance=1e-6)
    root.addObject('CollisionPipeline')
    root.addObject(
        'GpuCollisionBroadPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        logBoxesOnce=_env_bool('SOFA_GPU_BROAD_LOG_BOXES_ONCE', False),
    )
    root.addObject(
        'GpuCollisionNarrowPhase',
        enableGPU=True,
        allowCPUFallback=True,
        logBackendStatus=True,
        minGPUPairCount=_env_int('SOFA_GPU_NARROW_MIN_PAIR_COUNT', 8),
    )
    root.addObject('LocalMinDistance', alarmDistance=0.12, contactDistance=0.04, angleCone=0.05)
    root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')

    positions, tetrahedra, fixed_indices, surface_tris = generate_tissue_mesh()
    visual_surface = build_surface_mesh(positions, surface_tris)
    collision_surface = build_surface_subset_mesh(positions, surface_tris, stride=3)

    root.addObject(
        'GpuPipelineBenchmarkController',
        name='GpuDenseCollisionPhase3Timing',
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
        total_mass=1.0,
        young_modulus=800,
        poisson_ratio=0.45,
    )
    add_gpu_collision_subset_surface(tissue, collision_surface)
    add_gpu_visual_surface(tissue, visual_surface, '0.88 0.34 0.34 0.8')

    blade_verts, blade_tris = create_blade_geometry()
    static_overlap_mode = _env_bool('SOFA_BLADE_STATIC_OVERLAP', False)
    settle_steps = _env_int('SOFA_BLADE_SETTLE_STEPS', 80)
    descend_steps = _env_int('SOFA_BLADE_DESCEND_STEPS', 260)
    sweep_steps = _env_int('SOFA_BLADE_SWEEP_STEPS', 420)
    lift_steps = _env_int('SOFA_BLADE_LIFT_STEPS', 140)
    total_down = _env_float('SOFA_BLADE_TOTAL_DOWN', 3.0)
    total_sweep_x = _env_float('SOFA_BLADE_TOTAL_SWEEP_X', 0.9)
    total_lift = _env_float('SOFA_BLADE_TOTAL_LIFT', 2.5)

    start_positions, sweep_signs = build_blade_grid(
        rows=4,
        cols=4,
        spacing_x=3.1,
        spacing_z=3.1,
        start_y=_env_float('SOFA_BLADE_START_Y', 3.6),
    )

    for blade_index, start_pos in enumerate(start_positions):
        blade = root.addChild(f'Blade{blade_index:02d}')
        blade.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.0, rayleighMass=0.1)
        blade.addObject('CGLinearSolver', name='LinearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)
        blade.addObject('MechanicalObject', name='rigidDof', template='Rigid3d', position=[start_pos],
                        showObject=True, showObjectScale=0.12)
        blade.addObject('UniformMass', name='mass', totalMass=0.25)
        blade.addObject('UncoupledConstraintCorrection', defaultCompliance=1.0e-6)
        if not static_overlap_mode:
            blade.addObject(
                'GpuKinematicRigidController',
                name='RigidPathController',
                startPosition=start_pos,
                sweepSign=sweep_signs[blade_index],
                settleSteps=settle_steps,
                descendSteps=descend_steps,
                sweepSteps=sweep_steps,
                liftSteps=lift_steps,
                totalDown=total_down,
                totalSweepX=total_sweep_x,
                totalLift=total_lift,
            )

        blade_col = blade.addChild('Collision')
        blade_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=blade_verts)
        blade_col.addObject('MeshTopology', name='colTopo', triangles=blade_tris)
        blade_col.addObject('TriangleCollisionModel', selfCollision=False)
        blade_col.addObject('RigidMapping', input='@../rigidDof', output='@colDofs')

        blade_vis = blade.addChild('Visual')
        blade_vis.addObject('OglModel', name='visualModel', position=blade_verts, triangles=blade_tris,
                            color='0.72 0.76 0.88 1.0')
        blade_vis.addObject('RigidMapping', input='@../rigidDof', output='@visualModel')

    return root
