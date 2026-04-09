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
from deterministic_blade_controller import (
    BenchmarkTimingController,
    MultiKinematicBladePathController,
)

BENCHMARK_LOG_DIR = default_benchmark_log_dir(current_dir)


def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, -9.81, 0.0]
    root.dt = 0.005

    root.addObject('RequiredPlugin', pluginName=[
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Mapping',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.LinearSolver.Direct',
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
        'MultiThreading',
    ])

    root.addObject('FreeMotionAnimationLoop', name='FreeMotion')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=1000, tolerance=1e-6)

    root.addObject('CollisionPipeline')
    root.addObject('ParallelBruteForceBroadPhase')
    root.addObject('ParallelBVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=0.12, contactDistance=0.04, angleCone=0.05)
    root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')
    root.addObject(BenchmarkTimingController(
        name='CpuDenseCollisionTiming',
        label='cpu_dense_collision_benchmark',
        output_dir=BENCHMARK_LOG_DIR,
        warmup_steps=50,
        flush_interval=50,
        log_interval=200,
    ))

    positions, tetrahedra, fixed_indices, surface_tris = generate_tissue_mesh()

    tissue = root.addChild('Tissue')
    tissue.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    tissue.addObject('SparseLDLSolver', name='LinearSolver', template='CompressedRowSparseMatrixd')
    tissue.addObject('TetrahedronSetTopologyContainer', name='topo', tetrahedra=tetrahedra)
    tissue.addObject('TetrahedronSetTopologyModifier')
    tissue.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')
    tissue.addObject('MechanicalObject', name='dofs', template='Vec3d', position=positions)
    tissue.addObject('UniformMass', name='mass', totalMass=1.0)
    tissue.addObject('TetrahedronFEMForceField', name='FEM', template='Vec3d',
                     youngModulus=800, poissonRatio=0.45, method='large')
    tissue.addObject('FixedProjectiveConstraint', name='fixEdges', indices=fixed_indices)
    tissue.addObject('LinearSolverConstraintCorrection', linearSolver='@LinearSolver')

    tissue_col = tissue.addChild('Collision')
    tissue_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=positions)
    tissue_col.addObject('MeshTopology', name='colTopo', triangles=surface_tris)
    tissue_col.addObject('IdentityMapping', input='@../dofs', output='@colDofs')
    tissue_col.addObject('TriangleCollisionModel', selfCollision=False)
    tissue_col.addObject('LineCollisionModel')
    tissue_col.addObject('PointCollisionModel')

    tissue_vis = tissue.addChild('Visual')
    tissue_vis.addObject('OglModel', name='visualModel', position=positions, triangles=surface_tris,
                         color='0.88 0.34 0.34 0.8')
    tissue_vis.addObject('IdentityMapping', input='@../dofs', output='@visualModel')

    blade_verts, blade_tris = create_blade_geometry()
    start_positions, sweep_signs = build_blade_grid(rows=4, cols=4, spacing_x=3.1, spacing_z=3.1, start_y=3.6)

    rigid_dofs_list = []
    for blade_index, start_pos in enumerate(start_positions):
        blade = root.addChild(f'Blade{blade_index:02d}')
        blade.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.0, rayleighMass=0.1)
        blade.addObject('SparseLDLSolver', name='LinearSolver', template='CompressedRowSparseMatrixd')
        rigid_dofs = blade.addObject(
            'MechanicalObject',
            name='rigidDof',
            template='Rigid3d',
            position=[start_pos],
            showObject=True,
            showObjectScale=0.12,
        )
        rigid_dofs_list.append(rigid_dofs)
        blade.addObject('UniformMass', name='mass', totalMass=0.25)
        blade.addObject('LinearSolverConstraintCorrection', linearSolver='@LinearSolver')

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

    root.addObject(MultiKinematicBladePathController(
        name='DenseBladePathController',
        rigid_dofs_list=rigid_dofs_list,
        start_positions=start_positions,
        sweep_signs=sweep_signs,
        settle_steps=80,
        descend_steps=260,
        sweep_steps=420,
        lift_steps=140,
        total_down=3.0,
        total_sweep_x=0.9,
        total_lift=2.5,
    ))

    return root
