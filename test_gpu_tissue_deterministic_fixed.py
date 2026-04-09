import os
import sys
import Sofa.Core

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from deterministic_blade_controller import (
    BenchmarkTimingController,
    DeterministicBladeForceController,
    KinematicBladePathController,
)

# -----------------------------------------------------------------------------
# Benchmark motion mode:
#   USE_KINEMATIC_PATH = False  -> deterministic force schedule (closer to current scene)
#   USE_KINEMATIC_PATH = True   -> exact position path (most repeatable benchmark)
# -----------------------------------------------------------------------------
USE_KINEMATIC_PATH = True
BENCHMARK_LOG_DIR = os.path.join(current_dir, "benchmark_logs")


def generate_tissue_mesh(nx=21, ny=3, nz=21,
                         sx=10.0, sy=0.3, sz=10.0,
                         border=2):
    positions = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = -sx / 2.0 + i * sx / (nx - 1)
                y = -sy / 2.0 + j * sy / (ny - 1)
                z = -sz / 2.0 + k * sz / (nz - 1)
                positions.append([x, y, z])

    def node(i, j, k):
        return i + j * nx + k * nx * ny

    tetrahedra = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = node(i,     j,     k)
                v1 = node(i + 1, j,     k)
                v2 = node(i + 1, j + 1, k)
                v3 = node(i,     j + 1, k)
                v4 = node(i,     j,     k + 1)
                v5 = node(i + 1, j,     k + 1)
                v6 = node(i + 1, j + 1, k + 1)
                v7 = node(i,     j + 1, k + 1)

                tetrahedra.extend([
                    [v0, v1, v2, v6],
                    [v0, v5, v1, v6],
                    [v0, v2, v3, v6],
                    [v0, v3, v7, v6],
                    [v0, v4, v5, v6],
                    [v0, v7, v4, v6],
                ])

    face_count = {}
    face_verts = {}
    for tet in tetrahedra:
        faces = [
            (tet[0], tet[2], tet[1]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[3], tet[2]),
            (tet[1], tet[2], tet[3]),
        ]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_verts:
                face_verts[key] = list(f)

    surface_tris = [face_verts[k] for k, c in face_count.items() if c == 1]

    fixed_indices = set()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (i < border or i >= nx - border or
                        k < border or k >= nz - border):
                    fixed_indices.add(node(i, j, k))

    return positions, tetrahedra, sorted(fixed_indices), surface_tris


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
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.MechanicalLoad',
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

    # Keep the root contact stack on CPU for the first GPU benchmark.
    root.addObject('CollisionPipeline')
    root.addObject('ParallelBruteForceBroadPhase')
    root.addObject('ParallelBVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=0.10, contactDistance=0.03, angleCone=0.05)
    root.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')
    root.addObject(BenchmarkTimingController(
        name='GpuTissueBenchmarkTiming',
        label='gpu_tissue_deterministic_fixed',
        output_dir=BENCHMARK_LOG_DIR,
        warmup_steps=50,
        flush_interval=50,
        log_interval=200,
    ))

    positions, tetrahedra, fixed_indices, surface_tris = generate_tissue_mesh()

    # -------------------------------------------------------------------------
    # GPU tissue branch
    # -------------------------------------------------------------------------
    tissue = root.addChild('Tissue')
    tissue.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    tissue.addObject('SparseLDLSolver', name='LinearSolver', template='CompressedRowSparseMatrixd')
    
    tissue.addObject('TetrahedronSetTopologyContainer', name='topo', tetrahedra=tetrahedra)
    tissue.addObject('TetrahedronSetTopologyModifier')
    tissue.addObject('TetrahedronSetGeometryAlgorithms', template='CudaVec3f')

    # Positions are still passed in from Python, but the mechanical state lives on GPU.
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=positions)
    tissue.addObject('UniformMass', name='mass', totalMass=0.5)
    tissue.addObject('TetrahedronFEMForceField', name='FEM', template='CudaVec3f',
                     youngModulus=500, poissonRatio=0.45, method='large',
                     computeGlobalMatrix=False)
    tissue.addObject('FixedProjectiveConstraint', name='fixEdges', indices=fixed_indices)
    tissue.addObject('LinearSolverConstraintCorrection', linearSolver='@LinearSolver')

    # Collision stays CPU-side but is mapped from the GPU tissue state.
    tissue_col = tissue.addChild('Collision')
    tissue_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=positions)
    tissue_col.addObject('MeshTopology', name='colTopo', triangles=surface_tris)
    tissue_col.addObject('IdentityMapping', input='@../dofs', output='@colDofs')
    tissue_col.addObject('TriangleCollisionModel', selfCollision=False)
    tissue_col.addObject('LineCollisionModel')
    tissue_col.addObject('PointCollisionModel')

    tissue_vis = tissue.addChild('Visual')
    tissue_vis.addObject('OglModel', name='visualModel', position=positions, triangles=surface_tris,
                         color='0.9 0.3 0.3 0.8')
    tissue_vis.addObject('IdentityMapping', input='@../dofs', output='@visualModel')

    # -------------------------------------------------------------------------
    # CPU rigid blade branch
    # -------------------------------------------------------------------------
    blade = root.addChild('Blade')
    blade.addObject('EulerImplicitSolver', name='OdeSolver', rayleighStiffness=0.0, rayleighMass=0.1)
    blade.addObject('SparseLDLSolver', name='LinearSolver', template='CompressedRowSparseMatrixd')
    blade.addObject('MechanicalObject', name='rigidDof', template='Rigid3d',
                    position=[[0.0, 3.0, 0.0, 0, 0, 0, 1]],
                    showObject=True, showObjectScale=0.3)
    blade.addObject('UniformMass', name='mass', totalMass=2.0)

    blade_force = blade.addObject('ConstantForceField',
                                  name='externalForce',
                                  forces=[[0, 0, 0, 0, 0, 0]])
    blade.addObject('LinearSolverConstraintCorrection', linearSolver='@LinearSolver')

    blade_col = blade.addChild('Collision')
    blade_verts = [
        [-1.0, -0.25, -0.04], [ 1.0, -0.25, -0.04],
        [ 1.0,  0.25, -0.04], [-1.0,  0.25, -0.04],
        [-1.0, -0.25,  0.04], [ 1.0, -0.25,  0.04],
        [ 1.0,  0.25,  0.04], [-1.0,  0.25,  0.04],
    ]
    blade_tris = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    blade_col.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=blade_verts)
    blade_col.addObject('MeshTopology', name='colTopo', triangles=blade_tris)
    blade_col.addObject('TriangleCollisionModel', selfCollision=False)
    blade_col.addObject('LineCollisionModel')
    blade_col.addObject('PointCollisionModel')
    blade_col.addObject('RigidMapping', input='@../rigidDof', output='@colDofs')

    blade_vis = blade.addChild('Visual')
    blade_vis.addObject('OglModel', name='visualModel', position=blade_verts, triangles=blade_tris,
                        color='0.75 0.75 0.85 1.0')
    blade_vis.addObject('RigidMapping', input='@../rigidDof', output='@visualModel')

    if USE_KINEMATIC_PATH:
        root.addObject(KinematicBladePathController(
            name='BladePathController',
            rigid_dofs=blade.rigidDof,
            start_pos=[0.0, 3.0, 0.0, 0, 0, 0, 1],
            settle_steps=100,
            descend_steps=350,
            sweep_steps=700,
            lift_steps=200,
            total_down=2.6,
            total_sweep_x=2.5,
            total_lift=2.0,
        ))
    else:
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
