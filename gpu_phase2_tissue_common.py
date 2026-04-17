def add_phase2_gpu_tissue(
    tissue,
    positions,
    tetrahedra,
    fixed_indices,
    *,
    total_mass,
    young_modulus,
    poisson_ratio,
    rayleigh_stiffness=0.1,
    rayleigh_mass=0.1,
    cg_iterations=80,
    cg_tolerance=1.0e-10,
    cg_threshold=1.0e-10,
    constraint_compliance=1.0e-6,
):
    """
    Experimental Phase 2 tissue stack.

    This does not replace SOFA's FEM implementation yet. It moves the scene away
    from the CPU direct solver and onto an iterative configuration that is
    compatible with `computeGlobalMatrix=False`, which is a better stepping stone
    toward a matrix-free GPU solve path.
    """

    tissue.addObject(
        'EulerImplicitSolver',
        name='OdeSolver',
        rayleighStiffness=rayleigh_stiffness,
        rayleighMass=rayleigh_mass,
    )
    tissue.addObject(
        'CGLinearSolver',
        name='LinearSolver',
        iterations=cg_iterations,
        tolerance=cg_tolerance,
        threshold=cg_threshold,
    )
    tissue.addObject('TetrahedronSetTopologyContainer', name='topo', tetrahedra=tetrahedra)
    tissue.addObject('TetrahedronSetTopologyModifier')
    tissue.addObject('TetrahedronSetGeometryAlgorithms', template='CudaVec3f')
    tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=positions)
    tissue.addObject('UniformMass', name='mass', totalMass=total_mass)
    tissue.addObject(
        'TetrahedronFEMForceField',
        name='FEM',
        template='CudaVec3f',
        youngModulus=young_modulus,
        poissonRatio=poisson_ratio,
        method='large',
        computeGlobalMatrix=False,
    )
    tissue.addObject('FixedProjectiveConstraint', name='fixEdges', indices=fixed_indices)
    tissue.addObject('UncoupledConstraintCorrection', defaultCompliance=constraint_compliance)


def add_cpu_collision_surface(node, positions, surface_tris):
    collision = node.addChild('Collision')
    collision.addObject('MechanicalObject', name='colDofs', template='Vec3d', position=positions)
    collision.addObject('MeshTopology', name='colTopo', triangles=surface_tris)
    collision.addObject('IdentityMapping', input='@../dofs', output='@colDofs')
    collision.addObject('TriangleCollisionModel', selfCollision=False)
    collision.addObject('LineCollisionModel')
    collision.addObject('PointCollisionModel')
    return collision


def add_tissue_visual(node, positions, surface_tris, color):
    visual = node.addChild('Visual')
    visual.addObject('OglModel', name='visualModel', position=positions, triangles=surface_tris, color=color)
    visual.addObject('IdentityMapping', input='@../dofs', output='@visualModel')
    return visual


PHASE2_EXPERIMENTAL_NOTES = (
    "Experimental Phase 2 tissue stack: CudaVec3f tissue state + "
    "TetrahedronFEMForceField(computeGlobalMatrix=false) + CGLinearSolver + "
    "UncoupledConstraintCorrection(defaultCompliance set). Collision mirrors and final response still "
    "remain on the CPU path."
)
