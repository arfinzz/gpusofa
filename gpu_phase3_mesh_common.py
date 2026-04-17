def _compress_indexed_mesh(vertex_indices, triangles, positions):
    index_map = {old_index: new_index for new_index, old_index in enumerate(vertex_indices)}
    compressed_positions = [positions[index] for index in vertex_indices]
    compressed_triangles = [
        [index_map[a], index_map[b], index_map[c]]
        for a, b, c in triangles
        if a in index_map and b in index_map and c in index_map
    ]
    return compressed_positions, compressed_triangles


def build_surface_mesh(positions, surface_tris):
    surface_vertex_indices = sorted({vertex for triangle in surface_tris for vertex in triangle})
    surface_positions, compressed_triangles = _compress_indexed_mesh(surface_vertex_indices, surface_tris, positions)
    return {
        'indices': surface_vertex_indices,
        'positions': surface_positions,
        'triangles': compressed_triangles,
    }


def build_surface_subset_mesh(positions, surface_tris, stride=2):
    surface_vertex_indices = sorted({vertex for triangle in surface_tris for vertex in triangle})
    stride = max(1, int(stride))

    while stride >= 1:
        selected = surface_vertex_indices[::stride]
        if surface_vertex_indices[-1] not in selected:
            selected.append(surface_vertex_indices[-1])

        selected_set = set(selected)
        filtered_tris = [
            triangle for triangle in surface_tris
            if triangle[0] in selected_set and triangle[1] in selected_set and triangle[2] in selected_set
        ]

        if filtered_tris or stride == 1:
            selected = sorted(set(selected))
            subset_positions, subset_triangles = _compress_indexed_mesh(selected, filtered_tris, positions)
            return {
                'indices': selected,
                'positions': subset_positions,
                'triangles': subset_triangles,
                'stride': stride,
            }

        stride -= 1


def add_gpu_visual_surface(node, full_surface_mesh, color):
    visual = node.addChild('Visual')
    visual.addObject('MechanicalObject', name='visualDofs', template='CudaVec3f', position=full_surface_mesh['positions'])
    visual.addObject('MeshTopology', name='visualTopo', triangles=full_surface_mesh['triangles'])
    visual.addObject('SubsetMapping', input='@../dofs', output='@visualDofs', indices=full_surface_mesh['indices'])
    render = visual.addChild('Render')
    render.addObject('OglModel', name='visualModel', position=full_surface_mesh['positions'],
                     triangles=full_surface_mesh['triangles'], color=color)
    render.addObject('IdentityMapping', input='@../visualDofs', output='@visualModel')
    return visual


def add_gpu_collision_subset_surface(node, collision_subset_mesh, *, add_line_model=False, add_point_model=False):
    collision = node.addChild('Collision')
    collision.addObject('MechanicalObject', name='colDofs', template='CudaVec3f', position=collision_subset_mesh['positions'])
    collision.addObject('MeshTopology', name='colTopo', triangles=collision_subset_mesh['triangles'])
    collision.addObject('SubsetMapping', input='@../dofs', output='@colDofs', indices=collision_subset_mesh['indices'])
    collision.addObject('TriangleCollisionModel', selfCollision=False)
    if add_line_model:
        collision.addObject('LineCollisionModel')
    if add_point_model:
        collision.addObject('PointCollisionModel')
    return collision


PHASE3_EXPERIMENTAL_NOTES = (
    "Phase 3 experimental dual-mesh stack: simulation stays on CudaVec3f volume DOFs, "
    "visual mesh is a CUDA surface subset mapped from the volume, and collision uses a "
    "reduced CUDA triangle subset mesh via SubsetMapping. Final response is still CPU-side."
)

PHASE45_EXPERIMENTAL_NOTES = (
    "Phase 4/5 experimental compact candidate pipeline: broad phase uses compact GPU pair "
    "generation instead of dense flag readback, narrow phase emits explicit GPU leaf-box "
    "contact candidates, and the dual-mesh CUDA collision surface remains GPU-resident. "
    "Final exact contact resolution still falls back to the CPU response path."
)
