import os


def generate_tissue_mesh(nx=31, ny=5, nz=31,
                         sx=14.0, sy=0.9, sz=14.0,
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


def create_blade_geometry(length=1.1, height=0.28, thickness=0.08):
    hx = 0.5 * length
    hy = 0.5 * height
    hz = 0.5 * thickness
    blade_verts = [
        [-hx, -hy, -hz], [hx, -hy, -hz],
        [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz],
        [hx, hy, hz], [-hx, hy, hz],
    ]
    blade_tris = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    return blade_verts, blade_tris


def build_blade_grid(rows=4, cols=4, spacing_x=3.1, spacing_z=3.1, start_y=3.6):
    start_positions = []
    sweep_signs = []
    x0 = -0.5 * spacing_x * (cols - 1)
    z0 = -0.5 * spacing_z * (rows - 1)

    for row in range(rows):
        for col in range(cols):
            x = x0 + col * spacing_x
            z = z0 + row * spacing_z
            start_positions.append([x, start_y, z, 0.0, 0.0, 0.0, 1.0])
            sweep_signs.append(1.0 if (row + col) % 2 == 0 else -1.0)

    return start_positions, sweep_signs


def default_benchmark_log_dir(base_dir):
    return os.environ.get("SOFA_BENCHMARK_LOG_DIR", os.path.join(base_dir, "benchmark_logs"))
