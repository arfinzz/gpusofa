# 02 — The Scene File, Line by Line

We'll read `test_gpu_one_tissue_one_blade_dense_grid_benchmark.py` from top to
bottom. This is the file SOFA loads to build the benchmark. Open it alongside
this tutorial.

A SOFA Python scene must define one function: `createScene(root)`. SOFA calls
it once at startup, handing you the empty `root` node. You hang everything off
`root`, return it, and SOFA takes over.

---

## 2.1 Reading the environment flags (top of file)

The file starts by reading a pile of environment variables:

```python
DETAILED_PROFILING = env_flag("SOFA_GPU_DETAILED_PROFILING", False)
COPY_CONTACTS_TO_HOST = env_flag("SOFA_COPY_CONTACTS_TO_HOST", False)
USE_INDEXED_DENSE_GRID_INPUT = env_flag("SOFA_USE_INDEXED_DENSE_GRID_INPUT", True)
USE_DIRECT_DEVICE_POSITIONS = env_flag("SOFA_USE_DIRECT_DEVICE_POSITIONS", True)
USE_FEATURE_BASED_PROXIMITY = env_flag("SOFA_USE_FEATURE_BASED_PROXIMITY", False)
# ... and more
```

`env_flag(name, default)` (from `dense_collision_benchmark_common.py`) reads an
environment variable and turns `"1"`, `"true"`, `"yes"`, `"on"` into `True`.
If the variable isn't set, it uses the default.

**Why do this?** So you can change the scene's behavior from the command line
without editing the file. The launcher scripts in `scripts/` set these
variables. For example, `run_fbp_smoke_test_wsl.sh` sets
`SOFA_USE_FEATURE_BASED_PROXIMITY=1` to turn on the feature-based proximity
path. The same scene file runs all the different modes.

The important defaults for our walkthrough:

- `COPY_CONTACTS_TO_HOST = False` → detection-only (don't copy results back).
- `USE_INDEXED_DENSE_GRID_INPUT = True` → use the cached-indices fast path.
- `USE_DIRECT_DEVICE_POSITIONS = True` → read GPU positions directly (zero-copy).
- `USE_FEATURE_BASED_PROXIMITY = False` → by default use the legacy
  exact-contact path; the smoke script flips this on.

---

## 2.2 Root configuration

```python
def createScene(root):
    root.name = "RootNode"
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 0.005
```

- `gravity = [0, 0, 0]` — **no gravity**. Nothing falls. This is part of making
  the benchmark static and deterministic.
- `dt = 0.005` — the timestep. Each frame advances simulated time by 5
  milliseconds. (It doesn't matter much here since nothing moves, but SOFA
  needs a value.)

---

## 2.3 Loading plugins

```python
root.addObject('RequiredPlugin', pluginName=[
    'SofaCUDA',
    'Sofa.Component.StateContainer',
    'Sofa.Component.Topology.Container.Constant',
    'Sofa.Component.Collision.Detection.Algorithm',
    'Sofa.Component.Collision.Detection.Intersection',
    'Sofa.Component.Collision.Geometry',
    'Sofa.Component.AnimationLoop',
])
```

`RequiredPlugin` tells SOFA to load shared libraries (`.so` files on Linux).
Loading a library makes its components available by name.

- `SofaCUDA` — gives us `CudaVec3f` and `CudaTriangleCollisionModel`. **This is
  the one that makes GPU memory possible.**
- The `Sofa.Component.*` entries are standard SOFA building blocks (state
  containers, topology, collision algorithms, animation loop).

Note: your own plugin, `SofaGpuCollision`, is **not** in this list. It's loaded
separately by the launcher script via `runSofa -l libSofaGpuCollision.so`. The
`-l` flag is just another way to load a plugin. Either way, by the time
`createScene` runs, `GpuCollisionBroadPhase` and `GpuCollisionNarrowPhase` are
registered and constructible by name.

---

## 2.4 Generating the geometry

```python
tissue_positions, tissue_triangles = generate_tissue_surface_grid()
blade_vertices, blade_triangles = create_blade_geometry(length=1.8, height=0.6, thickness=0.12)
blade_vertices = translate_vertices(blade_vertices, [0.0, 0.0, 0.0])
```

These come from `dense_collision_benchmark_common.py`:

### The tissue

`generate_tissue_surface_grid(nx=81, nz=81, sx=8.0, sz=8.0, y=0.0)` builds a
flat square sheet:

- 81 × 81 = **6,561 vertices**, laid out on a grid.
- Spanning X from −4 to +4 and Z from −4 to +4 (because `sx=sz=8.0`, centered).
- All at height `y = 0` — completely flat, lying in the XZ plane.
- Vertex spacing = 8.0 / 80 = **0.1 units** apart.
- Each of the 80 × 80 grid squares is split into 2 triangles → **12,800
  triangles**.

So the tissue is a flat 8×8 sheet finely tessellated into 12,800 triangles. No
thickness, no volume.

### The blade

`create_blade_geometry(length=1.8, height=0.6, thickness=0.12)` builds a simple
box (a rectangular cuboid):

- 8 corner vertices.
- **12 triangles** (2 per face × 6 faces).
- Spanning X ∈ [−0.9, 0.9], Y ∈ [−0.3, 0.3], Z ∈ [−0.06, 0.06].
- `translate_vertices(..., [0,0,0])` leaves it centered at the origin.

### The overlap

The blade is centered at the origin, and its Y-range is [−0.3, 0.3]. The tissue
lies at Y = 0. So **the blade box straddles the tissue plane** — its lower half
is below the sheet, its upper half is above, and the sheet passes through its
middle. That overlap is what produces collision contacts. This is the
"blade embedded in tissue" configuration, frozen in place.

---

## 2.5 The animation loop and pipeline

```python
root.addObject('DefaultAnimationLoop')
root.addObject('CollisionPipeline')
```

- `DefaultAnimationLoop` — the clock. Every time SOFA "steps," this object's
  `step()` runs and drives one frame.
- `CollisionPipeline` — SOFA's collision orchestrator. Each frame it calls the
  broad phase, then the narrow phase, then (if present) collision response. It
  doesn't know or care that ours run on the GPU; it just calls whatever
  broad/narrow components exist.

---

## 2.6 The broad phase component

```python
root.addObject(
    'GpuCollisionBroadPhase',
    enableGPU=True,
    allowCPUFallback=True,
    logBackendStatus=True,
    useObjectAabbCulling=False,
)
```

This is **your** component (from `GpuCollisionBroadPhase.cpp`). Key setting:

- `useObjectAabbCulling=False` — because there are only two objects, there's no
  point doing fancy GPU box-culling. The broad phase just pairs them up
  trivially. (More in file 04.)

---

## 2.7 The narrow phase component — the heart of it

```python
root.addObject(
    'GpuCollisionNarrowPhase',
    enableGPU=True,
    allowCPUFallback=True,
    useDenseGrid=True,
    copyContactsToHost=COPY_CONTACTS_TO_HOST,        # False → detection-only
    useIndexedDenseGridInput=USE_INDEXED_DENSE_GRID_INPUT,   # True
    useDirectDevicePositions=USE_DIRECT_DEVICE_POSITIONS,    # True
    useFeatureBasedProximity=USE_FEATURE_BASED_PROXIMITY,
    # ... grid configuration ...
    gridMinX=-4.5, gridMinY=-0.5, gridMinZ=-4.5,
    gridMaxX=4.5,  gridMaxY=0.5,  gridMaxZ=4.5,
    gridResolutionX=64, gridResolutionY=8, gridResolutionZ=64,
    contactDistance=0.03,
    maxTissueTrianglesPerCell=128,
    maxToolTrianglesPerCell=64,
    maxCandidatePairs=2000000,
)
```

This is the component that runs the GPU collision detection. The grid
parameters define the "dense grid" spatial structure (file 06):

- The grid is a box from (−4.5, −0.5, −4.5) to (4.5, 0.5, 4.5) — slightly bigger
  than the tissue so everything fits inside.
- It's chopped into 64 × 8 × 64 = **32,768 cells**.
- `contactDistance=0.03` — two triangles count as "in contact" if they come
  within 0.03 units of each other.
- `maxTissueTrianglesPerCell=128` — each cell can hold up to 128 tissue
  triangles (a safety capacity).

We'll dissect every one of these in file 06. For now, just note: this is where
you configure the GPU collision detector.

---

## 2.8 LocalMinDistance

```python
root.addObject('LocalMinDistance', alarmDistance=0.08, contactDistance=0.03, angleCone=0.0)
```

This is a standard SOFA "intersection method." The broad phase asks it "are
these two models even allowed to intersect?" It's mostly bookkeeping here; the
real distance math happens in your CUDA kernels.

---

## 2.9 The benchmark controller (the stopwatch)

```python
root.addObject(
    'GpuPipelineBenchmarkController',
    name='GpuOneTissueOneBladeTiming',
    label='gpu_one_tissue_one_blade_dense_grid_benchmark',
    outputDir=BENCHMARK_LOG_DIR,
    warmupSteps=10,
    flushInterval=20,
    logInterval=20,
    printProgress=True,
    # ... plus descriptive metadata fields ...
)
```

This is your stopwatch and CSV writer (`GpuPipelineBenchmarkController.cpp`). It
listens for SOFA's "frame begin" and "frame end" events, records per-frame
timings, and writes them to a CSV plus a summary file.

- `warmupSteps=10` — ignore the first 10 frames when averaging (the GPU needs a
  few frames to warm up its clocks and caches).
- `flushInterval=20` — write to disk every 20 frames.

File 11 explains every metric it records.

---

## 2.10 The actual objects

Finally, the two physical objects, each a child node of root:

```python
tissue = root.addChild('Tissue')
tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
tissue.addObject('TriangleCollisionModel', selfCollision=False)

blade = root.addChild('Blade')
blade.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=blade_vertices)
blade.addObject('MeshTopology', name='topo', triangles=blade_triangles)
blade.addObject('TriangleCollisionModel', selfCollision=False)
```

Each object has exactly three components — this trio is the whole story:

1. **`MechanicalObject` with `template='CudaVec3f'`** — the vertex positions,
   allocated in **GPU VRAM**. For the tissue, that's 6,561 points; for the
   blade, 8 points.

2. **`MeshTopology` with `triangles=...`** — the connectivity: the list of
   index-triples saying which vertices form each triangle. 12,800 triples for
   the tissue, 12 for the blade. This is **static** — it never changes.

3. **`TriangleCollisionModel`** — a flag that says "this object participates in
   collision." Because the `MechanicalObject` underneath uses `CudaVec3f`, SOFA
   automatically upgrades this to a **`CudaTriangleCollisionModel`** — the
   CUDA-aware variant. That upgrade is what lets your plugin grab the GPU
   pointer later.

- `selfCollision=False` — don't check the tissue against itself. (Setting this
  `True` is how you'd trigger the self-collision path in file 09.)

---

## 2.11 What this scene is NOT

To cement the "detection-only" idea, here's what's deliberately absent:

- **No solver** (no `EulerImplicitSolver`) — nothing integrates forces over
  time.
- **No force field** (no `TetrahedralCorotationalFEMForceField`) — the tissue
  has no squishiness.
- **No mass** — objects have no inertia.
- **No controller** — nothing moves the blade.
- **No `OglModel`** — nothing is drawn to a screen.
- **No `ContactManager`** — collisions are detected but never responded to.

So when SOFA "steps" this scene, the *only* substantial work is the collision
detection. Everything else is stripped away. That's the point: a clean
stopwatch on the GPU collision path.

---

## The shape of one frame

Putting it together, here's what `DefaultAnimationLoop::step()` does each frame
for this scene:

```text
1. (Physics)        nothing — no solver, no forces
2. CollisionPipeline::computeCollisionDetection()
     a. GpuCollisionBroadPhase  → emits the pair (Tissue, Blade)
     b. GpuCollisionNarrowPhase → runs the GPU kernels, finds contacts
3. (Collision response)  nothing — no ContactManager
4. GpuPipelineBenchmarkController records the frame's timings
```

The next files walk through each of those steps in detail. We start before the
clock even ticks — the one-time setup. Go to
[03_phase0_setup.md](03_phase0_setup.md).
