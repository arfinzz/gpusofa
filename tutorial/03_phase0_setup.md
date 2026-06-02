# 03 — Phase 0: Setup (Before the Clock Starts)

Everything in files 04–10 happens *every frame*. This file is different: it
describes the one-time setup that happens **once**, at scene load, before the
first frame ticks. Getting this setup right is what makes every later frame
cheap.

---

## 3.1 Loading the plugin

When SOFA processes `RequiredPlugin` (and the `-l libSofaGpuCollision.so` flag
from the launcher), it `dlopen`s the shared library. The moment the library
loads, two things happen automatically:

1. **`initExternalModule()` runs** (in `init.cpp`). For this plugin it's nearly
   empty — it just exists so SOFA knows the library is a valid plugin and can
   report its name, version, and component list.

2. **Static registration fires.** At the top of `GpuCollisionBroadPhase.cpp`:

   ```cpp
   int GpuCollisionBroadPhaseClass = sofa::core::RegisterObject(
       "GPU-first broad phase collision detection with a safe CPU fallback.")
       .add<GpuCollisionBroadPhase>();
   ```

   This line runs when the library loads (it's a global initializer). It tells
   SOFA's `ObjectFactory`: "the string `GpuCollisionBroadPhase` means *this*
   C++ class." The same happens for `GpuCollisionNarrowPhase` and
   `GpuPipelineBenchmarkController`.

After this, when your Python does `root.addObject('GpuCollisionNarrowPhase',
...)`, SOFA looks up that name in the factory and builds the C++ object.

---

## 3.2 Building the components

As `createScene` runs, each `addObject(...)` constructs a C++ component. The
constructor's main job is to declare its **Data fields** — the configurable
parameters you saw in the Python.

For example, in `GpuCollisionNarrowPhase`'s constructor:

```cpp
, d_contactDistance(initData(&d_contactDistance, 0.03, "contactDistance",
      "Inflation distance for dense-grid primitive AABBs."))
```

`initData` connects the C++ variable `d_contactDistance` to the Python keyword
`contactDistance`, with a default of `0.03` and a help string. When your scene
passed `contactDistance=0.03`, SOFA stuffed that value into this Data field.
This is the bridge between Python and C++.

A "Data field" in SOFA is a wrapped value with a name, default, and docstring;
SOFA can read/write it by name, expose it in GUIs, and serialize it. You read
it in C++ with `.getValue()`.

---

## 3.3 `init()` — probing the GPU

After the tree is built, SOFA walks it and calls `init()` on every component.

`GpuCollisionNarrowPhase::init()` does one important thing:

```cpp
void GpuCollisionNarrowPhase::init()
{
    // ... call base class init ...
    const auto status = backend::probe();
    m_backendAvailable = status.available;
    // ... log the result ...
}
```

`backend::probe()` (in `cuda/GpuCollisionBackend.cu`) asks "is there a usable
CUDA GPU?" by calling `cudaGetDeviceCount`. If yes, it returns
`{available: true, "CUDA runtime detected..."}`. The result is cached in
`m_backendAvailable`.

This is the single gate for all GPU work. If `probe()` had returned false (no
GPU, or the plugin built without CUDA), every later frame would fall back to
SOFA's CPU collision path (when `allowCPUFallback=True`). On your machine it
returns true, so the GPU path is live.

The broad phase does the same probe in its own `init()`.

---

## 3.4 `CudaVec3f` allocation — positions are born on the GPU

This is the most important setup step. Recall the scene:

```python
tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
```

When SOFA constructs this `MechanicalObject`, the `SofaCUDA` plugin sees the
`CudaVec3f` template and handles the allocation specially:

1. It takes the `tissue_positions` Python list (6,561 points = 19,683 floats).
2. It calls `cudaMalloc` to reserve a buffer for them in **VRAM**.
3. It copies the initial values into that VRAM buffer **once**.

From this point on, the canonical home of the tissue positions is the GPU. The
Python list is no longer needed. There is a small CPU-side wrapper object (the
`MechanicalObject` itself lives in CPU memory), but the *position payload* — the
actual coordinates — sits in VRAM.

Crucially: **this allocation happens once, at startup.** It is not a per-frame
cost. Every later frame just reads the buffer that already exists.

---

## 3.5 The topology cache — fingerprinting the connectivity

The narrow phase keeps a cache so it doesn't rebuild the triangle-index array
every frame. The cache is:

```cpp
std::unordered_map<const void*, CachedTriangleTopology> m_triangleTopologyCache;
```

It's a dictionary keyed by the collision-model pointer. Each entry holds:

```cpp
struct CachedTriangleTopology {
    std::size_t triangleCount;      // 12,800 for the tissue
    std::size_t vertexCount;        // 6,561
    std::uint64_t topologyHash;     // a fingerprint (see below)
    bool seenThisFrame;
    std::vector<std::uint32_t> triangleIndices;  // the flat (v0,v1,v2,...) array
    // ... scratch buffers for fallback paths ...
};
```

### The fingerprint (FNV-1a hash)

The first time the narrow phase sees a model, it computes a **hash** of its
topology — a single 64-bit number that fingerprints the entire triangle-index
list. The code (in `extractCudaIndexedSurface`) uses the FNV-1a algorithm:

```cpp
std::uint64_t hash = 1469598103934665603ull;   // FNV-1a offset basis
const auto mix = [&](const std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;                    // FNV-1a prime
};
mix(positions.size());
mix(triangles.size());
for (const auto& triangle : triangles) {
    mix(triangle[0]); mix(triangle[1]); mix(triangle[2]);
}
```

The idea: feed every index through the mix and you get one number. If *any*
index changes, the number changes. If the topology is identical, the number is
identical.

### Why this matters

Once the indices are flattened into a `std::vector<uint32_t>` of the form
`[v0,v1,v2, v0,v1,v2, ...]` (38,400 numbers for the tissue) and the hash is
recorded, the narrow phase can, on every subsequent frame, just compare the
hash. Same hash → topology unchanged → **skip rebuilding the array and skip
re-uploading it to the GPU.** This is the foundation of the "0 bytes H2D per
frame" claim.

In the benchmark, the topology never changes (nothing cuts the tissue), so
after frame 1 the cache always hits.

---

## 3.6 The persistent workspace — GPU memory that lives forever

The CUDA backend owns a single big struct called `DenseGridWorkspace`,
accessed through `denseGridWorkspace()`. It owns *all* the reusable GPU
allocations:

```text
grid            — the 32,768 cell buckets
cellTissueIds   — which tissue triangles are in each cell
cellToolIds     — which tool triangles are in each cell
candidatePairs  — the output list of pairs to check
pairHashKeys    — the dedupe hash table
contacts        — the output contact records
counters        — candidateCount, contactCount, overflowCount, etc.
indexedTissueIndices / indexedToolIndices — the device copies of the triangle indices
proximityContacts — the feature-based-proximity output buffer
fbpStartEvent / fbpEndEvent — reusable CUDA timing events
proximityCountersHostPinned — a pinned host buffer for fast counter readback
```

The workspace has an `ensure(...)` method that allocates these buffers, **but
only grows them when needed**:

```cpp
cudaError_t ensure(... sizes ...) {
    err = ensureDeviceArray(grid, gridCellCapacity, cellCount, newlyAllocatedBytes);
    // ensureDeviceArray only calls cudaMalloc if the requested size
    // is bigger than what's already allocated.
    ...
}
```

So on **frame 1**, `ensure` calls `cudaMalloc` for everything (the workspace
starts empty). On **every later frame**, the sizes are the same, nothing grows,
and `ensure` does nothing — it just hands back the existing pointers.

This is why the benchmark reports `avg_workspace_resize_count = 0` in steady
state: after the first frame, no GPU allocation happens. Allocation (`cudaMalloc`)
is slow and can stall the GPU, so doing it once and reusing forever is a big
win.

The workspace is a singleton (`static DenseGridWorkspace workspace;`), so it
persists for the whole process lifetime and is only freed when the plugin
unloads.

---

## 3.7 Summary of Phase 0

By the time the first frame is about to tick, the following are true:

| Thing | State | Cost per frame after this |
|---|---|---|
| Plugin registered | done | 0 |
| Components built, Data fields set | done | 0 |
| GPU probed, `m_backendAvailable=true` | done | 0 |
| Tissue + blade positions in VRAM | allocated once | 0 (read in place) |
| Topology hashed + flattened | done | 0 (cache hits) |
| GPU index buffers | uploaded on frame 1 | 0 (cached by hash) |
| Workspace GPU buffers | allocated on frame 1 | 0 (reused) |

Everything expensive — allocation, hashing, the first upload — is paid once,
up front. The per-frame loop that follows is almost pure compute with no
memory traffic. That's the design.

Next: the first actual per-frame step, the broad phase. Go to
[04_phase1_broad_phase.md](04_phase1_broad_phase.md).
