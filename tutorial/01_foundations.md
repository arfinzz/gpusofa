# 01 — Foundations

Before we trace the code, you need a handful of concepts. Take your time here;
everything later depends on these.

---

## 1.1 What is SOFA?

**SOFA** (Simulation Open Framework Architecture) is a C++ engine for physics
simulation, used heavily in medical and surgical simulation research. Think of
it as a stage on which you place "objects" and "rules", and it animates them
frame by frame.

You don't usually write C++ to use SOFA. You write a **scene file** in Python
that says "put a tissue here, put a blade there, use these physics rules."
SOFA reads your Python, builds the objects in C++, and runs them.

A SOFA scene is a **tree of nodes**. Each node holds **components**. A
component is a small unit of behavior:

- A component that stores vertex positions.
- A component that says how triangles connect.
- A component that detects collisions.
- A component that draws graphics.

You assemble components like LEGO bricks to make a simulation.

### Example: the smallest possible scene

```python
def createScene(root):
    root.addObject('DefaultAnimationLoop')   # a clock that ticks
    child = root.addChild('MyObject')        # a node
    child.addObject('MechanicalObject', position=[[0,0,0],[1,0,0]])  # two points
    return root
```

This scene has a clock and one object holding two points. It doesn't *do*
anything interesting, but it is a valid SOFA scene.

---

## 1.2 What is a mesh?

A **mesh** is how we represent a shape in a computer. The two parts:

1. **Vertices** — points in 3D space. Each is just three numbers `(x, y, z)`.
2. **Topology** (or "connectivity") — which vertices join up to form faces.

### Triangles

The simplest face is a **triangle**: three vertices. Almost all 3D shapes in
real-time graphics and collision are built from triangles, because triangles
are always flat and always convex, which makes the math simple.

A triangle is stored as **three indices** into the vertex list. For example:

```text
Vertices:  v0=(0,0,0)  v1=(1,0,0)  v2=(1,1,0)  v3=(0,1,0)
Triangle A: [0, 1, 2]   ← uses v0, v1, v2
Triangle B: [0, 2, 3]   ← uses v0, v2, v3
```

Those two triangles together make a square. Notice the vertices are stored
**once** and referenced by index. That's important later: positions change
every frame (the tissue moves), but the *indices* `[0,1,2]` stay the same
forever. This is the single most important optimization in the whole project.

### Surface mesh vs volume mesh

- A **surface mesh** is hollow — just the outer skin made of triangles. Like a
  paper lantern.
- A **volume mesh** is solid — the inside is packed with 3D pyramids called
  **tetrahedrons** (4 vertices each). Like a brick of jelly.

**Collisions only ever happen on surfaces.** A blade can't touch the *inside*
of tissue without first passing through its *surface*. So the collision system
only ever deals with surface triangles, even when the physics underneath uses
a volume mesh.

In this project's benchmark, the tissue is a flat surface mesh — a square sheet
of triangles. There is no volume, no jelly, no physics. That's deliberate: we
want to measure collision detection in isolation, with nothing else competing
for time.

---

## 1.3 CPU vs GPU — the core idea

Your computer has two very different processors:

### The CPU (Central Processing Unit)

- A handful of very powerful cores (say, 8).
- Each core is fast and smart — good at complicated, branchy logic.
- Works on tasks **one after another** (or a few at a time).

Analogy: a few expert chefs, each cooking a full meal start to finish.

### The GPU (Graphics Processing Unit)

- Thousands of small, simple cores. (Your GTX 1650 Ti has 1024 CUDA cores
  across 16 "streaming multiprocessors".)
- Each core is weak on its own and bad at branchy logic.
- But they all work **at the same time** on the same kind of task.

Analogy: a thousand line cooks, each chopping one onion. Useless for a
complicated recipe, unbelievable for chopping 10,000 onions.

### Why collision detection fits the GPU

Checking whether triangle #4,200 touches the blade is the *exact same
computation* as checking triangle #4,201. There are 12,800 tissue triangles.
Instead of one CPU checking them in a loop (12,800 steps, one after another),
the GPU assigns one core to each triangle and checks **all of them at once**.

That is the entire reason this project exists.

---

## 1.4 The memory problem (this is the crux)

Here's the catch that makes GPU programming hard: **the CPU and the GPU have
separate memory.**

- The CPU works on data in **System RAM** (your normal computer memory).
- The GPU works on data in **VRAM** (video memory, on the graphics card).

The two are connected by a cable on the motherboard called the **PCIe bus**.
Moving data across that cable is *slow* compared to how fast either processor
computes.

Two directions of transfer, and you'll see these abbreviations everywhere:

- **H2D** = Host-to-Device = copying from CPU RAM → GPU VRAM.
- **D2H** = Device-to-Host = copying from GPU VRAM → CPU RAM.

### The naive (slow) way

```text
Every frame:
  1. CPU has the 10,000 vertex positions in System RAM.
  2. CPU copies them across PCIe to VRAM.        ← H2D, ~5 milliseconds, SLOW
  3. GPU does the collision math.                ← fast
  4. GPU copies results back across PCIe.        ← D2H, slow
```

Steps 2 and 4 dominate. The actual math is fast; the *copying* is the
bottleneck. In an early version of this project, that H2D copy alone was
burning ~5 ms per frame.

### The fast way (what this project does)

What if the vertex positions never lived in System RAM in the first place?
What if they were *born* in VRAM and *stayed* there?

That's exactly what `CudaVec3f` does (next section). The positions live on the
GPU. The collision kernels read them directly. **Zero copying per frame.** This
is called **zero-copy**, and it's why the benchmark hits 700+ frames per
second.

---

## 1.5 `CudaVec3f` vs `Vec3d` — where data lives

In a SOFA scene, the component that holds vertex positions is called a
**`MechanicalObject`**. When you create one, you pick a **template** that
decides where the data lives and how precise it is:

| Template | Lives in | Precision | Who reads it |
|---|---|---|---|
| `Vec3d` | System RAM (CPU) | 64-bit double | CPU code |
| `CudaVec3f` | VRAM (GPU) | 32-bit float | GPU kernels (directly!) |

- `Vec3` = a 3D vector (x, y, z).
- `d` = double precision (64-bit, very accurate).
- `f` = single precision (32-bit float, less accurate but half the size and
  much faster on a GPU).
- `Cuda` prefix = "allocate this in VRAM."

When the scene says:

```python
tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
```

SOFA's `SofaCUDA` plugin intercepts the memory allocation and runs `cudaMalloc`
under the hood — the positions go straight into VRAM. The `tissue_positions`
Python list is used once to fill that VRAM buffer at startup, then forgotten.

### "Degrees of freedom" (DoF)

You'll see this phrase. A **degree of freedom** is just one number the
simulation can change. A single 3D vertex has 3 DoFs (its x, y, z). A
`MechanicalObject` holding 6,561 vertices has 6,561 × 3 = 19,683 DoFs. It is,
in essence, a big spreadsheet of these numbers. A `MechanicalObject` is "dumb":
it knows the *positions* but nothing about how they connect — that's the
topology component's job.

---

## 1.6 CUDA words you need

**CUDA** is NVIDIA's system for writing programs that run on the GPU. You write
a normal-looking C++ function, mark it specially, and the GPU runs thousands of
copies of it at once. A few terms:

### Kernel

A **kernel** is a function that runs on the GPU. It's marked with `__global__`
in the code. When you "launch" a kernel, you don't run it once — you run it
N times in parallel, once per "thread."

```cpp
__global__ void myKernel(...) { ... }   // runs on the GPU
```

### Thread

A **thread** is one running copy of the kernel. Each thread gets a unique ID so
it knows which piece of data it's responsible for. In this project, a typical
pattern is "one thread per triangle":

```cpp
const std::uint32_t triangleId = blockIdx.x * blockDim.x + threadIdx.x;
if (triangleId >= triangleCount) return;   // extra threads do nothing
```

`triangleId` is "my thread number." Thread 0 handles triangle 0, thread 4,200
handles triangle 4,200, and so on.

### Block and grid (launch shape)

Threads are grouped into **blocks**, and blocks form a **grid**. When you launch
a kernel you write `kernel<<<numBlocks, threadsPerBlock>>>(...)`. In this
project, `threadsPerBlock` is almost always 256. So to process 12,800
triangles you launch `ceil(12800 / 256) = 50` blocks of 256 threads = 12,800
threads.

(Don't confuse this "grid" of thread-blocks with the "dense grid" spatial data
structure in file 06. Same English word, totally different thing. CUDA grid =
how threads are organized. Dense grid = how 3D space is chopped into boxes.)

### `atomicAdd` and `atomicCAS`

When thousands of threads run at once, two of them might try to write to the
same memory at the same instant — a "race." **Atomic** operations are the fix:
they guarantee that even if 50 threads hit the same counter simultaneously,
each one's update lands correctly, one at a time, at the hardware level.

- `atomicAdd(&counter, 1)` — add 1 to a shared counter, safely. Returns the
  value *before* your add, so each thread gets a unique slot number.
- `atomicCAS(&slot, expected, new)` — "Compare-And-Swap." Atomically: *if* the
  slot currently equals `expected`, replace it with `new`. Returns the old
  value. Used to claim a hash-table slot — the first thread wins, others see
  it's taken.

These appear constantly in the kernels. When you see `atomicAdd`, read it as
"safely grab the next free slot."

### `cudaMemcpy` and `cudaDeviceSynchronize`

- `cudaMemcpy(dst, src, bytes, direction)` — the explicit copy across PCIe.
  This is the slow thing we want to avoid. `cudaMemcpyAsync` is the
  non-blocking version.
- `cudaDeviceSynchronize()` — "CPU, wait here until the GPU finishes everything
  queued." This *stalls* the CPU. Avoiding unnecessary syncs is a big theme.

### Asynchronous launches

Kernel launches are **non-blocking**: the CPU says "GPU, go run this," and
immediately moves on without waiting. The GPU works in the background. The CPU
only has to wait if it explicitly calls `cudaDeviceSynchronize` or a blocking
`cudaMemcpy`. This is the secret behind the fast benchmark — the CPU schedules
all the work and finishes the frame *while the GPU is still computing*.

---

## 1.7 The vocabulary of collision detection

Two stages, always in this order:

### Broad phase

The **broad phase** is the cheap first pass. It answers: "which *objects* (or
chunks of objects) are anywhere near each other?" It throws away the obviously-
far-apart pairs so the expensive stage doesn't waste time on them.

Analogy: before checking if two specific people shook hands, first check if
they were even in the same building.

### Narrow phase

The **narrow phase** is the expensive precise pass. For the pairs that survived
broad phase, it computes the exact answer: "*which triangles* are touching, at
what point, with what normal direction, how deep?"

Analogy: now that we know they were in the same room, check the exact handshake.

### Contact / DetectionOutput

A **contact** is the result: a record saying "these two primitives touch here."
In SOFA the standard record is called a `DetectionOutput` and it carries two
points (one on each object), a normal direction, and a distance value.

This project produces a richer record on the GPU called a `ProximityContact`
(it also carries barycentric weights — file 08), but it can convert down to a
SOFA `DetectionOutput` when needed.

---

## 1.8 Detection-only mode

Normally a simulation does three things with a collision: **detect** it,
**respond** to it (push the objects apart), and **render** it (draw the result).

This project's benchmark does **only the detection** — it finds the contacts,
writes them to GPU memory, and then *ignores them*. No pushing apart, no
drawing. Why?

Because we're measuring how fast detection is. If the blade actually pushed
into the tissue and deformed it, every frame would have a different number of
contacts, and the timing would be noisy. By freezing everything and only
detecting, we get a clean, repeatable stopwatch reading. This is called
**detection-only mode** and it's controlled by the flag
`copyContactsToHost=False` plus the absence of any response components.

---

## You now know enough

You understand: SOFA scenes, meshes (vertices + indices), surface vs volume,
CPU vs GPU, the memory/PCIe problem, `CudaVec3f` zero-copy, the CUDA words
(kernel, thread, block, atomic, sync, async), and the broad/narrow/contact
vocabulary.

Next: we open the actual scene file and read it line by line.
Go to [02_the_scene.md](02_the_scene.md).
