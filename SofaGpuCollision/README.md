# SofaGpuCollision

`SofaGpuCollision` is a standalone SOFA plugin scaffold for moving the collision
detection stack toward GPU execution in a controlled way.

It provides two SOFA components:

- `GpuCollisionBroadPhase`
- `GpuCollisionNarrowPhase`

Both components are designed to be GPU-first and benchmark-friendly:

- They expose explicit GPU intent at the component level.
- They keep a safe CPU fallback path so scenes remain runnable while the CUDA
  backend is being validated.
- They preserve the existing SOFA collision response path instead of replacing
  contact generation with a brittle custom implementation.

## Current state

This repository contains a robust plugin architecture and benchmark integration
point, but it does not pretend to be a fully validated production CUDA solver
yet.

What is implemented now:

- SOFA plugin build scaffold with install rules.
- Broad-phase and narrow-phase components registered with SOFA.
- Runtime backend probing.
- Explicit CPU fallback behavior.
- A benchmark scene that switches your GPU benchmark to the new plugin.

What still needs a compiled CUDA toolchain and SOFA SDK validation:

- Extracting collision-model AABBs into GPU buffers.
- Launching real broad-phase kernels and writing back candidate pairs.
- Launching narrow-phase pruning kernels before final SOFA contact generation.
- End-to-end validation on your exact SOFA and CUDA versions.

## Build

Set `SOFA_ROOT` or `CMAKE_PREFIX_PATH` so CMake can find your SOFA installation,
then configure the plugin:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\path\to\SOFA"
cmake --build build --config Release
```

Use the same architecture and compiler family as your SOFA build. On this
machine, SOFA was discovered as a 64-bit package, so a 32-bit MinGW generator
will not link against it correctly.

To compile the experimental CUDA backend once your environment exposes `nvcc`:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\path\to\SOFA" -DSOFAGPUCOLLISION_ENABLE_CUDA=ON
cmake --build build --config Release
```

## Scene usage

After building and making the plugin visible to SOFA, use:

```python
root.addObject("RequiredPlugin", pluginName=["SofaGpuCollision", ...])
root.addObject("GpuCollisionBroadPhase", enableGPU=True, allowCPUFallback=True)
root.addObject("GpuCollisionNarrowPhase", enableGPU=True, allowCPUFallback=True)
```

Ready-to-run benchmark scenes live in the repository's `testscenes/` directory
(e.g. `testscenes/one_tissue_one_blade.py` for the tri-tri path,
`testscenes/self_collision_vertex_triangle.py` for self-collision). Launch them
with the wrappers in `scripts/` (see `guide/setup.md`).
