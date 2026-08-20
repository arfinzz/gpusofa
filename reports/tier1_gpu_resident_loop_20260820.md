# Tier 1 — Closing the GPU Loop: Device-Resident Contact Response (2026-08-20)

The collision pipeline had a producer and no consumer. Contacts were computed on the
device at 0.29 ms and then either sat unread in a device buffer or were copied into SOFA's
host `DetectionOutput`. Every test scene was collision-only — no solver, no mass, no force
field — so nothing consumed them.

**This work closes the loop.** A frame now runs FEM tissue, collision detection, and contact
response with **zero device-to-host transfer of simulation state**, verified by assertion
rather than by profiler inspection.

Companion docs: plan in `PLAN_TIER1_TIER3.md`, mode/metric explainers in
`README_execution_modes.md`, collision performance in `performance_all_modes_20260715.md`.

---

## 1. What was built

| Component | File | Role |
|---|---|---|
| `contactVertexWeights` | `cuda/detail/ContactForces.cuh` | Decodes a contact's feature (VF / FV / EE) + local index + barycentrics into **weights on the triangle's 3 vertices**. Deliberately factored out — the future constraint path needs exactly this to build Jacobian rows |
| penalty force / dForce kernels | same | `F = max(0, k·(contactDistance − d) − c·vₙ)` scattered onto the 6 owning vertices with `atomicAdd`; `K = k·(n⊗n)` for implicit integration |
| `accumulateContactPenaltyForces` / `…DForces` | public API | Contact struct stays private to the CUDA TU (§7 boundary rule): callers pass device force pointers and get forces, never contacts |
| `RecordedContactHandle` | `cuda/detail/FbpKernels.cuh` | Every one of the five proximity drivers records where its contacts live, plus the device triangle indices needed to resolve owning vertices |
| `CudaContactPenaltyForceField` | new SOFA component | `PairInteractionForceField<CudaVec3fTypes>`; reaches state **only** through `deviceWrite()` / `deviceRead()` |
| `GpuResidencyChecker` | new SOFA component | Gate 5 — the zero-transfer assertion |
| `gpu_resident_fem_contact.py` | `testscenes/` | The first scene in the repo that actually simulates: solver + mass + FEM + boundary conditions + collision + response |

### The accessor rule everything depends on

`sofa::type::vector_device` holds two buffers and two flags. `helper::ReadAccessor` calls
`hostRead()`; `WriteAccessor` calls `hostWrite()`. **Either one copies the whole state
vector to the host and marks the device copy stale.** A single such call anywhere in the
frame silently reintroduces the per-frame transfer. Every access in the new code goes
through `deviceWrite()` / `deviceRead()` instead — and Gate 5 exists to catch any future
regression automatically.

## 2. Verification — 4 gates, all passing

### Gate 1 — GPU forces match an independent host reference

The host reference is written separately from the kernel (plain loops; it shares only the
weight *convention* being tested), then fed the identical downloaded contacts.

| stiffness | contacts | max abs error | max reference force | **relative error** |
|---:|---:|---:|---:|---:|
| 100 | 8018 | 1.5e-5 | 49.21 | **3.0e-7** |
| 1,000 | 8018 | 1.22e-4 | 492.10 | **2.5e-7** |
| 25,000 | 8018 | 2.93e-3 | 12302.59 | **2.4e-7** |

The absolute error scales linearly with stiffness while the relative error stays flat at
~2.5e-7 — the signature of float32 rounding, not a logic error. Threshold 1e-5: **PASS**.

### Gate 2 — Newton's third law

Sum of every force vector over both bodies. Tests the scatter alone, independent of the
penalty law, so sign and index errors have nowhere to hide.

| stiffness | net force | total force magnitude | **ratio** |
|---:|---:|---:|---:|
| 100 | 8.2e-5 | 14,929 | **5.5e-9** |
| 1,000 | 1.1e-3 | 149,295 | **7.4e-9** |
| 25,000 | 2.8e-2 | 3,732,379 | **7.5e-9** |

**PASS** — equal and opposite to nine significant figures.

### Gate 3 — live simulation

Contacts across a settling sequence: `27 → 44 → 51 → 53 → 54 → 45 → 55 → 60 → 66 → 80 → 78 → 85`,
all active. The count rising as the blade settles into the tissue is the physically correct
shape (deeper contact ⇒ more contacting triangles), and the simulation is stable across the
run rather than exploding or falling through.

### Gate 5 — the zero-transfer gate (the headline)

```
GPU residency @frame 20: clean=16 violations=0
GPU residency @frame 40: clean=36 violations=0
```

Every frame, for every `CudaVec3f` state vector, `isHostValid()` was **false** — meaning
nothing pulled position, velocity or force to the host. This is a deterministic in-process
assertion using SOFA's own validity flags: no profiler, works under WSL2 where nsys cannot
capture GPU timelines, and on failure it **names the offending mechanical object and
vector** instead of leaving a hunt through a trace.

## 3. Bugs found (all fixed)

1. **`sync_and_build_wsl.sh` never copied `CMakeLists.txt` to WSL.** Adding source files
   produced a **green build that silently omitted them** — caught only by checking exported
   symbols rather than trusting exit code 0. The script now syncs it and greps for the new
   component as a sync marker. This had been latent for the whole project.
2. **`MeshMatrixMass<CudaVec3f,CudaVec3f>` segfaults in `copyVertexMass()` during init** —
   a fault inside SOFA v25.12's own component, confirmed by backtrace, not a scene error.
   Worked around with `UniformMass` (also makes the equilibrium prediction cleaner).
3. **Two ODE solvers cannot couple through an interaction force field.** The first scene
   graph gave tissue and blade separate solvers. Restructured to a single solver over both,
   which also makes the contact coupling implicit via `addDForce`.
4. **SOFA emits broad-phase pairs in its own order, not scene order** — the recorded handle
   was reversed relative to the force field's request. The backend now accepts either order
   and swaps the force/velocity bindings internally, reporting both ids in the diagnostic
   when neither matches.
5. **The stub backend was missing sorted-grid and big-cell entries** — a latent CPU-only
   build break, now complete along with the new contact-force entry points.

## 4. What this does and does not prove

**Proven:** the frame is device-resident; contact forces are numerically correct against an
independent reference; the scatter conserves momentum exactly; the simulation is stable.

**Not claimed:** that penalty response is *accurate contact physics*. It is not — it
interpenetrates by construction and has no true friction. That is the documented trade for
staying on the GPU (see the constraint-vs-penalty analysis in `guide/architecture.md`).
Constraint-based response remains CPU-bound until a GPU constraint solver exists.

## 5. Next

Tier 3 (Ogden hyperelastic + SLS-Ogden viscoelastic on GPU) is planned in detail in
`PLAN_TIER1_TIER3.md`, including the Ogden-specific trap: the tangent stiffness contains
`1/(λᵢ² − λⱼ²)` terms that blow up when two principal stretches coincide, which uniaxial
and hydrostatic loading do constantly. Gate 4c is designed to sweep *through* the
degeneracy rather than only land on it.

Artifacts: `output/benchmark_logs/gpu_resident_*`. Gates 1+2 run from the standalone bench
(`SOFA_BACKEND_BENCH_RUN_CONTACT_FORCES=1`, on by default); Gates 3+5 from
`scripts/run_gpu_resident_scene_wsl.sh`.
