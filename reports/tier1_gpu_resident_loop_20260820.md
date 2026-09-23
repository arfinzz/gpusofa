# Tier 1 — Closing the GPU Loop: Device-Resident Contact Response (2026-08-20)

> **Correction (2026-09-23).** Two claims in the first version of this report were wrong.
> They are corrected in place below (Gate 3, Gate 5 and §4):
>
> 1. **"Zero device-to-host transfer" is wrong.** The residency checker only looked at the
>    end of each frame, and a CPU read followed by a GPU write later in the same frame hides
>    from that check. With a start-of-frame check added, `dofs.position` is copied from the
>    GPU to the CPU **every frame**. Switching parts of the scene off one at a time shows the
>    copy comes from SOFA's own solving loop, not from the collision or contact code. The
>    exact call is not pinned down yet.
> 2. **"The simulation is stable" is wrong.** That verdict came from a short run and looked
>    only at contact counts. Run for 300 frames, the blade comes apart: it flattens when it
>    lands, then its top face flies upward and its bottom face falls through the tissue.
>    Two causes: the blade is 8 unconnected points (nothing holds it together as a rigid
>    body), and the contact uses an unsigned distance, so it cannot tell which side of a
>    surface a point is on.
>
> Gates 1 and 2 (the force math and Newton's third law) are not affected and still pass.

The collision pipeline had a producer and no consumer. Contacts were computed on the
device at 0.29 ms and then either sat unread in a device buffer or were copied into SOFA's
host `DetectionOutput`. Every test scene was collision-only — no solver, no mass, no force
field — so nothing consumed them.

**This work closes the loop.** A frame now runs FEM tissue, collision detection, and contact
response on the GPU. (The first version said this happened with zero device-to-host
transfer of simulation state; that was wrong — see the correction above.)

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
| `gpu_resident_fem_contact.py` | `testscenes/collisiondetectiontests/` | The first scene in the repo that actually simulates: solver + mass + FEM + boundary conditions + collision + response |

### The accessor rule everything depends on

`sofa::type::vector_device` holds two buffers and two flags. `helper::ReadAccessor` calls
`hostRead()`; `WriteAccessor` calls `hostWrite()`. **Either one copies the whole state
vector to the host and marks the device copy stale.** A single such call anywhere in the
frame silently reintroduces the per-frame transfer. Every access in the new code goes
through `deviceWrite()` / `deviceRead()` instead — and Gate 5 exists to catch any future
regression automatically.

## 2. Verification — 4 gates (Gates 3 and 5 corrected 2026-09-23)

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

**⚠ Corrected 2026-09-23 — this verdict was wrong.** It rested on a short run and on
contact counts alone; it never checked the blade's shape, its penetration depth against
`mg/(k·n)`, or the energy. A 300-frame run (default drop height 0.6, dt 0.005) that logs
the blade's top and bottom faces shows the failure. The blade is 0.28 tall and the tissue
occupies y = −0.25 … 0.25:

| Frame | Bottom face y | Top face y | Blade height (should stay 0.28) |
|---:|---:|---:|---:|
| 1 | 0.46 | 0.74 | 0.28 |
| 40 | 0.26 | 0.53 | 0.27 (lands) |
| 50 | 0.24 | 0.43 | 0.19 (collapsing) |
| 80 | 0.26 | 1.15 | 0.89 |
| 200 | −0.17 | 5.65 | 5.82 |
| 300 | −5.89 | 9.85 | 15.74 |

Causes: (1) the blade is a `CudaVec3f` MechanicalObject with a mass and no force field,
so its 8 vertices are independent particles, not a rigid body; (2) the contact distance is
unsigned and the normal points from one contact point to the other, so once a vertex
crosses a surface the force pushes it further through, and beyond `contactDistance` there
is no force at all. The force math itself is still correct (Gates 1 and 2).

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

**⚠ Corrected 2026-09-23 — this result was wrong.** The checker sampled only at the end
of each frame. `hostRead()` marks the host copy valid without invalidating the device
copy, so a CPU read followed by any GPU write later in the frame leaves `isHostValid()`
false again by the end, and the transfer goes unseen. The checker now also samples at the
start of each frame, and the same scene reports a device-to-host copy of `dofs.position`
**every frame**. With the collision models and pipeline, the FEM force field, the
geometry algorithms and the benchmark controller all switched off, the copy is still
there, so it comes from SOFA's own solving loop (animation loop, ODE solver, linear
solver, mass, fixed constraint or MechanicalObject bookkeeping). The exact call is not
pinned down yet. The checker still cannot see a read that is hidden before both sample
points; proving that would need copy-level interception inside SofaCUDA.

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

**Proven:** contact forces are numerically correct against an independent reference; the
scatter conserves momentum exactly.

**Not proven (corrected 2026-09-23):** that the frame is device-resident — it is not, see
Gate 5; and that the simulation is stable — it is not, see Gate 3.

**Not claimed:** that penalty response is *accurate contact physics*. It is not — it
interpenetrates by construction and has no true friction. That is the documented trade for
staying on the GPU (see "Known problems and limits" in the root `README.md`).
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
