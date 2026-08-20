# TEMP PLAN — Tier 1 (contact consumer) + Tier 3 (GPU materials)

Working plan; delete once landed. Goal: close the loop so the frame never leaves the GPU,
then put realistic tissue material on the GPU too.

Grounded in verified facts from the install:
- `sofa::type::vector_device` exposes **`isHostValid()` / `isDeviceValid(gpu)`**
  (vector_device.h:139–145) → gives us a *deterministic in-process zero-transfer assertion*.
- All SofaCUDA transfers funnel through `mycudaMemcpyDeviceToHost` (mycuda.h:61) → a second,
  independent check.
- Hyperelastic material interface is `getStrainEnergy` / `deriveSPKTensor` /
  `applyElasticityTensor` / `ElasticityTensor` over a `StrainInformation` (HyperelasticMaterial.h:66–85).
- `SofaViscoElastic` ships Maxwell 1st/2nd, KelvinVoigt 1st/2nd, Burgers, and SLS variants
  fused with NeoHookean / StableNeoHookean / MooneyRivlin / Ogden, via
  `TetrahedronViscoelasticityFEMForceField` and `TetrahedronViscoHyperelasticityFEMForceField`.

---

## Design decisions already made (with reasons)

1. **Fused contact→force kernel, not SOFA's mapper+component pattern.** SOFA's
   `ContactMapper` creates an intermediate `MechanicalObject` per contact pair and
   instantiates components per frame — host-side object churn, exactly what we are removing.
   Instead: one kernel reads the device contact buffer and scatters forces onto parent DOFs.
   Same fusion insight as way 6.
2. **But the scatter is a reusable device function.** `contactVertexWeights(...)` decodes
   (featureKind, featureLocalIndex, barycentrics) into 3 per-triangle-vertex weights. The
   future constraint path needs exactly this to build J — so it does not get buried inside
   the penalty kernel. (Same discipline as extracting `fbpComputeClosestFeatureContact`.)
3. **`DefaultAnimationLoop` for this phase.** Its order is collision → integrate, so the
   contact buffer produced in the collision step is read by the force field during the solve.
   `FreeMotionAnimationLoop` is for the constraint path (Tier 2), not needed here.
4. **Forces written via `deviceWrite()`, never `WriteAccessor`.** `WriteAccessor` calls
   `hostWrite()` → forces a D2H and invalidates the device copy. Following
   `CudaSpringForceField.inl` as the reference for correct accessor discipline.

---

## Tier 1 — the contact consumer

| # | Item | Where | Notes |
|---|---|---|---|
| T1.1 | `DeviceContactHandle` + backend accessor | `GpuCollisionBackend.h`, way-6 driver | `{ const DeviceProximityContact* contacts; const uint32_t* countDevice; uint32_t capacity; surfaceIds }` for the last computed pair |
| T1.2 | Per-pair handle registry on the narrow phase | `GpuCollisionNarrowPhase` | `getContactHandle(cmA, cmB)`; multiple model pairs per frame |
| T1.3 | `contactVertexWeights` device helper | new `cuda/detail/ContactScatter.cuh` | decodes VF / FV / EE into 3 vertex weights per side |
| T1.4 | `CudaContactPenaltyForceField` | new component | `PairInteractionForceField<CudaVec3fTypes>`; `addForce` + `addDForce`; Data: `stiffness`, `damping` |

**`addForce` kernel** (one thread per contact): read contact → penalty magnitude
`F = k·max(0, −signedDistance)` plus damping `−c·(relative velocity · n)` → scatter
`+F·n·w_i` onto the 3 vertices of the first triangle and `−F·n·w_j` onto the second, with
`atomicAdd` into the two device force vectors.

**`addDForce` kernel**: apply `K·dx` with `K = k·(n ⊗ n)` for active contacts, needed for
implicit-integration stability. If it turns out to be a correctness rabbit hole it can ship
in a second pass (contact handled explicitly, smaller dt) — noted as a fallback, not a plan.

## Tier 3 — GPU materials — **DECIDED: Ogden, hyperelastic + viscoelastic**

| # | Item | Notes |
|---|---|---|
| T3.0 | Symmetric 3×3 eigensolver (device) | **Ogden is principal-stretch based**, so every element needs an eigendecomposition of `C = FᵀF`. Jacobi sweeps (robust, branch-light, converges fast for 3×3 symmetric) rather than closed-form Cardano, which loses accuracy exactly in the near-degenerate cases below |
| T3.1 | `CudaTetrahedronHyperelasticityFEMForceField` (Ogden) | per-tet: `F` → `C` → eigendecompose → principal stretches λᵢ → isochoric λ̄ᵢ = J^(−1/3)λᵢ → `W = Σₚ (μₚ/αₚ)(λ̄₁^αₚ + λ̄₂^αₚ + λ̄₃^αₚ − 3) + U(J)` → ∂W/∂λ → SPK → nodal forces. N term-pairs (typically 1–3) as a small runtime constant |
| T3.2 | `CudaTetrahedronViscoHyperelasticityFEMForceField` (SLS-Ogden) | T3.1 plus **per-element internal state** on device across frames: one symmetric tensor (6 floats) per Maxwell branch. First-order (1 branch) first, second-order (2 branches) if it falls out |

**The known trap, planned for up front:** the Ogden tangent stiffness contains terms with
`1/(λᵢ² − λⱼ²)`, which blows up when **two principal stretches are equal** — and equal
stretches are not exotic, they happen under uniaxial and hydrostatic loading, i.e. constantly.
Every robust Ogden implementation needs a limit (L'Hôpital) branch for the degenerate and
near-degenerate cases. This is the main correctness risk in Phase B and gets its own targeted
test (Gate 4c) rather than being discovered later as mystery NaNs.

CPU oracles for validation: `TetrahedronHyperelasticityFEMForceField` + `Ogden` for T3.1,
`TetrahedronViscoHyperelasticityFEMForceField` + `SLSOgdenFirstOrder` for T3.2 — both
confirmed present in the install.

Both force fields are per-element embarrassingly parallel: one thread per tetrahedron,
scatter-add into `f`. The work is tensor math and validation, not algorithm design.

---

## Test plan (6 gates, in order)

The discipline mirrors the collision work: nothing counts until a gate passes.

### Gate 1 — contact decoding correctness (bench-level, no SOFA)
New bench leg: run collision → compute forces on GPU → recompute the *same* forces on the
CPU from downloaded contacts → compare. Max relative error < 1e-5. Isolates T1.3/T1.4 math
from all SOFA plumbing.

### Gate 2 — Newton's third law
Reduce the total force over both bodies; must be ≈ 0. One reduction, catches every sign and
scatter error immediately.

### Gate 3 — analytic equilibrium
Blade of known mass resting on fixed tissue: settled penetration must match `d ≈ mg/(k·n)`
within a few percent. Catches magnitude errors that Gate 2 cannot (both sides wrong by the
same factor still sums to zero).

### Gate 4 — CPU oracle parity (Tier 3)

**4a — force parity.** Same tet mesh, same Ogden parameters, same imposed deformation:
compare `addForce` element-wise against SOFA's CPU `TetrahedronHyperelasticityFEMForceField`
with `Ogden`. Max relative error < 1e-5.

**4b — stiffness parity.** Validate `addDForce` independently by finite difference:
`(f(x+εdx) − f(x))/ε ≈ K·dx`. Independent of the CPU oracle, so it catches errors the
oracle and the port might share.

**4c — degenerate-stretch stress test (Ogden-specific).** Drive elements through exactly
the configurations that make `λᵢ² − λⱼ²` vanish: pure uniaxial stretch (λ₂ = λ₃),
hydrostatic (λ₁ = λ₂ = λ₃), and undeformed (all = 1). Assert finite forces, no NaN, and
continuity by sweeping through the degeneracy (ε = 1e−2 … 1e−8) rather than only landing on
it. This is the gate that catches the L'Hôpital branch being wrong.

**4d — viscoelastic time-series parity (T3.2 only).** History-dependent behaviour cannot be
validated from a single frame. Apply a step deformation and hold: compare the whole
**stress-relaxation curve** over N frames against the CPU `SLSOgdenFirstOrder`. Then a creep
test (hold force, watch deformation). Curves must match within tolerance at every sample,
which also proves the per-element device history is persisting and advancing correctly.

### Gate 5 — the zero-transfer gate (the headline result)
New tiny component `GpuResidencyChecker` running at `AnimateEndEvent`, asserting on every
mechanical state:

```
x.isHostValid() == false
v.isHostValid() == false
f.isHostValid() == false
```

If any CPU component touched the state, the flag flips true and the test **names the
offending vector**. Deterministic, in-process, no profiler needed — this is the actual proof
of the project's goal. Cross-checked against a `mycudaMemcpyDeviceToHost` call count.

### Gate 6 — performance
Only after 1–5 pass. Full GPU loop vs the same scene with CPU force fields + CPU penalty
contact, using the existing per-stage profiling + ncu. Report kernel time per stage.

---

## Phasing

- **Phase A (Tier 1)**: T1.1–T1.4 + Gates 1, 2, 3, 5 → *first closed GPU loop*, measurable.
- **Phase B (Tier 3a)**: T3.1 + Gate 4 → realistic hyperelastic tissue on GPU.
- **Phase C (Tier 3b)**: T3.2 + Gate 4 variant → viscoelasticity on GPU.
- Gate 6 after each phase; docs + report at the end, same pattern as the collision work.

## Test scene

`testscenes/dense_collision_benchmark_common.py` already has `generate_tissue_mesh()`
returning tets + fixed indices + surface triangles — so a proper FEM scene needs no new
geometry code. New scene: tet tissue + `EulerImplicitSolver` + `CGLinearSolver` +
`CudaMeshMatrixMass` + FEM force field + `CudaFixedProjectiveConstraint` + blade +
GPU collision + the new contact force field. This is also the first scene in the repo that
actually simulates anything.

## Open questions → see the two asked at plan time.
