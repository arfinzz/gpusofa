# The tissue on the GPU: the whole poke step on the GPU (2026-09-24)

The last CPU part of the GPU poke scene, the tissue, now runs on the GPU. The tissue's
material, mass, fixed base and implicit step, the collision detection, and the constraint
contact with friction all run on the GPU, and nothing of the tissue is copied between the
CPU and the GPU during a step. Only the probe, a 6-DOF rigid body, stays on the CPU. This
report records what was built, how it was compared with SOFA's CPU components on identical
inputs, a bug found in SOFA along the way, and the results. How to use it is in the root
`README.md` (sections 7.9, 9.5 and 9.6).

## 1. What was built

`GpuTissueSolver` is an ODE solver for a `CudaVec3f` tissue with tetrahedra. It takes the
place of this CPU set-up from the poke's CPU scene, stage for stage:

| Stage | Follows, in SOFA v25.12 | On the GPU |
|---|---|---|
| Material | `TetrahedronViscoHyperelasticityFEMForceField::addForce` with `SLSOgdenFirstOrder::deriveSPKTensor`, and `TetrahedronViscoelasticityFEMForceField::addForce` with `MaxwellFirstOrder::deriveCauchyGreenStressTensor`: the same deformation gradient (from the rest shape vectors), stresses, and viscous-strain updates (once per step each) | One thread per tetrahedron, double precision. |
| Stiffness | Both force fields' `updateTangentMatrix` and `applyElasticityTensor`: a 3×3 block per edge, (M + N)·V | The same thread writes its 6 edge blocks; each edge then adds up its tetrahedra's blocks in a fixed order. |
| Mass | `MeshMatrixMass` on tetrahedra, not lumped: ρV/10 per vertex, ρV/20 per edge, gravity on the lumped mass (× 2.5) | Computed once at start-up from the rest positions in double precision. |
| Fixed base | `FixedProjectiveConstraint` through the linear system (`MatrixLinearSystem`'s Dirichlet handling: row and column cleared, diagonal 1) and `projectResponse` on the right-hand side | Read once from the node's `CudaVec3f` constraint (by projecting a vector of ones). |
| Implicit step | `EulerImplicitSolver::solve`: b = h (f + (h + rS) K v − rM M v), projected; A = (1 + h rM) M − h (h + rS) K; v_free = v + dv; x_free = x + h v_free | K v edge by edge as `addDForce` does; A block by block in double precision. |
| Linear solve | `SparseLDLSolver` (exact) | A dense single-precision Cholesky (cuSOLVER), a solve, and one refinement step with the residual in double precision. |

Every sum over elements is a gather in a fixed order, with no atomic additions, so a step
gives the same bits every time.

The constraint contact (`GpuContactConstraintSolver`, `deformableGpuSolver` link) uses the
free motion's Cholesky factor for the compliance and the correction, reads the tissue's free
positions in place, and applies the correction to the tissue on the GPU. Before, with the
tissue on the CPU, each contact step copied the tissue's free positions (21.6 KB) and matrix
values (0.84 MB) to the GPU, factorised the matrix a second time, and copied the correction
back (21.6 KB).

Four smaller changes keep the tissue, and everything else, off the CPU during a step:

- **The logger.** The poke logger needs the surface height under the probe and the most
  squashed tetrahedron. The solver computes both on the GPU after the correction
  (`monitorPosition`, `minVolumeRatio`): 32 bytes instead of the tissue.
- **Bounding boxes.** SOFA's `CollisionPipeline` rebuilds every collision surface's bounding
  box on the CPU each frame (a copy of the surface to the CPU). `GpuCollisionPipeline` builds
  it once for GPU triangle surfaces; the broad phase's `testGpuModelBoxes=false` skips the
  box test for GPU surface pairs; the loop's `computeBoundingBox` is off.
- **Drawing.** SofaCUDA's `CudaVisualModel` draws the tissue surface and reads it only when
  the window is drawn. A visual mapping to an `OglModel`, the usual way, is applied after
  every step and would copy the tissue every step.
- **No force mapping on the surfaces.** In constraint mode the surfaces' mappings get
  `mapForces=false`, so SOFA's probe solver no longer maps the probe surface's (zero) forces
  up with two GPU read-backs per step (section 4.5).

## 2. How it was compared

`compareWithCpu` builds, outside the scene graph, a copy of the CPU set-up from SOFA's own
components (`EulerImplicitSolver`, `SparseLDLSolver`, `MechanicalObject`,
`TetrahedronSetTopologyContainer`, `MeshMatrixMass`, the two viscoelastic force fields and
`FixedProjectiveConstraint`), with the same rest positions, tetrahedra and parameters. At
the start of every step it is given the GPU tissue's positions and velocities and runs SOFA's
own free motion. Then the forces, the system matrix (SOFA's assembled matrix), dv, x_free
and v_free are compared with the GPU's. The copy runs every step, so its viscous strains
advance in step with the GPU's. The same copy also supplies the tissue's linear solver to
the constraint contact's own comparison with SOFA's CPU pipeline.

## 3. A bug in SofaViscoElastic's Ogden

The first comparison agreed at rest to 3e-6, but one step after the tissue started to move
the forces differed by 19% (30% with the viscous branches switched off), on identical
positions. Pinned down step by step:

1. A per-vertex dump of the positions and both sides' forces, and a separate NumPy version of
   the formulas as written in SOFA's source: NumPy agreed with the GPU to 1e-14, and not with
   SOFA.
2. The viscous branches were not the cause: with all viscosity off, the difference stayed
   (30%).
3. SOFA's per-tetrahedron stress (its `stressSPK` output) differed from the formula in every
   tetrahedron, most where the block bulges sideways near its fixed base, which is where C
   is far from diagonal.
4. `SLSOgdenFirstOrder` computes C^(α/2−1) from an eigen-decomposition:
   `Eigen::SelfAdjointEigenSolver<EigenMatrix> Vect(CEigen, true)`. In Eigen 3 (3.4.0 here)
   the second argument is `int options`, and `true` becomes 1, which does not contain
   `ComputeEigenvectors` (0x80). Eigen computes the eigenvalues only (correctly, in ascending
   order), and `eigenvectors()` returns its work matrix: C's lower triangle divided by its
   largest entry. (Eigen checks its options with an assertion, which SOFA's build compiles
   out.)
5. Rebuilt that way in NumPy, SOFA's forces were matched to 1e-14 at every step, and its
   stress diagonal to the printed precision.

So SofaViscoElastic's Ogden pairs the right eigenvalues with the wrong directions. Its
stress is right only at rest (C = I). Anywhere else it is wrong at first order in the strain,
and the error depends on how the tissue is turned in space. For C = I + 2E with small E, the
matrix it uses in place of C^p (p = α/2 − 1) is about
I + 2(E + diag E) − 4 max(E_ii) I + 2p diag(e₁, e₂, e₃), with e₁ ≤ e₂ ≤ e₃ the principal
strains, instead of I + 2p E (checked numerically: the difference from this formula shrinks
with the square of the strain, the difference from the true C^p only linearly).

SOFA's core `Ogden` material replaced the same call on 17/11/2025, with the comment
"incorrect eigenvector computation for 3x3 matrices". SofaViscoElastic v25.12 still has it,
in `SLSOgdenFirstOrder.h` (lines 99 and 151) and `SLSOgdenSecondOrder.h` (lines 102 and 154).
The fix is one argument: `Eigen::ComputeEigenvectors` instead of `true`.

The GPU tissue offers both: `ogdenEigenvectors="sofa"` (default) builds C^p exactly as
SofaViscoElastic does, in the stress and in the stiffness, so that the GPU and CPU scenes
have the same material; `ogdenEigenvectors="exact"` uses the true eigenvectors.

## 4. Results

All runs: the poke of `README.md` section 10.3 (1,800 nodes, 8,232 tetrahedra, 820 steps of
10 ms), on the GTX 1650 Ti in WSL2, 2026-09-24. Plot: `gpu_tissue_20260924.png`.

### 4.1 The tissue step against SOFA's CPU components

A whole poke with `SOFA_POKE_COMPARE=1 SOFA_POKE_COMPARE_EVERY=10`: the CPU copy runs every
step and is compared every 10th step (82 steps, contact steps included). Differences are the
largest over all vertices, divided by the largest value on SOFA's side:

| Quantity | Largest difference | Median |
|---|---:|---:|
| Forces (material + gravity) | 1.2e-14 | 6.9e-15 |
| System matrix A (SOFA's assembled matrix) | 1.7e-16 | 1.1e-16 |
| Velocity change dv | 2.9e-9 | 1.2e-10 |
| Free positions x_free (absolute) | 1.9 nm | 1.8 nm |
| Free velocities v_free (absolute) | 2.8e-8 m/s | 5.7e-11 m/s |

The forces and the matrix agree to the last bits of double precision. dv agrees to 3e-9: the
Cholesky is in single precision, and one refinement step (the residual in double precision
and a second solve) brings the solution most of the way to double precision. The free
positions differ by 1.9 nm because the GPU state stores positions in single precision (about
2 nm at the block's 3 cm half-width).

### 4.2 The contact step, with the tissue on the GPU

The same run compared every 10th contact step (39 steps) with SOFA's CPU constraint pipeline
on the same contacts (`GpuContactConstraintSolver`'s `compareWithCpu`; the tissue's linear
solver comes from the CPU copy above):

| Stage | Largest difference | Median |
|---|---:|---:|
| Constraint rows | 2.3e-7 | 1.7e-7 |
| Free violations | 1.2e-9 m | 5.6e-10 m |
| Compliance W (relative) | 8.7e-6 | 4.7e-6 |
| Multipliers (relative) | 2.9e-5 | 1.3e-5 |
| Multipliers, SOFA's solver on the GPU's W (relative) | 2.9e-6 | 1.2e-6 |
| Tissue correction | 13 nm (of up to 3.2 mm) | 5.4 nm |
| Probe correction | 1.1e-9 m | 2.3e-10 m |
| Gauss-Seidel sweeps equal | 39 of 39 | |
| Contact force on the probe | 3 µN (of 0.286 N) | |

These are the same as with the tissue on the CPU (`gpu_constraint_contact_20260924.md`): the
GPU tissue's factor serves the contact as well as SOFA's own direct solver does. SOFA's CPU
pipeline took 8.7 s per compared step for its compliance alone.

### 4.3 Whole pokes

| Result | Everything on the GPU | GPU scene with the CPU tissue | SOFA's CPU scene | Everything on the GPU, exact Ogden |
|---|---:|---:|---:|---:|
| Contact starts | 2.090 s | 2.090 s | 2.090 s | 2.070 s |
| Settled surface (t = 1 s) | −0.768 mm | −0.768 mm | −0.768 mm | −0.688 mm |
| Peak force, at the end of the press | 0.289143 N | 0.289146 N | 0.289689 N | 0.410948 N |
| Force at the start → end of the hold | 0.256445 → 0.206386 N | 0.256343 → 0.206390 N | 0.256982 → 0.207861 N | 0.378982 → 0.346261 N |
| Relaxed during the 1 s hold | 19.52% | 19.49% | 19.11% | 8.63% |
| Most squashed element (volume / rest volume) | 0.7611 | 0.7614 | 0.7530 | 0.6627 |
| Closest gap, probe tip to the surface node under it | 0.5025 mm | 0.5025 mm | 0.5025 mm | 0.5013 mm |
| Surface at the end, compared with its settled height | −1.140 mm | −1.140 mm | −1.081 mm | −0.305 mm |

Largest differences from "everything on the GPU", over the whole poke:

| Run | Force on the probe | Probe tip | Surface under the tip | Smallest volume ratio |
|---|---:|---:|---:|---:|
| GPU scene with the CPU tissue | 0.12 mN (0.041% of the peak) | 0.06 µm | 10 µm | 0.001 |
| SOFA's CPU scene | 1.7 mN (0.60%) | 0.9 µm | 77 µm | 0.013 |
| Exact Ogden | 140 mN (48%) | 70 µm | 1.8 mm | 0.15 |

- **The tissue's move to the GPU changes nothing.** The GPU scene gives the same poke with its
  tissue on the CPU (SOFA's components) or on the GPU, to 0.04% of the peak force.
- **SOFA's CPU scene: 0.6%, as before.** Its `LocalMinDistance` keeps different contacts
  (a median of 23 against the GPU's 533); both hold the probe out the same way.
- **The exact Ogden is a different tissue.** With SofaViscoElastic's version the tissue is
  softer: it settles further under its own weight, gives 30% less force at the end of the
  press, relaxes more than twice as much during the hold, and creeps back more slowly.

### 4.4 Time

Wall time per step (mean; "contact" = the 383 steps with contact rows). The first row is the
finished scene (a final run with stage timing, `SOFA_POKE_MEASURE_TIMES=1`); the others are
from the batch, before the last change of section 4.5, which saved about 1 ms per step:

| Run | Without contact | With contact | Whole poke |
|---|---:|---:|---:|
| Everything on the GPU (finished scene) | 36.1 ms | 60.7 ms | **47.6 ms** |
| Everything on the GPU (batch) | 36.5 ms | 63.0 ms | 48.9 ms |
| Everything on the GPU, exact Ogden (batch) | 37.6 ms | 68.9 ms | 53.7 ms |
| GPU scene with the CPU tissue | 275.3 ms | 362.1 ms | 315.9 ms |
| SOFA's CPU scene | 271.4 ms | 375.8 ms | 320.1 ms |

The finished scene runs the whole poke 6.7 times faster than SOFA's CPU scene: 7.5 times
without contact and 6.2 times with it. Its forces agree with the batch run's to 0.016% of the
peak (run-to-run noise from the contact correction's atomic additions).

GPU time per step in the final run:

| Stage | Mean | Of which |
|---|---:|---|
| Tissue step (every step) | 32.1 ms | material 1.2, assembly 0.8, Cholesky 27.4, solve and refinement 2.7 |
| Contact step, 1,200 rows or more (300 steps) | 28.9 ms | rows 0.8, compliance 4.2, Gauss-Seidel 22.4, correction 1.5 |
| Contact step, all 383 contact steps | 24.9 ms | rows 0.8, factorisation 0, compliance 4.0, Gauss-Seidel 18.6, correction 1.5 |

For comparison: SOFA's CPU tissue step (the CPU copy's free motion) takes a median of 259 ms,
and with the tissue on the CPU the GPU contact step took 81.6 ms at 1,200 rows or more, 39 ms
of it for its own Cholesky of the tissue matrix. The dense Cholesky is now the largest single
cost (27.4 of about 60 ms at contact); a sparse factorisation would cut it.

The first contact step used to take 863 ms, 823 ms of it in the compliance stage: cuBLAS
loading its kernels at its first call. With the warm-up (section 5) it takes 56.5 ms, and the
slowest step of the poke is 87.2 ms.

### 4.5 What still crosses between the CPU and the GPU

SofaCUDA prints every copy of a GPU state vector, with its call stack, when run with
`CUDA_VERBOSE=4`. Over the first 240 steps of the poke (30 of them with contact):

| Steps | CPU → GPU | GPU → CPU | What |
|---|---:|---:|---|
| Step 1 | 15 | 6 | Start-up: initial state uploads, the fixed DOFs (found once), each GPU surface's bounding tree (built once), the scene's first bounding box. |
| Steps 2 to 240 | **0** | **0** | Nothing, with and without contact. |

The first trace, before one last change, found 2 copies per step: SOFA's probe solver mapping
the probe surface's forces up to the probe (`RigidMapping::applyJT` in its force and
matrix-vector passes). SofaCUDA sums them on the GPU and reads back the per-block partial
sums, about 100 bytes each time, and each read makes the CPU wait for the GPU. The forces are
always zero: in constraint mode nothing acts on the surfaces through forces. So the scene now
sets `mapForces=false` on both surface mappings. SOFA's `FreeMotionAnimationLoop` still
propagates positions, velocities and corrections to them (it ignores that flag when it
propagates); the run was bit for bit identical to one without the change until contact
began, then within the usual run-to-run noise.

What SofaCUDA's trace can't see is the plugin's own reads of a few numbers per step. The
tissue solver reads whether the Cholesky succeeded (4 bytes) and the monitor values
(32 bytes); on a contact step the contact solver reads the contact and vertex counts
(16 bytes), the Gauss-Seidel's sweep count and error, and the probe's 6-DOF correction
(56 bytes). Each read makes the CPU wait for the GPU once; together they are well under
200 bytes per step.

The probe's surface costs no copy either: SofaCUDA computes `RigidMapping<Rigid3d,
CudaVec3f>` on the GPU, with the probe's rotation and position passed as kernel arguments.

## 5. Problems found and fixed on the way

- **SofaViscoElastic's Ogden computes no eigenvectors** (section 3). Reproduced on the GPU as
  the default, so the scenes stay comparable; `ogdenEigenvectors="exact"` for the real
  material.
- **The first contact step stalled for 0.8 s.** 823 ms of it went to the compliance stage:
  cuBLAS loading its kernels at its first call. The constraint workspace now runs each
  cuBLAS and cuSOLVER call once, on a small identity system, when it is created.
- **Bounding boxes copied the GPU surfaces to the CPU every frame** (SOFA's
  `CollisionPipeline` and the animation loop's `computeBoundingBox`): `GpuCollisionPipeline`,
  `testGpuModelBoxes=false`, `computeBoundingBox=false`.
- **SOFA mapped the probe surface's zero forces up to the probe twice per step**, each time
  reading GPU partial sums back: `mapForces=false` on the surfaces in constraint mode
  (section 4.5).
- **A visual mapping copies the tissue after every step**, even in a batch run: SOFA's
  animation loop applies visual mappings each step. `CudaVisualModel` draws from the GPU
  state and reads it only when the window is drawn.
- **Resizing a SofaCUDA vector needs OpenGL.** The GPU state's free-motion vectors are resized
  from the plugin, which pulls in SofaCUDA's OpenGL-buffer code: the plugin now links
  `Sofa.GL` (GLEW).

## 6. Limits

- One material family: `SLSOgdenFirstOrder`, with or without a `MaxwellFirstOrder` branch, on
  linear tetrahedra. The Ogden stiffness uses SOFA's formula in both modes; even with the true
  eigenvectors that formula is exact only for stretch changes lined up with the current
  stretch. That changes each implicit step a little (the path), not the state the tissue
  comes to rest in, which depends only on the stress.
- The system matrix is dense: 117 MB here, about 10,000 nodes at most on a 4 GB GPU, and the
  Cholesky is most of the step's time. A sparse factorisation is the next step for speed and
  size.
- Only fixed DOFs; the CPU comparison copy only supports fully fixed vertices.
- The probe (one rigid body) stays on the CPU, with its own implicit solver on a 6×6 system.
- `ogdenEigenvectors="exact"` has no CPU counterpart in SOFA v25.12 to compare with (the
  formula was checked against NumPy to 1e-14, and a whole poke runs stably).
- Not yet tried in SOFA's window (only batch runs).
