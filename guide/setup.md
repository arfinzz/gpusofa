# Setup

End-to-end instructions for running, building, profiling, and tuning the
GPU SOFA collision pipeline. Read top to bottom for a fresh machine; jump
to a section if you already know the lay of the land.

Last refreshed: 2026-05-25 (post Phase 12 cross-model wiring).

---

## 1. The big picture

You have **two checkouts** of this project, and they do not auto-sync:

| Checkout | Purpose | Path |
|---|---|---|
| Windows | Edit-of-truth; this is where git lives | `C:\Users\arfin\Desktop\GPU SOFA` |
| WSL2 (`wsl-gpu-proj`) | Build + run + profile (CUDA toolchain lives here) | `/home/arfin/gpu-sofa` |
| WSL view of Windows | Read-only mirror, used to rsync changes | `/mnt/c/Users/arfin/Desktop/GPU SOFA` |

The workflow loop:

```text
1. Edit on Windows (any editor, VS Code recommended)
2. Sync changed files into the WSL checkout (rsync or wsl cp)
3. Build in WSL (CMake-generated makefile)
4. Run scenes / benchmarks in WSL
5. (Optional) Copy CSVs and Nsight reports back to Windows for inspection
6. Commit on Windows
```

This split exists because:

- The CUDA toolchain (nvcc, Nsight tools, CUDA runtime, cuBLAS, etc.) is
  installed inside WSL.
- The SOFA install is also inside WSL (`/opt/sofa/install/v25.12`).
- Windows-side filesystem performance on `/mnt/c/...` is much slower than
  the WSL ext4 filesystem for builds. Keeping the build directory inside
  `/home/arfin/...` avoids the slowdown.
- The IDE experience on Windows is better for casual editing.

---

## 2. Hardware and software baseline

The numbers in `guide/plan.md` and `reports/...` were captured on:

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti (Turing, sm_75) |
| Driver | 526.98 (Windows host) |
| WSL distro | `wsl-gpu-proj` (Ubuntu 24.04 LTS) |
| Linux kernel | 6.6.87.2-microsoft-standard-WSL2 |
| SOFA | 25.12 at `/opt/sofa/install/v25.12` |
| CUDA toolkit | the one bundled in the SOFA install |
| Compiler | `gcc 13.3.0`, `nvcc` matching SOFA's expected toolchain |

If your machine has a different GPU, the kernels will still work but the
"this regressed on GTX 1650 Ti" decisions in `guide/plan.md` may not
apply — re-measure before assuming.

---

## 3. Zero-to-running walkthrough

The following sequence takes a fresh Windows machine + WSL distro and
gets the FBP fast path running. Skip steps that are already done on your
machine.

### 3.1  Confirm WSL distro and GPU access

From PowerShell on Windows:

```powershell
wsl -d wsl-gpu-proj -- uname -a
wsl -d wsl-gpu-proj -- nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Expected:

```text
Linux <hostname> 6.6.87.2-microsoft-standard-WSL2 ... GNU/Linux
NVIDIA GeForce GTX 1650 Ti, 526.98
```

### 3.2  Sync Windows source into WSL

If the WSL checkout doesn't exist yet:

```powershell
wsl -d wsl-gpu-proj -- mkdir -p /home/arfin/gpu-sofa
wsl -d wsl-gpu-proj -- rsync -a --exclude '.git' --exclude 'output' --exclude 'reports' --exclude 'SofaGpuCollision/build*' "/mnt/c/Users/arfin/Desktop/GPU SOFA/" /home/arfin/gpu-sofa/
```

For ongoing development, sync only what changed:

```powershell
wsl -d wsl-gpu-proj -- rsync -a /mnt/c/Users/arfin/Desktop/'GPU SOFA'/SofaGpuCollision/src/ /home/arfin/gpu-sofa/SofaGpuCollision/src/
wsl -d wsl-gpu-proj -- rsync -a /mnt/c/Users/arfin/Desktop/'GPU SOFA'/test_gpu_*.py /home/arfin/gpu-sofa/
wsl -d wsl-gpu-proj -- rsync -a /mnt/c/Users/arfin/Desktop/'GPU SOFA'/scripts/ /home/arfin/gpu-sofa/scripts/
wsl -d wsl-gpu-proj -- rsync -a /mnt/c/Users/arfin/Desktop/'GPU SOFA'/guide/ /home/arfin/gpu-sofa/guide/
```

A narrow per-file sync also works:

```powershell
wsl -d wsl-gpu-proj -- cp "/mnt/c/Users/arfin/Desktop/GPU SOFA/SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp" /home/arfin/gpu-sofa/SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp
```

### 3.3  First-time CMake configure (only once)

```powershell
wsl -d wsl-gpu-proj -- bash -c 'mkdir -p /home/arfin/gpu-sofa/SofaGpuCollision/build-profile && cd /home/arfin/gpu-sofa/SofaGpuCollision/build-profile && cmake -DSOFAGPUCOLLISION_ENABLE_CUDA=ON ..'
```

Expected output ends with a line like `-- Build files have been written to: /home/arfin/gpu-sofa/SofaGpuCollision/build-profile`.

### 3.4  Build

```powershell
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa/SofaGpuCollision/build-profile -- make -j2
```

Why `-j2`: the WSL2 Linux kernel + nvcc combination has memory-pressure
issues with higher parallelism on a typical 16 GB laptop. Bump to `-j4` if
you have plenty of RAM.

Expected end-of-output:

```text
[ 80%] Built target SofaGpuCollision
[100%] Built target SofaGpuCollisionDenseGridBackendBench
```

Build artifacts:

- `/home/arfin/gpu-sofa/SofaGpuCollision/build-profile/libSofaGpuCollision.so` — plugin shared library
- `/home/arfin/gpu-sofa/SofaGpuCollision/build-profile/SofaGpuCollisionDenseGridBackendBench` — standalone backend bench

Known non-blocking warnings (inherited from SOFA core, not project bugs):

- `RegisterObject is deprecated...` (deprecation path that SOFA v25.12 has scheduled)
- `noopKernel unused` (placeholder in the CUDA backend)

### 3.5  First benchmark

```powershell
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa -- bash scripts/run_fbp_smoke_test_wsl.sh
```

Expected ~5-10 second runtime ending with:

```text
[INFO]    [BatchGUI] 20 iterations done in 0.027... s ( 7XX FPS ).
```

The logs land in `/home/arfin/gpu-sofa/output/benchmark_logs/fbp_smoke_one_tissue_<timestamp>/`
with two files:

- `gpu_one_tissue_one_blade_dense_grid_benchmark_timings.csv` — one row per frame
- `gpu_one_tissue_one_blade_dense_grid_benchmark_summary.txt` — `avg_*` aggregates

### 3.6  Inspect the summary

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/peek_fbp_summary.sh /home/arfin/gpu-sofa/output/benchmark_logs/<run-dir>
```

You should see:

```text
collision_vertex_count=6569
collision_element_count=12812
avg_fps≈700-800
avg_narrow_wall_ms≈0.6-0.7
avg_narrow_kernel_ms≈0.5
avg_kernel_launch_count=7
avg_cuda_memset_count=1
avg_host_to_device_bytes=0
avg_device_to_host_bytes=0
```

If numbers are wildly off, see §8 (troubleshooting).

---

## 4. Scene catalogue

| Scene | What it tests | Expected output |
|---|---|---|
| `test_cpu_one_tissue_one_blade_benchmark.py` | CPU baseline | ~130 FPS, ~7-8 ms narrow wall |
| `test_gpu_one_tissue_one_blade_dense_grid_benchmark.py` | GPU dense-grid (exact-contact or FBP via env var) | Fast path: 633-775 FPS, ~0.7 ms narrow wall |
| `test_cpu_large_tissue_blade_benchmark.py` | CPU baseline, larger mesh | ~40 FPS |
| `test_gpu_large_tissue_blade_dense_grid_benchmark.py` | GPU dense-grid, larger mesh | 25-90 FPS depending on power state |
| `test_gpu_self_collision_vertex_triangle_smoke.py` | **Phase 12 self-collision** | 1300-1400 FPS, 2700 VertexFace contacts |
| `test_gpu_cross_model_vertex_triangle_smoke.py` | **Phase 12 cross-model** | 1500-2000 FPS, ~250 VertexFace contacts |

Validation-only scenes (not for production benchmarking):

| Scene | Purpose |
|---|---|
| `test_gpu_dense_phase45_validation.py` | Phase 4 + 5 caching/readback regressions |
| `test_gpu_phase5_overlap_validation.py` | Phase 5 overlap correctness |
| `test_gpu_tissue_phase45_validation.py` | Tissue-only Phase 4 + 5 |

Common scene helpers in `dense_collision_benchmark_common.py`:

- `generate_tissue_surface_grid(nx, nz, sx, sz, y)` — flat NxN triangle grid
- `generate_tissue_mesh(...)` — full 3D tetrahedralized tissue with surface skin
- `create_blade_geometry(length, height, thickness)` — 12-triangle box
- `create_subdivided_blade_geometry(...)` — finer blade for stress tests
- `build_blade_grid(rows, cols, spacing_x, spacing_z, start_y)` — array of blades for scaling studies
- `default_benchmark_log_dir(base_dir)` — resolves `SOFA_BENCHMARK_LOG_DIR` env var
- `env_flag(name, default)` — string-to-bool env var helper

---

## 5. Launcher scripts

Every benchmark scene has a wrapper script in `scripts/` that sets the
`SOFA_PLUGIN_PATH`, `LD_LIBRARY_PATH`, env-var defaults, and invokes
`runSofa -g batch`.

| Script | Scene | Default mode |
|---|---|---|
| `run_one_tissue_one_blade_benchmarks_wsl.sh` | CPU + GPU one-tissue/one-blade (back-to-back) | exact-contact, 60 steps |
| `run_large_tissue_blade_benchmarks_wsl.sh` | CPU + GPU large-tissue/blade | exact-contact, 60 steps |
| `run_scene_size_scaling_benchmarks_wsl.sh` | Scaling study across multiple sizes | exact-contact |
| `run_backend_dense_grid_benchmark_wsl_gpu_proj.sh` | Standalone backend bench (no SOFA scene) | direct kernel-only |
| `run_gpu_kernel_profile_wsl.sh` | NVTX-annotated profiling run | event timings on |
| `run_nsight_collision_profile_wsl_gpu_proj.sh` | Nsight Compute + Nsight Systems capture | full profiling |
| `run_fbp_smoke_test_wsl.sh` | **One-tissue/one-blade with FBP enabled** (Phase 11) | FBP, detection-only, readback off, tool-active-cell default-on |
| `run_fbp_large_tissue_wsl.sh` | **Large-tissue/subdivided-blade FBP** (Phase 15 large A/B) | FBP, detection-only, tool-active-cell default-on |
| `run_vertex_triangle_smoke_wsl.sh` | **V-t self-collision** (Phase 12) | FBP + v-t self, readback on |
| `run_cross_model_vt_smoke_wsl.sh` | **V-t cross-model** (Phase 12 cross-model) | FBP + v-t cross, readback on |
| `run_nsight_fbp_profile_wsl.sh` | **Nsight Compute on FBP + v-t kernels** | all three scenes, focused metric set |

Override the log directory or step count per run:

```powershell
wsl -d wsl-gpu-proj -- bash -c 'SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/my_run SOFA_BENCHMARK_STEPS=100 bash /home/arfin/gpu-sofa/scripts/run_fbp_smoke_test_wsl.sh'
```

Helper utility:

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/peek_fbp_summary.sh <run-dir>
```

prints the headline `avg_*` lines from a run's summary file.

---

## 6. Environment variable reference

These all default sensibly. Set them only to override.

### 6.1  Build / install

| Variable | Default | Purpose |
|---|---|---|
| `SOFA_ROOT` | `/opt/sofa/install/v25.12` | Root of the SOFA install. Used by every launcher script |
| `SOFA_GPU_COLLISION_LIB` | `${SOFA_ROOT}/plugins/SofaGpuCollision/lib/libSofaGpuCollision.so` (production) or `${REPO_DIR}/SofaGpuCollision/build-profile/libSofaGpuCollision.so` (smoke scripts) | The plugin to load |

### 6.2  Run control

| Variable | Default | Purpose |
|---|---|---|
| `SOFA_BENCHMARK_LOG_DIR` | `output/benchmark_logs/<scene>_<timestamp>` | Where the run writes its CSV + summary |
| `SOFA_BENCHMARK_STEPS` | 60 (production) / 20 (smoke scripts) | Number of frames to simulate |
| `SOFA_BENCHMARK_LABEL_SUFFIX` | empty (auto-timestamped in scripts) | String appended to run labels |
| `SOFA_LARGE_WARMUP_STEPS` | 10 | Frames to discard at start of stats |
| `SOFA_NSYS_STEPS` | 6 | NVTX capture-range step count |
| `SOFA_NSYS_FULL_STEPS` | 4 | Full-run Nsight Systems step count |
| `SOFA_NCU_STEPS` | 4 | Nsight Compute step count |
| `SOFA_NCU_LAUNCH_COUNT` | 24 | Max kernel launches to profile in NCU |

### 6.3  Backend tuning (legacy + dense-grid)

| Variable | Default | Purpose |
|---|---|---|
| `SOFA_GPU_DETAILED_PROFILING` | 0 | Enable per-stage CUDA event timings (adds syncs) |
| `SOFA_COPY_CONTACTS_TO_HOST` | 0 (smoke) / 1 (response-enabled scenes) | Publish contacts to SOFA `DetectionOutput` |
| `SOFA_DEDUPLICATE_PAIRS` | 1 | Master dedupe toggle |
| `SOFA_USE_GPU_HASH_DEDUPE` | 1 | GPU `atomicCAS` hash dedupe (vs legacy host sort/unique) |
| `SOFA_USE_PINNED_HOST_STAGING` | 1 | Pinned host buffers for fallback uploads |
| `SOFA_CANONICAL_PAIR_EMISSION` | 0 | Canonical-cell pair emission (legacy) |
| `SOFA_USE_INDEXED_DENSE_GRID_INPUT` | 1 | Use indexed surface input (the fast path) |
| `SOFA_USE_DIRECT_DEVICE_POSITIONS` | 1 | Use SOFA CUDA `deviceRead()` pointer directly |
| `SOFA_READ_COUNTERS_WHEN_CONTACTS_STAY_ON_DEVICE` | 0 | D2H counter reads in detection-only mode |
| `SOFA_COMPUTE_DEVICE_CONTACTS_WHEN_CONTACTS_STAY_ON_DEVICE` | 0 | Run exact-contact kernel even when contacts stay device-side |
| `SOFA_COMPACT_ACTIVE_CELLS` | 0 | Experimental active-cell compaction via a separate full-grid scan — **regressed**, superseded by `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION` |
| `SOFA_BATCH_TRIANGLE_INSERT` | 0 | Experimental fused tissue+tool insertion (regressed). Mutually exclusive with tool-active-cell generation |
| `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION` | **1** | **Phase 15 — DEFAULT ON (guide/plan.md §5.15).** Generate candidate pairs over tool-occupied (mixed) cells only — list built during the tool insert, no separate scan. **Measured 4.3× FPS (221 → 943) and 38× faster generation (300 → 7.9 µs) on one-tissue, 1.08× on large-tissue, contact counts bit-identical, never a regression.** Set to 0 only to A/B against the old all-cells path |

### 6.4  Feature-based proximity (Phase 11+12)

| Variable | Default | Purpose |
|---|---|---|
| `SOFA_USE_FEATURE_BASED_PROXIMITY` | 0 | Replace SAT exact-contact with VF + EE closest-feature kernel |
| `SOFA_USE_VERTEX_TRIANGLE_PROXIMITY` | 0 | Route self-collision and cross-model `(point, triangle)` pairs to the v-t kernel |
| `SOFA_PROXIMITY_COMPUTE_BARYCENTRICS` | 1 | Populate `ProximityContact.firstBarycentrics` / `secondBarycentrics` |
| `SOFA_PROXIMITY_READ_CONTACT_COUNTER` | 0 (production) / 1 (smoke scripts) | One batched D2H of (contactCount, overflowCount, vf, fv, ee). Adds one cudaDeviceSynchronize per frame |
| `SOFA_PROXIMITY_KEEP_CONTACTS_ON_DEVICE` | 1 | Skip copying `ProximityContact` array to host |
| `SOFA_PROXIMITY_MAX_CONTACTS` | 1000000 | Capacity of the device-side proximity contact buffer |

Not env-bridged (set on the `GpuCollisionNarrowPhase` component directly in scene Python):

| Data field | Default | Purpose |
|---|---|---|
| `proximityCounterReadbackInterval` | 0 | When > 0, read counters every Nth frame (sampled mode) |

### 6.5  Output paths

| Variable | Default | Purpose |
|---|---|---|
| `SOFA_BENCHMARK_LOG_DIR` | `output/benchmark_logs` | CSV + summary destination |
| `SOFA_SCALE_LOG_DIR` | `output/scene_size_scaling` | Scaling-study output |
| `SOFA_PROFILE_ROOT` | `output/benchmark_logs/<profile-run>` | Nsight profiling root |
| `SOFA_BACKEND_PROFILE_ROOT` | `output/backend_profile_<timestamp>` | Backend-bench profiling root |

`output/` is `.gitignore`d (except `output/README.md`).

---

## 7. Common workflows

### 7.1  Production detection-only run (tri-tri FBP fast path)

Use this when you want the headline FPS number.

```powershell
wsl -d wsl-gpu-proj -- bash -c 'export SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/my_tri_tri_fast; export SOFA_PROXIMITY_READ_CONTACT_COUNTER=0; bash /home/arfin/gpu-sofa/scripts/run_fbp_smoke_test_wsl.sh'
```

Expected: **~900-1000 FPS** (with Phase 15 tool-active-cell generation
default-on), ~0.5 ms narrow wall, 0 H2D + 0 D2H bytes. (Was ~700-800 FPS
before Phase 15.)

### 7.2  Validation mode (visible contact counts)

Use this when you want to see how many contacts the kernel emits.

```powershell
wsl -d wsl-gpu-proj -- bash -c 'export SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/my_tri_tri_validation; export SOFA_PROXIMITY_READ_CONTACT_COUNTER=1; bash /home/arfin/gpu-sofa/scripts/run_fbp_smoke_test_wsl.sh'
```

Expected: ~400-500 FPS, ~1.5 ms narrow wall, 56 EE contacts.

### 7.3  V-t self-collision

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_vertex_triangle_smoke_wsl.sh
```

Expected: ~1300-1400 FPS, 2700 VertexFace contacts.

### 7.4  V-t cross-model

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_cross_model_vt_smoke_wsl.sh
```

Expected: ~1500-2000 FPS, 254 VertexFace contacts.

### 7.5  Long profiling run with sampled counter readback

In your scene Python:

```python
root.addObject(
    "GpuCollisionNarrowPhase",
    ...,
    useFeatureBasedProximity=True,
    proximityReadContactCounter=False,           # off by default
    proximityCounterReadbackInterval=60,          # read every 60th frame
)
```

This gives you VF/FV/EE samples ~1×/sec at 60 FPS while keeping the
remaining frames on the fast path.

### 7.6  Nsight Compute capture targeting the new kernels

```powershell
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa -- bash scripts/run_nsight_collision_profile_wsl_gpu_proj.sh
```

Produces `output/benchmark_logs/<profile-run>/nsight/gpu_collision_ncu.ncu-rep`.
Open in `ncu-ui` (Windows host) or `ncu --import` (WSL CLI):

```powershell
wsl -d wsl-gpu-proj -- ncu --import <ncu-rep> --csv --page raw --metrics gpu__time_duration.sum,lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
```

The Nsight Systems sibling produces a `.nsys-rep` timeline; the script also
runs a full-run fallback in case the NVTX capture-range exits without a
report.

### 7.7  A/B protocol for tool-active-cell generation (§5.15, now default-on)

`SOFA_USE_TOOL_ACTIVE_CELL_GENERATION` is **implemented and default-on**.
This protocol re-measures it against the old all-cells generation (set the
flag to 0 for baseline) **on the same cool GPU, back to back** — useful for
re-validating on a new scene or after a kernel change. The GTX 1650 Ti has
overturned plausible optimizations before (`compactActiveCells` regressed),
so always measure on a cool GPU.

```powershell
# 0. Confirm the GPU is cool before starting (throttling invalidates A/B).
wsl -d wsl-gpu-proj -- nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv,noheader

# A. Baseline: current all-cells generation
wsl -d wsl-gpu-proj -- bash -c 'export SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/ab_gen_baseline; export SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=0; bash /home/arfin/gpu-sofa/scripts/run_fbp_smoke_test_wsl.sh'

# B. New: tool-active-cell generation
wsl -d wsl-gpu-proj -- bash -c 'export SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/ab_gen_active; export SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1; bash /home/arfin/gpu-sofa/scripts/run_fbp_smoke_test_wsl.sh'

# Compare the headline lines
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/peek_fbp_summary.sh /home/arfin/gpu-sofa/output/benchmark_logs/ab_gen_baseline
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/peek_fbp_summary.sh /home/arfin/gpu-sofa/output/benchmark_logs/ab_gen_active
```

What to look for (measured 2026-05-25 — these are the actual results, not
targets):

- `avg_narrow_wall_ms` dropped 4.00 → 0.56 ms (one-tissue fast path).
- `avg_fps` rose 221 → 943 (one-tissue), 108 → 116 (large-tissue).
- `avg_narrow_output_contact_count` (validation mode,
  `SOFA_PROXIMITY_READ_CONTACT_COUNTER=1`) was **bit-identical** between
  off/on: 56 (one-tissue), 8018 (large-tissue). This is the key correctness
  check — the optimization changes *which cells are scanned*, not *which
  contacts emit*. If the count changes, the active-cell list is dropping
  real candidate pairs.
- `avg_narrow_overflow_count` stayed 0.

For a large-tissue A/B use `scripts/run_fbp_large_tissue_wsl.sh` instead of
the one-tissue smoke script.

Nsight confirmation — the generation kernel grid dropped from 32 768 to
1 024 blocks (and the kernel from ~300 µs to 7.9 µs):

```powershell
wsl -d wsl-gpu-proj -- bash -c 'export SOFA_PROFILE_ROOT=/home/arfin/gpu-sofa/output/benchmark_logs/ab_gen_nsight; export SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1; bash /home/arfin/gpu-sofa/scripts/run_nsight_fbp_profile_wsl.sh'
# launch__grid_size for generateActiveDenseGridUniqueCandidatePairsKernel = 1024
```

### 7.8  Event-caching optimization (§5.16, landed)

Workspace-cached broad-cull events are **always on**. They removed
~40-80 µs/frame of `cudaEventCreate`/`cudaEventDestroy` churn with no change
to contact counts, byte transfers, or launch counts.

### 7.9  Large-tissue FBP run

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_fbp_large_tissue_wsl.sh
```

Expected (validation mode): ~110-120 FPS, 8018 contacts (5397 VF / 880 FV /
1741 EE), overflow 0. The subdivided blade produces ~322 560 candidate
pairs, so this scene exercises the FBP grid-stride loop (§5.17 in plan.md).

---

## 8. Troubleshooting

### Build errors

**`error: too few arguments in function call`** — A backend signature
changed and a call site wasn't updated. Look at the error line; the
recent change history is in `guide/plan.md` Phase notes.

**`error: RegisterObject is deprecated`** (warning) — Pre-existing SOFA
v25.12 warning. Not a project bug. Will need to migrate to
`ObjectRegistrationData` before SOFA v26.0.

**`ld: cannot find -lSofaCUDA`** — `SOFA_ROOT` not set, or `SofaCUDA` is
not in the SOFA install you point at. Check `${SOFA_ROOT}/plugins/SofaCUDA/lib/libSofaCUDA.so`.

**`nvcc fatal: unsupported gpu architecture`** — The CUDA toolchain
inside SOFA's install does not match the GPU. Reinstall SOFA with a
matching CUDA toolkit, or override `CMAKE_CUDA_ARCHITECTURES=75` for
sm_75 (the GTX 1650 Ti).

### Runtime errors

**`[PluginManager] Plugin not found: SofaGpuCollision`** — Set
`SOFA_GPU_COLLISION_LIB` to your built `.so`. The smoke scripts already
default this; if you're running `runSofa` directly, pass
`-l /path/to/libSofaGpuCollision.so`.

**`[GpuCollisionNarrowPhase] CUDA runtime detected. Kernel integration
points are compiled and ready for implementation.`** — That's the
backend probe success message. Not an error.

**`avg_fps` is 1/10th of the expected number** — Likely thermal
throttling. GTX 1650 Ti laptops drop core clock from 1485 MHz to
300-500 MHz under sustained load. Check:

```powershell
wsl -d wsl-gpu-proj -- nvidia-smi --query-gpu=temperature.gpu,clocks.gr,power.draw --format=csv,noheader
```

If `clocks.gr` is < 1 GHz, let the GPU cool, then re-run.

**`overflow_count > 0` in the summary** — Bucket capacities are too
small. Increase one or more of:

- `maxTissueTrianglesPerCell` (current default 64-128 depending on scene)
- `maxToolTrianglesPerCell` (current default 32-64)
- `maxCandidatePairs` (current default 1-2 million)
- `proximityMaxContacts` (current default 1 million)

**`hashDedupeProbeOverflowCount > 0`** — The pair-hash table got full.
The table is sized `nextPowerOfTwo(maxCandidatePairs * 2)`. Increase
`maxCandidatePairs`.

**CUDA out-of-memory at workspace `ensure(...)`** — The GTX 1650 Ti has
4 GB. The default workspace is well under that, but
`maxCandidatePairs=2000000 * sizeof(uint64) + 2× hash table = ~64 MB`
plus the proximity buffer `1000000 * sizeof(DeviceProximityContact) ≈
72 MB` totals ~140 MB just for the FBP path. If you push
`maxCandidatePairs` or `proximityMaxContacts` much higher, you can run
out. Drop the cap.

### Nsight quirks

**`gpu_collision_nsys.nsys-rep` is missing after a successful run** — The
NVTX capture-range mode in Nsight Systems sometimes exits with status 0
without writing the report. The launcher script
`run_nsight_collision_profile_wsl_gpu_proj.sh` now also runs a full-capture
fallback that produces `gpu_collision_nsys_full.nsys-rep`. Use that.

**Nsight Compute can't find shared library** — The launcher sets
`LD_LIBRARY_PATH` but the parent `ncu` process may not pass it through.
Re-run with explicit `LD_LIBRARY_PATH=...` in the same command:

```powershell
wsl -d wsl-gpu-proj -- bash -c 'export LD_LIBRARY_PATH=/opt/sofa/install/v25.12/lib:$LD_LIBRARY_PATH; bash /home/arfin/gpu-sofa/scripts/run_nsight_collision_profile_wsl_gpu_proj.sh'
```

### Sync gotchas

**Changes to `.cpp` not picked up after build** — You edited on Windows
but forgot to rsync to WSL. Run the §3.2 sync command.

**`rsync` deleted my output logs** — Don't use `--delete` when syncing
TO WSL. The smoke scripts write to WSL-side output dirs that the Windows
side doesn't have.

**Two `output/` directories diverged** — They're supposed to be separate.
Windows `output/` is for artifacts you intentionally copy back (e.g. to
include in a PR). WSL `output/` is for the raw run output that the
benchmark controller writes.

---

## 9. Scene-level configuration recipes

### 9.1  Detection-only one-tissue/one-blade (default benchmark)

```python
root.addObject('CollisionPipeline')
root.addObject('GpuCollisionBroadPhase',
    enableGPU=True, allowCPUFallback=True, useObjectAabbCulling=False)
root.addObject('GpuCollisionNarrowPhase',
    enableGPU=True,
    useDenseGrid=True,
    useIndexedDenseGridInput=True,
    useDirectDevicePositions=True,
    cacheTriangleTopology=True,
    copyContactsToHost=False,                  # detection-only
    detailedProfiling=False,
    # Grid sized for ±4 m XZ, ±0.5 m Y
    gridMinX=-4.5, gridMinY=-0.5, gridMinZ=-4.5,
    gridMaxX= 4.5, gridMaxY= 0.5, gridMaxZ= 4.5,
    gridResolutionX=64, gridResolutionY=8, gridResolutionZ=64,
    contactDistance=0.03,
    maxTissueTrianglesPerCell=128,
    maxToolTrianglesPerCell=64,
    maxCandidatePairs=2000000,
)
```

### 9.2  Same scene with tri-tri FBP

Add the FBP toggles:

```python
root.addObject('GpuCollisionNarrowPhase',
    ...,
    useFeatureBasedProximity=True,
    proximityComputeBarycentrics=True,
    proximityKeepContactsOnDevice=True,
    proximityReadContactCounter=False,         # off for wall-time runs
    proximityMaxContacts=1000000,
)
```

### 9.3  Self-collision via v-t

Two pieces:

```python
# 1. enable v-t on the narrow phase
root.addObject('GpuCollisionNarrowPhase',
    ...,
    useFeatureBasedProximity=True,
    useVertexTriangleProximity=True,
    proximityComputeBarycentrics=True,
    proximityKeepContactsOnDevice=True,
    contactDistance=0.06,
)

# 2. set selfCollision=True on the triangle model that should self-collide
tissue.addObject('TriangleCollisionModel', selfCollision=True)
```

The broad phase emits a `(cm, cm)` pair; the narrow phase routes it to v-t
self-collision.

### 9.4  Cross-model v-t (point cloud vs triangle mesh)

```python
# Tissue side (triangle mesh)
tissue = root.addChild('Tissue')
tissue.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tissue_positions)
tissue.addObject('MeshTopology', name='topo', triangles=tissue_triangles)
tissue.addObject('TriangleCollisionModel', selfCollision=False)

# Tool side (point cloud)
tool = root.addChild('Tool')
tool.addObject('MechanicalObject', name='dofs', template='CudaVec3f', position=tool_points)
tool.addObject('CudaPointCollisionModel')

# Narrow phase with both flags on
root.addObject('GpuCollisionNarrowPhase',
    ...,
    useFeatureBasedProximity=True,
    useVertexTriangleProximity=True,
    ...
)
```

The cross-model branch in `endNarrowPhase` detects the (`CudaPointCollisionModel`,
`CudaTriangleCollisionModel`) pair (or the reverse) and dispatches to the
v-t backend.

### 9.5  Sampled counter readback for long profiling

```python
root.addObject('GpuCollisionNarrowPhase',
    ...,
    useFeatureBasedProximity=True,
    proximityReadContactCounter=False,           # off by default
    proximityCounterReadbackInterval=60,           # read once per ~1 second at 60 FPS
)
```

### 9.6  Response-enabled (CPU contact processing)

```python
root.addObject('GpuCollisionNarrowPhase',
    ...,
    copyContactsToHost=True,                    # publish to SOFA DetectionOutput
    proximityKeepContactsOnDevice=False,        # download proximity contacts
    proximityReadContactCounter=True,           # we need the count to know how many to copy
)
# Add SOFA contact response components: ContactManager, RuleBasedContactManager, etc.
```

Slower than detection-only because of D2H traffic; the cost is needed if
SOFA's CPU response pipeline must consume the contacts.

Cross-model publication (Phase 12 follow-up, 2026-05-25): when
`copyContactsToHost=True` on a scene that uses both
`useFeatureBasedProximity=True` and `useVertexTriangleProximity=True`,
the narrow phase couples flags automatically — `keepContactsOnDevice` is
forced to false so the proximity-contact array is downloaded before
`publishCudaPointTriangleContacts` runs. Measured: 332 FPS at
254 contacts/frame on the cross-model smoke scene (vs 1968 FPS in
detection-only mode). The slowdown is the ~19 KB D2H per frame, not the
publication helper itself.

---

## 10. Metrics to watch

### 10.1  Per-frame benchmark CSV columns

```text
step, duration_seconds, included_in_stats
broad_wall_ms, broad_kernel_ms
narrow_wall_ms, narrow_kernel_ms
narrow_host_preparation_ms
narrow_sofa_triangle_extraction_ms
narrow_backend_triangle_pack_ms
narrow_h2d_ms
narrow_device_allocation_ms
narrow_clear_grid_ms, narrow_counter_clear_ms
narrow_tissue_aabb_ms, narrow_tool_aabb_ms     (legacy split-aabb path)
narrow_insert_tissue_ms, narrow_insert_tool_ms
narrow_generate_pairs_ms
narrow_candidate_readback_ms
narrow_sort_unique_ms, narrow_sort_unique_host_ms
narrow_exact_contact_ms                         (SAT kernel only)
narrow_contact_count_readback_ms
narrow_contact_download_ms
narrow_sofa_output_publish_ms
host_to_device_bytes, device_to_host_bytes, device_allocation_bytes
kernel_launch_count, cuda_memset_count, workspace_resize_count
broad_gpu_used, narrow_gpu_used
broad_input_primitive_count, broad_output_pair_count
narrow_input_primitive_count, narrow_output_pair_count, narrow_output_candidate_count
narrow_output_contact_count
narrow_raw_candidate_count, narrow_unique_candidate_count, narrow_duplicate_reduction_ratio
narrow_grid_cell_count, narrow_active_mixed_cell_count
narrow_tissue_insert_count, narrow_tool_insert_count
narrow_max_tissue_cell_occupancy, narrow_max_tool_cell_occupancy
narrow_overflow_count, narrow_hash_dedupe_probe_overflow_count

# Phase 11+12 additions
narrow_feature_based_proximity_kernel_ms
narrow_host_synchronization_ms
narrow_vf_contact_count, narrow_fv_contact_count, narrow_ee_contact_count
```

### 10.2  Headline summary `avg_*` fields

For wall-time comparisons, focus on:

- `avg_fps` and `avg_step_seconds`
- `avg_narrow_wall_ms` (CPU-observed narrow stage time)
- `avg_narrow_kernel_ms` (cudaEvent-measured GPU kernel time inside narrow stage)
- `avg_narrow_host_synchronization_ms` (derived: wall - kernel)
- `avg_host_to_device_bytes`, `avg_device_to_host_bytes`
- `avg_kernel_launch_count`, `avg_cuda_memset_count`

For correctness validation:

- `avg_narrow_output_contact_count` (total emitted contacts)
- `avg_narrow_vf_contact_count`, `avg_narrow_fv_contact_count`, `avg_narrow_ee_contact_count`
- `avg_narrow_overflow_count` (should always be 0)
- `avg_narrow_hash_dedupe_probe_overflow_count` (should always be 0)

### 10.3  Nsight Compute metrics worth watching

For each kernel of interest (insert, generate, FBP, v-t):

| Metric | What it tells you |
|---|---|
| `gpu__time_duration.sum` | Kernel duration |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Compute vs memory mix |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth used |
| `lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed` | L2 atomic pressure |
| `lts__t_sectors.avg.pct_of_peak_sustained_elapsed` | L2 sector traffic |
| `l1tex__t_sector_hit_rate.pct` | L1 hit rate |
| `launch__registers_per_thread` | Register usage (affects occupancy) |
| `smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed` | Memory-latency-bound indicator |

Query exact suffix names for your Nsight version:

```powershell
wsl -d wsl-gpu-proj -- ncu --query-metrics --query-metrics-mode suffix
```

---

## 11. Safety rules

These are operational rules that have caused trouble in the past:

- **Do not run benchmarks from the Windows checkout.** WSL is the only
  place that has a working CUDA toolchain.
- **Do not edit only the WSL checkout.** Always edit on Windows first;
  rsync to WSL second. Otherwise git on the Windows side won't see your
  changes and you'll lose them on the next sync from Windows.
- **Keep raw run outputs in `output/`**, not the repo root. They're
  `.gitignore`d there.
- **Keep reports and decks in `reports/`.** Generated PNGs, PPTX, etc.
- **Keep current project knowledge in `guide/`.** This file, the
  architecture doc, and the plan doc are the source of truth.
- **Treat `guide/archive/` as historical only.** It contains earlier
  plan iterations; do not use them as the active plan.
