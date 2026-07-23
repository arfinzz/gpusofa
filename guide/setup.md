# Setup

End-to-end instructions for running, building, profiling, and tuning the
GPU SOFA collision pipeline. Read top to bottom for a fresh machine; jump
to a section if you already know the lay of the land.

Last refreshed: 2026-07-23 (all **12 execution modes** measured — dense ×2, optimised
hash, simple hash, sorted grid ×4 toggle combos, big-cell fused ×4 build strategies —
with per-kernel profiling for every mode. Champion: `bigcell_sharedhash`. Canonical
numbers: `reports/performance_all_modes_20260715.md`; mode/metric explainers:
`reports/README_execution_modes.md`).

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
wsl -d wsl-gpu-proj -- rsync -a --delete /mnt/c/Users/arfin/Desktop/'GPU SOFA'/testscenes/ /home/arfin/gpu-sofa/testscenes/
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

All benchmark scenes live in **`testscenes/`** and import shared helpers from
`testscenes/dense_collision_benchmark_common.py`. These five are the canonical,
actively-maintained scenes (legacy CPU baselines and the phase 0–10 validation
scenes were removed in the 2026-06-09 cleanup — recover from git history if
needed):

| Scene (`testscenes/`) | What it tests | Expected output |
|---|---|---|
| `one_tissue_one_blade.py` | GPU dense-grid tri-tri **FBP**, small tool (the surgical default; exact-contact still selectable via env var) | fast warm ~720 FPS, ~0.47 ms narrow wall; validation **56 EdgeEdge** contacts |
| `large_tissue_blade.py` | GPU dense-grid FBP, large mesh (~79,520 elements) | fast ~146 FPS / validation ~114 FPS, **8018** contacts (5397/880/1741) |
| `self_collision_vertex_triangle.py` | **Phase 12 v-t self-collision** | 1300–2100 FPS, **2700 VertexFace** contacts |
| `cross_model_vertex_triangle.py` | **Phase 12 v-t cross-model** | 1500–4200 FPS, **254 VertexFace** contacts |
| `hash_prefixsum_large.py` | **Broad-cull way selector scene** (large tissue + large tool, ~14,368 elements) — all six ways via the `SOFA_USE_*_GENERATION` envs | **2354** contacts identical for every way; narrow kernel: bigcell 0.30 / simple 0.33 / sorted 0.34 / hash 0.34 / dense 1.3–1.5 ms |
| `collision_xlarge_200k.py` | **Extra-large canonical scene** (exactly 200,018 elements: 198,450 tissue + 1,568 blade triangles; same env toggles) | **12,178** contacts identical across ways. The historical standalone `SOFA_LARGE_TISSUE_NX=316` with its default 14,720-triangle tool is a different 213,170-element benchmark; set blade segments to 30/8/4 for the 200,018-count standalone analogue. |

Common scene helpers in `testscenes/dense_collision_benchmark_common.py`:

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

| Script | Scene(s) | Default mode |
|---|---|---|
| `run_full_benchmark_suite_wsl.sh` | **All canonical scenes** (small, large, v-t self, v-t cross, 5-way legs on the 14k scene, 3 xlarge-200k legs) in fast + validation modes | 160 steps/leg, one summary per leg; `SOFA_SUITE_ONLY='<regex>'` runs a filtered batch (useful under a wall-clock cap) |
| `run_bigcell_parity_wsl.sh` | **Way-6 parity gate** — backend bench across the big-cell toggle matrix (factors 4/2/1, forced chunk loop, hash-build legs) | contacts must equal the FBP leg; `bigcell_pairs_tested` must equal `sortedgrid_unique_pairs` |
| `run_ncu_bigcell_wsl.sh` | **Nsight Compute on the winning way-6 kernel** at the 14k / 80k / 200k triangle-count scales (standalone analogues) | fused duration, SM/DRAM throughput, achieved occupancy, and registers for every scale |
| `run_bigcell_detailed_profile_wsl.sh` | **Reproducible winner deep profile** at the 14k / 80k / 200k triangle-count scales (standalone analogues) | production graph timing, graph-off CUDA-event stage timing, opt-in `clock64()` fused-internal counters/cycles, occupancy/resources, CSVs |
| `run_mode_comparison_ab_wsl.sh` | **12-leg same-session comparison of EVERY execution mode** on `hash_prefixsum_large.py` (dense ×2, hash, simple, sorted ×4, bigcell ×4 builds) | counter-on, prints the kernel-time summary table |
| `run_fbp_smoke_test_wsl.sh` | **`one_tissue_one_blade.py` with FBP** (Phase 11) | FBP, detection-only, readback off, tool-active-cell default-on |
| `run_fbp_large_tissue_wsl.sh` | **`large_tissue_blade.py` FBP** (Phase 15 large A/B) | FBP, detection-only, tool-active-cell default-on |
| `run_vertex_triangle_smoke_wsl.sh` | **`self_collision_vertex_triangle.py`** (Phase 12) | FBP + v-t self, readback on |
| `run_cross_model_vt_smoke_wsl.sh` | **`cross_model_vertex_triangle.py`** (Phase 12 cross-model) | FBP + v-t cross, readback on |
| `run_hash_prefixsum_large_ab_wsl.sh` | **Hash A/B** — `hash_prefixsum_large.py`, dense vs hash+prefix-sum | counter-on, contact-count parity check |
| `run_branch_comparison_ab_wsl.sh` | **Branch comparison** — `hash_prefixsum_large.py` dense vs hash at two tool sizes (mixed + large) | counter-on, prints per-leg summary |
| `run_tiny_ab_wsl.sh` | **Tiny-scene A/B** — small tissue + small tool, dense vs hash (the regime where dense wins) | counter-on |
| `run_small_warm_ab_wsl.sh` | **Warm small-scene A/B** — discards a cold-clock warm-up leg, then fast + validation | use for a representative fast-path FPS |
| `run_nsight_fbp_profile_wsl.sh` | **Nsight Compute on FBP + v-t kernels** | three scenes, focused metric set |
| `run_nsight_collision_profile_wsl_gpu_proj.sh` | Nsight Compute + Nsight Systems capture (`one_tissue_one_blade.py`) | full profiling |
| `run_gpu_kernel_profile_wsl.sh` | NVTX-annotated profiling run | event timings on |
| `run_backend_dense_grid_benchmark_wsl_gpu_proj.sh` | Standalone backend bench (no SOFA scene) | direct kernel-only |

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
| `SOFA_USE_HASH_PREFIXSUM_GENERATION` | **0** | **EXPERIMENTAL (branch `experiment/hash-prefixsum-broadphase`), DEFAULT OFF.** Replace the dense grid with a spatial-hash + prefix-sum broad cull for the tri-tri FBP path. Read only by `testscenes/hash_prefixsum_large.py`; on the production scenes set the `useHashPrefixSumGeneration` Data field directly. Targets large-tissue + large-tool; **optimised 2026-06-17 to ~2.5–3× faster kernel than dense** (0.5–0.7 vs 1.8–2.1 ms), contacts bit-identical. Requires FBP on. See `reports/archive_pre_20260618/hash_optimized_broadphase_20260617.md` |
| `SOFA_HASH_TABLE_SIZE` | **0** | Hash table slot count for the hash / simple-hash paths. 0 = auto (~4 slots per input triangle, rounded to a power of two) |
| `SOFA_HASH_CUDA_GRAPH` | **1** | **DEFAULT ON (guide/plan.md §5.20).** Replay the hash broad cull's 11-kernel sequence as a captured CUDA Graph each steady-state frame (verified bit-identical, ~−7 % kernel / +10 % FPS). Set to 0 to launch the kernels individually (e.g. to A/B, or under detailed profiling, where it is auto-disabled). Only affects the `useHashPrefixSumGeneration` path |
| `SOFA_USE_SIMPLE_HASH_GENERATION` | **0** | **EXPERIMENTAL "4th way", DEFAULT OFF (2026-06-26).** Replace the dense grid with a *simple direct-bucket* spatial hash for the tri-tri FBP path — triangles stored straight into per-cell hash buckets in one insert pass (no mark/compact/fill), 7 kernels. Read by `testscenes/hash_prefixsum_large.py`; or set the `useSimpleHashGeneration` Data field directly. **Measured tied with the optimised hash (~0.35–0.38 ms, ~5× over dense), bit-identical, zero overflow** on the 14,368-element scene. Mutually exclusive with the hash flag (hash wins). Requires FBP on. See `reports/archive_pre_20260703/four_way_broadcull_comparison_20260626.md` |
| `SOFA_SIMPLE_HASH_CUDA_GRAPH` | **1** | **DEFAULT ON.** Replay the simple-hash 7-kernel sequence as a captured CUDA Graph each steady-state frame (same machinery as `SOFA_HASH_CUDA_GRAPH`, safe fallback). Set to 0 to launch individually. Only affects the `useSimpleHashGeneration` path |
| `SOFA_USE_SORTED_GRID_GENERATION` | **0** | **EXPERIMENTAL "5th way", DEFAULT OFF (2026-07-03).** Sorted-grid (tiled binning) broad cull: incidence expansion → counting sort by cell → block-per-mixed-cell generation with shared-memory staging; home-cell exactly-once dedup doubling as an exact AABB pre-cull. No per-cell caps. **Ties the hashes on the 14,368 scene (~0.35 ms); ~3× faster than everything on the 79,520 bench (0.65 ms), bit-identical.** Read by `testscenes/hash_prefixsum_large.py`, or set the `useSortedGridGeneration` Data field. See `reports/performance_all_modes_20260715.md` |
| `SOFA_SORTED_GRID_CUB_SORT` | **0** | Sorted-grid engine A/B: 1 = `cub::DeviceRadixSort` instead of the counting sort (measured slower: 0.48 vs 0.35 ms). Guarded by a frame-0 health probe — on this WSL2 stack CUB intermittently returns success with unsorted output; the probe detects it, prints a notice, and falls back to the counting sort for the process |
| `SOFA_SORTED_GRID_PAIRHASH_DEDUP` | **0** | Sorted-grid dedup A/B: 1 = atomicCAS pair-hash (like the hash ways; reproduces their 322,560-pair candidate set) instead of home-cell emission (measured slower: 0.51 vs 0.35 ms, and forfeits the pre-cull) |
| `SOFA_SORTED_GRID_CUDA_GRAPH` | **1** | **DEFAULT ON.** Replay the sorted-grid 9-kernel sequence as a captured CUDA Graph each steady-state frame. Set to 0 to launch individually |
| `SOFA_SORTED_GRID_VERIFY` | **0** | Diagnostic: run a device-side verifier over the sorted key stream every frame (ascending + run boundaries match the scanned histogram); violations land in the probe-overflow counter. The frame-0 CUB health probe runs regardless of this flag |
| `SOFA_USE_BIGCELL_FUSED_GENERATION` | **0** | **EXPERIMENTAL "6th way", DEFAULT OFF (2026-07-12).** Big-cell / small-cell FUSED generation + narrow phase: per-big-cell CSR table, one kernel per mixed big cell stages the tool side (ids + AABBs + vertices) in shared memory via an in-shared counting sort and runs the FBP math inline — no candidate-pair list, no separate FBP launch. Home-cell dedup at small-cell granularity ⇒ pair set identical to way 5's. **Fastest narrow kernel of all 8 configs on the 14,368 scene (0.309 ms); ties way 5 at 200k (1.09 ms).** Read by `testscenes/hash_prefixsum_large.py` + `testscenes/collision_xlarge_200k.py`, or set `useBigCellFusedGeneration` directly. Precedence: bigcell > hash > simple > sorted > dense. See `reports/performance_all_modes_20260715.md` |
| `SOFA_BIGCELL_FACTOR` | **2** | Big-cell edge in small cells (power of two ≤ 4). Factor 4 measured 40 % slower at 200k — only 48 mixed big cells → block-level parallelism collapse |
| `SOFA_BIGCELL_TOOL_TILE` | **256** | Tool entries staged in shared memory per chunk (1..256). Oversized big cells loop over chunks; parity-verified down to 32 (~17 % slower) |
| `SOFA_BIGCELL_HASH_BUILD` | **0** | Build A/B: 1 = literal per-big-cell open-addressing hash multi-map build instead of CSR. **Measured ~2.9× slower** (per-frame slot clear + atomicCAS probing + sparse sweeps); best-effort slot regions with drops reported via `buildOverflowCount` |
| `SOFA_BIGCELL_HASH_SLOTS` | **1024** | Hash-build slots per (big cell, side) region (pow2, clamped 64–4096). 2048 = zero-overflow on the 80k bench geometry |
| `SOFA_BIGCELL_CUDA_GRAPH` | **1** | **DEFAULT ON.** Replay the way-6 sequence (9 kernels CSR / 7 hash-build) as a captured CUDA Graph each steady-state frame |
| `SOFA_BIGCELL_SHARED_BUILD` | **1** | CSR-build shared-memory privatization A/B (2026-07-15): 0 = direct global atomics; **1 = per-block shared HASH TABLE then merge (DEFAULT — measured −12% at 80k / −26% at 200k full pipeline, and the block-grouped entry order speeds the fused consumer too)**; 2 = per-block shared SORTED LIST via in-shared bitonic sort (measured ~2.6× slower — the sort costs more than every atomic it avoids). Contacts identical in all modes; staging overflow falls back to the direct path (`sharedSpillCount`). See `reports/archive_pre_20260715/bigcell_shared_build_ab_20260715.md` |
| `SOFA_BIGCELL_PROFILE_INTERNALS` | **0** | Diagnostic-only fused-kernel instrumentation for SOFA scenes: launches a separate diagnostic kernel with `clock64()` phase/filter sampling and rejection counters; implies detailed profiling and disables graph replay. It perturbs registers, occupancy, and time—never use its elapsed time as production performance. Standalone bench equivalent: `SOFA_BACKEND_BENCH_BIGCELL_PROFILE_INTERNALS=1`. |

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

### 7.10  Full benchmark suite (all scenes, one command)

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_full_benchmark_suite_wsl.sh
```

Runs small, large, v-t self, v-t cross, the five broad-cull legs on the 14k
scene, and the three xlarge-200k legs — into
`output/benchmark_logs/full_suite_<stamp>/<leg>/`. When calling from Windows,
`wsl.exe` invocations are capped by the caller's timeout — run in batches with
`SOFA_SUITE_ONLY` (a leg-name regex) sharing one `SOFA_BENCHMARK_LOG_DIR`:

```powershell
wsl -d wsl-gpu-proj -- bash -c "SOFA_SUITE_ONLY='^xlarge_' SOFA_BENCHMARK_LOG_DIR=/home/arfin/gpu-sofa/output/benchmark_logs/my_run bash /home/arfin/gpu-sofa/scripts/run_full_benchmark_suite_wsl.sh"
```

The 2026-06-09 run is reported in `reports/archive_pre_20260618/benchmark_suite_20260609.md`.

### 7.10b  Way 6 (big-cell fused) — run + parity

```powershell
# SOFA scene, way 6 on the 14,368 selector scene (or collision_xlarge_200k.py for exactly 200,018):
wsl -d wsl-gpu-proj -- bash -c "SOFA_USE_BIGCELL_FUSED_GENERATION=1 SOFA_PROXIMITY_READ_CONTACT_COUNTER=1 /opt/sofa/install/v25.12/bin/runSofa -g batch -n 160 -l SofaPython3 -l SofaCUDA -l /home/arfin/gpu-sofa/SofaGpuCollision/build-profile/libSofaGpuCollision.so /home/arfin/gpu-sofa/testscenes/hash_prefixsum_large.py"
# Parity gate (backend bench, factors 4/2/1 + chunk loop + hash-build legs):
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_bigcell_parity_wsl.sh
# 8-leg same-session comparison (the summary table):
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_mode_comparison_ab_wsl.sh
```

Expected: contacts identical to every other way on the same scene (2354 on the
14k scene, 12,178 on the xlarge scene); `bigcell_pairs_tested` =
`sortedgrid_unique_pairs`; way-6 narrow kernel fastest (0.30 ms on 14k,
2.01 ms on xlarge). Numbers: `reports/performance_all_modes_20260715.md`.

> **Cold-clock caveat.** The *first* leg in a fresh process is contaminated by
> CUDA-context init + GPU clock ramp (the 1650 Ti idles at a low clock). For a
> representative small fast-path number, use `run_small_warm_ab_wsl.sh`, which
> discards a warm-up leg first. Today's warm small fast path: 723.8 FPS,
> 0.47 ms narrow wall, 0.28 ms narrow kernel.

### 7.11  Experimental hash + prefix-sum broad cull (A/B)

```powershell
wsl -d wsl-gpu-proj -- bash /home/arfin/gpu-sofa/scripts/run_hash_prefixsum_large_ab_wsl.sh
```

Runs the large-tissue + large-tool scene twice — dense grid (`useHashPrefixSumGeneration=0`)
then hash + prefix-sum (`=1`). Contact counts MUST match between legs (2354,
1119 VF / 428 FV / 807 EE); the FPS / narrow-wall delta is the payoff
(measured +11.8 % FPS, −15 % narrow wall on 2026-06-09). This path is
**default-off and opt-in**; the dense grid is untouched. Design + numbers:
`reports/archive_pre_20260618/hash_prefixsum_broadphase_experiment_20260609.md`.

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
