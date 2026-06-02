# Results Explanation

Companion to [`end_to_end_verification_20260525.md`](end_to_end_verification_20260525.md).
Read this if you opened the verification report and wondered "what does
that number actually mean and is it good or bad?"

This file is **plain-English**. Every metric, every column header, every
acronym used in the verification report and in the CSV files under
`output/benchmark_logs/` is defined here, in the same order you encounter
them when reading top-to-bottom.

For deep architectural context see `guide/architecture.md`. For the
project history and design decisions see `guide/plan.md`. For setup and
how-to-run see `guide/setup.md`.

---

## 1. What the system actually does

Surgical-simulation collision detection has to answer one question per
frame:

> Given a deformable tissue mesh and one or more surgical tools (rigid
> blade, point cloud of probe tips, …), which **pairs of geometric
> primitives** are close enough to interact?

That single question gets answered in two stages:

1. **Broad phase**: quickly cull pairs that can't possibly be close.
   Driven by axis-aligned bounding boxes (AABBs) and a uniform 3D cell
   grid. Output: a list of candidate (primitive-A, primitive-B) pairs.
2. **Narrow phase**: for each surviving candidate pair, compute the
   actual closest-feature geometry. Output: a list of `Contact` records
   with positions, normals, distances.

The plugin's narrow phase has **four output modes**, each with its own
kernel and SOFA dispatch:

| Mode | What it does | When to use |
|---|---|---|
| **Exact-contact (SAT)** | Boolean "do these two triangles intersect?" with a contact point on the deepest overlap axis | Legacy default; cheapest if you only care about intersection |
| **Tri-tri FBP** | Closest-feature proximity test (6 vertex-face + 9 edge-edge) per triangle pair, emits a contact when distance ≤ threshold | Recommended for any scene that will feed a constraint solver |
| **V-t self-collision** | Closest-point-on-triangle per (vertex, triangle) candidate, with own-corner exclusion. Used when a mesh tests against itself | Tissue self-folding / cutting / tearing detection |
| **V-t cross-model** | Same kernel as self-collision but without own-corner exclusion. Triggered by (CudaPointCollisionModel, CudaTriangleCollisionModel) pairs | Rigid tool point clouds vs deformable tissue |

The verification report exercises all four modes.

---

## 2. Where the numbers come from

Each SOFA scene runs for 20 frames (10 warmup + 10 measured by default).
The `GpuPipelineBenchmarkController` SOFA component captures one row per
frame into a CSV and writes a summary at the end. The standalone bench
tool (`SofaGpuCollisionDenseGridBackendBench`) does the same but
bypasses SOFA's scene loader and only calls the backend functions
directly.

**CSV files** under `output/benchmark_logs/<run>/` have one row per
frame.

**Summary files** have one `key=value` line per averaged metric across
the measured frames. The verification report copies these summary lines
verbatim.

**Nsight reports** under `output/benchmark_logs/<run>/07_nsight/<scene>/`
are produced by NVIDIA's profiling tool. Each `.ncu-rep` binary has been
exported to CSV with `ncu --import ... --csv --page raw` so the metric
rows are readable.

---

## 3. The wall-time metrics

### `avg_fps`

Frames per second, averaged across the measured frames. For a 20-frame
benchmark this is just `measured_steps / total_measured_seconds`.

**Higher is better.** The fast paths target 600+ FPS on the target
hardware. Below 100 FPS suggests something is misconfigured (most often:
counter readback left on, or `copyContactsToHost=True` without a
genuine response-pipeline use case).

### `avg_step_seconds`

The reciprocal of `avg_fps`. Seconds per simulation step, averaged. For
a 200 FPS run this is 0.005 s.

### `avg_narrow_wall_ms`

CPU-observed wall time around the narrow-phase stage. Measured with
`std::chrono::steady_clock` from the start of `endNarrowPhase()` to the
end. **This is the time the CPU sees**, not the time the GPU spends.

For an async pipeline (the production fast path), the GPU may still be
running the previous frame's kernels when the CPU finishes the current
frame's narrow phase. In that case `narrow_wall_ms` captures only the
host-side scheduling cost.

### `avg_narrow_kernel_ms`

Sum of all `cudaEventElapsedTime` measurements taken inside the
backend during the narrow phase. **This is the GPU time** measured via
CUDA events. Includes the broad-cull window (reset + insert + generate)
and, when measured, the FBP narrow pass.

In the production fast path, the FBP narrow pass's `cudaEventRecord`
pair is skipped, so this number represents only the broad cull. In
validation mode, both are measured.

### `avg_narrow_host_synchronization_ms`

Derived as `max(0, narrow_wall_ms - narrow_kernel_ms)`. This is the
host-side time that wasn't spent waiting for a measured GPU kernel.
Includes:

- CPU work to dispatch kernels
- `cudaMemcpy` synchronization (when counter readback is on)
- SOFA's overhead inside the narrow-phase function

**Lower is better.** For the fast path this should be a few hundred
microseconds. If it blows up to multiple milliseconds, look for
unintentional synchronization (counter readback, contact download,
`cudaDeviceSynchronize` calls).

### `avg_narrow_feature_based_proximity_kernel_ms`

Time spent specifically in `featureBasedProximityKernel` (tri-tri FBP)
or the v-t equivalent, measured via the workspace-owned `fbpStartEvent`
/ `fbpEndEvent` CUDA events. Only measured when `readContactCounter` is
true; otherwise reported as 0 (the kernel still runs, just isn't
timed).

---

## 4. The byte-transfer metrics

### `avg_host_to_device_bytes`

Bytes transferred from CPU memory to GPU memory per frame. Production
fast-path target: **0**.

The path achieves 0 because:

- Tissue and tool vertex positions are owned by SOFA's
  `MechanicalObject<CudaVec3f>` — they already live on the GPU.
  `useDirectDevicePositions=True` reads the SOFA CUDA vector's
  `deviceRead()` pointer directly.
- Triangle indices are uploaded **once** when the topology cache misses
  (first frame), then reused via `surfaceId` + `topologyVersion`
  matching in the device-side index buffers.

Non-zero values indicate either a fresh topology cache (first frame),
or that the scene falls into the legacy "packed" path that uploads
triangles each frame.

### `avg_device_to_host_bytes`

Bytes transferred from GPU memory to CPU memory per frame.

| Value | Meaning |
|---:|---|
| 0 | Detection-only fast path. No counter or contact readback. |
| 16-20 | One batched counter readback. 4-5 uint32 counters × 4 bytes. Pinned host buffer. |
| 64 | Validation mode tri-tri FBP: 5 proximity counters + 11 broad-cull counters. |
| Many KB (e.g. 19 320) | CPU publication mode: counters + the full ProximityContact array. |

### `avg_device_allocation_bytes`

Cumulative GPU memory allocated during measured frames. Should be 0 in
steady state (the workspace allocates on first frame, then reuses).

---

## 5. The launch and overflow metrics

### `avg_kernel_launch_count`

Number of `<<<...>>>` kernel launches per frame. Expected:

| Path | Launches | Sequence |
|---|---:|---|
| Tri-tri FBP | 7 | reset + memset + insertTri × 2 + generate + resetProx + FBP |
| V-t (self or cross) | 6 | reset + memset + insertTri + insertPoints + generate + v-t (the v-t function counts reset-proximity into the same step) |
| Exact-contact detection-only | 5 | reset + memset + insertTri × 2 + generate (no exact-contact kernel when contacts stay on device) |

### `avg_cuda_memset_count`

Number of `cudaMemset` calls per frame. Always 1 in steady state — the
pair-hash table reset uses `cudaMemset(0xff)` because it benchmarked
faster than a custom reset kernel on the GTX 1650 Ti.

### `avg_workspace_resize_count`

Number of times the workspace had to grow during measured frames.
Should be 0 in steady state. A non-zero value means the scene
configuration grew capacity mid-run, which forces a `cudaMalloc` /
`cudaFree` pair — visible spike in wall time.

### `avg_narrow_overflow_count`

Aggregate count of overflow events across all narrow-phase per-cell and
per-output buffers:

- Triangles tried to enter a cell bucket that was already full
  (`maxTissueTrianglesPerCell` or `maxToolTrianglesPerCell`).
- Candidate pairs tried to enter the `candidatePairs` array that was
  already full (`maxCandidatePairs`).
- Proximity contacts tried to enter the `proximityContacts` array that
  was already full (`proximityMaxContacts`).

**Should always be 0** in correctly-sized scenes. Non-zero means the
contact data is incomplete; bump the relevant capacity.

### `avg_narrow_hash_dedupe_probe_overflow_count`

The open-addressing pair-hash table overflowed its probe limit, meaning
a duplicate candidate pair might have slipped through. Table is sized
`nextPowerOfTwo(maxCandidatePairs * 2)`. **Should always be 0**.
Non-zero means increase `maxCandidatePairs`.

---

## 6. The candidate and contact counts

### `avg_narrow_input_primitive_count`

Total number of triangles fed into the narrow phase across all candidate
pairs. For one-tissue/one-blade scenes this is roughly
`tissueTris + bladeTris`.

### `avg_narrow_raw_candidate_count`

How many (tissue, tool) primitive pairs the dense-grid candidate-
generation kernel emitted **before** dedupe. Each cell bucket
contributes `tissueCount × toolCount` pairs. The same pair can be
emitted from multiple cells when both primitives' AABBs span multiple
cells.

### `avg_narrow_unique_candidate_count`

How many distinct candidate pairs survived dedupe (the `atomicCAS`-
based hash table). This is the number of pairs the **narrow kernel
actually processes**.

For the one-tissue/one-blade scene: 2 304 raw → 624 unique. The 73 %
reduction is from blade triangles spanning multiple cells; each unique
(tissue, blade) pair appears in ~4 cells on average.

### `avg_narrow_output_contact_count`

How many contacts the narrow kernel emitted (the proximity test passed
for ~9 % of candidate pairs in the tri-tri scene; ~5 contacts per vertex
in the v-t self scene).

For the production fast path this reads as 0 because the counter isn't
read back — the contacts still exist on the GPU.

### `avg_narrow_vf_contact_count` / `avg_narrow_fv_contact_count` / `avg_narrow_ee_contact_count`

Per-class breakdown of the proximity contacts, from the device-side
atomic counters incremented inside the FBP / v-t kernels.

- **VF** = Vertex-of-A vs Face-of-B was the closest feature pair.
- **FV** = Face-of-A vs Vertex-of-B was the closest feature pair.
- **EE** = Edge-of-A vs Edge-of-B was the closest feature pair.

Geometric truths:

- A flat tissue plane against a small blade box → all contacts will be
  **EE** (no vertex penetrates the other face plane at small contact
  distances).
- A vertex cloud against a triangle mesh → all contacts will be **VF**
  by construction (the v-t kernel only computes that test).

If a tri-tri scene reports all VF or all FV, that probably means the
geometry actually has a vertex piercing a face — interesting signal but
not a bug.

---

## 7. The dense-grid occupancy metrics

### `avg_narrow_grid_cell_count`

Total cell count = `gridResolutionX × gridResolutionY × gridResolutionZ`.
For the one-tissue scene: 64 × 8 × 64 = 32 768. For the v-t smoke
scenes: 32 × 4 × 32 = 4 096.

### `avg_narrow_active_mixed_cell_count`

Cells that contained at least one tissue triangle AND at least one tool
triangle (or vertex). Only meaningful when `compactActiveCells=True`.

### `avg_narrow_tissue_insert_count` / `avg_narrow_tool_insert_count`

Total per-cell insertions of tissue / tool primitives. A primitive
spanning N cells contributes N insertions. Reads as 0 when stats
collection is off (`detailedProfiling=false`).

### `avg_narrow_max_tissue_cell_occupancy` / `avg_narrow_max_tool_cell_occupancy`

The deepest cell bucket reached during insertion. Should stay below the
respective `maxTissueTrianglesPerCell` / `maxToolTrianglesPerCell`
capacity. If it exceeds, `overflow_count` will be non-zero.

---

## 8. The Nsight Compute kernel metrics

The Nsight CSVs contain per-launch metrics for individual kernels. The
explanations below are what each metric tells you about that kernel.

### `gpu__time_duration.sum`

Microseconds the kernel actually ran on the GPU. Different from the
host-observed launch time because launch overhead is excluded.

### `launch__grid_size`

Number of blocks in the launch (the first dimension of the grid).
**Indicates how parallel the workload is**.

- For `featureBasedProximityKernel` this is 256 blocks (the capped over-
  launch). With 624 actual candidate pairs and 256 threads per block,
  almost all threads early-exit.
- For `insertIndexedTrianglesKernel` this is `ceil(triangleCount / 256)`.
  For 12 800 tissue triangles: 50 blocks. For 12 blade triangles: 1
  block (with 244 idle threads).

### `launch__registers_per_thread`

Registers each thread uses. Lower is better for occupancy. The GTX
1650 Ti has 64 KB of register file per SM, so 32 registers/thread caps
occupancy at 64 warps/SM (the architectural max), while 68
registers/thread caps it at ~30 warps/SM (about 47% of peak).

- `featureBasedProximityKernel`: **68** (high, capped at ~50% occupancy)
- `featureBasedVertexTriangleProximityKernel`: 32 (good)
- `insertIndexedTrianglesKernel`: 36 (good)
- `generateDenseGridUniqueCandidatePairsKernel`: 22 (excellent)

### `sm__throughput.avg.pct_of_peak_sustained_elapsed`

How busy the streaming multiprocessors were during the kernel, as a
percentage of theoretical peak. 100% means every SM was at full ALU
utilization every cycle. Real values for these kernels are 4-26 %.

Low SM throughput means either (a) the kernel has lots of idle threads
(over-launch with early-exit), or (b) the kernel is waiting on memory
not compute.

### `sm__warps_active.avg.pct_of_peak_sustained_active`

Percentage of the kernel's runtime during which warps were active on
each SM. Different from SM throughput: a kernel can have high warp
activity (warps are scheduled) but low compute throughput (warps are
stalled).

Typical values: 43-62 % for the new kernels. The 43% on FBP is the
68-reg occupancy cap. The 52-62% on v-t is comfortable.

### `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`

The dominant of compute vs memory throughput. High value = the kernel
is using either ALU or memory subsystem efficiently. The v-t self-
collision kernel hits 41-43% here, which is healthy.

### `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`

How much of the DRAM bandwidth the kernel consumed. Low values (< 5 %)
for the FBP/v-t kernels — they don't push the memory bus.

### `lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed`

Percentage of L2 cache cycles spent processing atomic operations
(`atomicAdd`, `atomicCAS`). High values indicate atomic contention.
**For the FBP/v-t kernels this is 0.01-0.5 %** — atomics are not the
bottleneck. (For the `insertIndexedTrianglesKernel` it's 20%, but the
absolute kernel time is small.)

### `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`

L2 cache "tag sector" traffic. Indicates how much memory the kernel
reads through L2. The v-t self-collision kernel hits 30%, which is the
triangle-index lookups for each (vertex, triangle) candidate.

### `sm__inst_executed.avg.pct_of_peak_sustained_elapsed`

Percentage of instruction issue cycles where the kernel actually issued
an instruction. Related to but not the same as compute throughput.

---

## 9. What "good" and "bad" look like

### Production fast path (detection-only)

| Metric | Good | Bad | Why |
|---|---|---|---|
| `avg_fps` | > 500 | < 100 | Low FPS = sync somewhere |
| `avg_narrow_wall_ms` | < 1 ms | > 5 ms | Wall blowing up means CPU is waiting on GPU |
| `avg_host_to_device_bytes` | 0 | > 0 | Position uploads are wasted work |
| `avg_device_to_host_bytes` | 0 (fast) or 16-20 (sampled) | > 64 | Counter readback should be off in prod |
| `avg_kernel_launch_count` | exactly 6 (v-t) or 7 (tri-tri FBP) | varies | Wrong launch count = path detection bug |
| `avg_cuda_memset_count` | 1 | > 1 | Extra memsets = extra reset work |
| `avg_workspace_resize_count` | 0 | > 0 in steady state | Resize means scene config grew capacity |
| `avg_narrow_overflow_count` | 0 | > 0 | Contact data incomplete |
| `avg_narrow_hash_dedupe_probe_overflow_count` | 0 | > 0 | Dedupe may have missed duplicates |

### Validation mode (counter readback on)

| Metric | Good | Bad | Why |
|---|---|---|---|
| `avg_fps` | 200-500 | < 50 | Readback sync limits to ~500 FPS |
| `avg_narrow_feature_based_proximity_kernel_ms` | < 1 ms | > 3 ms | FBP kernel cost |
| `avg_narrow_host_synchronization_ms` | < 1 ms | > 3 ms | Sync overhead — investigate where |
| `avg_device_to_host_bytes` | 16-64 | > 1000 | Should be a single batched read |
| `avg_narrow_output_contact_count` | > 0 in proximity scenes | 0 | If you expect contacts and see 0, kernel disabled |

### CPU publication mode

| Metric | Good | Bad | Why |
|---|---|---|---|
| `avg_fps` | 100-400 (depends on contact count) | < 50 | Publication is unavoidable cost |
| `avg_device_to_host_bytes` | counters + contacts × ~76 B | only counters | Publication didn't fire |
| `avg_narrow_sofa_output_publish_ms` | < 1 ms | > 5 ms | SOFA publication overhead |

---

## 10. Per-scene expected results

| Scene | FPS range | narrow_wall (ms) | Expected contacts | VF / FV / EE |
|---|---|---|---|---|
| Tri-tri FBP fast (one-tissue/one-blade) | 500-800 | 0.5-1.0 | 0 (not read back) | 0 / 0 / 0 |
| Tri-tri FBP validation | 300-500 | 1.5-2.5 | 56 | 0 / 0 / 56 |
| V-t self-collision (2-layer slab) | 1300-2100 | 0.4-0.7 | 2700 | 2700 / 0 / 0 |
| V-t cross-model detection-only | 1300-2000 | 0.4-0.6 | 254 | 254 / 0 / 0 |
| V-t cross-model publication | 300-1400 | 0.5-3.0 | 254 | 254 / 0 / 0 |
| Backend bench exact-contact (12 800 + 224 tris) | n/a | 3-5 | 0 | n/a |
| Backend bench tri-tri FBP | n/a | 2-3 | 488 | 0 / 133 / 355 |
| Backend bench v-t cross-model | n/a | 1.5-2.5 | 38 | 38 / 0 / 0 |

If a scene measures outside these ranges, the likely causes (in priority
order):

1. **GPU thermal throttling**. Check
   `nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv`.
   Below 1 GHz core clock = throttled. Let it cool, re-run.
2. **Counter readback flag stuck on**. Check
   `SOFA_PROXIMITY_READ_CONTACT_COUNTER` env var.
3. **Scene config wrong**. Inspect the scene Python for any non-default
   `useFeatureBasedProximity` / `useVertexTriangleProximity` / `copyContactsToHost` setting.
4. **Different SOFA scene file**. Make sure the launcher is pointing
   at the expected `test_gpu_*.py`.

---

## 11. Reading the Nsight numbers in two sentences

> The new kernels (`featureBasedProximityKernel`,
> `featureBasedVertexTriangleProximityKernel`, `insertIndexedPointsKernel`)
> are not bottlenecks. The FBP kernel uses 68 registers per thread, which
> caps occupancy at ~50% of peak, but the kernel only takes 17 µs per
> launch so this doesn't matter for the user-visible FPS.

---

## 12. Quick acronym lookup

| Acronym | Meaning |
|---|---|
| **AABB** | Axis-Aligned Bounding Box |
| **SAT** | Separating Axis Theorem (boolean intersection test) |
| **FBP** | Feature-Based Proximity (Ericson-style closest-feature) |
| **VF / FV / EE** | Vertex-Face / Face-Vertex / Edge-Edge feature pair |
| **v-t** | Vertex-Triangle (a simpler variant of FBP) |
| **SOFA** | Simulation Open Framework Architecture |
| **CUDA** | NVIDIA's GPU compute platform |
| **WSL** | Windows Subsystem for Linux (we run inside `wsl-gpu-proj`) |
| **SM** | Streaming Multiprocessor (the GPU's compute unit) |
| **L2 / LTS** | Level-2 cache (the GPU has ~1 MB shared between SMs) |
| **DRAM** | The GPU's main memory (4 GB on the 1650 Ti) |
| **H2D / D2H** | Host-to-Device / Device-to-Host (CPU↔GPU memory transfers) |
| **NCU** | Nsight Compute (NVIDIA's kernel-level profiler) |
| **NVTX** | NVIDIA Tools Extension (annotations for Nsight Systems) |

---

## 13. Where to dig next

Got a number that doesn't match the expected ranges?

1. **`guide/setup.md` §8 Troubleshooting** — quick answers for build
   errors, runtime errors, Nsight quirks, sync gotchas.
2. **`guide/architecture.md` §11 Failure modes** — table of overflow
   and fallback symptoms, what each one means, what to bump.
3. **`guide/plan.md` §6 follow-ups** — the kernel-level profile findings
   that explain why the current numbers look the way they do.
4. **`output/benchmark_logs/<run>/<scene>_timings.csv`** — per-frame
   numbers that the summary files average out. Useful for spotting a
   single misbehaving frame.

Got a number that matches and you want to make it better?

1. **`guide/plan.md` §7 Remaining work** — the prioritized list of
   tuning options.
2. **`reports/end_to_end_verification_20260525.md` §7 Nsight Compute
   kernel profile** — what we know about each kernel's bottleneck.

Got a code question?

1. **`guide/architecture.md` §13 Quick cross-reference** — file +
   function pointers for every concept in the codebase.
