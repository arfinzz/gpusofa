# 11 — Profiling: How the Stopwatch Works

The whole point of the benchmark is to *measure* speed. This file explains how
the timing is captured and what every number in the output means. Two files do
the work:

- `GpuPipelineProfiling.{h,cpp}` — the in-memory accumulator.
- `GpuPipelineBenchmarkController.{h,cpp}` — the SOFA component that writes the
  CSV and summary.

---

## 11.1 The two clocks: wall time vs kernel time

There are two fundamentally different ways to measure GPU work, and confusing
them is the #1 beginner mistake.

### Wall time (`chrono::steady_clock`)

This is ordinary CPU stopwatch time: "how long between when the CPU entered the
narrow phase and when it left?" The code wraps the phase like this:

```cpp
const auto phaseStart = std::chrono::steady_clock::now();
// ... do the narrow phase ...
stageSnapshot.wallMilliseconds =
    std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phaseStart).count();
```

**Important:** because the CPU doesn't wait for the GPU (file 10), wall time
measures *CPU orchestration only*. It does NOT include the GPU compute that
overlaps the next frame. In the fast path, `narrow_wall_ms ≈ 0.56` is the cost
of extracting surfaces, building config, and issuing launches — not the kernels.

### Kernel time (CUDA events)

To measure actual *GPU* time, you need CUDA's own clock: **events**. An event is
a marker you drop into the GPU's command stream. The GPU timestamps it when it
reaches that point.

```cpp
cudaEventRecord(startEvent);
// ... launch kernels ...
cudaEventRecord(endEvent);
cudaEventSynchronize(endEvent);                  // wait for the GPU to reach endEvent
cudaEventElapsedTime(&ms, startEvent, endEvent); // GPU time between the two markers
```

This measures real GPU execution time — but notice `cudaEventSynchronize`
*forces a wait*. So measuring kernel time costs you the sync you were trying to
avoid. That's the tension: you can't measure GPU time for free.

The project resolves this by only taking event timings when you've *asked* for
them (validation/profiling mode), and using persistent, reused events (stored on
the workspace) so you don't pay `cudaEventCreate`/`Destroy` every frame.

### The derived "host sync" number

The CSV also reports `narrow_host_synchronization_ms`, computed as:

```cpp
hostSynchronizationMilliseconds = max(0, wallMilliseconds - gpuKernelMilliseconds);
```

This is "how much of the wall time was *not* GPU kernel time" — i.e. pure
host-side overhead and any synchronization waiting. A small number here means
the CPU orchestration is lean.

---

## 11.2 The accumulator — `GpuPipelineProfiling`

Each frame, the broad phase and narrow phase each fill a `StageSnapshot` (a
struct of ~50 numbers — timings, byte counts, contact counts) and hand it to:

```cpp
profiling::recordBroadPhase(stageSnapshot);
profiling::recordNarrowPhase(stageSnapshot);
```

These store the snapshot in a thread-safe global `RuntimeState`. Once per frame,
the benchmark controller calls `finishStep()`, which folds the frame's two
snapshots into a running `AggregateSnapshot` (sums across all frames). A
`std::mutex` guards the writes (the SOFA step is single-threaded, so this is
cheap insurance).

---

## 11.3 The controller — writing the CSV and summary

`GpuPipelineBenchmarkController` is a SOFA component that listens for animation
events:

```cpp
void handleEvent(Event* event) {
    if (dynamic_cast<AnimateBeginEvent*>(event)) onAnimateBegin();   // start timer
    if (dynamic_cast<AnimateEndEvent*>(event))   onAnimateEnd();     // stop timer, record
}
```

- **`onAnimateBegin`** — note the frame start time.
- **`onAnimateEnd`** — compute this frame's duration, pull the broad/narrow
  snapshots, build a `PendingRow`, and (if past the warmup) add it to the
  running averages.

### Warmup

The first `warmupSteps` frames (10 in the benchmark) are recorded but **excluded
from the averages**. The GPU's clock speed ramps up over the first few frames,
and caches need to fill, so early frames are unrepresentative. Excluding them
gives a stable steady-state number.

### Flushing

Every `flushInterval` (20) frames, the buffered rows are appended to the CSV
file on disk. This avoids writing every single frame (slow) while not losing
much if the run is interrupted.

### Two output files

For a run labeled `gpu_one_tissue_one_blade_dense_grid_benchmark`, you get:

- `..._timings.csv` — one row per frame, ~50 columns.
- `..._summary.txt` — the `avg_*` aggregates plus scene metadata.

---

## 11.4 The summary fields, decoded

Open a `_summary.txt` and here's what each line means. (Run
`scripts/peek_fbp_summary.sh <run-dir>` to print the headline ones.)

### Throughput

| Field | Meaning |
|---|---|
| `avg_step_seconds` | average wall time of a full frame |
| `avg_fps` | 1 / avg_step_seconds — frames per second |

### Narrow-phase timing

| Field | Meaning |
|---|---|
| `avg_narrow_wall_ms` | CPU wall time in the narrow phase (orchestration) |
| `avg_narrow_kernel_ms` | GPU kernel time (only meaningful when readback/events on) |
| `avg_narrow_host_synchronization_ms` | derived: wall − kernel (host overhead) |
| `avg_narrow_sofa_triangle_extraction_ms` | time in `extractCudaIndexedSurface` |
| `avg_narrow_h2d_ms` | time spent on host→device copies |
| `avg_narrow_feature_based_proximity_kernel_ms` | the FBP kernel's measured time (FBP path) |

### Data movement (the zero-copy proof)

| Field | Fast-path value | Meaning |
|---|---|---|
| `avg_host_to_device_bytes` | **0** | bytes copied CPU→GPU per frame |
| `avg_device_to_host_bytes` | **0** | bytes copied GPU→CPU per frame |
| `avg_kernel_launch_count` | 7 (FBP) / 5 (exact) | GPU operations per frame |
| `avg_cuda_memset_count` | 1 | the pair-hash clear |
| `avg_workspace_resize_count` | **0** | GPU allocations per frame (0 = reuse working) |

### Contact results (validation mode)

| Field | Meaning |
|---|---|
| `avg_narrow_raw_candidate_count` | candidate pairs before dedupe (~2304 one-tissue) |
| `avg_narrow_unique_candidate_count` | after dedupe (~624 one-tissue; 322,560 large-tissue) |
| `avg_narrow_duplicate_reduction_ratio` | fraction removed by dedupe (~0.73) |
| `avg_narrow_output_contact_count` | contacts emitted (~56 one-tissue; 8018 large-tissue) |
| `avg_narrow_vf_contact_count` / `_fv_` / `_ee_` | feature-type breakdown (all EE for one-tissue) |
| `avg_narrow_overflow_count` | dropped due to capacity (should be 0) |
| `avg_narrow_grid_cell_count` | total cells (32768) |

Note: with the default active-cell generation, the unique-candidate and contact
counts are **identical** to the all-cells path — the optimization changes which
cells are scanned, not which pairs survive. If a code change makes these numbers
drift between the flag on/off, that's the correctness alarm bell (it's how the
Phase 17 grid-stride bug was caught — file 07 §7.6).

The contact-count fields are **0 in the fast path** because counters aren't read
back. You only see real values in validation mode
(`SOFA_PROXIMITY_READ_CONTACT_COUNTER=1`).

---

## 11.5 Reading a real summary

Here's an annotated excerpt from the verified fast-path run (active-cell
generation default-on):

```text
avg_fps=942.7                       ← ~940 frames/sec
avg_narrow_wall_ms=0.559            ← CPU spent 0.56 ms orchestrating
avg_narrow_kernel_ms=0.349          ← (broad-cull window timing)
avg_narrow_host_synchronization_ms=0.210   ← lean host overhead
avg_host_to_device_bytes=0          ← zero-copy positions + cached indices
avg_device_to_host_bytes=0          ← contacts stay on GPU
avg_kernel_launch_count=7           ← the 7-op FBP cascade
avg_cuda_memset_count=1             ← the hash clear
avg_workspace_resize_count=0        ← no per-frame allocation
```

And the validation run that actually counts contacts:

```text
avg_fps=475.1                       ← slower: one sync per frame
avg_narrow_output_contact_count=56  ← 56 contacts found
avg_narrow_ee_contact_count=56      ← all of them edge-edge (file 08 explains why)
avg_narrow_vf_contact_count=0
avg_device_to_host_bytes=64         ← the counter readback
```

The drop from ~940 to ~475 FPS is the *entire* cost of asking the GPU "how many
contacts did you find?" once per frame. That single number tells you why the
fast path keeps everything on the device.

---

## 11.6 Nsight — the deep profiler

For per-kernel internals (occupancy, memory throughput, register usage), the CSV
isn't enough; you need NVIDIA's **Nsight Compute**. The script
`scripts/run_nsight_fbp_profile_wsl.sh` runs it against the three FBP scenes and
exports per-kernel metrics. The findings live in
`reports/gpu_collision_phase11_12_kernel_profile_20260525.md`.

Key things Nsight told us that the CSV couldn't:

- The FBP kernel uses **68 registers/thread**, which caps occupancy at ~50%.
- On the small scene it runs at only **~5% SM throughput** — with 624 pairs and
  a 1,024-block grid, most threads have no pair to process (this is expected and
  accepted; the kernel is only ~17 µs).
- **Atomic pressure is < 0.5%** everywhere — confirming that atomics are *not* a
  bottleneck (a question the team specifically investigated and put to rest).
- Nsight also **confirmed the Phase 15 win directly**: the candidate-generation
  kernel's launched grid dropped from **32,768 blocks (~300 µs)** to
  **1,024 blocks (~8 µs)** when `useToolActiveCellGeneration` is on — visible in
  the `launch__grid_size` and `gpu__time_duration` metrics. This is the kind of
  before/after a CSV throughput number can't show you.

This is the difference between "how long did the frame take?" (CSV) and "why did
each kernel take that long?" (Nsight).

---

## 11.7 Why measure this way at all?

The deepest lesson of the profiling design: **the thing you measure changes the
thing you measure.** Asking the GPU for results (a readback) forces a sync, which
slows the frame. So the benchmark is careful to:

1. Default to no readback (fast path) for honest throughput numbers.
2. Offer opt-in readback for *correctness* validation (counts, overflow).
3. Use Nsight separately for *kernel-internal* analysis.

You pick the tool that answers your question without distorting it. A throughput
number comes from the fast path; a contact count comes from validation mode; a
register-usage number comes from Nsight. Mixing them up leads to wrong
conclusions — like thinking FBP is slow because the validation run shows ~475
FPS, when the production path is actually ~940.

---

## 11.8 Summary

```text
Wall time (chrono) = CPU orchestration; doesn't include overlapping GPU work.
Kernel time (CUDA events) = real GPU time, but measuring it forces a sync.
host_synchronization_ms = wall - kernel = host overhead.
The controller listens to AnimateBegin/End, skips warmup frames, averages,
  and writes a per-frame CSV + an avg summary.
0 H2D / 0 D2H / 0 resizes in the summary = the zero-copy design working.
Contact counts only appear in validation mode (readback on).
Nsight Compute answers the "why" that the CSV can't.
```

That completes the per-frame story. The last file is a glossary you can come
back to. Go to [12_glossary.md](12_glossary.md).
