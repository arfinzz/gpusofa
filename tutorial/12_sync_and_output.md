# 12 — The sync bypass & frame end (the speed secret)

The kernels are launched. Now comes the step that makes or breaks the
benchmark's speed: what the CPU does *after* launching them. This file explains
why doing **nothing** is the fast answer, and what the alternatives cost.

---

## 12.1 The naive expectation

A beginner's mental model of the frame is:

```text
CPU: launch kernels
CPU: WAIT for the GPU to finish      ← "I need the results, right?"
CPU: read the contacts
CPU: end the frame
```

That "WAIT" is the killer. Waiting for the GPU (`cudaDeviceSynchronize`) means
the CPU sits idle while the GPU computes, and then the GPU sits idle while the
CPU does its next-frame setup. The two never overlap. You've serialized them.

---

## 12.2 What actually happens (the fast path)

In detection-only mode (`copyContactsToHost=False`,
`readCountersWhenContactsStayOnDevice=False`), the narrow phase does this:

```text
CPU: launch all 7 GPU operations (non-blocking)
CPU: end the frame immediately — DO NOT WAIT
```

That's it. The CPU fires the kernels and returns. It does **not** call
`cudaDeviceSynchronize`. It does **not** call `cudaMemcpy` to read results.
The GPU keeps churning through the kernels in the background while the CPU
starts the *next* frame's bookkeeping.

Because kernel launches are asynchronous (file 01), the CPU and GPU run
**concurrently**. The CPU is always a little ahead, queuing up work; the GPU is
always a little behind, executing it. Neither waits for the other.

### The consequence for timing

The benchmark's `narrow_wall_ms` measures CPU wall-clock time around the narrow
phase — but since the CPU doesn't wait for the GPU, that measurement captures
only the **CPU-side orchestration time** (extracting surfaces, building config,
issuing launches), *not* the GPU compute time.

This is why you see, in the fast path (with active-cell generation default-on):

```text
avg_narrow_wall_ms = 0.56       ← CPU orchestration only
avg_narrow_kernel_ms = 0.35     ← the broad-cull window the code does time
```

The FBP math kernel runs ~17 µs *on the GPU*, but it overlaps the next frame's
CPU work, so it never adds to the wall time the CPU observes. The ~940 FPS
number comes from this overlap. (Before the Phase 15 active-cell generation, the
same path was ~775 FPS — the difference is the candidate-generation kernel
dropping from ~300 µs to ~8 µs, see file 06 §6.8 and file 09 §9.4.)

---

## 12.3 Why the over-launch + grid-stride is essential here

Recall from file 09 §9.6 that the FBP kernel launches a **fixed grid** and
grid-strides over the candidate pairs, reading the pair count *from GPU memory*.
The reason connects directly to the sync bypass.

To launch *exactly* the right number of threads, the CPU would need to know
`candidateCount` — but that number lives in GPU memory. Reading it means a
`cudaMemcpy(D2H)`, which is **synchronous**: it forces the CPU to wait for the
candidate-generation kernel to finish before it can launch the math kernel. That
single readback would re-introduce the stall we're trying to avoid.

By launching a fixed, generous grid and letting the kernel read the count itself
and grid-stride over the work, the CPU never needs to know the count. No
readback, no sync, no stall. This is the design choice that took the FBP path
from 109 FPS (with the readback) to ~940 FPS (without it, and with the
active-cell generation of file 09 §9.4). The grid-stride loop is also what keeps
the kernel *correct* when a large scene has more pairs than launched threads
(file 09 §9.6).

---

## 12.4 The data-transfer scorecard

In the fast path, tally the bytes crossing PCIe for the *whole frame*:

| Transfer | Bytes | When |
|---|---|---|
| Positions H2D | 0 | never (zero-copy `deviceRead`) |
| Indices H2D | 0 | cached after frame 1 |
| Counters D2H | 0 | not read in fast path |
| Contacts D2H | 0 | kept on device |

**Total: 0 bytes H2D, 0 bytes D2H per frame.** This is the CSV's
`avg_host_to_device_bytes = 0` and `avg_device_to_host_bytes = 0`. The frame is
pure GPU compute with zero memory traffic across the bus.

---

## 12.5 When you DO want the results — the readback modes

Detection-only is great for benchmarking, but a real simulation eventually needs
the contacts. There are three escalating levels of "give me the data," each
costing more:

### Level 1 — counter readback (validation/profiling)

`readCountersWhenContactsStayOnDevice=True` (or `proximityReadContactCounter=True`
for FBP). The backend copies back a handful of small counters — how many
candidates, how many contacts, the VF/FV/EE breakdown, overflow flags.

The FBP path does this efficiently: five `cudaMemcpyAsync` into one **pinned
host buffer**, followed by a single `cudaDeviceSynchronize`. (Pinned memory is
page-locked host memory that transfers faster.) Cost: ~20 bytes D2H + one sync.
You use this to *validate* that the kernel is finding the right number of
contacts, not for production.

In the benchmark, turning this on drops one-tissue FBP from ~940 → ~475 FPS —
the one sync per frame is the cost.

### Level 2 — sampled counter readback

`proximityCounterReadbackInterval = N`. Read the counters only every Nth frame.
This lets a long profiling run collect occasional VF/FV/EE samples without
paying the sync every single frame. The narrow phase tracks `m_frameCounter` and
only takes the readback path when `m_frameCounter % N == 0`.

### Level 3 — full contact download + publication

`copyContactsToHost=True`. The backend downloads the entire contact array and
the narrow phase publishes it into SOFA's `DetectionOutput` so a CPU response
pipeline (contact manager, constraint solver) can consume it.

This is the expensive mode. For the cross-model scene it drops 1968 → 332 FPS
because ~19 KB of contact records cross PCIe each frame. You only enable it if
you genuinely need the contacts on the CPU — e.g. for SOFA's standard collision
response.

### The flag coupling

For FBP, `copyContactsToHost=True` automatically forces
`keepContactsOnDevice=False` (the narrow phase does this in code), so the
contacts actually get downloaded before publication. You don't have to remember
to set both.

---

## 12.6 The frame ends

After the narrow phase returns (having launched the kernels and, in fast mode,
not waited), control goes back up to SOFA:

```text
CollisionPipeline::computeCollisionResponse()   → no-op (no ContactManager)
DefaultAnimationLoop finishes the step
GpuPipelineBenchmarkController's "AnimateEnd" handler fires:
    → records this frame's timings (file 14)
    → every 20 frames, flushes the CSV to disk
The frame is over. The clock ticks to the next frame.
```

Meanwhile the GPU may still be finishing this frame's FBP kernel — that's fine,
it'll be done before the next frame's kernels need the workspace, and CUDA's
stream ordering guarantees the operations run in the order they were queued.

---

## 12.7 Putting the whole frame together

Here's the complete frame for the one-tissue/one-blade FBP fast path:

```text
DefaultAnimationLoop::step()
│
├─ (physics)            nothing
│
├─ CollisionPipeline::computeCollisionDetection()
│   │
│   ├─ GpuCollisionBroadPhase
│   │    beginBroadPhase / addCollisionModel×2 / endBroadPhase
│   │    → emits pair (tissue, blade)                          [~0.01 ms CPU]
│   │
│   └─ GpuCollisionNarrowPhase
│        beginNarrowPhase                                      [bump frame counter]
│        addCollisionPair(tissue, blade)                       [queue it]
│        endNarrowPhase:
│           extractCudaIndexedSurface ×2  → zero-copy pointers [0 bytes H2D]
│           build DenseGridConfig
│           launch 7 GPU ops (non-blocking):
│              reset grid, memset hash, insert tissue,
│              insert blade (+ build active list),
│              generate+dedupe pairs (over ~30 active cells),
│              reset counters, FBP math
│           RETURN without waiting                              [~0.56 ms CPU wall]
│
├─ CollisionPipeline::computeCollisionResponse()  nothing
│
└─ GpuPipelineBenchmarkController records the frame             [logs timings]

GPU (in the background, overlapping the next frame): finishes the 7 ops,
   writes ~56 contacts to VRAM (from 624 candidate pairs),     [0 bytes D2H]
   never copies them out.
```

Total per-frame PCIe traffic: **0 bytes both ways**. CPU wall time: ~0.56 ms.
Frame rate: ~940 per second.

---

## 12.8 Summary

```text
The fast path launches kernels and DOES NOT WAIT (no sync, no readback).
CPU and GPU overlap → the GPU's compute time hides behind the next frame's CPU work.
The over-launch + grid-stride removes the one readback that would have forced a sync.
Result: 0 bytes H2D, 0 bytes D2H, ~0.56 ms CPU wall, ~940 FPS.
Three opt-in readback levels (counters / sampled / full publication) trade FPS
  for getting data back to the CPU — used only when you actually need it.
```

That completes the per-frame story. Part IV is about **running and measuring** it:
first the catalogue of every scene and every kind of run. Go to
[13_scenes_and_runs.md](13_scenes_and_runs.md).
