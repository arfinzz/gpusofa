# Reading the reports: every metric, what it means, how it's computed

This file is the companion to every benchmark report in `reports/`. It explains
**what each number is, where it comes from, exactly how it is calculated, and how
much to trust it.** If a report quotes `avg_narrow_kernel_ms=1.254` or
`vf/fv/ee=1119/428/807` and you want to know what that *means*, this is the file.

All formulas below are taken directly from the measurement code:
- `SofaGpuCollision/src/SofaGpuCollision/GpuPipelineBenchmarkController.cpp`
  (the controller that times each step and writes the CSV + summary),
- `GpuPipelineProfiling.h` (the per-step snapshot struct),
- `cuda/GpuCollisionBackend.cu` (where the GPU-side counters are bumped).

---

## 1. Where the numbers live

Each benchmark leg writes two things into its log directory:

| File | What it is |
|---|---|
| `<label>.csv` | **one row per simulation step** — the raw per-frame measurements |
| `<label>_summary.txt` | **aggregates** over the *measured* steps (warmup excluded) — `key=value` lines |

The reports quote the `_summary.txt` aggregates. The helper
`scripts/summarize_full_suite.sh` just greps these `key=value` lines into a table.

**Warmup vs measured.** The first `warmup_steps` steps (10 in the suite) are run
but **not** counted — they cover process start-up, the first cold kernel launch,
and the GPU clock ramping from idle. Everything in the summary is averaged over
the remaining `measured_steps` only.

---

## 2. The timing model (read this before the metric tables)

Each frame, the collision narrow phase is measured **two different ways at once**:

1. **Wall time** — a CPU stopwatch (`std::chrono`) around the call to
   `endNarrowPhase`. This is *orchestration* time: how long the CPU spent
   issuing the work. Because the fast path issues GPU kernels **without waiting**
   for them (see [tutorial/10](../tutorial/10_phase4_sync_and_output.md)), this
   wall time does **not** include the GPU actually finishing — it's mostly the
   cost of setting up and launching.

2. **Kernel time** — a *GPU* stopwatch (CUDA events recorded immediately before
   and after the narrow kernel). This is how long the collision math actually ran
   on the GPU. It is only recorded when contact-counter readback is on (otherwise
   taking the event would force a sync and defeat the no-wait design).

From these two, a third is derived per step:

> **`host_synchronization_ms = max(0, narrow_wall_ms − narrow_kernel_ms)`**

i.e. the part of the wall time that was *not* the kernel — launch latency,
counter readback, any incidental waiting. (`narrowHostSynchronizationMilliseconds`
in the snapshot.)

**Which number should you trust?** The **kernel time** is the robust signal: it's
measured on the GPU's own clock, it's nearly identical run-to-run, and it
isolates the collision work. **FPS and wall time are noisier** because they
include the whole SOFA scene graph and are sensitive to this laptop's GPU
power-state and temperature. When comparing two approaches, compare kernel time
first; treat FPS deltas under ~10% as inside the noise band.

---

## 3. Headline metrics

| Summary key | Plain meaning | Exact formula | Trust |
|---|---|---|---|
| `avg_fps` | whole-pipeline frames/sec | `1.0 / avg_step_seconds` | noisy (laptop thermal) |
| `avg_step_seconds` | mean wall time of one full simulation step | `total_measured_seconds / measured_steps` | noisy |
| `min_step_seconds` / `max_step_seconds` | best / worst single step | running min/max over measured steps | — |
| `avg_narrow_wall_ms` | mean CPU time orchestrating the narrow phase | `Σ narrow_wall_ms / measured_steps` | medium |
| `avg_narrow_kernel_ms` | mean GPU time of the narrow-phase kernel | `Σ narrow_kernel_ms / measured_steps` | **robust** |
| `host_synchronization_ms` (CSV col) | wall not attributable to the kernel | `max(0, wall − kernel)` per step | medium |
| `avg_kernel_launch_count` | CUDA kernel launches issued per frame | mean of per-step launch count | exact (integer) |

> **Important subtlety about `avg_fps`.** It is `1 / mean(step_time)`, **not**
> `mean(1 / step_time)`. The mean is taken over *step times*, then inverted. That
> means one slow step drags FPS down more than a naive average of per-step FPS
> would. This is the honest way (it equals total frames ÷ total time), but it's
> why a single stutter visibly dents the reported FPS.

**Why FPS can be *lower* even when the kernel got *faster*.** FPS covers the
entire frame — physics integration, the SOFA scene-graph traversal, collision,
and GPU power-state transitions — not just the collision kernel. On this GTX
1650 Ti the clock wanders with temperature, so `avg_fps` can drop while
`avg_narrow_kernel_ms` (the thing this project actually optimizes) holds or
improves. That's why the reports lead with kernel time.

---

## 4. Correctness metrics (the contact counts)

These come from GPU counters that the narrow kernel bumps with `atomicAdd`, then
copied back once per frame **only in "validation" mode**
(`proximityReadContactCounter` on). In the detection-only "fast" mode they show
as `—` / not read, because reading them would force a CPU↔GPU sync.

| Summary key | Meaning | How produced |
|---|---|---|
| `avg_narrow_output_contact_count` | total contacts emitted | `proximityContactCount` counter |
| `avg_narrow_vf_contact_count` | of those, **Vertex–Face** contacts | `proximityVfCount` |
| `avg_narrow_fv_contact_count` | **Face–Vertex** contacts | `proximityFvCount` |
| `avg_narrow_ee_contact_count` | **Edge–Edge** contacts | `proximityEeCount` |
| `avg_narrow_overflow_count` | contacts/triangles **dropped** for lack of capacity | `overflowCount` |

- **VF / FV / EE** classify *which features were closest* on the two triangles.
  VF = a vertex of mesh A nearest a face of mesh B; FV = the reverse; EE = the
  closest points lie on an edge of each. The narrow kernel runs 6 VF + 9 EE tests
  per triangle pair and keeps the single closest. (Full geometry:
  [tutorial/08](../tutorial/08_the_math.md).) The three always sum to the total.
- **These counts are the correctness fingerprint.** Two runs of the *same*
  geometry must produce *identical* VF/FV/EE numbers. That's how we prove the
  hash broad cull is equivalent to the dense one (same `2354 = 1119/428/807`) and
  how the Phase 17 grid-stride fix was validated (large scene: the buggy `~3700`
  became the correct `8018`).
- **`overflow_count` must be 0.** Non-zero means a per-cell bucket
  (`maxTissue/ToolTrianglesPerCell`) or the contact array (`maxContacts`) was too
  small and real contacts were silently dropped — the result is then incomplete.
  All shipped scenes are sized so this stays 0.

For integer counters that are constant every frame (as contact counts are on a
fixed scene), the "average" is just that constant — `1119` means exactly 1119
every measured step.

---

## 5. Broad-cull metrics (how the candidate set was built)

The narrow phase only ever sees *candidate pairs* that survived the broad cull.
These metrics describe that funnel.

| Key / stat | Meaning | How produced |
|---|---|---|
| `raw_candidate_count` | candidate pairs **before** dedup | `rawCandidateCount` — every (tissueTri, toolTri) from every shared cell, counting duplicates |
| `unique_candidate_count` | candidate pairs **after** dedup | `candidateCount` — the deduped set actually tested |
| `active_mixed_cell_count` | grid cells holding **both** tissue and tool | `DeviceDenseGridStats.activeMixedCellCount` |
| `max_tissue/tool_cell_occupancy` | most triangles any one cell held | per-cell max, for sizing bucket capacities |

- **Why raw > unique.** A triangle that straddles several cells is inserted into
  each, so the same pair can be emitted from multiple shared cells. The GPU
  open-addressing dedup table (`pairHashKeys`) collapses these. Example from the
  large-tool scene: `raw 61,920 → unique 33,408`. The narrow phase runs on the
  33,408, not the 61,920.
- **The dense and hash paths must agree here too.** Same geometry ⇒ same
  `raw` and `unique` counts on both legs (verified: 61,920 / 33,408 on both).
  This is a second, independent correctness check beyond the contact counts.

---

## 6. Hash-path-only metrics (experiment)

When the spatial-hash + prefix-sum broad cull is enabled
(`useHashPrefixSumGeneration`), the standalone backend bench and the experiment
report quote a few extra numbers from `HashPrefixSumStats`:

| Stat | Meaning |
|---|---|
| `hash_table_size` | number of slots in the open-addressing table; auto = `nextPow2((firstTris+secondTris)×4)` |
| `occupied_slots` | slots that actually got claimed = number of distinct occupied cells (vs the dense grid's fixed 32,768) |
| `unique_pairs` | deduped candidate pairs the hash path produced (must equal the dense `unique_candidate_count`) |
| `probe_overflow_count` | triangle insertions that exceeded `maxProbe` linear-probe steps and gave up (must be 0) |
| `bucket_overflow_count` | triangles dropped from a full slot bucket (must be 0) |

`occupied_slots` is the whole point of the hash path: it scales with how much
geometry there actually is, not with a fixed cell count. On the verification run
it was `16,536` occupied slots in a `65,536`-slot table — the dense grid would
have materialized all `32,768` cells regardless.

---

## 7. Data-movement metrics (the zero-copy proof)

| Key | Meaning | Expected on the fast path |
|---|---|---|
| `avg_host_to_device_bytes` | bytes copied CPU→GPU per frame | **0** (positions are read in place via `devicePositions`; topology is cached) |
| `avg_device_to_host_bytes` | bytes copied GPU→CPU per frame | **0** in detection-only; only counters (a few bytes) in validation |
| `avg_narrow_device_allocation_ms` / `_bytes` | GPU memory allocated this frame | **~0** in steady state — the workspace is persistent and grows only once |

Seeing `0 / 0` here is the evidence for the zero-copy claim: the collision runs
entirely on the simulation's existing GPU buffers, copying nothing each frame.
(Detail: [tutorial/05](../tutorial/05_phase2_narrow_prep.md) and
[tutorial/10](../tutorial/10_phase4_sync_and_output.md).)

---

## 8. Per-stage breakdown columns (optional, detailed profiling)

With `detailedProfiling` on, the CSV also carries a fine-grained breakdown of the
broad cull — `avg_narrow_clear_grid_ms`, `avg_narrow_insert_tissue_ms`,
`avg_narrow_insert_tool_ms`, `avg_narrow_generate_pairs_ms`, etc. Each is the
arithmetic mean over measured steps of that sub-stage's time. These are how the
Phase 15 win was localized (`generate_pairs` 300 µs → 7.9 µs). They are off by
default because taking those intermediate timings requires syncs that perturb the
very thing being measured; use them to *diagnose*, not to quote headline FPS.

---

## 9. How to read an A/B comparison correctly

When a report shows two legs (e.g. `dense` vs `hash`, or `today` vs
`documented`), apply this checklist:

1. **Correctness first.** Do `output_contact_count` and `vf/fv/ee` match exactly?
   Do `raw`/`unique` candidate counts match? Is `overflow` 0 on both? If any
   differ, the two are *not* computing the same thing and a speed comparison is
   meaningless.
2. **Then kernel time.** Compare `avg_narrow_kernel_ms`. This is the real,
   low-noise verdict on which approach does less GPU work.
3. **Then wall / FPS, with skepticism.** Compare `avg_narrow_wall_ms` and
   `avg_fps`, but discount swings under ~10% as thermal noise — especially the
   first leg of a fresh process (cold GPU clock; the small fast-path first leg
   famously reads ~431 FPS cold vs ~724 warm).
4. **Same session, back-to-back, same geometry.** Only trust an A/B where both
   legs ran in the same session under the same thermal conditions on identical
   geometry. The `branch_comparison` and `hash_prefixsum` reports are structured
   this way on purpose.

---

## 10. One-line glossary

- **Step / frame** — one tick of simulated time (`dt = 0.005 s` in the scenes).
- **Broad phase / broad cull** — cheap rejection of pairs that can't touch.
- **Narrow phase** — exact per-pair geometry that produces contacts.
- **Candidate pair** — a (triangleA, triangleB) that shared a grid cell and so is worth testing.
- **Contact** — an emitted `ProximityContact`: where two features are closest, plus normal and signed distance.
- **Fast / detection-only mode** — kernels issued, nothing read back, CPU doesn't wait. Highest FPS, no contact counts.
- **Validation mode** — counters read back each frame so contact counts are visible; slightly slower, used to prove correctness.
- **Kernel time vs wall time** — GPU-clock measurement of the kernel vs CPU-clock measurement of orchestration. Trust kernel time.

For the deeper, tutorial-style version of all of this, see
[tutorial/11_profiling.md](../tutorial/11_profiling.md) and the new
[tutorial/00_high_level_flow.md](../tutorial/00_high_level_flow.md).
