# Phase 11/12 Kernel-Level Profile — 2026-05-25

Nsight Compute capture targeting the new feature-based proximity kernels
introduced in Phase 11 (`featureBasedProximityKernel`) and Phase 12
(`insertIndexedPointsKernel`, `featureBasedVertexTriangleProximityKernel`).
GTX 1650 Ti, sm_75. Profile root:
`output/benchmark_logs/fbp_nsight_20260525/`.

Captured via `scripts/run_nsight_fbp_profile_wsl.sh`. Each scene contributes
~4-8 kernel launches across the 4 measured steps. The numbers below are
per-launch averages.

## Per-kernel summary

### `featureBasedProximityKernel` (tri-tri FBP)

Launched in: `tri_tri_fbp` scene (one-tissue + one-blade).

| Metric | Value | Reading |
|---|---:|---|
| Duration | 17.5 µs | Cheap in absolute terms |
| Grid size | 256 blocks × 256 threads = 65 536 threads | Capped over-launch |
| Active workload | ~624 candidate pairs | Only ~0.95% of threads have work |
| `launch__registers_per_thread` | **68** | High; limits occupancy |
| `sm__throughput.pct` | 4.8% | Very low |
| `sm__warps_active.pct` | 43% | Modest occupancy (register-limited) |
| `gpu__compute_memory_throughput.pct` | 2.8% | Not compute-bound, not memory-bound |
| `gpu__dram_throughput.pct` | 1.4% | DRAM idle |
| `lts__d_atomic_input_cycles_active.pct` | **0.01%** | Atomics are nothing |
| `lts__t_sectors.pct` | 1.5% | L2 traffic minimal |
| `sm__inst_executed.pct` | 4.7% | Most cycles are early-exit |

**What bounds this kernel:** register pressure + the over-launch pattern.
The 6 VF + 9 EE = 15 closest-point tests per pair use ~68 registers worth
of float-precision intermediate state (positions, edge vectors, dot
products, barycentrics, current-best). At 68 regs/thread, GTX 1650 Ti can
host only ~32 KB regs / 68 regs ≈ 30 warps per SM, which is 47% of theoretical
peak. So even with a fully populated workload the kernel would top out at
~50% occupancy.

**Why not tune yet:** the absolute kernel time is 17 µs. At 775 FPS that's
~1.3% of the frame budget. The fast-path FPS already matches the pre-FBP
detection-only baseline (775 vs 633). Burning engineering time to shave
the FBP kernel down to ~5 µs would not change the user-visible FPS.

**Future tuning options if this kernel ever dominates:**

1. **Warp-per-pair**: Launch with `gridDim = (uniqueCandidateCount + 31) / 32`
   blocks of 32 threads. Each warp cooperatively handles one pair, with
   each thread taking one of the 15 feature tests. Use `__shfl_sync` to
   share triangle vertices. Should ~5-8× SM throughput.
2. **Register spill to shared memory**: Move intermediate vectors (best
   point on first, best point on second, best barys) to shared memory.
   Drops registers/thread to ~40 and unlocks more occupancy at the cost
   of shared bank conflicts.
3. **Split kernels**: Run VF tests in one kernel and EE tests in another,
   merging results in a third pass. Halves register usage but adds two
   launches.

### `featureBasedVertexTriangleProximityKernel` (v-t)

Launched in: `vt_self_collision` and `vt_cross_model` scenes.

| Metric | Self-collision | Cross-model | Reading |
|---|---:|---:|---|
| Duration | 10.5 µs | 5.6 µs | Both cheap |
| Grid size | 256×256 = 65 536 | 256×256 = 65 536 | Same over-launch |
| Active workload | ~2700 pairs | ~254 pairs | More work → longer |
| `launch__registers_per_thread` | **32** | 32 | Less than half of FBP's 68 |
| `sm__throughput.pct` | 13.5% | 14.0% | Better than FBP |
| `sm__warps_active.pct` | 53% | 62% | Better occupancy |
| `gpu__compute_memory_throughput.pct` | 41% | 14% | Mixed compute / memory |
| `gpu__dram_throughput.pct` | 4.1% | 4.9% | Touches DRAM modestly |
| `lts__d_atomic_input_cycles_active.pct` | 0.47% | 0.30% | Negligible |
| `lts__t_sectors.pct` | 29% | 6.5% | L2 traffic dominated by triangle index lookups |
| `sm__inst_executed.pct` | 9.3% | 9.5% | About 2× FBP density |

**Reading:** v-t kernels are 2× cheaper and 2× better-utilized than tri-tri
FBP because there's only ONE closest-point test per (vertex, triangle)
pair instead of 15. Register count drops from 68 to 32, occupancy doubles.

**What bounds the self-collision case at 41% compute-memory throughput:**
that's actually a healthy mix — the kernel reads 3 triangle vertex
positions + 1 point position per pair (4 × 12 bytes = 48 B), runs ~25 dot
products + region-selection branches, writes a 64-byte ProximityContact.
The 29% L2 sector activity is the triangle index lookups (one indexed read
per triangle).

**No tuning needed.**

### `insertIndexedPointsKernel` (Phase 12)

| Metric | Self-collision (512 verts) | Cross-model (64 tool verts) | Reading |
|---|---:|---:|---|
| Duration | 6.9 µs | 5.8 µs | Bounded by launch overhead at this size |
| Grid size | 2 blocks × 256 | 1 block × 256 | 512 / 256 = 2; 64 < 256 = 1 |
| `launch__registers_per_thread` | 30 | 30 | Reasonable |
| `sm__throughput.pct` | 1.1% | 0.15% | Tiny workload underfills the SMs |
| `sm__warps_active.pct` | 24% | 8.5% | Underfilled |
| `gpu__compute_memory_throughput.pct` | 14% | 2% | Memory-bound when work exists |
| `lts__d_atomic_input_cycles_active.pct` | 2.9% | 0.28% | Atomic-adds for cell bucket counters |

**Reading:** the insertion kernel is launch-overhead-bound for small point
clouds (which is the surgical-tool case). For a tool with 64 points the
SMs see one block of 256 threads (192 of them idle); duration is dominated
by ~5-6 µs of launch+ramp overhead.

**No tuning needed.** The kernel is doing exactly the right thing — it's
just that a 64-point cloud doesn't have enough work to fill the GPU.

### Comparison: `generateDenseGridUniqueCandidatePairsKernel` across scenes

This is the heaviest broad-cull kernel. Same code, different scene
geometry:

| Scene | Cells | Duration | SM throughput | L2 atomic |
|---|---:|---:|---:|---:|
| tri-tri FBP | 32 768 | 300 µs | 26% | 2.2% |
| v-t self-collision | 4 096 | 61 µs | 17% | 7.5% |
| v-t cross-model | 4 096 | 41 µs | 23% | 2.0% |

The 8× cell-count difference between tri-tri (64×8×64) and v-t (32×4×32)
grids drives most of the 5× duration spread. SM throughput is highest in
the cross-model case because tool points are sparse → cells with both
classes are few → less work per cell.

The v-t self-collision case has the highest atomic pressure (7.5%)
because the self-collision pair has *both* sides inserting into many
shared cells, so the candidate-generator's atomicAdd into pair-hash and
candidate-count is more contended. Still well below "atomic-bound".

## Headline findings

1. **FBP and v-t kernels are not bottlenecks.** Total kernel time for the
   one-tissue tri-tri FBP scene is ~360 µs (broad cull + FBP), well under
   the 0.7 ms narrow-wall budget.

2. **Atomics are negligible everywhere.** The 2026-05-24 atomic decision
   stands: rewriting atomics into warp-aggregated or prefix-sum variants
   would save microseconds at best.

3. **FBP register pressure (68/thread) is the only kernel-internal
   constraint** that bounds future scaling. Mitigation strategies exist
   (warp-per-pair, register-to-shared spill, kernel split) but are not
   needed at current scene sizes.

4. **Over-launching with per-thread early exit is the right design** for
   the production fast path. It eliminates a synchronous count readback;
   the wasted-block cost is invisible against the broad-cull's 300 µs
   floor.

5. **V-t kernels are cleaner than FBP** in every metric (lower regs,
   better occupancy, better SM throughput). The vertex-triangle test is
   a single closest-point evaluation per pair vs 15 for tri-tri. This
   makes the v-t paths intrinsically faster — the measured 1385 / 1968
   FPS for the v-t scenes vs 775 for tri-tri is consistent with the
   per-kernel breakdown.

## Source artifacts

- `output/benchmark_logs/fbp_nsight_20260525/tri_tri_fbp/profile.ncu-rep`
- `output/benchmark_logs/fbp_nsight_20260525/vt_self_collision/profile.ncu-rep`
- `output/benchmark_logs/fbp_nsight_20260525/vt_cross_model/profile.ncu-rep`
- Per-scene CSV exports alongside each `.ncu-rep`.
- Extraction helper: `scripts/extract_fbp_kernel_metrics.sh`.

To rerun:

```powershell
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa -- bash scripts/run_nsight_fbp_profile_wsl.sh
```
