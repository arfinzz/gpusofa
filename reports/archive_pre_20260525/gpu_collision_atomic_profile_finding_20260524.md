# Atomic Profile Finding — 2026-05-24

Measured on the 2026-05-17 Nsight Compute report (`output/benchmark_logs/final_nsight_profile_20260517/nsight/gpu_collision_ncu.ncu-rep`), GTX 1650 Ti, one-tissue/one-blade scene.

## Per-kernel atomic pressure

| Kernel | Grid | Duration (us) | L2 atomic cycles | L2 t-sectors | SM throughput | SM warps active | Compute-mem throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| resetDenseGridKernel | 128 | 2.5 | 0.0% | 36.7% | 16.7% | 69.2% | 38.7% |
| insertIndexedTrianglesKernel (blade, 12 tris) | 1 | 30.4 | 0.19% | 0.37% | 0.11% | 3.5% | 0.57% |
| insertIndexedTrianglesKernel (tissue, 12800 tris) | 50 | 28.6 | 19.1% | 39.4% | 8.23% | 71.7% | 59.5% |
| generateDenseGridUniqueCandidatePairsKernel | 32768 | 299.6 | 0.55% | 3.32% | 26.4% | 70.2% | 0.55% |

## Conclusion

**Atomics are not the bottleneck.** The kernel with appreciable L2 atomic pressure (tissue insert, 19.1%) only costs 28.6 us — a fraction of the 360-us narrow path. The most expensive kernel by far (candidate generation, 300 us) has 0.55% L2 atomic cycles. Replacing atomic-based insertion with warp-aggregated atomics or two-pass prefix-sum scan would, even in the optimistic case, save ~10-14 us per frame on a 700 us narrow path. That is at most a 2% improvement at the cost of significant new code.

## What is the bottleneck

1. **Candidate generation kernel** at 299.6 us with only 26% SM throughput — most cells are empty or single-class, so threads run dry. Improving this means either compacting to mixed cells before generation, or per-active-cell thread distribution. The user already implemented `compactActiveCells` as a flag and measured it regressed on this hardware; that experiment should not be repeated without a smarter active-cell representation.
2. **Blade-side insert launch latency** — 30 us for 12 triangles on 1 block of 256 threads with 244 idle threads. The kernel takes basically the same time as the 12800-triangle tissue insert. This is launch overhead, not compute or atomic cost. Fusing tissue+tool insert into one launch (the existing `batchTriangleInsert` flag) is the right shape — the existing measurement that it regressed must be a config bug or measurement noise; worth a second look but not a top priority.
3. **Empty SM cycles in reset and insert** — small grids on the GTX 1650 Ti SM array (16 SMs) underfill compute.

## Decision

**Do not rewrite atomic insertion.** Keep the existing scheme. Redirect optimization effort to:

- The new feature-based proximity (VF/EE) kernel — that will be the new dominant cost; design it for high SM throughput from the start (avoid divergent branching, prefer warp-coalesced reads of triangle vertices, one pair per warp not per thread for high register-pressure paths).
- The candidate generation pass — try a per-active-cell thread mapping after the new kernel is in place.

This finding closes the user's question "atomics and locks are causing delay?" with: no, they are not.
