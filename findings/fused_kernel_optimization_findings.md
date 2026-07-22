# Fused-kernel optimization findings

## Final decision

Keep exactly one fused-kernel improvement: **reuse exact raw triangle AABBs**.

Reject and do not reintroduce the following without new hardware, a new workload, and a fresh hypothesis:

- two-pass closest-feature winner selection;
- warp-aggregated contact allocation;
- factor-2 eight-bin template specialization;
- counter-free production behavior as the default;
- cached-AABB experimental build fill;
- CUB block-scan staging in the shared-hash builder;
- heavy-first mixed-cell sorting;
- a single hard-coded 128-thread fused launch.

The production code now has one fused kernel, not a runtime experiment dispatcher. There are no optimization bit masks and no experimental template combinations.

## Why raw-AABB reuse won

For every tissue/tool triangle pair, the closest-feature calculation first needs the exact triangle AABBs to reject pairs whose boxes are farther apart than `contactDistance`.

Previously, the fused kernel already had the six triangle vertices, but the closest-feature helper recomputed both exact boxes for every pair. A tool triangle can be paired with many tissue triangles, so its same three min/max calculations were repeated many times.

The winning implementation changes the ownership of that work:

- when a tool triangle enters the 256-entry shared-memory tile, compute and store its exact raw AABB once;
- when a tissue entry begins scanning a matching tool run, compute its exact raw AABB once;
- use those two saved boxes for every pair's squared-gap rejection;
- call the closest-feature helper with `aabbAlreadyRejected=true`, avoiding duplicate box construction;
- keep the original inflated-AABB overlap and home-cell ownership tests unchanged;
- keep pair, contact, overflow, VF, FV, and EE counters unchanged.

This is exact reuse, not an approximation. It cannot remove a valid contact because every triangle is contained in its raw AABB and the test compares the squared box gap to the exact squared contact threshold.

## Campaign coverage

The experiment campaign executed 824 benchmark/profile legs:

| Campaign | Legs |
|---|---:|
| Original plus fused masks 0-15 parity | 17 |
| Initial 80k/200k timing | 34 |
| Fused/build factorial parity | 85 |
| Fused/build factorial timing | 170 |
| Warp masks 16-31 parity | 16 |
| Scalar and warp overflow stress | 2 |
| Four-round focused fused comparison | 112 |
| Three-round fused/build cross | 180 |
| Threads/blocks/heavy-first sweep | 160 |
| Nsight deep profiles | 10 |
| SOFA scene A/B runs | 30 |
| Factor/tile/build raw-AABB parity | 8 |

Correctness invariants across the relevant matrices were:

- 80k: 43,584 tested pairs and 8,018 contacts;
- 80k subtype split: 2,828 VF, 3,469 FV, 1,721 EE;
- 200k: 89,856 tested pairs and 17,040 contacts;
- 200k subtype split: 8,805 VF, 4,914 FV, 3,321 EE;
- scalar and warp overflow stress: 8,018 generated, capacity 100, overflow 7,918;
- no CSR entry overflow or shared spill in the champion shared-hash configuration.

The final production implementation was then rebuilt and independently revalidated across factors 1/2/4, tiles 32/256, CSR, fixed hash, shared-hash, and shared-sort configurations.

## Performance result

Clock and thermal drift on the GTX 1650 Ti were large, so individual process timings were not treated as proof. Selection used order-rotated repetitions and end-to-end scene repetitions.

The stats-preserving raw-AABB variant produced approximately:

- 2.58% lower fused-kernel time on the stable 14k SOFA rounds;
- 1.24% lower fused-kernel time on the 200k SOFA scene;
- about 3% lower round-normalized backend time at 200k.

The cleaned production kernel uses 122 registers/thread, 17,936 bytes of static shared memory, no stack frame, and no local-memory allocation in the final `cuobjdump` resource report.

## Rejected ideas

### Two-pass closest-feature winner

**Idea:** first find only the winning feature and distance, then run a second pass to build its complete contact.

**Why it looked attractive:** register allocation dropped from roughly 124 to 112 registers/thread.

**Why it failed:** it repeated feature computations, introduced a 72-byte stack frame, and reduced eligible warp issue. Nsight duration regressed about 19% at 80k and 29% at 200k.

**Decision:** reject. Lower register count is not automatically faster.

### Warp-aggregated output allocation

**Idea:** one leader performs a single `contactCount` atomic for all active lanes in a warp.

**Why it looked attractive:** fewer global atomics.

**Why it failed:** hits occur on sparse, divergent lanes. The aggregation logic increased allocation to 129 registers/thread, reduced theoretical occupancy to 25%, and regressed Nsight duration about 58% at 80k and 44% at 200k.

**Decision:** reject. The scalar output atomic is not the dominant problem here.

### Eight local bins for factor 2

**Idea:** factor 2 has only eight subcells, so shrink shared bin arrays from 64 to 8.

**Result:** static shared memory fell by only 448 bytes, from 17,936 to 17,488. This did not change the occupancy limit and produced no consistent speedup.

**Decision:** reject the separate template. The generic 64-bin implementation is simpler and correct for factors 1, 2, and 4.

### Counter-free kernel

**Idea:** remove tested-pair and VF/FV/EE atomics.

**Result:** total contacts remained correct and the 200k scene improved slightly, but diagnostic counters became zero by design.

**Decision:** reject as production behavior. Reports, parity checks, and callers depend on those counters. If a future API explicitly makes statistics optional, revisit it as an API-level choice rather than an environment-mask trick.

### Cached-AABB build fill

**Idea:** reuse AABBs created by the count pass instead of reconstructing them in the fill pass.

**Result:** no repeatable improvement across both scales. The builder is not the same bottleneck on every workload, and added variants enlarged code without a stable win.

**Decision:** reject.

### CUB block-scan staging

**Idea:** replace per-entry shared staging atomics with a block exclusive scan.

**Result:** the build kernel rose from 29 to 31 registers/thread and from 28,688 to 29,872 bytes of shared memory. Extra scan synchronization outweighed saved atomics.

**Decision:** reject.

### Heavy-first mixed-cell order

**Idea:** sort mixed big cells by `tissueCount × toolCount` so expensive cells start first.

**Result:** it added a kernel and bitonic barriers, only handled lists up to 256 cells, and was neutral or slower.

**Decision:** reject. A real load-balancing redesign would need a work queue, not a small pre-sort.

### One universal 128-thread launch

**Result:** 128 threads often helped the 80k case, while 256 threads were safer at 200k. One launch shape traded one scale for another.

**Decision:** retain 256 threads and the established 1,024-block cap. Revisit only with an independently replicated adaptive rule.

## Profile interpretation

The fused kernel is not DRAM-bandwidth bound:

- DRAM throughput was roughly 0.7-2.0% of peak;
- only about 7-8 lanes per warp were active in the irregular pair loop;
- eligible warps were roughly 0.16-0.33 per scheduler;
- achieved occupancy was roughly 20-29% for the viable kernels.

The dominant problem is irregular control flow and dependency latency. Optimizations aimed only at memory bandwidth or atomic count are therefore unlikely to help unless they also improve lane coherence or eligible-warp supply.

## Rule for future work

Do not repeat any rejected experiment merely because it is a common CUDA optimization. Start with a new profile on the target GPU and scene.

The next materially different research direction is a two-kernel survivor pipeline:

1. broad/home-cell/raw-AABB rejection and survivor compaction;
2. a separate closest-feature kernel over a denser survivor queue.

That is an architectural experiment with extra memory traffic. It is not an already-proven improvement.
