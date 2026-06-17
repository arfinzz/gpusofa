# Three more hash-path optimizations + FBP kernel profiling

**Date:** 2026-06-17
**Hardware:** NVIDIA GeForce GTX 1650 Ti (Turing sm_75), WSL2, SOFA 25.12.
**Build:** `SofaGpuCollision/build-profile/libSofaGpuCollision.so` (clean rebuild).
**Status:** All three working, **correctness-verified bit-identical**, faster.

Three optimizations applied on top of the already-optimised hash broad cull
([hash_optimized_broadphase_20260617.md](hash_optimized_broadphase_20260617.md)),
plus an Nsight Compute profile of the narrow kernel to guide what's next. All
changes are in `cuda/GpuCollisionBackend.cu` (no new files; no build-system
change). Contacts stay **2354 (1119/428/807), overflow 0** on the large scene and
match the documented baseline on every suite scene.

---

## 1. The three optimizations

### #3 — Drop the vestigial CUB scan
The optimised hash cull still ran `cub::DeviceScan::ExclusiveSum` + a 1-thread
`setCompactHashRawTotalKernel` every frame, producing `pairOffsets`/`rawTotal`.
But the candidate generator is now **block-per-bucket** (one block per mixed
bucket, divide/modulo indexing) and needs **no global offsets**, and the raw-pair
total is already accumulated by `computeCompactHashPairsPerBucketKernel` into
`rawCandidateCount`. So the scan was pure dead work. Removed it (and the CUB
temp-storage allocation in `ensure()`). A follow-up cleanup then deleted the dead
`setCompactHashRawTotalKernel` definition, the inert `pairOffsets`/`rawTotal`/
`scanTempStorage` workspace buffers, and the now-unused `<cub/cub.cuh>` include, so
**no CUB dependency remains**. **Launches 13 → 11.**

### #2 — Cheap AABB pre-reject in the FBP narrow kernel
`featureBasedProximityKernel` previously ran all **15 closest-feature tests**
(3 VF + 3 FV + 9 EE) for *every* candidate pair, then checked the distance at the
end. But same-cell pairs can be far apart (a grid cell is ~4× the contact
distance), so much of that math was wasted. Added a **conservative squared-AABB
gap test** before the 15 tests: build both triangles' boxes, compute the per-axis
gap, and `continue` if `gx²+gy²+gz² > contactDistance²`. This is exact — the
triangles are contained in their boxes — so it **never drops a real contact**
(output bit-identical). Benefits **both** the hash and the dense FBP paths (shared
kernel).

### #1 — CUDA Graphs for the per-frame kernel sequence
The hash broad cull is now **11 sub-10 µs kernels**; per-launch CPU overhead and
inter-kernel GPU scheduling gaps are a meaningful fraction of the ~0.4 ms total.
Added a `launchAll(stream, fullClear)` helper that records the whole sequence on a
stream, and **captures it once into a `cudaGraphExec` and replays it** each
steady-state frame (`cudaGraphLaunch`). Details:
- First frame (and after any workspace resize) runs **directly** with a full
  dedup-table init; the graph (touched-clear topology) is captured on the next
  steady-state frame.
- A **signature** (table size, bucket capacity, triangle counts, max contacts,
  contact distance, compact-pair mode) triggers re-capture if it changes; a
  resize (`newlyAllocatedBytes > 0`) invalidates the graph.
- The graph is **launched on the default stream**, so the optional counter
  readback stays correctly ordered after it.
- Any capture/instantiate/launch failure **falls back** to the direct sequence.
- Default **ON**; disable with `SOFA_HASH_CUDA_GRAPH=0`. Skipped under detailed
  profiling (which needs per-stage events).

---

## 2. Measured effect (large tissue + large tool, 14,368 elements)

All same-session, contacts **bit-identical (2354)**, overflow 0, launches 11.
Trust the kernel time (GPU-event measured).

**#2 + #3 together** (vs the prior optimised hash, same session):

| Path | kernel ms before | kernel ms after | Δ |
|---|---:|---:|---|
| hash (large+large) | 0.700 | **0.410** | **−41 %** |
| dense + Phase 15 (pre-reject helps it too) | 1.787 | 1.680 | −6 % |

**#1 CUDA Graphs** (warm, same session, graphs off vs on):

| metric | graphs off | graphs on | Δ |
|---|---:|---:|---|
| narrow kernel (ms) | 0.359 | **0.333** | **−7 %** |
| narrow wall (ms) | 0.819 | **0.774** | −5 % |
| FPS | 625 | **688** | **+10 %** |

**Cumulative this round:** the hash broad cull's kernel time went from ~0.70 ms
(start of session) → 0.41 ms (#2+#3) → **~0.33 ms (#1)** — roughly **2× faster**
again on top of the earlier 2.5–3× — all with bit-identical output.

---

## 3. Nsight Compute profile of the FBP narrow kernel (Tier 2)

`featureBasedProximityKernel` is the dominant compute on large scenes, so I
profiled it with `ncu` (standalone backend bench, large geometry, 8 launches) to
find the limiter before doing more narrow-phase work:

| Metric | Value (avg over launches) |
|---|---:|
| `sm__throughput` (compute SOL) | **~27 %** |
| `gpu__dram_throughput` (DRAM SOL) | **~9 %** |
| `sm__warps_active` (achieved occupancy) | **~44 %** |
| `launch__registers_per_thread` | **79** |
| `launch__occupancy_limit_registers` | **3 blocks/SM** |
| `launch__waves_per_multiprocessor` | 21.3 |

**Reading:** the kernel is **neither compute- nor memory-bound** (27 % / 9 % SOL).
It is **register/occupancy-limited** — 79 registers/thread caps it to 3 blocks/SM
and ~44 % occupancy, so it is latency-bound on the dependent FP chains of the
closest-point tests with too few warps to hide that latency.

**What this rules in/out for future narrow-phase work:**
- ❌ **Memory-layout changes don't help** — SoA/`float4` vertices, shared-memory
  triangle caching, `__ldg` are pointless here (DRAM is only 9 % utilised).
- ✅ **Raising occupancy is the lever** — cut register pressure with
  `__launch_bounds__`, split the 9 edge-edge tests into their own kernel, or
  shrink the live `best*` state; each could lift occupancy past the 3-block
  register cap. This is the recommended next narrow-phase optimization.
- ✅ The **pre-reject (#2)** already helps by removing per-pair work for far pairs.

---

## 3b. Full-suite parity (graphs default-on)

Confirms the three changes + graphs-on don't alter any scene's output (validation
legs, `full_suite_20260617_110614`):

| Scene | contacts (VF/FV/EE) | overflow | launches |
|---|---|---:|---:|
| one-tissue / one-blade | **56** (0/0/56) | 0 | 7 |
| large tissue / blade | **8018** (5397/880/1741) | 0 | 7 |
| v-t self-collision | **2700** (2700/0/0) | 0 | 6 |
| v-t cross-model | **254** (254/0/0) | 0 | 6 |
| hash — dense leg | **2354** (1119/428/807) | 0 | 7 |
| hash — **graphs leg** | **2354** (1119/428/807) | 0 | **11** |

Every count matches the documented baseline; the hash graphs leg ran +91 % FPS
vs the dense leg (386 vs 202) on the suite's thermal conditions. A final 3-mode
comparison (graphs default-on) read hash **0.377 ms kernel / 434 FPS** vs
dense+Phase-15 1.63 ms — the hash path is now ~4× faster kernel than dense on the
both-large scene.

---

## 4. Correctness verification

- **Standalone backend bench** (no SOFA, clean rebuild): `hash_contacts = fbp_contacts = 8018`, overflow 0, probe-overflow 0.
- **CUDA-graph A/B** (200 steps × graphs off/on, interleaved): contacts **2354 (1119/428/807)** identical on every leg, overflow 0.
- **Full suite** (graphs default-on): every scene matches the documented baseline (see §3).
- **Pre-reject** is conservative by construction (squared box gap ≤ true closest distance), so it can only skip pairs that cannot produce a contact.

---

## 5. Reproduce

```bash
# Backend parity (no SOFA):
SOFA_BACKEND_BENCH_RUN_HASH=1 ./SofaGpuCollisionDenseGridBackendBench

# CUDA-graph A/B on the large hash scene (toggle SOFA_HASH_CUDA_GRAPH):
SOFA_USE_HASH_PREFIXSUM_GENERATION=1 SOFA_HASH_CUDA_GRAPH=0 runSofa -g batch ... testscenes/hash_prefixsum_large.py
SOFA_USE_HASH_PREFIXSUM_GENERATION=1 SOFA_HASH_CUDA_GRAPH=1 runSofa -g batch ... testscenes/hash_prefixsum_large.py

# 3-mode comparison + full suite (hash now uses graphs by default):
scripts/run_mode_comparison_ab_wsl.sh
scripts/run_full_benchmark_suite_wsl.sh
```
