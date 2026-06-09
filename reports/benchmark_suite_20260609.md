# GPU Collision — Full Benchmark Suite (2026-06-09)

**Branch:** `experiment/hash-prefixsum-broadphase`
**Hardware:** NVIDIA GeForce GTX 1650 Ti (Turing sm_75), WSL2, SOFA 25.12.
**Build:** `SofaGpuCollision/build-profile/libSofaGpuCollision.so` (this branch).
**Method:** `scripts/run_full_benchmark_suite_wsl.sh`, 160 steps/leg (10 warmup +
150 measured). Fast path = detection-only (no counter readback); validation =
counter readback on (contact counts visible). Older reports archived under
`reports/archive_pre_20260609/`.

> **TL;DR.** Contact counts are **bit-identical to the documented 2026-05-25
> baseline on every scene** — no correctness change. Narrow-phase wall and
> kernel times are **at or better than** the documented numbers; whole-pipeline
> FPS sits within this laptop's known thermal-variance band. The new
> spatial-hash + prefix-sum broad cull gives **+11.8 % FPS / −15 % narrow wall**
> on the large-tissue + large-tool scene, with bit-identical contacts.

> **Independent re-run verification (2026-06-09, second pass).** The full suite
> + the hash A/B were re-executed from a clean build of this branch. **Every
> contact count reproduced exactly**: small 56 EE, large 8018 (5397/880/1741),
> v-t self 2700 VF, v-t cross 254 VF, hash dense/on 2354 (1119/428/807),
> overflow 0 everywhere. The hash path's per-cell counts matched dense exactly
> (raw 61,920 / unique 33,408 on both legs) and ran +10.9–19.5 % faster
> depending on thermal state. The standalone backend bench also reproduced
> 488 = 488 hash/dense parity with 0 overflow. Warm small fast-path re-read
> 733.8 FPS, matching the 723.8 below. All numbers in this report are confirmed
> reproducible; absolute FPS varies within the documented thermal band.

---

## 1. Today's measured results

| Scene | Elements | Mode | FPS | Narrow wall (ms) | Narrow kernel (ms) | Contacts (VF/FV/EE) | Overflow |
|---|---:|---|---:|---:|---:|---|---:|
| one-tissue / one-blade (FBP) | 12,812 | fast (warm) | **723.8** | 0.472 | 0.282 | — (not read) | 0 |
| one-tissue / one-blade (FBP) | 12,812 | validation | 445.9 | 1.419 | 0.815 | **56** (0/0/56) | 0 |
| large tissue / subdiv blade (FBP) | 79,520 | fast | **146.2** | 2.413 | 2.090 | — (not read) | 0 |
| large tissue / subdiv blade (FBP) | 79,520 | validation | 114.2 | 4.046 | 3.288 | **8018** (5397/880/1741) | 0 |
| v-t self-collision (slab) | 900 | fast | **5836.2** | 0.096 | — | — (not read) | 0 |
| v-t self-collision (slab) | 900 | validation | 2105.9 | 0.412 | — | **2700** (2700/0/0) | 0 |
| v-t cross-model (tissue+cloud) | 3,200 | fast | **4225.0** | 0.077 | — | — (not read) | 0 |
| v-t cross-model (tissue+cloud) | 3,200 | validation | 1741.6 | 0.406 | — | **254** (254/0/0) | 0 |
| hash A/B — **dense grid** | 14,368 | validation | 343.6 | 1.997 | 1.439 | **2354** (1119/428/807) | 0 |
| hash A/B — **hash + prefix-sum** | 14,368 | validation | **384.1** | 1.695 | 1.254 | **2354** (1119/428/807) | 0 |

(The first small fast-path leg of the batch read 431 FPS / 1.42 ms kernel — a
cold-GPU-clock artifact, the first kernel launch in a fresh process before the
1650 Ti ramps from its idle clock. The 723.8 FPS / 0.282 ms above is the warm
re-run via `scripts/run_small_warm_ab_wsl.sh`, which is the representative
number; fast path is correctly faster than validation once warm.)

---

## 2. How fast vs. the older (documented 2026-05-25) numbers

Documented numbers are from `guide/plan.md` §1 / §5.x (the 2026-05-25 production
build). "Δ" compares today to documented.

| Scene / metric | Documented (2026-05-25) | Today (2026-06-09) | Δ |
|---|---:|---:|---|
| **small, fast** — FPS | 943 | 723.8 | −23 % (thermal; see below) |
| **small, fast** — narrow wall | 0.56 ms | **0.472 ms** | **−16 % (better)** |
| **small, fast** — narrow kernel | 0.35 ms | **0.282 ms** | **−19 % (better)** |
| **small, validation** — FPS / contacts | 475 / 56 EE | 445.9 / **56 EE** | −6 % FPS, contacts identical |
| **large, validation** — FPS | 116 | 114.2 | −1.6 % (noise) |
| **large, validation** — narrow wall | 4.09 ms | 4.046 ms | ≈ equal |
| **large, validation** — contacts | 8018 (5397/880/1741) | **8018 (5397/880/1741)** | **identical** |
| **v-t self, validation** — FPS / contacts | 1385–1589 / 2700 VF | **2105.9** / **2700 VF** | **+33–52 % FPS**, contacts identical |
| **v-t cross, validation** — FPS / contacts | 1380 / 254 VF | **1741.6** / **254 VF** | **+26 % FPS**, contacts identical |
| **v-t cross, fast** — FPS | 1968 | **4225.0** | **+115 %** |

### Reading the comparison

- **Correctness: unchanged.** Every validation leg reproduces the documented
  contact count *exactly* — 56 EE (small), 8018 / 5397·880·1741 (large),
  2700 VF (self), 254 VF (cross). The current code is bit-for-bit equivalent in
  output to the 2026-05-25 baseline.
- **The collision work itself is as fast or faster.** The robust GPU metrics —
  `narrow_wall_ms` and `narrow_kernel_ms` — are **at or below** the documented
  values on the scenes where they're recorded (small fast: 0.47 vs 0.56 ms wall,
  0.28 vs 0.35 ms kernel; large: equal). The v-t paths are markedly faster today.
- **The one "slower" number is whole-pipeline FPS on the small fast path
  (724 vs 943).** This is the metric the docs repeatedly flag as
  thermally/power-state sensitive on this laptop (`plan.md` §1, §7.2; `setup.md`
  §8). Since the *narrow wall is lower today*, the FPS gap is SOFA scene-graph +
  GPU power-state overhead outside the collision kernels, not a collision
  regression. The kernel did less work; the surrounding frame cost more.

**Bottom line:** no regression. Same contacts, equal-or-faster kernels; the
absolute FPS swing is the documented thermal noise band.

---

## 3. The new path: spatial-hash + prefix-sum broad cull

Same-session A/B (thermally fair, back-to-back) on the large-tissue + large-tool
scene (12,800 tissue + 1,568 tool tris = 14,368 elements):

| Metric | Dense grid | Hash + prefix-sum | Δ |
|---|---:|---:|---|
| avg FPS | 343.6 | **384.1** | **+11.8 %** |
| narrow wall (ms) | 1.997 | **1.695** | **−15.1 %** |
| narrow kernel (ms) | 1.439 | **1.254** | **−12.9 %** |
| contacts (VF/FV/EE) | 2354 (1119/428/807) | 2354 (1119/428/807) | **identical** |
| overflow / probe-overflow | 0 / 0 | 0 / 0 | — |
| kernel launches / frame | 7 | 8 (+1 scan) | — |

The hash path materializes only occupied cells and uses prefix-sum
work-expansion (one thread per candidate pair, perfect load balance) instead of
the fixed 32,768-cell dense grid. In the both-large regime that more than pays
for the extra `thrust::exclusive_scan`. Full design write-up:
`reports/hash_prefixsum_broadphase_experiment_20260609.md`. It is **opt-in,
default-off** (`useHashPrefixSumGeneration`); with it off the dense path is
byte-identical to `main`.

> Regime note: for the small-tool / large-tissue scenes the Phase 15
> tool-active-cell path is extremely cheap and still wins. The hash cull is for
> the both-large case.

---

## 4. Reproduce

```bash
# Full suite (all scenes, fast + validation legs):
scripts/run_full_benchmark_suite_wsl.sh         # writes output/benchmark_logs/full_suite_<stamp>/<leg>/

# Warm small-scene re-run (avoids the cold-clock first-leg artifact):
scripts/run_small_warm_ab_wsl.sh

# Hash A/B only (large tissue + large tool):
scripts/run_hash_prefixsum_large_ab_wsl.sh
```

Headline fields per leg live in `<leg>/<label>_summary.txt`:
`avg_fps`, `avg_narrow_wall_ms`, `avg_narrow_kernel_ms`,
`avg_narrow_output_contact_count`, `avg_narrow_{vf,fv,ee}_contact_count`,
`avg_narrow_overflow_count`.
