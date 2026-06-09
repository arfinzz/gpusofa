# Branch comparison — dense (feature) vs hash (experiment), small & large scenes

**Date:** 2026-06-09
**Hardware:** NVIDIA GeForce GTX 1650 Ti (Turing sm_75), WSL2, SOFA 25.12.
**Build:** `SofaGpuCollision/build-profile/libSofaGpuCollision.so` (merged `main`).
**Reproduce:** `scripts/run_branch_comparison_ab_wsl.sh` (tiny-tool + large-tool
on a large tissue) and `scripts/run_tiny_ab_wsl.sh` (small tissue + small tool).
Metric definitions: [README_metrics_explained.md](README_metrics_explained.md).

---

## What "both branches" means here

The repository merged two lines of work into `main`:

- **`feature/feature-based-proximity-vertex-triangle`** — the stable collision
  engine: dense-grid broad cull + Phase 15 tool-active-cell generation, feeding
  the feature-based proximity (FBP) and vertex-triangle narrow kernels.
- **`experiment/hash-prefixsum-broadphase`** — a *strict superset* that adds one
  thing: an **alternative** broad cull (spatial hash + prefix-sum work expansion)
  for the tri-tri FBP path, behind a default-off flag
  (`useHashPrefixSumGeneration`).

They are the **same binary**. The hash flag toggles only the broad cull; the
narrow phase is identical, so **the contacts are bit-identical** and the only
question is speed. Therefore:

> **"dense vs hash" *is* the feature-vs-experiment comparison** — the dense leg
> is exactly how the feature branch behaves, the hash leg is exactly what the
> experiment adds. With the flag off, `main` runs the feature-branch path
> byte-for-byte.

This report measures both legs on **three scene sizes**, same session,
back-to-back (thermally fair), contacts verified identical on every pair.

---

## 1. The headline: a small→large sweep (same session, A/B per row)

All three rows are the **same scene file** (`test_gpu_hash_prefixsum_large.py`)
at three geometry sizes. Each row is a dense leg and a hash leg run back-to-back.
Contacts are **bit-identical** within every row (the correctness check); the
numbers below are the speed payoff.

| Scene | tissue tris | tool tris | total | **dense FPS** | **hash FPS** | FPS Δ | **dense kernel ms** | **hash kernel ms** | kernel Δ | contacts (VF/FV/EE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Tiny** (small tissue, small tool) | 800 | 32 | 832 | **499.6** | 487.8 | **−2.4 %** | **1.381** | 1.481 | **+7.2 % (hash slower)** | 166 (35/0/131) |
| **Mixed** (large tissue, small tool) | 12,800 | 32 | 12,832 | 387.1 | **443.8** | **+14.6 %** | 1.250 | **1.065** | **−14.8 % (hash faster)** | 1096 (108/457/531) |
| **Large** (large tissue, large tool) | 12,800 | 1,568 | 14,368 | 338.5 | **345.5** | **+2.1 %** | 1.487 | **1.368** | **−8.0 % (hash faster)** | 2354 (1119/428/807) |

Read it as a story:

- **Tiny scene → dense wins.** Few cells are occupied, so there is almost no
  candidate work to expand; the hash path's extra `exclusive_scan` + binary-search
  decode is pure overhead. Dense is ~2–7 % ahead.
- **Large tissue (mixed or large tool) → hash wins.** Once the tissue is big, the
  dense grid pays to reset and scan all 32,768 fixed cells while the hash table
  only materializes the occupied slots; the prefix-sum gives perfectly balanced
  one-thread-per-pair narrow work. Hash is 8–15 % faster on the kernel.

**The deciding factor is occupancy/tissue size, not tool size.** Hash overtakes
the dense grid when a large fraction of work comes from a large mesh; it loses on
genuinely small scenes where fixed overheads dominate and the scan doesn't pay
for itself.

> Every row: `overflow = 0`, `probe_overflow = 0`, dense launches **7** kernels,
> hash launches **8** (the extra one is the `thrust::exclusive_scan`).

---

## 2. A cooler-GPU large-scene A/B (the dedicated experiment run)

The sweep above ran four legs in one process, so the GPU warmed through it and
the *large* row's FPS is thermally suppressed (the kernel delta, −8 %, is the
trustworthy figure). A dedicated two-leg run on a **cool** GPU — the canonical
hash experiment measurement — shows the larger FPS gap:

| metric | dense grid | hash + prefix-sum | Δ |
|---|---:|---:|---|
| avg FPS | 359.5 | **405.2** | **+12.7 %** |
| avg narrow wall (ms) | 1.974 | **1.698** | −14.0 % |
| avg narrow kernel (ms) | 1.436 | **1.268** | **−11.7 %** |
| contacts (VF/FV/EE) | 2354 (1119/428/807) | 2354 (1119/428/807) | **identical** |
| raw / unique candidates | 61,920 / 33,408 | 61,920 / 33,408 | **identical** |
| overflow / probe-overflow | 0 / 0 | 0 / 0 | — |

Full design + correctness write-up:
[hash_prefixsum_broadphase_experiment_20260609.md](hash_prefixsum_broadphase_experiment_20260609.md).
Standalone backend parity (no SOFA): `hash_contacts = fbp_contacts = 488`
(0/133/355), occupied_slots 16,536, unique_pairs 6,720, overflow 0.

**Take the kernel time as the verdict** (it's GPU-clock-measured and barely
varies): hash is **8–15 % faster on a large tissue, ~7 % slower on a tiny scene**,
with identical output throughout.

---

## 3. The genuine production scenes (dense path, both branches)

These are the four standard scenes the project actually ships, all on the
**default** (dense) path — i.e. how *both* branches run out of the box. Contact
counts are the correctness fingerprint and are **bit-identical to the documented
2026-05-25 baseline**. From `scripts/run_full_benchmark_suite_wsl.sh`.

| Scene | Elements | Mode | FPS | Narrow kernel (ms) | Contacts (VF/FV/EE) | Overflow |
|---|---:|---|---:|---:|---|---:|
| one-tissue / one-blade (FBP) | 12,812 | fast (warm) | **723.8** | 0.282 | — (not read) | 0 |
| one-tissue / one-blade (FBP) | 12,812 | validation | 445.9 | 0.815 | **56** (0/0/56) | 0 |
| large tissue / subdiv blade (FBP) | 79,520 | fast | **146.2** | 2.090 | — (not read) | 0 |
| large tissue / subdiv blade (FBP) | 79,520 | validation | 114.2 | 3.288 | **8018** (5397/880/1741) | 0 |
| v-t self-collision (slab) | 900 | fast | **5836.2** | — | — | 0 |
| v-t self-collision (slab) | 900 | validation | 2105.9 | — | **2700** (2700/0/0) | 0 |
| v-t cross-model (tissue+cloud) | 3,200 | fast | **4225.0** | — | — | 0 |
| v-t cross-model (tissue+cloud) | 3,200 | validation | 1741.6 | — | **254** (254/0/0) | 0 |

> The genuine **one-tissue / one-blade** surgical scene (a small-footprint blade
> touching ~30 grid cells) is the Phase 15 sweet spot: candidate generation is
> ~8 µs and the dense path runs at 724 FPS. This is the regime where the hash
> path is *not* used and would *not* help — like the Tiny row in §1, the scan
> overhead would cost more than it saves. The hash path is reserved for the
> large-tissue case (§1 Mixed/Large, §2).

---

## 4. Which branch / path should I use?

| If your scene is… | Use | Why |
|---|---|---|
| a small tool in a small/local region (the surgical default) | **dense** (flag off) | Phase 15 culls to ~30 cells; the scan would be wasted overhead |
| a large tissue (with any tool) | **hash** (`useHashPrefixSumGeneration=True`) | only occupied slots are materialized; prefix-sum balances the narrow work; 8–15 % faster kernel |
| self-collision or point-cloud-vs-mesh | **dense** (v-t path) | the hash cull is wired only for the tri-tri FBP path |

Because the hash path is **opt-in and default-off**, merging the experiment into
`main` costs the default path nothing: with the flag off the runtime is
byte-identical to the feature branch, and the faster broad cull is available for
the large-tissue regime by setting one flag. That is why `main` now carries both.

---

## 5. Reproduce

```bash
# Small→large sweep (Tiny is a separate small-tissue script):
scripts/run_tiny_ab_wsl.sh                 # Tiny  row  (small tissue + small tool)
scripts/run_branch_comparison_ab_wsl.sh    # Mixed + Large rows (large tissue)

# Dedicated cool-GPU large A/B (§2):
scripts/run_hash_prefixsum_large_ab_wsl.sh

# Full production suite (§3):
scripts/run_full_benchmark_suite_wsl.sh

# Standalone backend correctness, no SOFA:
SOFA_BACKEND_BENCH_RUN_HASH=1 ./SofaGpuCollisionDenseGridBackendBench
```

Per-leg headline fields are in `<leg>/<label>_summary.txt`; every field is
defined in [README_metrics_explained.md](README_metrics_explained.md).
