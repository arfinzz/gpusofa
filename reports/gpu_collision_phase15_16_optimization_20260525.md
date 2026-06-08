# Phase 15 + 16 Optimization Report — 2026-05-25

Implementation and verification of the two optimizations that target the
measured bottleneck of the tri-tri FBP fast path:

- **Phase 15** — tool-active-cell candidate generation (the 300 µs
  generation kernel).
- **Phase 16** — workspace-cached broad-cull CUDA events (the ~80 µs/frame
  event churn).

GTX 1650 Ti, sm_75, WSL2. Both designed in `guide/plan.md` §5.15-5.16
before implementation; this report records what was built and measured.

---

## 1. The problem (recap)

`reports/end_to_end_verification_20260525.md` §7 measured the tri-tri FBP
fast path spending **~300 µs (≈80% of GPU time)** in
`generateDenseGridUniqueCandidatePairsKernel`. Root cause: the kernel
launches **one block per grid cell** (`<<<32768, 256>>>`), but the blade
only occupies ~30 cells, so ~99% of blocks read an empty bucket and exit.
The 300 µs is block-scheduling overhead, not pair work (26% SM throughput).

Separately, the broad-cull functions created and destroyed four CUDA timing
events every frame (~40-80 µs of driver churn).

---

## 2. What was implemented

### Phase 15 — tool-active-cell candidate generation

**Mechanism: build the mixed-cell list during the tool insert (no separate
scan).** In `insertTriangleAabbIntoGrid` (the shared insert helper), the
first tool triangle to claim slot 0 in a cell that already contains tissue
appends the cell id to an active list:

```cpp
const std::uint32_t localIndex = atomicAdd(&grid[cellId].toolCount, 1u);
if (localIndex < bucketCapacity) {
    cellIds[cellId * bucketCapacity + localIndex] = triangleId;
    if (buildToolActiveList && !insertTissue && localIndex == 0u &&
        grid[cellId].tissueCount > 0u) {
        const std::uint32_t activeIndex = atomicAdd(activeCellCount, 1u);
        activeCellIds[activeIndex] = cellId;   // sized to cellCount, cannot overflow
    }
    ...
}
```

Correctness:
- `localIndex == 0` ⇒ each cell appended at most once (natural dedupe).
- `grid[cellId].tissueCount > 0` is valid because the tissue insert kernel
  completes before the tool insert kernel on the serialized default stream,
  so the list contains exactly the **mixed** cells.
- `activeCellIds` is allocated to `cellCount`; distinct cells ≤ cellCount,
  so the append can never overflow.
- `resetDenseGridKernel` already zeroes `activeCellCount` each frame.

**Generation reuses the existing
`generateActiveDenseGridUniqueCandidatePairsKernel`**, which device-reads
`*activeCellCount` and grid-strides over the active list. Launched with a
fixed grid of `min(cellCount, 1024)` blocks — no host readback. The ~30
active cells run in the first ~30 blocks; the rest exit immediately.

**Scope:** the indexed contact path
(`computeDenseGridIndexedTriangleContacts`), which is the production FBP and
exact-contact broad cull. The packed fallback path ignores the flag (its
`insertPackedTrianglesKernel` call uses the defaulted `buildToolActiveList
= false` and its generation still checks only `compactActiveCells`), so it
stays correct, just unoptimized.

**How this differs from the regressed `compactActiveCells` (Phase 9):** that
path ran a separate `compactActiveDenseGridCellsKernel` that scans **all
32 768 cells** to find mixed cells — the scan costs what it saves. Phase 15
**never scans all cells**; it builds the list as a side effect of the insert
it already runs.

**Guards:**
- New flag `useToolActiveCellGeneration` (`DenseGridConfig` + narrow-phase
  Data field + `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION`), distinct from
  `compactActiveCells`.
- Rejected in combination with `batchTriangleInsert` (which fuses the
  inserts and breaks the tissue-before-tool ordering the active list relies
  on) — diagnostic returned.

### Phase 16 — workspace-cached broad-cull events

Four events (`broadStageStart/End`, `broadTotalStart/End`) moved onto
`DenseGridWorkspace`, lazy-created once via `ensureBroadEvents()`, freed in
the workspace destructor. Both contact functions alias the cached handles
(`cudaEvent_t& stageStart = workspace.broadStageStart; ...`), so the rest of
each function is unchanged; `destroyStageEvents()` is now a no-op. Mirror of
the FBP-event fix that shipped in Phase 14. Always on.

---

## 3. Measured results

All A/B pairs ran back-to-back on the same thermal state (GPU 44 °C at
start). Absolute FPS is depressed by repeated-run heating, but the
**relative** improvement and the kernel-level numbers are the signal.

### 3.1 Fast path (production, no counter readback)

| Metric | Baseline (flag off) | Active (flag on) | Ratio |
|---|---:|---:|---:|
| avg_fps | 221 | **943** | **4.3×** |
| avg_narrow_wall_ms | 4.00 | **0.56** | 7.2× |
| avg_narrow_kernel_ms | 3.83 | **0.35** | **10.9×** |
| avg_host_to_device_bytes | 0 | 0 | — |
| avg_device_to_host_bytes | 0 | 0 | — |
| avg_kernel_launch_count | 7 | 7 | — |
| avg_narrow_overflow_count | 0 | 0 | — |

943 FPS exceeded the ~900-1000 target.

### 3.2 Validation mode (counter readback on — contact counts visible)

| Metric | Baseline (flag off) | Active (flag on) |
|---|---:|---:|
| avg_fps | 119 | **475** |
| avg_narrow_wall_ms | 7.75 | **1.42** |
| avg_narrow_kernel_ms | 6.83 | **0.82** |
| **avg_narrow_output_contact_count** | **56** | **56** |
| **avg_narrow_ee_contact_count** | **56** | **56** |
| avg_narrow_vf_contact_count | 0 | 0 |
| avg_narrow_overflow_count | 0 | 0 |

**Contact counts are bit-identical.** The optimization changes which cells
are scanned, not which contacts emit. This is the key correctness result.

### 3.3 Nsight Compute — the generation kernel transformation

| | Baseline | Active |
|---|---|---|
| Kernel | `generateDenseGridUniqueCandidatePairsKernel` | `generateActiveDenseGridUniqueCandidatePairsKernel` |
| `launch__grid_size` | **32 768** | **1 024** |
| `gpu__time_duration` | **~300 µs** | **7.9 µs** |

32× fewer launched blocks, 38× faster generation.

Full per-frame GPU kernel breakdown with the flag on:

```text
resetDenseGridKernel                                 2.5 µs
insertIndexedTrianglesKernel (blade)                31.7 µs  (launch-bound)
generateActiveDenseGridUniqueCandidatePairsKernel    7.9 µs  ← was ~300 µs
resetProximityCountersKernel                         1.8 µs
featureBasedProximityKernel                         17.7 µs
                                                    -------
total                                              ~61 µs   (was ~380 µs)
```

Generation went from ~80% of GPU time to ~13%. Candidate generation is no
longer the bottleneck — the blade insert (launch-bound) and the FBP kernel
now dominate.

### 3.4 Regression check (Phase 16 affects all paths)

The event-caching change touches every dense-grid contact function, so the
v-t paths were re-verified:

| Scene | Contacts | VF/FV/EE | Overflow |
|---|---:|---|---:|
| v-t self-collision | 2700 | 2700 / 0 / 0 | 0 |
| v-t cross-model | 254 | 254 / 0 / 0 | 0 |

Unchanged from the 2026-05-25 verification. No regression.

---

## 4. Files changed

| File | Change |
|---|---|
| `GpuCollisionBackend.h` | `useToolActiveCellGeneration` in `DenseGridConfig` |
| `cuda/GpuCollisionBackend.cu` | active-list build in `insertTriangleAabbIntoGrid` + insert kernels; active-generation dispatch + `batchTriangleInsert` rejection in `computeDenseGridIndexedTriangleContacts`; `broad*` events + `ensureBroadEvents()` in `DenseGridWorkspace`; both contact functions alias cached events |
| `GpuCollisionNarrowPhase.{h,cpp}` | `d_useToolActiveCellGeneration` Data field; set `denseGridConfig.useToolActiveCellGeneration` in the main dispatch |
| `test_gpu_one_tissue_one_blade_dense_grid_benchmark.py` | `SOFA_USE_TOOL_ACTIVE_CELL_GENERATION` env wiring |
| `scripts/run_fbp_smoke_test_wsl.sh` | flag pass-through |

Clean rebuild, no new warnings beyond the pre-existing SOFA `RegisterObject`
deprecation.

---

## 5. Large-tissue A/B + default flip + grid-stride fix (2026-05-25)

The large-tissue/blade scene (79 520 collision elements, subdivided blade)
was the gate for flipping the default. It cleared the flip **and** exposed a
separate latent bug.

### 5.1 The exposed bug

The first large-tissue A/B showed baseline 3691 contacts vs active 3790 —
**different counts** despite identical `unique_candidate_count` (322 560).
Root cause: `featureBasedProximityKernel` and
`featureBasedVertexTriangleProximityKernel` processed one pair per thread
(`candidatePairs[tid]`) with **no grid-stride loop**, and the fixed
over-launch grid is only 65 536 threads. On the large scene (322 560 pairs)
they silently dropped ~80% of pairs; baseline and active dropped a different
80% because the pair order differs (cell order vs active-list order).

This bug pre-dated Phase 15 (it shipped with the Phase 11 over-launch sync
fix); Phase 15 merely exposed it. All scenes with < 65 536 pairs (one-tissue
624, v-t self 2700, v-t cross 254) were unaffected.

### 5.2 The fix

Both proximity kernels now grid-stride over all candidate pairs:

```cpp
const uint32_t pairCount = *candidatePairCount;
const uint32_t stride = gridDim.x * blockDim.x;
for (uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < pairCount; idx += stride) {
    ... process candidatePairs[idx] ...   // the two `return` became `continue`
}
```

The launch grid is now a pure GPU-saturation target (bumped to 1024 blocks);
correctness is independent of it. After the fix the large-tissue scene emits
**8018 contacts (5397 VF / 880 FV / 1741 EE)** — the *correct* count, and
bit-identical between baseline and active.

### 5.3 Large-tissue A/B (after the fix)

| Metric | Baseline (off) | Active (on) |
|---|---:|---:|
| avg_fps | 108 | **116** (1.08×) |
| avg_narrow_wall_ms | 4.63 | 4.09 |
| avg_narrow_kernel_ms | 3.91 | 3.33 |
| **output_contact_count** | **8018** | **8018** |
| vf / fv / ee | 5397 / 880 / 1741 | 5397 / 880 / 1741 |
| unique_candidate_count | 322 560 | 322 560 |
| overflow | 0 | 0 |

The large-scene win is smaller (1.08×) than the one-tissue win (4.3×)
because the subdivided blade touches many cells — the tool/tissue asymmetry
is weaker and the FBP kernel itself dominates. But it is **still faster and
never a regression**, with bit-identical output.

### 5.4 Default flipped to ON

On the strength of: 4.3× one-tissue, 1.08× large-tissue, bit-identical
output on both, zero overflow, and the structural guarantee that the active
list is ≤ cellCount (grid-strided, never more work than all-cells), the
default was flipped to ON:

- `DenseGridConfig.useToolActiveCellGeneration { true }`
- `GpuCollisionNarrowPhase` Data field default `true`
- Benchmark-scene env-flag defaults `true`
- Launcher script defaults `1`

`compactActiveCells` is effectively superseded (it achieved the same idea
via a full-grid scan that regressed).

## 6. Status — fully closed

- **Phase 15** (active-cell generation): done, **default ON**.
- **Phase 16** (event caching): done, always on.
- **Phase 17** (FBP/v-t grid-stride correctness fix): done.

Reproduction:

```bash
# One-tissue A/B (default-on; set =0 for the old baseline)
env SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=0 bash scripts/run_fbp_smoke_test_wsl.sh
env SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1 bash scripts/run_fbp_smoke_test_wsl.sh

# Large-tissue A/B (exercises the grid-stride loop)
env SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=0 bash scripts/run_fbp_large_tissue_wsl.sh
env SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1 bash scripts/run_fbp_large_tissue_wsl.sh

# Nsight confirmation of the grid-size drop
env SOFA_USE_TOOL_ACTIVE_CELL_GENERATION=1 bash scripts/run_nsight_fbp_profile_wsl.sh
```
