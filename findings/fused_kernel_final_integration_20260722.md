# Final fused-kernel production integration report

## Outcome

The experiment framework was removed and the only stats-preserving winner was fused directly into the production `fusedBigCellNarrowKernel`.

The final source has:

- one fused narrow-phase kernel;
- no `SOFA_GPU_BIGCELL_OPT_MASK`;
- no experimental fused dispatcher;
- no two-pass or warp-output helper;
- no experimental build mask;
- no heavy-first scheduling flag;
- no duplicated factor-2 kernel template;
- the original 256-thread launch and 1,024-block cap;
- complete tested-pair, contact, overflow, VF, FV, and EE reporting.

## Source changes

### `BigCellGrid.cuh`

The production fused kernel now:

1. loads a tool triangle's three vertices into shared memory;
2. creates its exact raw AABB once and stores that beside the vertices;
3. creates a tissue triangle's exact raw AABB once per tissue entry/tool tile;
4. reconstructs the tool's inflated AABB from its raw box for the unchanged broad/home-cell tests;
5. performs the exact raw-box squared-gap rejection;
6. calls the common closest-feature helper without asking it to rebuild the same boxes.

The now-unused tool inflated-AABB pointer was removed from the fused-kernel parameter list.

### `FbpKernels.cuh`

`fbpComputeClosestFeatureContact` gained an `aabbAlreadyRejected` argument with a default value of `false`.

- Normal FBP callers use the default and retain the original internal exact-AABB rejection.
- The fused kernel passes `true` only after performing the same exact rejection using cached raw boxes.

Because the helper is force-inlined and the argument is a compile-time literal at each call site, the fused path does not retain a runtime branch.

### Benchmark output

`DenseGridBackendBench.cpp` now prints `bigcell_total_overflow`. This exposed proximity-output overflow that was already included in `BackendExecutionStats` but was not printed in the big-cell summary.

## Removed files and mechanisms

The following rejected implementation families were deleted:

- experimental fused-kernel templates and launch dispatchers;
- experimental two-pass closest-feature helper;
- warp-aggregated output emitter;
- experimental shared-hash build variants;
- heavy-first scheduling kernel;
- factorial/warp/build campaign drivers tied to removed flags.

The raw results remain under `output/verification_20260719/fused_optimization_final/` so the negative findings can be audited without keeping dead production code.

## Final validation

### Build

The CUDA library and `SofaGpuCollisionDenseGridBackendBench` rebuilt successfully in `/home/arfin/gpu-sofa/SofaGpuCollision/build-profile`.

### Production backend repetitions

| Round | 80k kernel | 80k contacts | 200k kernel | 200k contacts |
|---:|---:|---:|---:|---:|
| 0 | 0.544023 ms | 8,018 | 0.904856 ms | 17,040 |
| 1 | 0.680864 ms | 8,018 | 0.872314 ms | 17,040 |
| 2 | 0.595262 ms | 8,018 | 0.853169 ms | 17,040 |
| 3 | 0.583351 ms | 8,018 | 0.879836 ms | 17,040 |

Every round retained:

- 43,584 pairs and 2,828/3,469/1,721 VF/FV/EE at 80k;
- 89,856 pairs and 8,805/4,914/3,321 VF/FV/EE at 200k;
- zero total overflow.

### Factor/tile/build parity

Eight production permutations passed exact contact, pair, and subtype parity:

- CSR factor 4, tiles 256 and 32;
- CSR factor 2, tile 256;
- CSR factor 1, tile 256;
- fixed hash factor 4, tiles 256 and 32;
- shared-hash CSR factor 2;
- shared-sort CSR factor 2.

The fixed hash-table legs retain their pre-existing 256 build overflows. They still emit the correct contacts and this condition is unrelated to raw-AABB reuse.

### SOFA report scenes

| Scene | Contacts | VF/FV/EE | Overflow |
|---|---:|---:|---:|
| `hash_prefixsum_large.py` big-cell | 2,354 | 1,119/428/807 | 0 |
| `collision_xlarge_200k.py` big-cell | 12,178 | 3,615/4,774/3,789 | 0 |

### Final kernel resources

`cuobjdump` reports:

- 122 registers/thread;
- 17,936 bytes static shared memory/block;
- zero stack bytes;
- zero local-memory bytes.

## Reproduction

Run:

```bash
bash scripts/run_fused_winner_validation_wsl.sh
```

The final evidence captured during integration is in `output/verification_20260722/fused_winner_final/`.
