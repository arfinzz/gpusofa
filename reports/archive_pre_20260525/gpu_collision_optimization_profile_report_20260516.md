# GPU Collision Optimization And Profiling Report

Date: 2026-05-16

Device: NVIDIA GeForce GTX 1650 Ti, compute capability 7.5

## Scope

This pass implemented the requested integrated SOFA collision optimizations and profiled the result:

- Cache SOFA triangle topology and avoid rebuilding full packed triangle arrays when topology is unchanged.
- Keep collision surfaces on the GPU-oriented SOFA path through indexed dense-grid input.
- Cache device-side triangle indices by surface id/topology version.
- Upload only vertex positions each frame for indexed input.
- Remove/defer small readbacks in detection-only mode.
- Reduce small launches by fusing reset/counter work and fusing AABB generation with grid insertion.
- Keep the pair hash clear on `cudaMemset`, because profiling showed it was faster than the custom kernel reset on GTX 1650 Ti.

## Implemented Files

Core implementation:

- `SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp`

Benchmark/profile wiring:

- `test_gpu_one_tissue_one_blade_dense_grid_benchmark.py`
- `test_gpu_large_tissue_blade_dense_grid_benchmark.py`
- `scripts/run_backend_dense_grid_benchmark_wsl_gpu_proj.sh`
- `scripts/run_nsight_collision_profile_wsl_gpu_proj.sh`

Guide updates:

- `guide/plan.md`
- `guide/architecture.md`

## Build And Profiling Commands

Build:

```text
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa/SofaGpuCollision/build-profile make -j2
```

Result: build passed. The final incremental rebuild after cleanup completed without CUDA warning output; earlier full builds only showed non-blocking SOFA deprecation warnings.

Nsight run:

```text
SOFA_GPU_COLLISION_LIB=/home/arfin/gpu-sofa/SofaGpuCollision/build-profile/libSofaGpuCollision.so
SOFA_PROFILE_ROOT=/home/arfin/gpu-sofa/output/benchmark_logs/nsight_indexed_fused_hashmemset_20260516
SOFA_USE_INDEXED_DENSE_GRID_INPUT=1
SOFA_NSYS_STEPS=6
SOFA_NCU_STEPS=4
SOFA_NCU_LAUNCH_COUNT=24
scripts/run_nsight_collision_profile_wsl_gpu_proj.sh
```

Result:

- `nsys_status=0`
- `ncu_status=0`
- `ncu_launch_status=0`
- Nsight Compute report generated.
- Nsight Systems completed, but the installed `nsys` could not import `.qdstrm` into `.nsys-rep` because the importer binary/dependencies were missing.

## Benchmark Summary

All rows are detection-only dense-grid GPU narrow phase with direct CUDA collision surfaces and contacts kept device-side.

| Run | Scene | Avg step ms | FPS | Narrow wall ms | Narrow kernel ms | Host prep ms | H2D bytes/frame | D2H bytes/frame | Launches | Memsets | Candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-change control | one tissue/blade | 10.526 | 95.00 | 6.686 | 2.180 | 3.080 | 512480 | 8 | 12 | 5 | 624 |
| Indexed + fused | one tissue/blade | 7.530 | 132.81 | 3.585 | 1.891 | 0.687 | 78828 | 8 | 6 | 1 | 624 |
| Packed + fused comparison | one tissue/blade | 10.223 | 97.82 | 5.685 | 1.695 | 2.904 | 512480 | 8 | 6 | 1 | 624 |
| Pre-change control | large tissue/blade | 64.872 | 15.41 | 44.685 | 24.910 | 16.968 | 3180800 | 8 | 12 | 5 | 322560 |
| Indexed + fused | large tissue/blade | 38.910 | 25.70 | 20.600 | 15.954 | 3.180 | 489012 | 8 | 6 | 1 | 322560 |

## Main Findings

The indexed path is the right integrated SOFA optimization. On the one-tissue scene, same-session step time improved from 10.526 ms to 7.530 ms, a 28.5% reduction. Narrow-phase wall time dropped 46.4%, host preparation dropped 77.7%, and H2D bytes dropped 84.6%.

The large scene improved more strongly because it had more CPU triangle extraction and packing to remove. Step time improved from 64.872 ms to 38.910 ms, a 40.0% reduction. FPS increased 66.7%. Narrow-phase wall time dropped 53.9%, host preparation dropped 81.3%, H2D bytes dropped 84.6%, and H2D time dropped 52.6%.

Launch reduction landed cleanly. The normal detection-only path now reports 6 kernel launches and 1 `cudaMemset` per frame, down from 12 launches and 5 memsets. The packed fallback also benefits from fused AABB/insert, but it still pays the SOFA triangle extraction cost, so the indexed path is better for integrated wall time.

The remaining 8-byte D2H transfer is not the real cost by itself. The measured `candidate_readback_ms` still includes synchronization with prior GPU work. Large scene readback/sync dropped from 22.97 ms to 14.80 ms, but it remains the biggest measured wait in the integrated path.

Nsight Compute confirms the expected memory/launch shape. `insertIndexedTrianglesKernel` on the tissue launch reached about 60.2% memory throughput with only 8.3% compute throughput, so insertion is memory dominated. The tiny tool insertion launch used only 1 block and achieved about 3.5% occupancy, confirming that small tool work is underfilled. `exactDenseGridIndexedContactKernel` used 67 registers/thread and was underfilled in the small scene, but it is not the integrated wall-time bottleneck yet.

The older 2026-05-14 one-tissue best run in the guide remains lower at 4.026 ms average step and 2.583 ms narrow wall. This pass should be judged against the same-session control and large-scene result, because system/load and binary path differed from that older run. The architectural win is still real: no CPU packed triangle arrays, far fewer bytes uploaded, and half the launches.

## Artifacts

Windows-side copied artifacts:

- `output/benchmark_logs/one_tissue_one_blade_20260515_224035`
- `output/benchmark_logs/one_tissue_indexed_fused_hashmemset_20260516`
- `output/benchmark_logs/one_tissue_packed_fused_hashmemset_20260516`
- `output/benchmark_logs/large_tissue_indexed_opt_20260516`
- `output/benchmark_logs/large_tissue_indexed_fused_hashmemset_20260516`
- `output/benchmark_logs/nsight_indexed_fused_hashmemset_20260516`

Nsight Compute report:

- `output/benchmark_logs/nsight_indexed_fused_hashmemset_20260516/nsight/gpu_collision_ncu.ncu-rep`
- `output/benchmark_logs/nsight_indexed_fused_hashmemset_20260516/nsight/ncu_details_header.txt`

## Remaining Work

Highest value next:

- Reuse SOFA-owned CUDA position buffers directly instead of extracting positions through the CPU-visible model API.
- If direct device pointers are not available, use pinned/asynchronous staging for large position uploads.
- Keep candidate/contact counts device-side longer; read them asynchronously or every N frames when only logging needs them.
- Batch tiny tool work or tune block size for small-scene tool/exact-contact launches.
- Revisit SoA/coalesced layout for insertion/candidate data before spending time on exact-contact register pressure.
