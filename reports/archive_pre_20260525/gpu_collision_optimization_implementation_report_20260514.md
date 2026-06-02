# GPU Collision Optimization Implementation Report

Date: 2026-05-14

## Scope

Implemented the first production optimization pass for the SOFA dense-grid GPU collision path:

- Cache stable SOFA triangle topology in `GpuCollisionNarrowPhase`.
- Rebuild only per-frame vertex/packed triangle data on the default SOFA path.
- Add an indexed-surface backend API and gate it behind `useIndexedDenseGridInput=false` until true GPU-side indexed packing is implemented.
- Keep dense-grid counters device-side in detection-only runs unless detailed profiling or CPU logging requests them.
- Reduce triangle upload preparation cost with layout-checked direct upload / bulk copy.
- Skip pinned staging for small triangle uploads under 1 MiB; pinned staging remains available for larger transfers.
- Skip dense-grid stats memset and stat atomics when those stats will not be read back.

## Key Files

- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.h`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionNarrowPhase.cpp`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackend.h`
- `SofaGpuCollision/src/SofaGpuCollision/GpuCollisionBackendStub.cpp`
- `SofaGpuCollision/src/SofaGpuCollision/cuda/GpuCollisionBackend.cu`

## Verification

Build:

```text
wsl -d wsl-gpu-proj --cd /home/arfin/gpu-sofa/SofaGpuCollision/build-profile make -j2
```

Result: build passed. Remaining warning is the pre-existing unused `noopKernel`.

Backend smoke benchmark:

```text
./SofaGpuCollisionDenseGridBackendBench
```

Result:

```text
tissue_triangles=64800
blade_triangles=14720
measured_steps=30
wall_avg_ms=7.540589
gpu_kernel_avg_ms=3.647167
raw_candidates=462848
unique_candidates=322560
contacts=1756
overflow=0
hash_dedupe_probe_overflow=0
```

SOFA one-tissue/one-blade detection-only benchmark, final verified run:

```text
log_dir=/home/arfin/gpu-sofa/output/benchmark_logs/optimized_final_sofa_20260514
label=gpu_one_tissue_one_blade_dense_grid_benchmark_20260514_090414
avg_step_seconds=0.004025695
avg_fps=248.404306625
avg_narrow_wall_ms=2.583371200
avg_narrow_kernel_ms=1.117225605
avg_narrow_sofa_triangle_extraction_ms=0.988028900
avg_narrow_backend_triangle_pack_ms=0.000900000
avg_narrow_h2d_ms=0.298036700
avg_narrow_candidate_readback_ms=0.838610500
avg_device_to_host_bytes=8
avg_kernel_launch_count=12
avg_cuda_memset_count=5
avg_workspace_resize_count=0
avg_narrow_unique_candidate_count=624
avg_narrow_output_contact_count=0
```

## Notes

- The conservative indexed backend API is present but intentionally opt-in. The tested CPU repack implementation was slower than the default cached path, so the default remains `useIndexedDenseGridInput=false`.
- The largest remaining clean SOFA cost is now wrapper-side triangle extraction/packing at roughly 1 ms in the final measured run.
- A true GPU-native indexed path should upload/cache indices once, stream only positions, and launch a small GPU pack/indexed-AABB stage. That is the next meaningful step beyond this pass.
