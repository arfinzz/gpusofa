# Scene Size Scaling Benchmark

Generated from scaling benchmark logs.

- CSV: `/mnt/c/Users/arfin/Desktop/GPU SOFA/reports/scene_size_scaling_summary.csv`
- SVG graph: `/mnt/c/Users/arfin/Desktop/GPU SOFA/reports/scene_size_scaling_comparison.svg`

| Case | Collision triangles | CPU wall ms | GPU wall ms | GPU compute-only ms | H2D bytes/step | Raw candidates | Unique candidates | Contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| medium | 37536 | 4.964149 | 8.219249 | 3.383548 | 1501440 | 156096 | 97440 | 1268 |
| large | 79520 | 8.053165 | 11.756998 | 3.997251 | 3180800 | 462848 | 322560 | 1756 |
| xlarge | 146496 | 10.587752 | 42.425398 | 10.669280 | 5859840 | 2420736 | 1142400 | 2576 |
| xxlarge | 234016 | 24.752219 | 71.250486 | 23.899769 | 9360640 | 4488960 | 2116608 | 6416 |

Interpretation:

- `GPU wall ms` is the full measured step time from the GPU benchmark scene.
- `GPU compute-only ms` is `broad_kernel_ms + narrow_kernel_ms` from CUDA event timing.
- H2D transfer bytes scale directly with uploaded triangle records because the current path uploads packed `DeviceTriangle` arrays every step.
- If GPU compute-only approaches CPU wall time while GPU wall time remains much higher, the missing speedup is mostly transfer/orchestration rather than raw device kernel throughput.
