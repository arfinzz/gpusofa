#include <SofaGpuCollision/GpuCollisionBackend.h>

#include <cuda_runtime.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SOFAGPUCOLLISION_HAS_NVTX 1
#else
#define SOFAGPUCOLLISION_HAS_NVTX 0
#endif
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
// CUB is used ONLY by the sorted-grid ("5th way") CUB-radix-sort toggle
// (SortedGridConfig::useCubRadixSort). The default counting-sort route keeps
// the hash/dense paths CUB-free, as established by the 2026-06-17b cleanup.
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

// ============================================================================
// This file is intentionally a thin umbrella. The whole CUDA backend compiles
// as ONE translation unit: kernels must be visible to their launch sites
// (avoiding -rdc/device-linking keeps cross-function inlining and codegen
// byte-identical to the pre-split monolith), so the modules below are textual
// includes, NOT separate compilation units. Include order is dependency order:
// common device math -> dense grid (owns the shared cell math + pair dedup) ->
// legacy paths -> FBP kernels -> hash -> simple hash -> sorted grid.
// Each module holds one concern's workspace + kernels + host driver(s).
// ============================================================================

#include "detail/BackendCommon.cuh"
#include "detail/DenseGrid.cuh"
#include "detail/BroadPhaseLegacy.cuh"
#include "detail/FbpKernels.cuh"
#include "detail/HashGrid.cuh"
#include "detail/SimpleHash.cuh"
#include "detail/SortedGrid.cuh"
