# 05 — Phase 2: Narrow-Phase Data Preparation (Zero-Copy)

The broad phase handed us one pair: `(tissue, blade)`. The narrow phase now has
to feed that geometry to the CUDA kernels. The clever part is that it does this
**without copying any vertex data**. This file explains exactly how.

File: `GpuCollisionNarrowPhase.cpp`, function `endNarrowPhase()` and its helper
`extractCudaIndexedSurface()`.

---

## 5.1 The three calls (mirror of the broad phase)

Just like the broad phase, SOFA drives the narrow phase with three calls:

```text
beginNarrowPhase()                 → reset for a new frame
addCollisionPair(pair) for each    → "here's a pair to check"
endNarrowPhase()                   → do the actual detection
```

### `beginNarrowPhase()`

```cpp
void GpuCollisionNarrowPhase::beginNarrowPhase() {
    m_pendingPairs.clear();
    ++m_frameCounter;
    markTriangleTopologyCacheUnused();   // mark all cache entries "not seen yet"
    // ... base class ...
}
```

It clears the pending pairs, bumps a frame counter (used for sampled profiling),
and marks every topology-cache entry as "unused this frame." Entries that don't
get touched this frame will be pruned at the end (so deleted objects don't leak
cache memory).

### `addCollisionPair(pair)`

```cpp
void GpuCollisionNarrowPhase::addCollisionPair(const ...& cmPair) {
    // ...
    m_pendingPairs.push_back(cmPair);   // just queue it
}
```

It doesn't process the pair yet — it just queues it. All the real work is
deferred to `endNarrowPhase`, so the kernels can be launched in one batch.

For our scene, `m_pendingPairs = [(tissue, blade)]` after this.

---

## 5.2 `endNarrowPhase()` — the per-pair loop

`endNarrowPhase` loops over the queued pairs. For each pair it:

1. **Extracts** each side into a GPU-ready surface description.
2. **Builds** a configuration struct from the Data fields.
3. **Dispatches** to the right backend function (one of four paths).
4. **Records** the timing.

This file covers steps 1 and 2 (the data prep). Step 3 (the kernels) is files
07–09. Step 4 (timing) is file 11.

---

## 5.3 Extraction — `extractCudaIndexedSurface`

For each model in the pair, the narrow phase calls:

```cpp
firstOk  = extractCudaIndexedSurface(pair.first,  firstModel,  firstIndexedSurface);
secondOk = extractCudaIndexedSurface(pair.second, secondModel, secondIndexedSurface);
```

Let's walk through what `extractCudaIndexedSurface` does, because this is where
zero-copy happens.

### Step A — confirm it's a CUDA model

```cpp
auto* cudaModel = dynamic_cast<CudaTriangleCollisionModel*>(collisionModel->getLast());
if (cudaModel == nullptr || cudaModel->empty()) return false;
```

`dynamic_cast` is a runtime type check. It asks "is this collision model
actually a CUDA triangle model?" Because our scene used `CudaVec3f`, the answer
is yes. (If it were a plain CPU model, the cast returns `nullptr` and the code
falls back to a CPU path.)

### Step B — get or build the cached topology

```cpp
auto& cacheEntry = m_triangleTopologyCache[static_cast<const void*>(cudaModel)];
cacheEntry.seenThisFrame = true;
```

It looks up this model in the topology cache (from file 03). It marks the entry
"seen this frame." Then it decides whether the topology changed:

```cpp
const bool sizeChanged = /* triangle or vertex count differs */;
const std::uint64_t topologyHash =
    (sizeChanged || d_validateTriangleTopologyCache.getValue())
        ? computeTopologyHash()       // recompute the FNV-1a fingerprint
        : cacheEntry.topologyHash;    // reuse the cached one
const bool topologyChanged = sizeChanged || cacheEntry.topologyHash != topologyHash;

if (topologyChanged) {
    // rebuild the flat [v0,v1,v2,...] index array
    cacheEntry.triangleIndices.resize(triangles.size() * 3u);
    for (each triangle) { /* copy its 3 indices in */ }
    cacheEntry.missCount += 1;
} else {
    cacheEntry.hitCount += 1;        // nothing to do — cache hit!
}
```

By default `validateTriangleTopologyCache=false`, so after frame 1 it doesn't
even recompute the hash (the size hasn't changed). It just takes the
cache-hit branch: the flat index array already exists and is reused as-is.

**This is the per-frame win:** no rebuilding the 38,400-element index array for
the tissue, every frame.

### Step C — get the positions WITHOUT copying (the zero-copy magic)

```cpp
const backend::TriangleVertex* directDevicePositions = nullptr;
if (d_useDirectDevicePositions.getValue()) {
    const auto* sofaDevicePositions = positions.deviceRead();
    static_assert(sizeof(SofaHostPosition) == sizeof(backend::TriangleVertex));
    directDevicePositions =
        reinterpret_cast<const backend::TriangleVertex*>(sofaDevicePositions);
}
```

This is the heart of it. `positions.deviceRead()` is a SofaCUDA function that
returns a **raw pointer into the VRAM buffer** where the positions live. No
copy — just the address.

The `static_assert` confirms that SOFA's vertex layout (3 floats) is byte-for-
byte identical to our backend's `TriangleVertex` struct (also 3 floats), so we
can safely `reinterpret_cast` (reinterpret the bytes as a different type without
converting them). The pointer now points at the same VRAM, viewed as our type.

### Step D — fill the `TriangleIndexedSurface`

```cpp
outSurface.positions = nullptr;                 // host pointer unused
outSurface.devicePositions = directDevicePositions;   // the GPU pointer
outSurface.vertexCount = positions.size();      // 6,561 for tissue
outSurface.triangleIndices = cacheEntry.triangleIndices.data();  // the cached flat array
outSurface.triangleCount = cacheEntry.triangleCount;             // 12,800
outSurface.surfaceId = (uint64) cudaModel;      // identity = model pointer
outSurface.topologyVersion = cacheEntry.topologyHash;            // the fingerprint
return true;
```

The result is a `TriangleIndexedSurface` (defined in `GpuCollisionBackend.h`):

```cpp
struct TriangleIndexedSurface {
    const TriangleVertex* positions;       // host pointer (null here)
    const TriangleVertex* devicePositions; // GPU pointer (used here)
    std::uint32_t vertexCount;
    const std::uint32_t* triangleIndices;  // flat (v0,v1,v2,...) — still on host
    std::uint32_t triangleCount;
    std::uint64_t surfaceId;               // for caching + self-collision detection
    std::uint64_t topologyVersion;         // for the device-side index cache
};
```

Notice the asymmetry:

- **Positions** → a *GPU* pointer (already in VRAM, zero copy).
- **Indices** → a *host* pointer to the cached flat array. These get uploaded to
  GPU once and then cached device-side (next section).

---

## 5.4 The `surfaceId` and `topologyVersion` — the device-side index cache

Inside the backend, the workspace remembers which surface's indices it has
already uploaded:

```text
indexedTissueSurfaceId        — which model's indices are currently in VRAM
indexedTissueTopologyVersion  — the version (hash) of those indices
```

When the backend receives the surface, it checks:

```cpp
const bool uploadTissueTopology =
    workspace.indexedTissueSurfaceId != tissueSurface.surfaceId ||
    workspace.indexedTissueTopologyVersion != tissueSurface.topologyVersion;
```

- Frame 1: the workspace has never seen this surface → `uploadTissueTopology =
  true` → it copies the index array to VRAM with `cudaMemcpyAsync`.
- Frame 2+: same surfaceId, same version → `uploadTissueTopology = false` → it
  **skips the upload** and reuses the device buffer.

So the indices cross PCIe exactly once, on frame 1. After that: zero H2D for
indices too.

### Why surfaceId is the model pointer

`surfaceId = (uint64) cudaModel` — the memory address of the collision model.
This guarantees a unique ID per object (two different objects have different
addresses). It's used for two things:

1. The device-index cache above (don't re-upload the same model's indices).
2. Self-collision detection in the v-t path (file 09): if both surfaces have
   the *same* surfaceId, they're the same mesh, and the kernel enables
   "own-corner exclusion."

---

## 5.5 Building the config

After extraction, the narrow phase fills a `DenseGridConfig` from the Data
fields:

```cpp
backend::DenseGridConfig denseGridConfig;
denseGridConfig.gridMinX = d_gridMinX.getValue();   // -4.5
denseGridConfig.gridResolutionX = d_gridResolutionX.getValue();  // 64
denseGridConfig.contactDistance = d_contactDistance.getValue();  // 0.03
denseGridConfig.copyContactsToHost = d_copyContactsToHost.getValue();  // false
denseGridConfig.readCountersWhenContactsStayOnDevice =
    d_readCountersWhenContactsStayOnDevice.getValue() || d_detailedProfiling.getValue();
// ... etc — every grid parameter from the scene ...
```

This struct is a plain bundle of numbers that gets passed to the backend. It's
how the scene's Python parameters reach the CUDA code.

---

## 5.6 The data-transfer scorecard for Phase 2

Let's tally the bytes crossing PCIe during preparation, in steady state
(frame 2+):

| Data | Direction | Bytes | Why |
|---|---|---|---|
| Tissue positions | H2D | **0** | Already in VRAM; we use the `deviceRead()` pointer |
| Blade positions | H2D | **0** | Same |
| Tissue indices | H2D | **0** | Cached device-side (uploaded frame 1 only) |
| Blade indices | H2D | **0** | Same |

**Total: 0 bytes.** The CSV's `avg_host_to_device_bytes = 0` is this fact.

On frame 1 only, the indices upload once: ~38,400 × 4 bytes ≈ 154 KB for the
tissue, a few hundred bytes for the blade. After that, nothing.

---

## 5.7 Summary of Phase 2

```text
Input:  the pair (tissue, blade) as SOFA collision models
Work:   - dynamic_cast to confirm CUDA models
        - topology cache: reuse the flat index array (cache hit)
        - deviceRead(): grab the VRAM position pointer (zero copy)
        - fill TriangleIndexedSurface for each side
        - build DenseGridConfig from the Data fields
Output: two TriangleIndexedSurface structs + a config, ready for the GPU
Cost:   ~0.005 ms; 0 bytes H2D in steady state
```

The data is ready. Now the GPU does the actual collision math. But to understand
the kernels, you first need to understand the **dense grid** — the spatial data
structure they all operate on. That's the next file, with a fully worked numeric
example. Go to [06_the_dense_grid.md](06_the_dense_grid.md).
