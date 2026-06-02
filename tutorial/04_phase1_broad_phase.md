# 04 — Phase 1: The Broad Phase

The clock ticks. `DefaultAnimationLoop::step()` runs, and it calls
`CollisionPipeline::computeCollisionDetection()`, which calls your broad phase.

File: `GpuCollisionBroadPhase.cpp`.

---

## 4.1 What the broad phase is *for*

The broad phase answers a cheap question: **"which pairs of objects are worth
checking in detail?"**

In a big scene with 50 organs and 5 instruments, you do NOT want to run the
expensive narrow phase on all 55 × 54 / 2 = 1,485 possible pairs. Most are
nowhere near each other. The broad phase prunes them down to the few that might
actually touch.

In our benchmark there are only **two** objects (tissue and blade), so there's
exactly **one** possible pair. The broad phase has almost nothing to do — but
let's see how it works anyway, because the mechanism matters for bigger scenes.

---

## 4.2 The three calls SOFA makes

SOFA drives the broad phase with three method calls each frame:

```text
beginBroadPhase()                  → reset for a new frame
addCollisionModel(cm) for each cm  → "here is an object that can collide"
endBroadPhase()                    → "now figure out the pairs"
```

### `beginBroadPhase()`

```cpp
void GpuCollisionBroadPhase::beginBroadPhase() {
    m_pendingModels.clear();
    // ... base class setup ...
}
```

It empties the list of models it's collecting. Fresh start.

### `addCollisionModel(cm)`

```cpp
void GpuCollisionBroadPhase::addCollisionModel(sofa::core::CollisionModel* cm) {
    // ...
    m_pendingModels.push_back(cm);
}
```

SOFA calls this once per collidable object. It just stashes the pointer. For
our scene it's called twice: once with the tissue's collision model, once with
the blade's. After both calls, `m_pendingModels = [tissue, blade]`.

### `endBroadPhase()` — the actual work

This is where pairs are decided. Our scene has `useObjectAabbCulling=False`, so
the code takes the simple branch:

```cpp
// (simplified from the actual code)
std::vector<CollisionModelEntry> entries;
for (auto* cm : m_pendingModels) {
    if (cm == nullptr || cm->empty()) continue;
    entries.push_back({ cm, cm->getLast() });
    if (doesSelfCollide(cm))
        this->cmPairs.emplace_back(cm, cm);    // self-collision pair!
}

// O(n^2): every model against every other model
for (i in entries)
    for (j in 0..i)
        appendPotentialPair(entries[i], entries[j]);
```

For 2 models, the double loop runs exactly once (i=1, j=0): it considers the
(blade, tissue) pair. `appendPotentialPair` does a couple of sanity checks
(are they allowed to collide? does the intersection method accept them?) and
then pushes the pair into `this->cmPairs`.

`cmPairs` is the inherited list that SOFA reads to know which pairs need the
narrow phase. After `endBroadPhase`, `cmPairs = [(tissue, blade)]`.

---

## 4.3 The "rough radar" metaphor — and when it's wrong

You may have heard the broad phase described as a "rough radar" that knows
*where* objects roughly are. That's true **only** when `useObjectAabbCulling`
is on. In that mode (`endBroadPhase`'s other branch), the code:

1. Computes one **AABB** (axis-aligned bounding box — the smallest box that
   contains the object) per model.
2. Hands all the boxes to `backend::computeBroadPhasePairs(...)`, which checks
   on the GPU which boxes overlap.
3. Keeps only the overlapping pairs.

That's the "radar." But **our benchmark turns it off** (`useObjectAabbCulling=False`),
because with two objects it's faster to just emit the one pair than to compute
boxes and call the GPU. So in our scene the broad phase is *not* doing spatial
culling — it's a trivial pair enumerator. The real spatial work (chopping space
into a grid) happens in the narrow phase.

**Takeaway:** in this benchmark, the broad phase's only job is to say "check the
tissue against the blade." That's it.

---

## 4.4 Self-collision pairs

Notice this line in `endBroadPhase`:

```cpp
if (doesSelfCollide(cm))
    this->cmPairs.emplace_back(cm, cm);
```

If a `TriangleCollisionModel` was created with `selfCollision=True`, the broad
phase emits a pair where *both sides are the same model* — `(cm, cm)`. This is
how an object gets checked against itself (think tissue folding onto itself).

Our benchmark uses `selfCollision=False`, so this doesn't fire here. But it's
the trigger for the vertex-triangle self-collision path in file 09 — remember
this line.

---

## 4.5 Profiling the broad phase

At the end, the broad phase records a timing snapshot:

```cpp
stageSnapshot.inputPrimitiveCount = entries.size();   // 2 models
stageSnapshot.outputPairCount = this->cmPairs.size(); // 1 pair
stageSnapshot.wallMilliseconds = /* elapsed time */;
profiling::recordBroadPhase(stageSnapshot);
```

In the benchmark CSV you'll see `broad_input_primitive_count=2` and
`broad_output_pair_count=1`, and `broad_wall_ms` will be tiny (around
0.01 ms) — because, as we said, there's almost nothing to do.

---

## 4.6 Summary of Phase 1

```text
Input:  the scene's collision models (tissue, blade)
Output: a list of model pairs to check → [(tissue, blade)]
Cost:   negligible (~0.01 ms); it's just pair bookkeeping in this scene
GPU:    none used in this scene (culling is off)
```

The broad phase handed the narrow phase a single pair. Now the narrow phase has
to do the real work: take that pair and find the exact contacts. But first it
has to get the triangle data ready for the GPU — without copying it. That's
Phase 2. Go to [05_phase2_narrow_prep.md](05_phase2_narrow_prep.md).
