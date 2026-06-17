# GPU SOFA Collision — Beginner Tutorial

This tutorial explains, from the ground up, exactly what happens when you run
the GPU collision benchmark in this project. It assumes **no prior knowledge**
of SOFA, GPUs, or CUDA. Every claim is grounded in the actual code in this
repository — file names and function names are given so you can open them and
follow along.

If you have already read `guide/architecture.md`, this tutorial is the gentle,
example-driven companion to it. The guide is a reference; this is a lesson.

## How to read this

**New here? Read [00_high_level_flow.md](00_high_level_flow.md) first** — it's the
easy, no-jargon big picture of the whole system on one page. Then read the rest
in order; each builds on the previous. The final file,
[13_kernels_and_data_structures_reference.md](13_kernels_and_data_structures_reference.md),
is the complete engineering lookup table (every kernel, data structure, and
thread count) — use it as a reference, not a bedtime read.

| File | What you'll learn | Time |
|---|---|---|
| [00_high_level_flow.md](00_high_level_flow.md) | **Start here.** The whole system in plain words, one frame top to bottom | 10 min |
| [01_foundations.md](01_foundations.md) | What SOFA is, what a mesh is, CPU vs GPU, the CUDA words you need | 15 min |
| [02_the_scene.md](02_the_scene.md) | Line-by-line tour of the benchmark scene file | 15 min |
| [03_phase0_setup.md](03_phase0_setup.md) | What happens before the clock starts: plugin load, GPU memory, caches | 15 min |
| [04_phase1_broad_phase.md](04_phase1_broad_phase.md) | The broad phase — finding which objects *might* touch | 10 min |
| [05_phase2_narrow_prep.md](05_phase2_narrow_prep.md) | Preparing data for the GPU without copying anything (zero-copy) | 15 min |
| [06_the_dense_grid.md](06_the_dense_grid.md) | The core data structure, with a fully worked numeric example | 20 min |
| [07_phase3_kernels.md](07_phase3_kernels.md) | The seven CUDA kernels that do the actual work | 25 min |
| [08_the_math.md](08_the_math.md) | The geometry: closest points, barycentrics, VF and EE tests | 20 min |
| [09_vertex_triangle.md](09_vertex_triangle.md) | Self-collision and point-cloud-vs-mesh paths | 15 min |
| [10_phase4_sync_and_output.md](10_phase4_sync_and_output.md) | Why the benchmark is so fast: the synchronization bypass | 10 min |
| [11_profiling.md](11_profiling.md) | How timing works and what every number in the CSV means | 15 min |
| [12_glossary.md](12_glossary.md) | Every term, defined in one place | reference |
| [13_kernels_and_data_structures_reference.md](13_kernels_and_data_structures_reference.md) | **Reference.** Every kernel, its inputs/outputs, the data structures, kernel counts, and exactly how threads are allocated | reference |
| [14_optimizations.md](14_optimizations.md) | **Every optimization** explained easy + hard, with mermaid diagrams — the dense grid, Phase 15, the whole spatial-hash rework, CUDA graphs, **and the ones that were tried and *failed*** (so you don't retry them) | 30 min |
| [15_profiling_and_tuning.md](15_profiling_and_tuning.md) | How to **find a kernel's bottleneck** with Nsight (compute/memory/latency, stall reasons) and why FPS lies — the method behind every win and dead end | 20 min |

> Companion outside the tutorial: [reports/README_metrics_explained.md](../reports/README_metrics_explained.md)
> defines every number the benchmarks print (FPS, kernel time, contact counts, …);
> the current measured numbers + full optimization history live in
> [reports/performance_and_optimizations_20260618.md](../reports/performance_and_optimizations_20260618.md).

> **Diagrams:** files 00, 14, and 15 use `mermaid` flow diagrams — they render as
> real flowcharts on GitHub, VS Code, and most markdown viewers (plain-text
> fallback boxes are kept elsewhere).

## The one-sentence summary

> The project takes the slowest part of a surgical simulation — figuring out
> when a surgical tool touches tissue — and moves it onto the graphics card,
> where thousands of tiny processors check all the triangles at once instead
> of one CPU checking them one at a time.

## The mental picture you should hold

```text
        A SOFA SCENE (described in a Python file)
        ┌─────────────────────────────────────────┐
        │  Tissue  (a sheet of 12,800 triangles)   │
        │  Blade   (a box of 12 triangles)         │
        │                                          │
        │  + instructions: "detect collisions      │
        │     on the GPU, don't bother responding" │
        └─────────────────────────────────────────┘
                          │
                          │  every frame (every 5 ms of simulated time)
                          ▼
        ┌─────────────────────────────────────────┐
        │  BROAD PHASE  — which objects are near?   │  (cheap)
        │  NARROW PHASE — exactly which triangles   │  (the GPU work)
        │                 are touching, and where?  │
        └─────────────────────────────────────────┘
                          │
                          ▼
        Contacts (kept on the GPU, never copied back to the CPU)
        + a stopwatch reading written to a CSV file
```

Everything in this tutorial is an expansion of that picture.

## A note on "phases"

The word "phase" is used two different ways in this project, and it's worth
clearing up now so you don't get confused:

1. **Execution phases** (Phase 0, 1, 2, 3, 4) — the steps that run *every
   frame* while the simulation is going. This tutorial is organized around
   these.

2. **Project phases** (Phase 11, Phase 12, etc.) — the chronological
   milestones of *building* the software, recorded in `guide/plan.md`. These
   are the history of the project, not the runtime steps.

When this tutorial says "Phase 3," it means the third runtime step (the CUDA
kernels). When `guide/plan.md` says "Phase 11," it means the eleventh thing
the developers built (feature-based proximity).

Now go to [01_foundations.md](01_foundations.md).
