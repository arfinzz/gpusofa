# GPU SOFA Collision — the complete tutorial

This tutorial explains, from the ground up, **exactly** what this project does and how —
assuming **no prior knowledge** of SOFA, GPUs, or CUDA. Every claim is grounded in the
real code (file and function names are given). It is the gentle, example-driven
companion to `guide/architecture.md` (the terse reference).

It reflects the **current** state of the project: the **five broad-cull ways** (dense,
Phase-15 dense, optimised hash, simple direct-bucket hash, sorted grid), the four
narrow-phase paths, CUDA graphs, the profiling story, and the optimizations that worked
**and the ones that failed**.

## How to read this

Read **[00_high_level_flow.md](00_high_level_flow.md) first** — the whole system on one
page, no jargon. Then follow the order below; each chapter builds on the previous. Two
chapters are **references** (look-up tables, not bedtime reading): 17 and 18.

### Part I — Orientation
| # | Chapter | What you'll learn |
|---|---|---|
| 00 | [The high-level flow](00_high_level_flow.md) | **Start here.** The whole system in plain words, one frame top to bottom (+ a flow diagram) |
| 01 | [Foundations](01_foundations.md) | What SOFA is, what a mesh is, CPU vs GPU, the CUDA words you need |
| 02 | [The scene files](02_the_scene.md) | Line-by-line tour of a benchmark scene |
| 03 | [Setup & the zero-copy data path](03_setup_and_data_path.md) | Before the clock starts: plugin load, GPU memory, the topology cache |

### Part II — The broad cull (finding candidate pairs)
| # | Chapter | What you'll learn |
|---|---|---|
| 04 | [The broad phase](04_broad_phase.md) | Finding which *objects* might touch (the cheap first pass) |
| 05 | [Zero-copy data prep](05_zero_copy_prep.md) | Handing mesh data to the GPU without copying anything |
| 06 | [The dense grid (+ optimised dense, Phase 15)](06_the_dense_grid.md) | The default broad cull, with a fully worked numeric example and the tool-active-cell optimization |
| 07 | [The spatial-hash broad cull + the hashing in detail](07_the_hash_broad_cull.md) | The alternative for big tissues — and **exactly what kind of hashing** it uses (open addressing, MurmurHash, atomicCAS, linear probing) |
| 08 | [Optimising the hash + the 4th way](08_optimising_the_hash.md) | The six tricks (compact buckets, no binary search, touched-clear, 32-bit pairs, scan drop) **+ CUDA graphs** — and the **simple direct-bucket "4th way"** that ties them with 7 kernels instead of 11 |

### Part III — The narrow phase (exact contacts)
| # | Chapter | What you'll learn |
|---|---|---|
| 09 | [The kernels](09_the_kernels.md) | The CUDA kernels that build the grid and resolve contacts |
| 10 | [The narrow-phase math](10_the_math.md) | Closest points, barycentrics, the VF / FV / EE feature tests |
| 11 | [Vertex-triangle paths](11_vertex_triangle.md) | Self-collision and point-cloud-vs-mesh |
| 12 | [Sync & output: the speed secret](12_sync_and_output.md) | Why it's so fast: detection-only, never wait for the GPU |

### Part IV — Running, measuring, optimizing
| # | Chapter | What you'll learn |
|---|---|---|
| 13 | [Every scene & every run](13_scenes_and_runs.md) | All five scenes, fast vs validation, the dense↔hash A/B runs, what each measures |
| 14 | [Benchmark metrics](14_benchmark_metrics.md) | How timing works and what every CSV number means |
| 15 | [Profiling & tuning](15_profiling_and_tuning.md) | Finding a kernel's bottleneck with Nsight (compute/memory/latency, stall reasons) and why FPS lies |
| 16 | [Optimizations & dead ends](16_optimizations_and_dead_ends.md) | **Every optimization** easy + hard with diagrams — *and the ones that were tried and failed*, so you don't retry them |

### References (look up, don't read end-to-end)
| # | Chapter | What it is |
|---|---|---|
| 17 | [Kernels & data-structures reference](17_kernels_reference.md) | Every kernel, its inputs/outputs, data structures, kernel counts, exact thread allocation |
| 18 | [Glossary](18_glossary.md) | Every term, defined in one place |

> **Companion documents (outside the tutorial):**
> [reports/README_metrics_explained.md](../reports/README_metrics_explained.md) defines
> every benchmark number; the current measured results + full optimization history are in
> [reports/performance_six_ways_20260713.md](../reports/performance_six_ways_20260713.md);
> `guide/architecture.md` and `guide/plan.md` are the terse engineering references.

> **Diagrams.** Chapters 00, 07, 08, 13, 15, 16 use ```mermaid``` blocks — they render as
> real flowcharts on GitHub, VS Code, and most markdown viewers. Chapters 07–08 also embed
> rendered **SVG diagrams** (in [assets/hash/](assets/hash/)) — the step-by-step
> allocation, hashing, and optimisation walkthroughs, including a full variable-state
> trace. Other chapters use plain-text diagrams that render anywhere.

## The one-sentence summary

> The project takes the slowest part of a surgical simulation — figuring out when a
> surgical tool touches tissue — and moves it onto the graphics card, where thousands of
> tiny processors check all the triangles at once instead of one CPU checking them one
> at a time.

## The mental picture to hold

```text
        A SOFA SCENE (a Python file): tissue mesh + tool mesh, on the GPU
                          │  every frame (5 ms of simulated time)
                          ▼
        BROAD PHASE   — which objects are near?            (cheap CPU)
        BROAD CULL    — which triangle PAIRS might touch?  (GPU: dense grid OR hash)
        NARROW PHASE  — do they actually touch, and where? (GPU: the heavy math)
                          ▼
        Contacts (kept on the GPU, never copied back) + a CSV timing line
```

Everything in this tutorial expands one box of that picture. Start with
[00_high_level_flow.md](00_high_level_flow.md).
