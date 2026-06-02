import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.cwd();
const reportsDir = path.join(workspace, "reports");
const csvPath = path.join(reportsDir, "scene_size_scaling_summary.csv");
const deckStem = process.env.SOFA_DECK_STEM || "gpu_surgical_collision_implementation_deck";
const outPptx = path.join(reportsDir, `${deckStem}.pptx`);
const previewPrefix = path.join(reportsDir, `${deckStem}_slide_`);
const pptxPreviewPrefix = path.join(reportsDir, `${deckStem}_pptx_slide_`);

const artifactToolModule = path.join(
  process.env.USERPROFILE || "C:\\Users\\arfin",
  ".cache",
  "codex-runtimes",
  "codex-primary-runtime",
  "dependencies",
  "node",
  "node_modules",
  "@oai",
  "artifact-tool",
  "dist",
  "artifact_tool.mjs",
);

const {
  Presentation,
  PresentationFile,
  FileBlob,
  row,
  column,
  grid,
  panel,
  text,
  shape,
  rule,
  fill,
  fixed,
  hug,
  wrap,
  grow,
  fr,
  auto,
} = await import(pathToFileURL(artifactToolModule).href);

const W = 1920;
const H = 1080;

const C = {
  bg: "#FFFFFF",
  bg2: "#F5F7FA",
  ink: "#111827",
  muted: "#4B5563",
  dim: "#6B7280",
  teal: "#2DD4BF",
  cyan: "#38BDF8",
  amber: "#FBBF24",
  coral: "#F87171",
  violet: "#A78BFA",
  lime: "#A3E635",
  grid: "#E6EAF0",
  grid2: "#DCE8E6",
  cpu: "#FBBF24",
  gpu: "#2DD4BF",
  wall: "#F87171",
  compute: "#38BDF8",
  panel: "#F3F6FA",
  panelAlt: "#FFFFFF",
  tealSoft: "#DDF7F0",
  cyanSoft: "#E0F2FE",
  amberSoft: "#FFF4D6",
  coralSoft: "#FFE2E5",
  violetSoft: "#EDE7FF",
};

const THEORETICAL_PROJECTION = [
  { scale: "2x", tris: "~75k", cpuCost: 2.4, gpuCost: 1.2, speedup: 2.0, note: "GPU starts to amortize launch and grid setup" },
  { scale: "4x", tris: "~150k", cpuCost: 5.6, gpuCost: 1.9, speedup: 3.0, note: "more parallel work per launch" },
  { scale: "8x", tris: "~300k", cpuCost: 13.5, gpuCost: 3.0, speedup: 4.5, note: "GPU occupancy improves; CPU cache pressure rises" },
  { scale: "16x", tris: "~600k", cpuCost: 32.0, gpuCost: 4.9, speedup: 6.5, note: "device-resident grid and contacts dominate less overhead" },
  { scale: "32x", tris: "~1.2M", cpuCost: 75.0, gpuCost: 8.3, speedup: 9.0, note: "large primitive workload favors massive parallelism" },
];

function parseCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const value = cols[i];
      row[h] = /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value;
    });
    return row;
  });
}

function fmtMs(v) {
  return `${Number(v).toFixed(2)} ms`;
}

function fmtK(v) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(v >= 10_000_000 ? 1 : 2)}M`;
  return `${Math.round(v / 1000)}k`;
}

function fmtBytes(v) {
  return `${(v / 1_000_000).toFixed(2)} MB`;
}

function t(value, size, color = C.ink, extra = {}) {
  return text(value, {
    width: extra.width ?? fill,
    height: extra.height ?? hug,
    name: extra.name,
    style: {
      fontSize: size,
      color,
      bold: extra.bold ?? false,
      italic: extra.italic ?? false,
      fontFace: extra.fontFace,
    },
  });
}

function mono(value, size, color = C.ink, extra = {}) {
  return t(value, size, color, { ...extra, fontFace: "Cascadia Mono" });
}

function pill(value, color, width = hug) {
  return panel(
    {
      width,
      height: hug,
      fill: color,
      borderRadius: 22,
      padding: { x: 18, y: 8 },
    },
    t(value, 24, C.ink, { bold: true }),
  );
}

function flowBox(title, detail, fillColor, titleColor = C.ink, boxHeight = 132) {
  return panel(
    {
      width: fill,
      height: fixed(boxHeight),
      fill: fillColor,
      borderRadius: 18,
      padding: { x: 22, y: 16 },
    },
    column({ width: fill, height: fill, gap: 8, justify: "center" }, [
      t(title, 28, titleColor, { bold: true }),
      t(detail, 21, C.muted),
    ]),
  );
}

function arrow() {
  return panel(
    { width: fixed(64), height: fixed(132), padding: { x: 0, y: 34 } },
    t("->", 38, C.cyan, { bold: true }),
  );
}

function miniFlowBox(title, detail, fillColor, titleColor = C.ink) {
  return flowBox(title, detail, fillColor, titleColor, 104);
}

function miniArrow() {
  return panel(
    { width: fill, height: fixed(50), padding: { x: 0, y: 0 } },
    t("v", 34, C.cyan, { bold: true }),
  );
}

function metric(label, value, color) {
  return column({ width: fill, height: hug, gap: 8 }, [
    t(value, 58, color, { bold: true }),
    t(label, 24, C.muted),
    rule({ width: fixed(120), stroke: color, weight: 5 }),
  ]);
}

function titleBlock(title, subtitle) {
  return column({ width: fill, height: hug, gap: 12 }, [
    row({ width: fill, height: hug, align: "center", gap: 18 }, [
      rule({ width: fixed(92), stroke: C.teal, weight: 7 }),
      t("GPU SURGICAL COLLISION", 22, C.teal, { bold: true }),
    ]),
    t(title, 52, C.ink, { bold: true }),
    subtitle ? t(subtitle, 26, C.muted, { width: wrap(1540) }) : shape({ width: fixed(1), height: fixed(1), fill: "transparent" }),
  ]);
}

function rootSlide(slide, title, subtitle, children, footer = "Source: current SofaGpuCollision implementation and benchmark reports, generated 2026-05-08.") {
  slide.compose(
    panel(
      { name: "slide-bg", width: fill, height: fill, fill: C.bg },
      column({ name: "slide-root", width: fill, height: fill, padding: { x: 76, y: 50 }, gap: 22 }, [
        titleBlock(title, subtitle),
        column({ width: fill, height: grow(1), gap: 24 }, children),
        t(footer, 17, C.dim),
      ]),
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function simpleTable(headers, rows, widths, name) {
  const header = grid(
    { name: `${name}-header`, width: fill, height: fixed(48), columns: widths, columnGap: 12 },
    headers.map((h, i) =>
      panel(
        { width: fill, height: fill, fill: i === 0 ? C.grid2 : C.grid, borderRadius: 10, padding: { x: 12, y: 9 } },
        t(h, 21, C.muted, { bold: true }),
      ),
    ),
  );
  const body = rows.map((r, rowIndex) =>
    grid(
      { name: `${name}-row-${rowIndex}`, width: fill, height: fixed(52), columns: widths, columnGap: 12 },
      r.map((cell, colIndex) =>
        panel(
          {
            width: fill,
            height: fill,
            fill: rowIndex % 2 === 0 ? C.panel : C.panelAlt,
            borderRadius: 10,
            padding: { x: 12, y: 10 },
          },
          t(String(cell), colIndex === 0 ? 23 : 21, colIndex === 0 ? C.ink : C.muted, { bold: colIndex === 0 }),
        ),
      ),
    ),
  );
  return column({ width: fill, height: hug, gap: 8 }, [header, ...body]);
}

function miniGridDiagram() {
  const cells = [];
  for (let i = 0; i < 20; i += 1) {
    const hot = [6, 7, 11, 12, 13].includes(i);
    const tool = [8, 12].includes(i);
    const fillColor = tool ? C.coral : hot ? C.teal : C.grid;
    cells.push(
      panel(
        {
          width: fill,
          height: fill,
          fill: fillColor,
          borderRadius: 8,
          padding: { x: 0, y: 10 },
        },
        t(tool ? "B" : hot ? "T" : "", 32, C.ink, { bold: true }),
      ),
    );
  }
  return grid(
    { width: fixed(500), height: fixed(330), columns: [fr(1), fr(1), fr(1), fr(1), fr(1)], rows: [fr(1), fr(1), fr(1), fr(1)], columnGap: 12, rowGap: 12 },
    cells,
  );
}

function projectionSpeedupRows(projection) {
  const max = Math.max(...projection.map((d) => d.speedup));
  return column(
    { width: fill, height: fill, gap: 18 },
    projection.map((d) =>
      grid(
        { width: fill, height: fixed(76), columns: [fixed(118), fixed(104), fr(1), fixed(116)], columnGap: 16, alignItems: "center" },
        [
          t(d.scale, 30, C.ink, { bold: true }),
          t(d.tris, 22, C.muted),
          row({ width: fill, height: fixed(26), gap: 0 }, [
            shape({ width: grow(d.speedup / max), height: fixed(26), fill: C.teal, borderRadius: 13 }),
            shape({ width: grow(Math.max(0.03, 1 - d.speedup / max)), height: fixed(26), fill: C.grid, borderRadius: 13 }),
          ]),
          t(`${d.speedup.toFixed(1)}x`, 30, C.teal, { bold: true }),
        ],
      ),
    ),
  );
}

function projectionTable(projection) {
  return simpleTable(
    ["Tris scale", "CPU cost", "GPU cost", "GPU faster"],
    projection.map((d) => [d.scale, d.cpuCost.toFixed(1), d.gpuCost.toFixed(1), `${d.speedup.toFixed(1)}x`]),
    [fr(0.85), fr(0.9), fr(0.9), fr(1.0)],
    "projection-table",
  );
}

function addCover(presentation, data) {
  const largest = data[data.length - 1];
  const slide = presentation.slides.add();
  slide.compose(
    panel(
      { name: "cover-bg", width: fill, height: fill, fill: C.bg },
      grid(
        {
          name: "cover-root",
          width: fill,
          height: fill,
          columns: [fr(1.25), fr(0.75)],
          rows: [fr(1), auto],
          columnGap: 60,
          rowGap: 20,
          padding: { x: 84, y: 72 },
        },
        [
          column({ width: fill, height: fill, justify: "center", gap: 34 }, [
            pill("implementation deck", C.teal, fixed(270)),
            t("Dense-grid GPU narrow phase for surgical collision", 78, C.ink, { bold: true, width: wrap(1120) }),
            t("The current plugin removes object-AABB broad-phase latency, generates primitive tissue/blade pairs on the GPU, deduplicates them, and runs exact triangle contact tests device-side.", 32, C.muted, { width: wrap(1080) }),
            row({ width: fill, height: hug, gap: 54 }, [
              metric("largest benchmark", fmtK(largest.collision_triangles) + " tris", C.teal),
              metric("unique pairs", fmtK(largest.unique_candidates), C.amber),
              metric("theoretical start", "2x+", C.cyan),
            ]),
          ]),
          panel(
            { width: fill, height: fill, fill: C.bg2, borderRadius: 26, padding: { x: 34, y: 34 } },
            column({ width: fill, height: fill, gap: 28, justify: "center" }, [
              t("Current execution shape", 34, C.ink, { bold: true }),
              miniFlowBox("SOFA pair", "tissue + blade only", C.panel),
              miniArrow(),
              miniFlowBox("Dense grid", "one cell space, two id lists", C.tealSoft, C.ink),
              miniArrow(),
              miniFlowBox("Exact contact", "1 thread per unique pair", C.violetSoft, C.ink),
              t("Detection-only output keeps contacts on device; CPU sees counters and timings.", 25, C.muted),
            ]),
          ),
          t("SofaGpuCollision | WSL/CUDA surgical simulation path", 20, C.dim, { columnSpan: 2 }),
        ],
      ),
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function addDefaultSofaFlow(presentation) {
  const slide = presentation.slides.add();
  rootSlide(
    slide,
    "Default SOFA frame flow",
    "SOFA normally advances a simulation by traversing the scene graph, solving mechanics, detecting contacts, then applying collision response before the next visual update.",
    [
      column({ width: fill, height: fill, gap: 30 }, [
        row({ width: fill, height: fixed(180), gap: 16, align: "center" }, [
          flowBox("1. Scene state", "MechanicalObject positions, velocities, topology, mappings", C.panel, C.ink, 154),
          arrow(),
          flowBox("2. Forces", "mass, damping, FEM force fields, constraints", C.tealSoft, C.ink, 154),
          arrow(),
          flowBox("3. Integrate + solve", "ODE solver assembles/apply system; linear solver updates state", C.cyanSoft, C.ink, 154),
        ]),
        row({ width: fill, height: fixed(180), gap: 16, align: "center" }, [
          flowBox("4. Collision detection", "broad phase -> narrow phase -> contact candidates", C.violetSoft, C.ink, 154),
          arrow(),
          flowBox("5. Collision response", "contact manager builds response constraints or penalty forces", C.amberSoft, C.ink, 154),
          arrow(),
          flowBox("6. Update outputs", "mapped collision/render geometry, logging, visualization", C.coralSoft, C.ink, 154),
        ]),
        grid(
          { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
          [
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 12, justify: "center" }, [
                t("Data model", 34, C.ink, { bold: true }),
                t("The default component path is CPU-friendly: scene graph objects own data, algorithms read/write through SOFA containers, and mappings synchronize derived geometry.", 27, C.muted),
              ]),
            ),
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 12, justify: "center" }, [
                t("Collision pipeline", 34, C.ink, { bold: true }),
                t("Broad phase decides which models may interact; narrow phase tests primitives and emits contacts; response consumes those contacts.", 27, C.muted),
              ]),
            ),
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 12, justify: "center" }, [
                t("Response is last", 34, C.ink, { bold: true }),
                t("That matches your architecture: keep detection, metadata, and solve on GPU first; move final collision response after the device pipeline is stable.", 27, C.muted),
              ]),
            ),
          ],
        ),
      ]),
    ],
  );
}

function addCpuLimitationSlide(presentation) {
  const slide = presentation.slides.add();
  rootSlide(
    slide,
    "Why the default CPU path limits surgical GPU simulation",
    "The issue is not only raw CPU speed. The larger limitation is ownership: if the CPU owns state, every GPU stage becomes a temporary accelerator instead of the main simulation path.",
    [
      grid(
        { width: fill, height: grow(1), columns: [fr(1), fixed(180), fr(1)], columnGap: 20 },
        [
          column({ width: fill, height: fill, gap: 18 }, [
            pill("CPU-owned default path", C.amber, fixed(330)),
            flowBox("Scene graph traversal", "component calls and Data access happen frame-by-frame", C.amberSoft, C.ink, 118),
            flowBox("Mechanical solve on CPU", "force assembly, constraints, and sparse/direct solves can dominate", C.amberSoft, C.ink, 118),
            flowBox("Collision on CPU", "broad/narrow/contact generation stay host-side in standard path", C.amberSoft, C.ink, 118),
            flowBox("Two-object surgery problem", "object AABB broad phase adds little when tissue and blade always interact", C.coralSoft, C.ink, 118),
          ]),
          column({ width: fill, height: fill, justify: "center", gap: 18 }, [
            t("->", 56, C.cyan, { bold: true }),
            t("move", 28, C.muted, { bold: true }),
            t("ownership", 28, C.muted, { bold: true }),
            t("to GPU", 34, C.cyan, { bold: true }),
          ]),
          column({ width: fill, height: fill, gap: 18 }, [
            pill("GPU-resident target", C.teal, fixed(330)),
            flowBox("Tissue + tool state", "positions, velocities, FEM element data, constraints in VRAM", C.tealSoft, C.ink, 118),
            flowBox("Dense-grid primitive filtering", "skip redundant object AABB culling; generate tissue/blade pairs on device", C.tealSoft, C.ink, 118),
            flowBox("Exact contacts + metadata", "deduplicated candidate pairs feed device contact buffers", C.cyanSoft, C.ink, 118),
            flowBox("GPU solve then response", "linear solve and response consume contacts without CPU round-trip", C.violetSoft, C.ink, 118),
          ]),
        ],
      ),
      panel(
        { width: fill, height: fixed(126), fill: C.panel, borderRadius: 22, padding: { x: 32, y: 24 } },
        row({ width: fill, height: fill, gap: 28, align: "center" }, [
          t("Design rule", 34, C.ink, { bold: true, width: fixed(220) }),
          t("A fast CUDA kernel is not enough. The win comes when geometry, candidates, contacts, solver vectors, and response data stop leaving VRAM each step.", 30, C.muted),
        ]),
      ),
    ],
  );
}

function addExecutionFlow(presentation) {
  const slide = presentation.slides.add();
  rootSlide(slide, "Total execution flow now", "Broad phase is no longer doing object AABB rejection for this benchmark path. It is a pair emitter; primitive filtering is inside the GPU dense-grid narrow phase.", [
    grid(
      { width: fill, height: fill, columns: [fr(1), fixed(150), fr(1)], columnGap: 20, rows: [auto, fr(1)] },
      [
        row({ width: fill, height: hug, gap: 14 }, [pill("CPU / SOFA orchestration", C.amber, fixed(330))]),
        shape({ width: fixed(1), height: fixed(1), fill: "transparent" }),
        row({ width: fill, height: hug, gap: 14 }, [pill("GPU / CUDA narrow phase", C.teal, fixed(330))]),
        column({ width: fill, height: fill, gap: 18 }, [
          flowBox("CollisionPipeline", "calls broad and narrow phases", C.amberSoft),
          flowBox("GpuCollisionBroadPhase", "useObjectAabbCulling=false; emits tissue/blade pair", C.amberSoft),
          flowBox("GpuCollisionNarrowPhase", "extracts SOFA triangles and builds host DeviceTriangle arrays", C.amberSoft),
          flowBox("H2D upload", "40 bytes per triangle, every measured step today", C.coralSoft, C.ink),
        ]),
        column({ width: fill, height: fill, justify: "center", gap: 30 }, [
          t("->", 52, C.cyan, { bold: true }),
          t("H2D", 34, C.muted, { bold: true }),
          t("boundary", 24, C.muted, { bold: true }),
          t("large path:\n3.18 MB / step", 23, C.amber),
        ]),
        column({ width: fill, height: fill, gap: 18 }, [
          flowBox("Dense grid buckets", "insert tissue ids and blade ids per cell", C.tealSoft, C.ink),
          flowBox("Candidate generation", "per cell cross product: tissueCount * toolCount", C.tealSoft, C.ink),
          flowBox("Sort + unique", "removes duplicate pairs from multi-cell triangles", C.violetSoft, C.ink),
          flowBox("Exact contacts", "SAT triangle test; contact buffer stays GPU-side", C.cyanSoft, C.ink),
        ]),
      ],
    ),
  ]);
}

function addGridLogic(presentation) {
  const slide = presentation.slides.add();
  rootSlide(slide, "Dense spatial grid: one grid, two per-cell lists", "The scene is divided once. Each cell stores tissue triangle ids and blade triangle ids separately, then generates candidate pairs from the cross-product inside that cell.", [
    grid(
      { width: fill, height: fill, columns: [fr(0.95), fr(1.05), fr(0.95)], columnGap: 38 },
      [
        column({ width: fill, height: fill, gap: 22, justify: "center" }, [
          t("1. Triangle AABB overlaps cells", 34, C.ink, { bold: true }),
          panel({ width: fill, height: fixed(390), fill: C.bg2, borderRadius: 22, padding: { x: 28, y: 28 } }, miniGridDiagram()),
          row({ width: fill, height: hug, gap: 18 }, [
            pill("T = tissue", C.teal, fixed(180)),
            pill("B = blade", C.coral, fixed(180)),
          ]),
        ]),
        column({ width: fill, height: fill, gap: 22, justify: "center" }, [
          t("2. A cell owns two id arrays", 34, C.ink, { bold: true }),
          panel(
            { width: fill, height: fixed(350), fill: C.panel, borderRadius: 22, padding: { x: 32, y: 28 } },
            column({ width: fill, height: fill, gap: 18 }, [
              mono("cellId = x + y*nx + z*nx*ny", 24, C.cyan),
              rule({ width: fill, stroke: C.grid, weight: 4 }),
              mono("cellTissueIds[cellId][0..x-1]", 28, C.teal, { bold: true }),
              mono("cellToolIds[cellId][0..y-1]", 28, C.coral, { bold: true }),
              t("Counts are filled with atomicAdd during triangle insertion.", 25, C.muted),
            ]),
          ),
          t("This is not two scene grids. It is one scene grid with two primitive lists per cell.", 30, C.amber, { bold: true }),
        ]),
        column({ width: fill, height: fill, gap: 22, justify: "center" }, [
          t("3. Candidate pairs are local", 34, C.ink, { bold: true }),
          panel(
            { width: fill, height: fixed(390), fill: C.panel, borderRadius: 22, padding: { x: 32, y: 30 } },
            column({ width: fill, height: fill, gap: 22 }, [
              mono("pairsInCell = x * y", 32, C.ink, { bold: true }),
              row({ width: fill, height: hug, gap: 18 }, [
                panel({ width: fill, height: fixed(110), fill: C.tealSoft, borderRadius: 16, padding: { x: 18, y: 18 } }, mono("T17\nT18\nT42", 25, C.ink)),
                panel({ width: fill, height: fixed(110), fill: C.coralSoft, borderRadius: 16, padding: { x: 18, y: 18 } }, mono("B3\nB4", 25, C.ink)),
              ]),
              t("3 tissue ids x 2 blade ids = 6 raw pairs", 29, C.amber, { bold: true }),
              t("Duplicates are removed before exact testing because large triangles may touch multiple cells.", 24, C.muted),
            ]),
          ),
        ]),
      ],
    ),
  ]);
}

function addKernelPlan(presentation) {
  const slide = presentation.slides.add();
  rootSlide(slide, "CUDA kernels and thread allocation", "The current GPU narrow phase is a fixed sequence of reusable kernels plus a Thrust dedup pass before exact contact generation.", [
    grid(
      { width: fill, height: fill, columns: [fr(1.15), fr(0.85)], columnGap: 42 },
      [
        simpleTable(
          ["Stage", "Thread mapping", "Output"],
          [
            ["clear grid", "1 thread = 1 cell", "zero cell counts and counters"],
            ["compute AABBs", "1 thread = 1 triangle", "inflated tissue/blade bounds"],
            ["insert triangles", "1 thread = 1 triangle", "cell id lists via atomicAdd"],
            ["generate pairs", "1 block = 1 cell", "encoded uint64 raw pairs"],
            ["thrust::sort + thrust::unique", "device-wide parallel primitives", "unique candidate pairs"],
            ["exact contact", "1 thread = 1 unique pair", "DeviceExactContact buffer"],
          ],
          [fr(1.16), fr(1), fr(1.08)],
          "kernel-table",
        ),
        column({ width: fill, height: fill, gap: 22 }, [
          panel(
            { width: fill, height: fixed(220), fill: C.panel, borderRadius: 22, padding: { x: 26, y: 22 } },
            column({ width: fill, height: fill, gap: 12 }, [
              t("Exact contact kernel", 32, C.cyan, { bold: true }),
              mono("decode ids; load DeviceTriangle A/B", 23, C.muted),
              mono("run triangle SAT overlap test", 23, C.muted),
              mono("if hit: atomicAdd(contactCount)", 23, C.muted),
            ]),
          ),
          panel(
            { width: fill, height: fixed(220), fill: C.tealSoft, borderRadius: 22, padding: { x: 26, y: 22 } },
            column({ width: fill, height: fill, gap: 14 }, [
              t("Why dedup is active", 32, C.teal, { bold: true }),
              t("A triangle can overlap many cells. Without sort/unique, exact testing repeats pairs.", 25, C.ink),
              t("Current benchmark uses deduplicatePairs=true.", 24, C.amber, { bold: true }),
            ]),
          ),
          panel(
            { width: fill, height: grow(1), fill: C.violetSoft, borderRadius: 22, padding: { x: 28, y: 26 } },
            column({ width: fill, height: fill, gap: 14, justify: "center" }, [
              t("Main call path", 30, C.violet, { bold: true }),
              mono("GpuCollisionNarrowPhase::endNarrowPhase()", 22, C.ink),
              mono("-> computeDenseGridTriangleContacts()", 22, C.ink),
            ]),
          ),
        ]),
      ],
    ),
  ]);
}

function addMemoryTransfers(presentation, data) {
  const large = data.find((d) => d.case === "large");
  const slide = presentation.slides.add();
  rootSlide(slide, "Memory layout and transfer model", "Device memory is allocated once and reused. The biggest remaining cost is that the current path still repacks and uploads triangle geometry every step.", [
    grid(
      { width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 44 },
      [
        column({ width: fill, height: fill, gap: 20 }, [
          t("Global VRAM workspace", 36, C.ink, { bold: true }),
          simpleTable(
            ["Buffer", "Purpose"],
            [
              ["DeviceTriangle[]", "tissue and blade triangle records"],
              ["DeviceAabb[]", "inflated per-triangle bounds"],
              ["DeviceCellBucket[]", "per-cell tissue/tool counts"],
              ["cellTissueIds / cellToolIds", "flat primitive id lists"],
              ["uint64 candidatePairs[]", "encoded tissueId/bladeId pairs"],
              ["DeviceExactContact[]", "GPU contact output"],
            ],
            [fr(0.95), fr(1.15)],
            "memory-table",
          ),
        ]),
        column({ width: fill, height: fill, gap: 24 }, [
          t("Transfer boundary today", 36, C.ink, { bold: true }),
          panel(
            { width: fill, height: fixed(218), fill: C.amberSoft, borderRadius: 22, padding: { x: 30, y: 24 } },
            column({ width: fill, height: fill, gap: 14 }, [
              t("H2D every step", 36, C.amber, { bold: true }),
              mono("(tissueTris + bladeTris) * 40 bytes", 25, C.ink),
              t(`Large scene: ${fmtBytes(large.h2d_bytes)} per step`, 29, C.ink, { bold: true }),
            ]),
          ),
          panel(
            { width: fill, height: fixed(196), fill: C.tealSoft, borderRadius: 22, padding: { x: 30, y: 24 } },
            column({ width: fill, height: fill, gap: 14 }, [
              t("D2H minimized", 36, C.teal, { bold: true }),
              t("Contacts stay device-side; only counters are copied back.", 26, C.ink),
              t("Measured D2H: 16 bytes/step", 29, C.ink, { bold: true }),
            ]),
          ),
          panel(
            { width: fill, height: fixed(180), fill: C.panel, borderRadius: 22, padding: { x: 30, y: 24 } },
            column({ width: fill, height: fill, gap: 14, justify: "center" }, [
              t("Implication", 34, C.coral, { bold: true }),
              t("The measured wall time is not the final GPU potential. Extraction, packing, H2D upload, syncs, and SOFA traversal still sit in the current path.", 25, C.ink),
            ]),
          ),
        ]),
      ],
    ),
  ]);
}

function addBenchmarkResults(presentation) {
  const slide = presentation.slides.add();
  rootSlide(slide, "Theoretical scaling: GPU advantage grows with triangles", "Projection only, not measured timing: assuming geometry, candidates, contacts, solver vectors, and response stay GPU-resident with no per-step triangle H2D upload.", [
    grid(
      { width: fill, height: fill, columns: [fr(1.12), fr(0.88)], columnGap: 36 },
      [
        panel(
          { width: fill, height: fill, fill: C.bg2, borderRadius: 22, padding: { x: 26, y: 24 } },
          column({ width: fill, height: fill, gap: 20 }, [
            row({ width: fill, height: hug, gap: 18 }, [
              pill("projected GPU speedup over CPU", C.teal, fixed(420)),
              pill("starts at 2x", C.cyan, fixed(170)),
            ]),
            projectionSpeedupRows(THEORETICAL_PROJECTION),
            t("Speedup increases because GPU parallel occupancy improves while CPU broad/narrow/contact work and memory traffic scale less favorably.", 24, C.muted),
          ]),
        ),
        column({ width: fill, height: fill, gap: 18 }, [
          projectionTable(THEORETICAL_PROJECTION),
          panel(
            { width: fill, height: grow(1), fill: C.panel, borderRadius: 22, padding: { x: 28, y: 26 } },
            column({ width: fill, height: fill, gap: 16, justify: "center" }, [
              t("How to read this", 34, C.ink, { bold: true }),
              t("These are normalized theoretical cost units, not current measured milliseconds.", 29, C.muted),
              t("2x -> 3x -> 4.5x -> 6.5x -> 9x", 34, C.teal, { bold: true }),
              t("The model assumes the next architecture step removes per-step CPU/GPU transfers from the hot path.", 28, C.muted),
            ]),
          ),
        ]),
      ],
    ),
  ]);
}

function addInterpretation(presentation) {
  const slide = presentation.slides.add();
  rootSlide(slide, "Why current wall time is not the final GPU story", "The current implementation proves the GPU path exists, but the theoretical speedup appears only when the remaining CPU-owned pieces are removed from the hot path.", [
    grid(
      { width: fill, height: fill, columns: [fr(0.95), fr(1.05)], columnGap: 44 },
      [
        column({ width: fill, height: fill, gap: 22 }, [
          t("Current bottleneck", 36, C.ink, { bold: true }),
          panel(
            { width: fill, height: fixed(360), fill: C.panel, borderRadius: 22, padding: { x: 30, y: 30 } },
            column({ width: fill, height: fill, gap: 18, justify: "center" }, [
              t("CPU still prepares the GPU work", 38, C.coral, { bold: true }),
              t("SOFA traversal, triangle extraction, host packing, per-step H2D upload, synchronization, and logging still sit around the kernels.", 29, C.muted),
            ]),
          ),
          t("So the measured GPU wall time is a transitional integration cost, not the theoretical end-state cost.", 27, C.muted),
        ]),
        column({ width: fill, height: fill, gap: 22 }, [
          panel(
            { width: fill, height: fixed(226), fill: C.tealSoft, borderRadius: 22, padding: { x: 30, y: 24 } },
            column({ width: fill, height: fill, gap: 14 }, [
              t("What is already good", 34, C.teal, { bold: true }),
              t("Object broad phase removed. Grid filtering, dedup, exact contacts, and contact residency are active.", 26, C.ink),
              t("The plugin now exposes the right GPU-side stages to optimize.", 25, C.ink),
            ]),
          ),
          panel(
            { width: fill, height: fixed(286), fill: C.amberSoft, borderRadius: 22, padding: { x: 30, y: 24 } },
            column({ width: fill, height: fill, gap: 14 }, [
              t("What blocks speedup", 34, C.amber, { bold: true }),
              t("1. Per-step triangle upload scales with geometry.", 25, C.ink),
              t("2. Dense grid launches over many empty cells.", 25, C.ink),
              t("3. Sort/unique grows with duplicate candidates.", 25, C.ink),
              t("4. GTX 1650 Ti launch/sync overhead is visible.", 25, C.ink),
            ]),
          ),
          panel(
            { width: fill, height: fixed(176), fill: C.cyanSoft, borderRadius: 22, padding: { x: 30, y: 22 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("Next implementation step", 32, C.cyan, { bold: true }),
              t("Keep geometry resident, build active-cell lists, use CUB compaction, then connect GPU response so contacts never return to CPU.", 25, C.ink),
            ]),
          ),
        ]),
      ],
    ),
  ]);
}

function addPluginProgressSlide(presentation) {
  const slide = presentation.slides.add();
  rootSlide(
    slide,
    "Why this is already a real SOFA plugin",
    "This is not just a standalone CUDA demo. It is integrated into SOFA's collision pipeline with configurable components, CUDA backend execution, fallback controls, and benchmark instrumentation.",
    [
      grid(
        { width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 24, rowGap: 24 },
        [
          panel(
            { width: fill, height: fill, fill: C.tealSoft, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("SOFA components", 34, C.ink, { bold: true }),
              mono("GpuCollisionBroadPhase", 23, C.ink),
              mono("GpuCollisionNarrowPhase", 23, C.ink),
              mono("GpuPipelineBenchmarkController", 23, C.ink),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.cyanSoft, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("Pipeline integration", 34, C.ink, { bold: true }),
              t("The plugin plugs into SOFA CollisionPipeline and replaces the broad/narrow phase path for the tissue/blade benchmark scenes.", 27, C.muted),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.violetSoft, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("CUDA backend", 34, C.ink, { bold: true }),
              t("Dense grid kernels, device candidate buffers, Thrust sort/unique, and exact triangle contact generation are implemented in the backend.", 27, C.muted),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.amberSoft, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("Configurable runtime", 34, C.ink, { bold: true }),
              t("Grid bounds, resolution, capacities, deduplication, host-copy behavior, fallback, and logging are exposed as plugin settings.", 27, C.muted),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.coralSoft, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("Benchmark proof", 34, C.ink, { bold: true }),
              t("CPU reference scenes, GPU dense-grid scenes, transfer metrics, stage timing, scaling runs, and report outputs are already built.", 27, C.muted),
            ]),
          ),
          panel(
            { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
            column({ width: fill, height: fill, gap: 12, justify: "center" }, [
              t("Strong progress", 34, C.teal, { bold: true }),
              t("The important architecture shift is done: primitive candidate generation and exact contact work are now GPU-side, with CPU transfers measured instead of hidden.", 27, C.muted),
            ]),
          ),
        ],
      ),
    ],
  );
}

function addCollisionResponseNextStep(presentation) {
  const slide = presentation.slides.add();
  rootSlide(
    slide,
    "Next step: GPU collision response integration",
    "To make the pipeline fully integrated, the next milestone is to consume the GPU contact buffer directly in collision response instead of stopping at detection-only output.",
    [
      column({ width: fill, height: fill, gap: 30 }, [
        row({ width: fill, height: fixed(170), gap: 16, align: "center" }, [
          flowBox("GPU contacts", "DeviceExactContact buffer from narrow phase", C.tealSoft, C.ink, 146),
          arrow(),
          flowBox("Response metadata", "normal, gap, feature ids, barycentric data, material ids", C.cyanSoft, C.ink, 146),
          arrow(),
          flowBox("Contact forces", "penalty/compliant response first; constraints later", C.amberSoft, C.ink, 146),
          arrow(),
          flowBox("Tissue state update", "apply forces/constraints without contact D2H round-trip", C.violetSoft, C.ink, 146),
        ]),
        grid(
          { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
          [
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 14, justify: "center" }, [
                t("MVP response path", 34, C.ink, { bold: true }),
                t("Start with compliant or penalty response on GPU. It is the fastest way to validate tool reaction force and penetration control.", 27, C.muted),
              ]),
            ),
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 14, justify: "center" }, [
                t("Full integration target", 34, C.ink, { bold: true }),
                t("Contacts, response forces, solver vectors, and tissue state should remain resident in VRAM during the steady-state simulation step.", 27, C.muted),
              ]),
            ),
            panel(
              { width: fill, height: fill, fill: C.panel, borderRadius: 22, padding: { x: 30, y: 26 } },
              column({ width: fill, height: fill, gap: 14, justify: "center" }, [
                t("Exit criteria", 34, C.ink, { bold: true }),
                t("No per-step contact-copy to CPU, stable contact forces, bounded penetration, and benchmark logs showing response time as a measured GPU stage.", 27, C.muted),
              ]),
            ),
          ],
        ),
        panel(
          { width: fill, height: fixed(118), fill: C.tealSoft, borderRadius: 22, padding: { x: 32, y: 24 } },
          row({ width: fill, height: fill, gap: 28, align: "center" }, [
            t("Final goal", 36, C.ink, { bold: true, width: fixed(210) }),
            t("Detection and response become one GPU-resident surgical collision pipeline: tissue/tool state -> contacts -> response -> solver update.", 30, C.muted),
          ]),
        ),
      ]),
    ],
  );
}

async function saveSlidePng(slide, filePath) {
  const png = await slide.export({ format: "png" });
  await fs.writeFile(filePath, Buffer.from(await png.arrayBuffer()));
}

async function main() {
  await fs.mkdir(reportsDir, { recursive: true });
  const data = parseCsv(await fs.readFile(csvPath, "utf8"));

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  addCover(presentation, data);
  addDefaultSofaFlow(presentation);
  addCpuLimitationSlide(presentation);
  addExecutionFlow(presentation);
  addGridLogic(presentation);
  addKernelPlan(presentation);
  addMemoryTransfers(presentation, data);
  addBenchmarkResults(presentation);
  addInterpretation(presentation);
  addPluginProgressSlide(presentation);
  addCollisionResponseNextStep(presentation);

  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(outPptx);

  const previewPaths = [];
  for (let i = 0; i < presentation.slides.items.length; i += 1) {
    const file = `${previewPrefix}${String(i + 1).padStart(2, "0")}.png`;
    await saveSlidePng(presentation.slides.items[i], file);
    previewPaths.push(file);
  }

  const imported = await PresentationFile.importPptx(await FileBlob.load(outPptx));
  const pptxPreviewPaths = [];
  for (let i = 0; i < imported.slides.items.length; i += 1) {
    const file = `${pptxPreviewPrefix}${String(i + 1).padStart(2, "0")}.png`;
    await saveSlidePng(imported.slides.items[i], file);
    pptxPreviewPaths.push(file);
  }

  const manifest = {
    pptx: outPptx,
    sourceCsv: csvPath,
    previews: previewPaths,
    pptxPreviews: pptxPreviewPaths,
    slideCount: presentation.slides.items.length,
    generatedAt: new Date().toISOString(),
  };
  await fs.writeFile(path.join(reportsDir, `${deckStem}_manifest.json`), JSON.stringify(manifest, null, 2));
  console.log(JSON.stringify(manifest, null, 2));
}

await main();
