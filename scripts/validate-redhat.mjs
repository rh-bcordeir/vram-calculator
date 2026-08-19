/**
 * Checks every preset against the vRAM minimums Red Hat publishes in
 * "Red Hat AI 3 — Validated models", Table 3.1.
 *
 *   node scripts/validate-redhat.mjs [--verbose]
 *
 * Two things are checked per published row:
 *
 *   size — the capacity the calculator says the model needs to start, against
 *          the "Min. vRAM (GB)" column. Red Hat's figure is device capacity;
 *          the calculator reports memory used, so it is divided by the default
 *          gpu-memory-utilization of 0.9 before comparing. A minimum is a
 *          starting figure, so this is weights + overhead with no workload on
 *          top — the KV pool is whatever is left.
 *
 *   shape — every GPU configuration in the "Supported GPUs" column has to be
 *           one the calculator would let a user pick (tensor parallelism has to
 *           divide the KV heads, or be divisible by them) and has to hold the
 *           weights. This is what catches a wrong KV-head count or a Count
 *           field that rules out a machine Red Hat ships on.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { PRESETS, PRECISIONS, GPUS, tpSizesFor, keptFor, GIB } from "../src/lib/constants.js";
import { computeVram } from "../src/lib/vram.js";

const here = dirname(fileURLToPath(import.meta.url));
const rows = JSON.parse(readFileSync(join(here, "redhat-minimums.json"), "utf8"));
const verbose = process.argv.includes("--verbose");

/** The default the app opens with, and what Red Hat's capacity figures assume. */
const UTILIZATION = 0.9;
/** vLLM V1 chunks prefill and caps a forward pass here however long the context. */
const BATCHED_TOKENS = 2048;
/** Anything past 10% of the published figure is treated as a disagreement. */
const TOLERANCE = 0.1;

/** "8XH200" -> { count: 8, gpu: <GPUS entry> } */
const GPU_BY_TAG = {
  L4: "l4", A10G: "a10g", L40S: "l40s", "A100-40": "a100-40", "A100-80": "a100-80",
  H100: "h100", H200: "h200", B200: "b200", MI300X: "mi300x",
};
function parseGpu(tag) {
  const m = /^(\d+)X(.+)$/.exec(tag);
  if (!m) return null;
  const id = GPU_BY_TAG[m[2].toUpperCase()] ?? GPU_BY_TAG[m[2]];
  const gpu = GPUS.find((g) => g.id === id);
  return gpu ? { count: Number(m[1]), gpu } : null;
}

const gib = (gb) => (gb * 1e9) / GIB;
const gb = (g) => (g * GIB) / 1e9;

const results = [];
for (const row of rows) {
  const preset = PRESETS.find((p) => p.id === row.preset);
  if (!preset) {
    results.push({ ...row, status: "MISSING", detail: "no preset with that id" });
    continue;
  }
  if (!preset.precisions.includes(row.precision)) {
    results.push({
      ...row,
      status: "BLOCKED",
      detail: `preset offers ${preset.precisions.join("/")} — Red Hat publishes a ${row.precision} checkpoint`,
    });
    continue;
  }

  const weightBytes = PRECISIONS.find((p) => p.id === row.precision).bytes;
  const configs = row.gpus.map(parseGpu).filter(Boolean);
  const smallest = configs
    .filter((c) => tpSizesFor(preset.kvHeads, preset.kvLatent).includes(c.count))
    .sort((a, b) => a.count * a.gpu.gib - b.count * b.gpu.gib)[0];

  const at = (count, gpuGib) =>
    computeVram({
      ...preset,
      kvLayers: preset.kvLayers ?? preset.layers,
      kvLatent: preset.kvLatent ?? null,
      swaLayers: preset.swaLayers ?? 0,
      swaWindow: preset.swaWindow ?? 0,
      keptParams: keptFor(preset, row.precision),
      weightBytes,
      kvBytes: Math.max(weightBytes, 1),
      context: 4096,
      concurrency: 1,
      batchedTokens: BATCHED_TOKENS,
      gpuGib,
      gpuCount: count,
      utilization: UTILIZATION,
    });

  // What the calculator says you have to buy: everything that is not KV cache,
  // divided by the share of the card vLLM is allowed to touch.
  const base = at(smallest?.count ?? 1, smallest?.gpu.gib ?? 80);
  const neededGB = gb((base.weights + base.overhead) / UTILIZATION);
  const ratio = neededGB / row.minVramGB;

  // Shape: every published machine has to be selectable and hold the weights.
  const shape = [];
  for (const tag of row.gpus) {
    const c = parseGpu(tag);
    if (!c) continue; // a GPU this calculator does not carry
    if (!tpSizesFor(preset.kvHeads, preset.kvLatent).includes(c.count))
      shape.push(`${tag}: Count ${c.count} is disabled for ${preset.kvHeads} KV heads`);
    else if (at(c.count, c.gpu.gib).weightsAloneTooBig)
      shape.push(`${tag}: weights alone do not fit`);
  }

  results.push({
    ...row,
    status: Math.abs(ratio - 1) <= TOLERANCE && !shape.length ? "OK" : "CHECK",
    ratio,
    neededGB,
    shape,
    where: smallest ? `${smallest.count}x${smallest.gpu.name}` : "n/a",
  });
}

const width = Math.max(...results.map((r) => r.repo.length));
for (const r of results) {
  if (!verbose && r.status === "OK") continue;
  const head = `${r.status.padEnd(7)} ${r.repo.padEnd(width)} ${r.precision.padEnd(4)}`;
  if (r.ratio === undefined) console.log(`${head} ${r.detail}`);
  else
    console.log(
      `${head} needs ${r.neededGB.toFixed(0).padStart(4)} GB vs ${String(r.minVramGB).padStart(4)} GB published` +
        `  ratio ${r.ratio.toFixed(3)}  on ${r.where}` +
        (r.shape.length ? `\n          ${r.shape.join("\n          ")}` : "")
    );
}

const sized = results.filter((r) => r.ratio !== undefined);
const mean = sized.reduce((a, r) => a + r.ratio, 0) / sized.length;
const sd = Math.sqrt(sized.reduce((a, r) => a + (r.ratio - mean) ** 2, 0) / sized.length);
const within = sized.filter((r) => Math.abs(r.ratio - 1) <= TOLERANCE).length;
const shapeFails = results.filter((r) => r.shape?.length).length;

console.log(
  `\n${sized.length} published minimums · mean ratio ${mean.toFixed(3)} · ` +
    `CoV ${((sd / mean) * 100).toFixed(1)}% · within ±10% ${within}/${sized.length} · ` +
    `GPU-shape failures ${shapeFails} · unmapped ${results.length - sized.length}`
);
process.exit(results.some((r) => r.status === "MISSING" || r.status === "BLOCKED") || within < sized.length || shapeFails ? 1 : 0);
