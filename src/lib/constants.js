export const GIB = 1024 ** 3;

/** Bytes stored per attention element, before context and batch scaling. */
export const PRECISIONS = [
  { id: "fp16", name: "FP16 / BF16", bytes: 2 },
  { id: "fp8", name: "FP8 / INT8", bytes: 1 },
  { id: "int4", name: "INT4 / MXFP4", bytes: 0.5 },
];

export const KV_DTYPES = [
  { id: "auto", name: "Match model dtype", bytes: null },
  { id: "fp16", name: "FP16 / BF16", bytes: 2 },
  { id: "fp8", name: "FP8", bytes: 1 },
];

/**
 * Usable device memory in GiB, not the marketing number. A "80GB" A100
 * reports roughly 79.2 GiB to the driver.
 */
export const GPUS = [
  { id: "l4", name: "NVIDIA L4", gib: 22.3 },
  { id: "a10g", name: "NVIDIA A10G", gib: 22.5 },
  { id: "l40s", name: "NVIDIA L40S", gib: 44.3 },
  { id: "a100-40", name: "NVIDIA A100 40GB", gib: 39.4 },
  { id: "a100-80", name: "NVIDIA A100 80GB", gib: 79.2 },
  { id: "h100", name: "NVIDIA H100 80GB", gib: 79.6 },
  { id: "h200", name: "NVIDIA H200", gib: 139.8 },
  { id: "b200", name: "NVIDIA B200", gib: 178.0 },
  { id: "mi300x", name: "AMD MI300X", gib: 191.5 },
  { id: "gaudi3", name: "Intel Gaudi 3", gib: 126.0 },
];

/**
 * AWS EC2 GPU instances, as a shortcut that fills in Accelerator + Count.
 *
 * `count` is GPUs *inside one instance*. Running several instances does not
 * add up here — separate instances are separate machines with only a network
 * between them, which is not what tensor parallelism wants. To get more GPUs
 * you size up to a bigger instance, which is exactly what this list shows.
 *
 * Sizes that differ only in vCPU and host RAM are collapsed: g6.xlarge through
 * g6.16xlarge are all one L4, so they are one row here.
 *
 * Left out on purpose: the g6f fractional-GPU sizes (a slice of an L4, which
 * the Count field cannot express), and p6-b300 / p6e-gb200 / the UltraServers
 * (a GPU this calculator does not carry, and NVLink domains of 36-72 GPUs that
 * break the single-node assumption behind Count).
 */
export const AWS_INSTANCES = [
  {
    label: "G6 · NVIDIA L4",
    instances: [
      { id: "g6.xlarge-16xlarge", name: "g6.xlarge → g6.16xlarge", gpu: "l4", count: 1 },
      { id: "g6.12xlarge", name: "g6.12xlarge / g6.24xlarge", gpu: "l4", count: 4 },
      { id: "g6.48xlarge", name: "g6.48xlarge", gpu: "l4", count: 8 },
    ],
  },
  {
    label: "G6e · NVIDIA L40S",
    instances: [
      { id: "g6e.xlarge-16xlarge", name: "g6e.xlarge → g6e.16xlarge", gpu: "l40s", count: 1 },
      { id: "g6e.12xlarge", name: "g6e.12xlarge / g6e.24xlarge", gpu: "l40s", count: 4 },
      { id: "g6e.48xlarge", name: "g6e.48xlarge", gpu: "l40s", count: 8 },
    ],
  },
  {
    label: "G5 · NVIDIA A10G",
    instances: [
      { id: "g5.xlarge-16xlarge", name: "g5.xlarge → g5.16xlarge", gpu: "a10g", count: 1 },
      { id: "g5.12xlarge", name: "g5.12xlarge / g5.24xlarge", gpu: "a10g", count: 4 },
      { id: "g5.48xlarge", name: "g5.48xlarge", gpu: "a10g", count: 8 },
    ],
  },
  {
    label: "P4 · NVIDIA A100",
    instances: [
      { id: "p4d.24xlarge", name: "p4d.24xlarge", gpu: "a100-40", count: 8 },
      { id: "p4de.24xlarge", name: "p4de.24xlarge", gpu: "a100-80", count: 8 },
    ],
  },
  {
    label: "P5 · NVIDIA H100 / H200",
    instances: [
      { id: "p5.4xlarge", name: "p5.4xlarge", gpu: "h100", count: 1 },
      { id: "p5.48xlarge", name: "p5.48xlarge", gpu: "h100", count: 8 },
      { id: "p5e.48xlarge", name: "p5e.48xlarge / p5en.48xlarge", gpu: "h200", count: 8 },
    ],
  },
  {
    label: "P6 · NVIDIA B200",
    instances: [{ id: "p6-b200.48xlarge", name: "p6-b200.48xlarge", gpu: "b200", count: 8 }],
  },
];

/** Flat lookup, for resolving the <select> value back to an instance. */
export const AWS_INSTANCE_LIST = AWS_INSTANCES.flatMap((f) => f.instances);

/**
 * GPUs per machine worth offering. Multi-GPU nodes are built in powers of two —
 * 3, 5, 6 and 7 are not shapes anyone can rent or rack, and tensor parallelism
 * wants a power of two anyway.
 */
export const TP_SIZES = [1, 2, 4, 8];

/**
 * vLLM splits the KV heads evenly across the GPUs and refuses to start when the
 * division is not exact, so a count is only usable if it divides kvHeads.
 */
export const tpSizesFor = (kvHeads) => TP_SIZES.filter((n) => kvHeads % n === 0);

/**
 * Preset footnotes. These mirror the warnings huggingface.js raises when it
 * reads a config, so a preset says the same thing the Load button would.
 */
const MOE = "Mixture of Experts: every expert weight counts, not just the active ones.";

const MLA =
  "Latent attention (MLA): the real KV cache is far smaller than this estimate.";

/** Only some layers of a hybrid model hold a KV cache. */
const hybrid = (attn, total, rest) =>
  `Hybrid architecture: only ${attn} of the ${total} layers use attention — the rest are ${rest}. ` +
  `Set Layers to ${attn} for a realistic KV figure.`;

const NO_NVFP4 =
  "Red Hat ships this one in NVFP4 too, which is not offered here: the format " +
  "stores a scale factor per block of weights, so it costs nearer 4.5 bits per " +
  "parameter than 4, and the INT4 setting would land about 20% under the " +
  "published minimum.";

const swaWindow = (tokens, swa, total) =>
  `Sliding-window attention: ${swa} of the ${total} layers are capped at ${tokens.toLocaleString("en-US")} tokens, ` +
  `so past that context only the remaining ${total - swa} layers keep growing.`;

/**
 * The Red Hat AI 3 validated models, grouped by family.
 *
 * Every entry is one *architecture*, not one Hugging Face repo. Red Hat ships
 * most of these several times over (FP8, w4a16, w8a8, NVFP4), but a quantized
 * checkpoint has the same layers, KV heads and head dim as its baseline — only
 * the weight bytes change, and that is the "Weight precision" control. So the
 * FP8 and INT4 variants collapse into the row they were quantized from.
 *
 * Numbers come from each model's published config.json, read the same way
 * huggingface.js reads it: `params` is the safetensors total in billions,
 * `headDim` is `head_dim` or hidden_size / num_attention_heads.
 *
 * `keptB` is how much of the model quantization leaves at 16 bits, in billions
 * — taken from the dtype breakdown the Hugging Face API publishes for Red Hat's
 * own quantized checkpoint, so it is measured, not derived. It is nowhere near
 * just the embedding table on MoE and multimodal models: 6% of Llama 4 Scout
 * (its vision tower), 12% of Kimi K2. Two entries could not be measured because
 * their repos are gated (TinyLlama, Mistral Large 3) and keep the embedding
 * estimate, which is a floor.
 *
 * `hidden` is hidden_size, which sets how much activation memory a token in
 * flight costs.
 *
 * Enabled-but-not-validated models are left out, as are the embedding,
 * reranker, speech and geospatial entries — none of them are decoder LLMs this
 * calculator can model.
 *
 * Also left out: models Red Hat ships only as NVFP4 (Qwen3 Coder Next, Qwen3
 * VL 235B). NVFP4 carries a scale factor per block of weights, so it costs
 * closer to 4.5 bits per parameter than the 4 the INT4 setting assumes, and
 * the estimate lands 15-20% under the published minimum with no other
 * precision to fall back on. Models that merely *offer* an NVFP4 variant stay,
 * with `precisions` narrowed to the ones they estimate accurately at.
 *
 * `precisions` lists the weight precisions a preset can be trusted at; the
 * rest are disabled while it is selected. Absent means all of them. It only
 * ever constrains presets — a repo loaded by hand is never restricted.
 */
export const PRESET_GROUPS = [
  {
    label: "Granite",
    models: [
      { id: "granite-31-8b", name: "Granite 3.1 8B Instruct", params: 8.17, layers: 40, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 0.4 },
      { id: "granite-32-2b", name: "Granite 3.2 2B Instruct", params: 2.53, layers: 40, kvHeads: 8, headDim: 64, hidden: 2048, keptB: 0.2 },
      { id: "granite-33-8b", name: "Granite 3.3 8B Instruct", params: 8.17, layers: 40, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 0.4 },
      {
        id: "granite-40-h-tiny",
        name: "Granite 4.0 H Tiny",
        params: 6.94,
        layers: 40,
        kvHeads: 4,
        headDim: 128, hidden: 1536, keptB: 0.17,
        notes: [MOE, hybrid(4, 40, "Mamba")],
      },
      {
        id: "granite-40-h-small",
        name: "Granite 4.0 H Small",
        params: 32.21,
        layers: 40,
        kvHeads: 8,
        headDim: 128, hidden: 4096, keptB: 0.44,
        notes: [MOE, hybrid(4, 40, "Mamba")],
      },
      { id: "granite-41-8b", name: "Granite 4.1 8B", params: 8.79, layers: 40, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 0.44 },
    ],
  },
  {
    label: "Llama",
    models: [
      { id: "tinyllama-11b", name: "TinyLlama 1.1B Chat", params: 1.1, layers: 22, kvHeads: 4, headDim: 64, hidden: 2048, keptB: 0.13 },
      { id: "llama-32-1b", name: "Llama 3.2 1B Instruct", params: 1.24, layers: 16, kvHeads: 8, headDim: 64, hidden: 2048, keptB: 0.53 },
      { id: "llama-31-8b", name: "Llama 3.1 8B Instruct", params: 8.03, layers: 32, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 1.05 },
      { id: "llama-33-70b", name: "Llama 3.3 70B Instruct", params: 70.55, layers: 80, kvHeads: 8, headDim: 128, hidden: 8192, keptB: 2.11 },
      {
        id: "llama-31-nemotron-70b",
        name: "Llama 3.1 Nemotron 70B Instruct",
        params: 70.55,
        layers: 80,
        kvHeads: 8,
        headDim: 128, hidden: 8192, keptB: 2.11,
      },
      {
        id: "llama-4-scout-17b",
        name: "Llama 4 Scout 17B 16E",
        params: 108.64,
        layers: 48,
        kvHeads: 8,
        headDim: 128, hidden: 5120, keptB: 5.98,
        notes: [MOE],
      },
      {
        id: "llama-4-maverick-17b",
        name: "Llama 4 Maverick 17B 128E",
        params: 401.58,
        layers: 48,
        kvHeads: 8,
        headDim: 128, hidden: 5120, keptB: 15.1,
        notes: [MOE],
      },
    ],
  },
  {
    label: "Mistral",
    models: [
      { id: "ministral-3-3b", name: "Ministral 3 3B Instruct 2512", params: 3.85, layers: 26, kvHeads: 8, headDim: 128, hidden: 3072, keptB: 0.82 },
      { id: "ministral-3-14b", name: "Ministral 3 14B Instruct 2512", params: 13.95, layers: 40, kvHeads: 8, headDim: 128, hidden: 5120, keptB: 3.88 },
      { id: "mistral-small-24b", name: "Mistral Small 24B Instruct 2501", params: 23.57, layers: 40, kvHeads: 8, headDim: 128, hidden: 5120, keptB: 1.35 },
      { id: "mistral-small-31-24b", name: "Mistral Small 3.1 24B Instruct 2503", params: 24.01, layers: 40, kvHeads: 8, headDim: 128, hidden: 5120, keptB: 1.78 },
      { id: "devstral-small-2-24b", name: "Devstral Small 2 24B Instruct 2512", params: 24.01, layers: 40, kvHeads: 8, headDim: 128, hidden: 5120, keptB: 1.35 },
      { id: "mixtral-8x7b", name: "Mixtral 8x7B Instruct v0.1", params: 46.7, layers: 32, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 0.26, notes: [MOE] },
      {
        id: "mistral-large-3-675b",
        name: "Mistral Large 3 675B Instruct 2512",
        params: 675,
        layers: 61,
        kvHeads: 128,
        headDim: 192, hidden: 7168, keptB: 1.88,
        precisions: ["fp16", "fp8"],
        notes: [MOE, MLA, NO_NVFP4],
      },
    ],
  },
  {
    label: "Phi",
    models: [
      { id: "phi-4-mini", name: "Phi-4 Mini Instruct", params: 3.84, layers: 32, kvHeads: 8, headDim: 128, hidden: 3072, keptB: 1.23 },
      { id: "phi-4", name: "Phi-4 14B", params: 14.66, layers: 40, kvHeads: 10, headDim: 128, hidden: 5120, keptB: 1.03 },
      { id: "phi-4-reasoning", name: "Phi-4 Reasoning", params: 14.66, layers: 40, kvHeads: 10, headDim: 128, hidden: 5120, keptB: 1.03 },
    ],
  },
  {
    label: "Qwen",
    models: [
      { id: "qwen25-7b", name: "Qwen2.5 7B Instruct", params: 7.62, layers: 28, kvHeads: 4, headDim: 128, hidden: 3584, keptB: 1.09 },
      { id: "qwen3-8b", name: "Qwen3 8B", params: 8.19, layers: 36, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 1.25 },
      {
        id: "qwen35-35b-a3b",
        name: "Qwen3.5 35B A3B",
        params: 35.95,
        layers: 40,
        kvHeads: 2,
        headDim: 256, hidden: 2048, keptB: 2.53,
        notes: [MOE, hybrid(10, 40, "linear attention")],
      },
      {
        id: "qwen3-next-80b-a3b",
        name: "Qwen3 Next 80B A3B Instruct",
        params: 81.32,
        layers: 48,
        kvHeads: 2,
        headDim: 256, hidden: 2048, keptB: 1.96,
        notes: [MOE, hybrid(12, 48, "gated DeltaNet")],
      },
      {
        id: "qwen35-122b-a10b",
        name: "Qwen3.5 122B A10B",
        params: 125.09,
        layers: 48,
        kvHeads: 2,
        headDim: 256, hidden: 3072, keptB: 5.27,
        notes: [MOE, hybrid(12, 48, "linear attention")],
      },
      {
        id: "qwen35-397b-a17b",
        name: "Qwen3.5 397B A17B",
        params: 403.4,
        layers: 60,
        kvHeads: 2,
        headDim: 256, hidden: 4096, keptB: 14.71,
        notes: [MOE, hybrid(15, 60, "linear attention")],
      },
      {
        id: "qwen3-coder-480b-a35b",
        name: "Qwen3 Coder 480B A35B Instruct",
        params: 480.15,
        layers: 62,
        kvHeads: 8,
        headDim: 128, hidden: 6144, keptB: 1.96,
        notes: [MOE],
      },
    ],
  },
  {
    label: "Nemotron",
    models: [
      {
        id: "nemotron-nano-9b-v2",
        name: "Nemotron Nano 9B v2",
        params: 8.89,
        layers: 56,
        kvHeads: 8,
        headDim: 128, hidden: 4480, keptB: 1.18,
        notes: [hybrid(4, 56, "Mamba")],
      },
      {
        id: "nemotron-3-nano-30b-a3b",
        name: "Nemotron 3 Nano 30B A3B",
        params: 31.58,
        layers: 52,
        kvHeads: 2,
        headDim: 128, hidden: 2688, keptB: 1.09,
        notes: [MOE, hybrid(6, 52, "Mamba")],
      },
      {
        id: "nemotron-3-super-120b-a12b",
        name: "Nemotron 3 Super 120B A12B",
        params: 123.61,
        layers: 88,
        kvHeads: 2,
        headDim: 128, hidden: 4096, keptB: 4.72,
        precisions: ["fp16", "fp8"],
        notes: [MOE, hybrid(8, 88, "Mamba"), NO_NVFP4],
      },
    ],
  },
  {
    label: "Other",
    models: [
      { id: "smollm3-3b", name: "SmolLM3 3B", params: 3.08, layers: 36, kvHeads: 4, headDim: 128, hidden: 2048, keptB: 0.26 },
      {
        id: "gemma-3n-e4b",
        name: "Gemma 3n E4B IT",
        params: 7.85,
        layers: 35,
        kvHeads: 2,
        headDim: 256, hidden: 2048, keptB: 3.96,
        notes: [swaWindow(512, 28, 35)],
      },
      { id: "apertus-8b", name: "Apertus 8B Instruct 2509", params: 8.05, layers: 32, kvHeads: 8, headDim: 128, hidden: 4096, keptB: 1.08 },
      {
        id: "gpt-oss-20b",
        name: "gpt-oss 20B",
        params: 21.51,
        layers: 24,
        kvHeads: 8,
        headDim: 64, hidden: 2880, keptB: 1.8,
        notes: [MOE, swaWindow(128, 12, 24)],
      },
      {
        id: "gpt-oss-120b",
        name: "gpt-oss 120B",
        params: 120.41,
        layers: 36,
        kvHeads: 8,
        headDim: 64, hidden: 2880, keptB: 2.17,
        notes: [MOE, swaWindow(128, 18, 36)],
      },
      {
        id: "minimax-m25",
        name: "MiniMax M2.5",
        params: 228.7,
        layers: 62,
        kvHeads: 8,
        headDim: 128, hidden: 3072, keptB: 1.29,
        notes: [MOE],
      },
      {
        id: "deepseek-r1-0528",
        name: "DeepSeek R1 0528",
        params: 684.53,
        layers: 61,
        kvHeads: 128,
        headDim: 56, hidden: 7168, keptB: 22.23,
        notes: [MOE, MLA],
      },
      {
        id: "kimi-k2",
        name: "Kimi K2 Instruct",
        params: 1026.41,
        layers: 61,
        kvHeads: 64,
        headDim: 112, hidden: 7168, keptB: 19.65,
        notes: [MOE, MLA],
      },
    ],
  },
];

/** Flat lookup, for resolving the <select> value back to a model. */
export const PRESETS = PRESET_GROUPS.flatMap((g) => g.models);

export const fmt = (n, d = 1) =>
  Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
