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

/** Values taken from each model's published config.json. */
export const PRESETS = [
  { id: "granite33-8b", name: "Granite 3.3 8B", params: 8.17, layers: 40, kvHeads: 8, headDim: 128 },
  { id: "granite33-2b", name: "Granite 3.3 2B", params: 2.53, layers: 40, kvHeads: 8, headDim: 128 },
  { id: "llama31-8b", name: "Llama 3.1 8B", params: 8.03, layers: 32, kvHeads: 8, headDim: 128 },
  { id: "llama33-70b", name: "Llama 3.3 70B", params: 70.6, layers: 80, kvHeads: 8, headDim: 128 },
  { id: "qwen3-8b", name: "Qwen3 8B", params: 8.19, layers: 36, kvHeads: 8, headDim: 128 },
  { id: "qwen3-32b", name: "Qwen3 32B", params: 32.8, layers: 64, kvHeads: 8, headDim: 128 },
  { id: "mistral-7b", name: "Mistral 7B v0.3", params: 7.25, layers: 32, kvHeads: 8, headDim: 128 },
  { id: "phi4-14b", name: "Phi-4 14B", params: 14.7, layers: 40, kvHeads: 10, headDim: 128 },
];

export const fmt = (n, d = 1) =>
  Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
