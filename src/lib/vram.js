import { GIB } from "./constants.js";

/**
 * Total VRAM = weights + KV cache + overhead, checked against the memory the
 * serving engine is actually allowed to touch (capacity x gpu-memory-utilization).
 *
 * Assumes standard multi-head or grouped-query attention. Latent attention
 * (MLA) and sliding-window models cache considerably less.
 */
export function computeVram({
  params,
  layers,
  kvHeads,
  headDim,
  weightBytes,
  kvBytes,
  context,
  concurrency,
  gpuGib,
  gpuCount,
  utilization,
  overheadPct,
}) {
  const weights = (params * 1e9 * weightBytes) / GIB;

  // 2 covers the key and the value tensor.
  const kvPerTokenBytes = 2 * layers * kvHeads * headDim * kvBytes;
  const kv = (kvPerTokenBytes * context * concurrency) / GIB;

  // A CUDA context costs roughly 0.8 GiB per device before anything is loaded.
  const cudaContext = 0.8 * gpuCount;
  const overhead = cudaContext + (overheadPct / 100) * (weights + kv);
  const total = weights + kv + overhead;

  const capacity = gpuGib * gpuCount;
  const usable = capacity * utilization;

  // Solve the same equation backwards for the two headroom questions.
  const f = 1 + overheadPct / 100;
  const kvBudget = (usable - cudaContext - weights * f) / f;
  const perSequence = (kvPerTokenBytes * context) / GIB;
  const maxConcurrency = Math.max(0, Math.floor(kvBudget / perSequence));
  const maxContext = Math.max(
    0,
    Math.floor(kvBudget / ((kvPerTokenBytes * concurrency) / GIB) / 256) * 256
  );

  return {
    weights,
    kv,
    overhead,
    total,
    capacity,
    usable,
    fits: total <= usable,
    slack: usable - total,
    kvPerTokenKiB: kvPerTokenBytes / 1024,
    kvPerRequestGib: perSequence,
    maxConcurrency,
    maxContext,
    weightsAloneTooBig: weights + cudaContext > usable,
  };
}

export function vllmFlags({ context, concurrency, utilization, gpuCount, kvBytes }) {
  return [
    `--max-model-len ${context}`,
    `--max-num-seqs ${concurrency}`,
    `--gpu-memory-utilization ${utilization}`,
    gpuCount > 1 ? `--tensor-parallel-size ${gpuCount}` : null,
    kvBytes === 1 ? "--kv-cache-dtype fp8" : null,
  ].filter(Boolean);
}
