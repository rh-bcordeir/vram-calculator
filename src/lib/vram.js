import { GIB } from "./constants.js";

/**
 * Total VRAM = weights + KV cache + overhead, checked against the memory the
 * serving engine is actually allowed to touch (capacity x gpu-memory-utilization).
 *
 * Assumes standard multi-head or grouped-query attention. Latent attention
 * (MLA) and sliding-window models cache considerably less.
 */
/*
 * The two overhead constants below are a straight line fitted through two
 * measurements of Granite 3.3 8B (hidden 4,096) on an L4 under vLLM 0.10.1,
 * with a 21.37 GiB budget and 15.25 GiB of weights:
 *
 *   max-num-batched-tokens  2,048 -> 5.32 GiB KV pool -> 0.80 GiB overhead
 *   max-num-batched-tokens 16,384 -> 3.86 GiB KV pool -> 2.26 GiB overhead
 *
 * Note this is what vLLM *reserves*, which is what sets the KV pool size, not
 * what it ends up touching — nvidia-smi showed only 0.48 GiB in use at 2,048.
 *
 * Two points on one model, so the shape is trusted more than the coefficients:
 * activation is linear in tokens in flight and in model width, which is what
 * the forward pass actually does. Deriving the multiplier from geometry instead
 * gave 10, a third under what the second measurement showed.
 */

/** CUDA context, captured graphs, and the margin vLLM keeps, per device. */
const FIXED_PER_GPU = 0.59;

/** Live activation buffers, as a multiple of hidden size per token in flight. */
const ACT_PER_TOKEN = 13.35;

export function computeVram({
  params,
  layers,
  kvHeads,
  headDim,
  hidden,
  keptParams = 0,
  weightBytes,
  kvBytes,
  context,
  concurrency,
  batchedTokens,
  gpuGib,
  gpuCount,
  utilization,
}) {
  // Quantization never touches the whole model. The embedding table and output
  // head always stay at 16 bits, and MoE routers, shared experts and vision
  // towers usually do too — 22 of DeepSeek R1's 684B parameters, for instance.
  // Counting all of it at the quantized width understates such a checkpoint by
  // well over 10%.
  const kept = Math.min(Math.max(keptParams, 0), params);
  const quantized = params - kept;
  const weights = ((quantized * weightBytes + kept * Math.max(weightBytes, 2)) * 1e9) / GIB;

  // 2 covers the key and the value tensor.
  const kvPerTokenBytes = 2 * layers * kvHeads * headDim * kvBytes;
  const kv = (kvPerTokenBytes * context * concurrency) / GIB;

  // Activation memory tracks tokens in flight and model width — not weight
  // count. Tensor parallelism shards the intermediate tensors, so the total
  // stays flat as GPUs are added while the fixed cost repeats per device.
  const fixed = FIXED_PER_GPU * gpuCount;
  const activation = (batchedTokens * hidden * ACT_PER_TOKEN * 2) / GIB;
  const overhead = fixed + activation;
  const total = weights + kv + overhead;

  const capacity = gpuGib * gpuCount;
  const usable = capacity * utilization;

  // Everything except the KV cache is fixed, so the headroom questions are a
  // plain subtraction now rather than solving through a percentage.
  const kvBudget = usable - weights - overhead;
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
    kvBudget,
    maxConcurrency,
    maxContext,
    // vLLM exits on startup if there is no room for a KV cache at all.
    weightsAloneTooBig: weights + overhead > usable,
  };
}

export function vllmFlags({ context, concurrency, utilization, gpuCount, kvBytes, batchedTokens }) {
  return [
    `--max-model-len ${context}`,
    `--max-num-seqs ${concurrency}`,
    `--max-num-batched-tokens ${batchedTokens}`,
    `--gpu-memory-utilization ${utilization}`,
    gpuCount > 1 ? `--tensor-parallel-size ${gpuCount}` : null,
    kvBytes === 1 ? "--kv-cache-dtype fp8" : null,
  ].filter(Boolean);
}
