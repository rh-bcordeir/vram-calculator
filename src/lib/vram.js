import { GIB } from "./constants.js";

/**
 * Total VRAM = weights + KV cache + overhead, checked against the memory the
 * serving engine is actually allowed to touch (capacity x gpu-memory-utilization).
 *
 * The KV term covers four shapes vLLM treats differently:
 *
 *   - grouped-query attention, the default: 2 tensors x kvHeads x headDim per
 *     layer per token, sharded across the GPUs;
 *   - latent attention (MLA), where a single compressed vector of `kvLatent`
 *     elements stands in for both tensors and every rank keeps its own copy;
 *   - hybrid stacks (Mamba, gated DeltaNet, linear attention), where only
 *     `kvLayers` of the `layers` hold a KV cache at all;
 *   - sliding-window layers, whose cache stops growing at `swaWindow` tokens.
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

/**
 * How many copies of the KV cache a tensor-parallel deployment ends up holding.
 *
 * vLLM shards the KV heads across the ranks while there are enough to go round.
 * Below that it replicates: eight GPUs serving a model with two KV heads keep
 * four copies of the cache, not a quarter each. MLA is the extreme case — the
 * latent vector is a single head, so every rank carries the whole thing, which
 * is the reason large MLA deployments reach for data-parallel attention.
 */
export function kvReplicas({ kvHeads, kvLatent, gpuCount }) {
  const shards = kvLatent ? 1 : kvHeads;
  return gpuCount > shards ? gpuCount / shards : 1;
}

export function computeVram({
  params,
  layers,
  kvLayers = layers,
  kvHeads,
  headDim,
  kvLatent = null,
  swaLayers = 0,
  swaWindow = 0,
  hidden,
  keptParams = 0,
  keptBytes = 2,
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
  // head always stay at the checkpoint's own width — 16 bits usually, 32 on the
  // Sarvam releases, which is what `keptBytes` carries — and MoE routers, shared experts and vision
  // towers usually do too — 22 of DeepSeek R1's 684B parameters, for instance.
  // Counting all of it at the quantized width understates such a checkpoint by
  // well over 10%.
  const kept = Math.min(Math.max(keptParams, 0), params);
  const quantized = params - kept;
  const weights = ((quantized * weightBytes + kept * Math.max(weightBytes, keptBytes)) * 1e9) / GIB;

  // One layer, one token. Standard attention stores a key and a value tensor
  // for every KV head; MLA stores one compressed vector instead.
  const elems = kvLatent || 2 * kvHeads * headDim;
  const replicas = kvReplicas({ kvHeads, kvLatent, gpuCount });
  const perLayerTokenBytes = elems * kvBytes * replicas;

  // Windowed layers stop growing once the context passes the window; the rest
  // grow with it. `slope` is what one more token of context costs.
  const windowed = Math.min(swaLayers, kvLayers);
  const growing = Math.max(kvLayers - windowed, 0);
  const windowTokens = swaWindow ? Math.min(context, swaWindow) : context;
  const tokensCached = growing * context + windowed * windowTokens;
  const kv = (perLayerTokenBytes * tokensCached * concurrency) / GIB;

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
  const perSequence = (perLayerTokenBytes * tokensCached) / GIB;
  const maxConcurrency = Math.max(0, Math.floor(kvBudget / perSequence));

  // Beyond the window only the non-windowed layers keep charging for context,
  // so the ceiling is solved on whichever side of the window it lands.
  const perTokenGib = (perLayerTokenBytes * concurrency) / GIB;
  const maxContext = (() => {
    if (kvBudget <= 0) return 0;
    const flat = swaWindow ? (windowed * swaWindow * perTokenGib) : 0;
    const belowWindow = kvBudget / (perTokenGib * kvLayers);
    if (!swaWindow || belowWindow <= swaWindow) return floor256(belowWindow);
    return growing ? floor256((kvBudget - flat) / (perTokenGib * growing)) : Infinity;
  })();

  return {
    weights,
    kv,
    overhead,
    total,
    capacity,
    usable,
    fits: total <= usable,
    slack: usable - total,
    // An average over the sequence: windowed layers stop charging for context
    // past the window, so this drifts below the full per-layer cost.
    kvPerTokenKiB: (perLayerTokenBytes * tokensCached) / context / 1024,
    kvPerRequestGib: perSequence,
    kvReplicas: replicas,
    kvBudget,
    maxConcurrency,
    maxContext,
    // vLLM exits on startup if there is no room for a KV cache at all.
    weightsAloneTooBig: weights + overhead > usable,
  };
}

const floor256 = (n) => Math.max(0, Math.floor(n / 256) * 256);

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
