/**
 * Reads model architecture from a Hugging Face repo.
 *
 * Nothing here runs on page load. `fetchModel` is called only when the user
 * asks for a repo, and every repo is read at most once per session.
 */

// import.meta.env only exists under Vite; guard it so this module can also be
// imported by plain Node for tests.
const HF_BASE = import.meta.env?.VITE_HF_BASE || "https://huggingface.co";

const cache = new Map();

/** Accepts "owner/name", a full model URL, or a URL with /tree/main appended. */
export function parseRepoId(input) {
  let s = (input || "").trim();
  if (!s) return null;

  s = s
    .replace(/^https?:\/\//, "")
    .replace(/^(www\.)?huggingface\.co\//, "")
    .replace(/^models\//, "")
    .split(/[?#]/)[0]
    .replace(/\/+$/, "");

  const parts = s.split("/").filter(Boolean);
  const cut = parts.findIndex((p) => ["tree", "blob", "resolve", "raw"].includes(p));
  const clean = cut === -1 ? parts : parts.slice(0, cut);

  return clean.length >= 2 ? `${clean[0]}/${clean[1]}` : null;
}

/**
 * Mistral publishes its own repos in the `params.json` /
 * `consolidated.safetensors` layout rather than the transformers one, so there
 * is no config.json to read — Mistral Large 3 and the Ministral 3 line included.
 * The fields are the same numbers under different names.
 */
export function fromParamsJson(p) {
  return {
    num_hidden_layers: p.n_layers,
    num_attention_heads: p.n_heads,
    num_key_value_heads: p.n_kv_heads ?? p.n_heads,
    head_dim: p.head_dim,
    hidden_size: p.dim,
    vocab_size: p.vocab_size,
    max_position_embeddings: p.max_position_embeddings,
    tie_word_embeddings: p.tied_embeddings ?? false,
    sliding_window: p.sliding_window ?? null,
    kv_lora_rank: p.kv_lora_rank ?? null,
    qk_rope_head_dim: p.qk_rope_head_dim ?? null,
    num_experts: p.moe?.num_experts ?? null,
    quantization_config: p.quantization_config,
    vision_config: p.vision_encoder ?? null,
  };
}

/**
 * How many of the layers hold a KV cache, and how many of those are windowed.
 *
 * Hybrid stacks put attention on a minority of their layers and something with
 * a small fixed state — Mamba, gated DeltaNet, linear attention — on the rest.
 * Charging every layer for a KV cache overstates such a model several times
 * over, so the layer map is read wherever the config publishes one.
 */
export function readLayerMix(root) {
  const total = root.num_hidden_layers ?? root.n_layer ?? root.num_layers;
  const types = root.layer_types || root.attn_type_list;
  const pattern = root.hybrid_override_pattern;

  // vLLM ignores the window unless the model asks for it (Qwen2.5 and Phi-4
  // Mini both carry a nominal `sliding_window` the size of their full context).
  const windowOff = root.use_sliding_window === false;
  const window = windowOff ? null : root.sliding_window || null;

  if (Array.isArray(types) && types.length) {
    const attn = types.filter((t) => t !== "mamba" && t !== "linear_attention").length;
    const swa = window ? types.filter((t) => t === "sliding_attention").length : 0;
    return { kvLayers: attn, swaLayers: swa, swaWindow: swa ? window : 0 };
  }
  // Nemotron writes the same thing as a string: * is attention, M is Mamba.
  if (typeof pattern === "string" && pattern.length)
    return { kvLayers: pattern.split("*").length - 1, swaLayers: 0, swaWindow: 0 };
  // Qwen3 Next says it with a stride instead of a map.
  if (root.full_attention_interval && total)
    return {
      kvLayers: Math.floor(total / root.full_attention_interval),
      swaLayers: 0,
      swaWindow: 0,
    };

  // Every layer attends, and a window applies to all of them or to none.
  return { kvLayers: total, swaLayers: window ? total : 0, swaWindow: window || 0 };
}

/**
 * Pulls the fields the KV cache formula needs out of a config.json, working
 * around the things that are routinely missing, nested or written three
 * different ways.
 */
export function readConfig(cfg, apiData, repoId) {
  // Multimodal repos nest the language model config one level down.
  const root = cfg.text_config || cfg.llm_config || cfg;

  const layers = root.num_hidden_layers ?? root.n_layer ?? root.num_layers;
  const attnHeads = root.num_attention_heads ?? root.n_head;
  // No num_key_value_heads means plain MHA: every query head keeps its own KV.
  const kvHeads = root.num_key_value_heads ?? attnHeads;
  // head_dim is frequently absent and has to be derived.
  const headDim =
    root.head_dim ?? (root.hidden_size && attnHeads ? root.hidden_size / attnHeads : null);

  // Latent attention caches one compressed vector per layer per token — the
  // rank of the down-projection plus the rotary part — rather than a key and a
  // value tensor for every head. Two orders of magnitude apart on DeepSeek.
  const kvLatent = root.kv_lora_rank
    ? root.kv_lora_rank + (root.qk_rope_head_dim ?? 0)
    : null;
  const { kvLayers, swaLayers, swaWindow } = readLayerMix(root);

  if (!layers || !kvHeads || !headDim) {
    throw new Error(
      "config.json is missing the attention fields this calculator needs. Enter them by hand."
    );
  }

  // config.json never carries a parameter count; the API does.
  let params = null;
  const st = apiData?.safetensors;
  if (st?.total) params = st.total / 1e9;
  else if (st?.parameters) params = Object.values(st.parameters).reduce((a, b) => a + b, 0) / 1e9;
  if (!params) {
    const m = repoId.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/);
    if (m) params = parseFloat(m[1]);
  }

  // How much of the model quantization left alone. The API breaks the tensor
  // count down by dtype, so on a quantized repo this is measured rather than
  // guessed — and guessing is not good enough: the 16-bit share is 3% on Llama
  // 3.3 70B but 6% on Llama 4 Scout (its vision tower) and 12% on Kimi K2.
  const byDtype = Object.entries(st?.parameters || {});
  const isQuantized = byDtype.some(([d]) => /F8|F4|I32|U8|I8/i.test(d));
  const keptDtypes = byDtype.filter(([d]) => /BF16|^F16|^F32$/i.test(d));
  const measuredKept = keptDtypes.reduce((a, [, v]) => a + v, 0) / 1e9;
  // Not every checkpoint leaves that part at 16 bits: the Sarvam releases are
  // float32 throughout, which is 4 bytes a parameter and a 13% difference on
  // the total.
  const keptBytesMeasured = measuredKept
    ? keptDtypes.reduce((a, [d, v]) => a + v * (/F32/i.test(d) ? 4 : 2), 0) / (measuredKept * 1e9)
    : 2;
  // Fall back to the embedding table and output head, which is the floor: they
  // are the part no quantization scheme ever touches.
  const vocab = root.vocab_size ?? cfg.vocab_size;
  const tied = root.tie_word_embeddings ?? cfg.tie_word_embeddings ?? false;
  const keptParams =
    isQuantized && measuredKept
      ? measuredKept
      : vocab && root.hidden_size
        ? ((vocab * root.hidden_size) / 1e9) * (tied ? 1 : 2)
        : 0;

  let precision = "fp16";
  const q = cfg.quantization_config || root.quantization_config;
  if (q) {
    const method = String(q.quant_method || q.format || "").toLowerCase();
    // compressed-tensors — what every RedHatAI w4a16 and NVFP4 repo uses —
    // nests the width per tensor group instead of exposing it at the top
    // level. Missing it silently reads a 4-bit checkpoint as 8-bit, which
    // doubles the weight estimate.
    const grouped = Object.values(q.config_groups || {})
      .map((g) => g?.weights?.num_bits)
      .filter((n) => typeof n === "number");
    const bits = q.bits ?? q.w_bit ?? q.weight_bits ?? (grouped.length ? Math.min(...grouped) : null);
    if (method.includes("mxfp4") || method.includes("nvfp4") || bits === 4) precision = "int4";
    else if (method.includes("fp8") || method.includes("compressed") || bits === 8) precision = "fp8";
  }

  // Cases where the standard formula over- or under-counts. Say so rather than
  // quietly returning a wrong number.
  const notes = [];
  if (q) notes.push(`Quantized checkpoint detected (${q.quant_method || q.format || "unknown"}).`);
  if (root.num_local_experts || root.num_experts || root.n_routed_experts)
    notes.push("Mixture of Experts: every expert weight counts, not just the active ones.");
  if (kvLatent)
    notes.push(
      `Latent attention (MLA): each layer caches one ${kvLatent}-element latent vector per token instead of a key and a value tensor per head. vLLM cannot shard a single head, so every GPU keeps a copy — the KV figure multiplies by the count.`
    );
  if (kvLayers < layers)
    notes.push(
      `Hybrid architecture: only ${kvLayers} of the ${layers} layers use attention — the rest hold a small fixed recurrent state. Attention layers is set to ${kvLayers}.`
    );
  if (swaLayers)
    notes.push(
      `Sliding-window attention: ${swaLayers} of the ${kvLayers} attention layers stop growing at ${swaWindow.toLocaleString("en-US")} tokens. The KV figure accounts for it.`
    );
  if (!params) notes.push("Parameter count unavailable — set it manually.");

  return {
    layers,
    kvLayers,
    keptBytes: Math.round(keptBytesMeasured * 100) / 100,
    swaLayers,
    swaWindow,
    kvLatent,
    kvHeads,
    headDim: Math.round(headDim),
    hidden: root.hidden_size ?? Math.round(headDim) * (attnHeads || 1),
    keptParams: Math.round(keptParams * 100) / 100,
    params: params ? Math.round(params * 100) / 100 : null,
    precision,
    maxLen: root.max_position_embeddings ?? null,
    notes,
  };
}

/** Returns { model, cached }. Throws with a message meant for the UI. */
export async function fetchModel(repoId) {
  if (cache.has(repoId)) return { model: cache.get(repoId), cached: true };

  const [cfgRes, apiRes] = await Promise.all([
    fetch(`${HF_BASE}/${repoId}/resolve/main/config.json`),
    fetch(`${HF_BASE}/api/models/${repoId}`).catch(() => null),
  ]);

  if (cfgRes.status === 401 || cfgRes.status === 403)
    throw new Error("This repo is gated or private. Public repos only.");
  if (!cfgRes.ok && cfgRes.status !== 404)
    throw new Error(`Hugging Face returned ${cfgRes.status}.`);

  let cfg;
  if (cfgRes.ok) {
    cfg = await cfgRes.json();
  } else {
    // Mistral's own repos carry params.json instead.
    const alt = await fetch(`${HF_BASE}/${repoId}/resolve/main/params.json`);
    if (!alt.ok)
      throw new Error(
        "No config.json or params.json at that path. Check the repo id, or the model may be GGUF-only."
      );
    cfg = fromParamsJson(await alt.json());
  }
  const apiData = apiRes && apiRes.ok ? await apiRes.json() : null;
  const model = readConfig(cfg, apiData, repoId);

  cache.set(repoId, model);
  return { model, cached: false };
}
