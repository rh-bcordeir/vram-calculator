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
 * Pulls the fields the KV cache formula needs out of a config.json, working
 * around the three things that are routinely missing or nested.
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

  let precision = "fp16";
  const q = cfg.quantization_config || root.quantization_config;
  if (q) {
    const method = String(q.quant_method || q.format || "").toLowerCase();
    const bits = q.bits ?? q.w_bit ?? q.weight_bits;
    if (method.includes("mxfp4") || bits === 4) precision = "int4";
    else if (method.includes("fp8") || method.includes("compressed") || bits === 8) precision = "fp8";
  }

  // Cases where the standard formula over- or under-counts. Say so rather than
  // quietly returning a wrong number.
  const notes = [];
  if (q) notes.push(`Quantized checkpoint detected (${q.quant_method || q.format || "unknown"}).`);
  if (root.num_local_experts || root.num_experts || root.n_routed_experts)
    notes.push("Mixture of Experts: every expert weight counts, not just the active ones.");
  if (root.kv_lora_rank || (cfg.architectures || []).some((a) => /Deepseek/i.test(a)))
    notes.push("Latent attention (MLA): the real KV cache is far smaller than this estimate.");
  if (root.sliding_window)
    notes.push(
      `Sliding-window attention (${root.sliding_window} tokens): KV cache is capped by the window, not by max context.`
    );
  if (!params) notes.push("Parameter count unavailable — set it manually.");

  return {
    layers,
    kvHeads,
    headDim: Math.round(headDim),
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
  if (cfgRes.status === 404)
    throw new Error("No config.json at that path. Check the repo id, or the model may be GGUF-only.");
  if (!cfgRes.ok) throw new Error(`Hugging Face returned ${cfgRes.status}.`);

  const cfg = await cfgRes.json();
  const apiData = apiRes && apiRes.ok ? await apiRes.json() : null;
  const model = readConfig(cfg, apiData, repoId);

  cache.set(repoId, model);
  return { model, cached: false };
}
