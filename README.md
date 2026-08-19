# How much VRAM

A GPU sizing calculator for LLM inference on vLLM. Paste a Hugging Face repo, set your
workload, and see whether the model fits — broken down into weights, KV cache and overhead,
checked against the memory the serving engine is actually allowed to use.

Built with React and Vite. No backend, no build-time data, no tracking.

## Why

The rule of thumb everyone quotes — roughly 2 bytes per parameter at 16-bit precision — only
covers the weights. In production the KV cache is often the larger number, and it scales with
context length _and_ concurrency. A model that loads fine will still fall over under load if
the cache pool runs dry.

This tool computes all three components and solves the equation backwards too: how many
concurrent requests fit at your context length, and how much context fits at your concurrency.

## Running it

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # static output in dist/
npm run preview  # serve the production build locally
```

## How the numbers are produced

```
weights      = (params − kept) × bytes_per_param + kept × kept_bytes
kv_per_layer = elems × replicas × kv_bytes          per token
kv_total     = kv_per_layer × tokens_cached × concurrency
overhead     = 0.59 GiB per device + batched_tokens × hidden × 13.35 × 2
usable       = gpu_capacity × gpu_count × gpu_memory_utilization
```

- **`kept`** is the share of the model quantization leaves alone — embeddings, output head,
  MoE routers, vision towers. It is 3% of Llama 3.3 70B but 12% of Kimi K2, so ignoring it
  understates a quantized MoE checkpoint badly. `kept_bytes` is its width: 2 nearly always,
  4 on the float32 releases.
- **`elems`** is `2 × kv_heads × head_dim` for standard attention — the 2 covers the key and
  the value tensor, and `kv_heads` means `num_key_value_heads`, not `num_attention_heads`.
  Latent-attention models cache one compressed vector instead, two orders of magnitude less.
- **`replicas`** is how many copies tensor parallelism ends up holding. vLLM shards the KV
  heads across the ranks while there are enough to go round and replicates below that, so a
  2-KV-head model on 8 GPUs keeps 4 copies and a single-head latent cache keeps 8.
- **`tokens_cached`** counts only the layers that hold a KV cache — 4 of Granite 4.0 H's 40,
  the rest being Mamba — and stops charging windowed layers once the context passes their
  window.
- **Overhead** is a line fitted through two measurements on a live cluster; activation memory
  tracks tokens in flight and model width, not weight count. `CALIBRATION.md` shows the work.

Accelerator capacities are the values the driver reports, not the marketing number: an
"80GB" A100 has about 79.2 GiB.

## Checking it against Red Hat's published minimums

```bash
npm run validate              # summary, plus any row outside ±10%
npm run validate -- --verbose # all 61 rows
```

Every model in the preset list is one Red Hat validates, and 61 of those rows publish a
minimum vRAM figure and a list of supported GPU configurations.
`scripts/redhat-minimums.json` holds them; the script checks both the size and the shape —
that each published machine is one the app would let you pick, and that it holds the weights.
All 61 land within 10% of the published figure, 58 within 5%, with no shape failures.

## Hugging Face lookup

Nothing is fetched on page load. When you press **Load**, two requests go out:

| Request | Supplies |
| --- | --- |
| `/{repo}/resolve/main/config.json` | layers, KV heads, head dim, latent rank, layer mix, quantization, trained context |
| `/api/models/{repo}` | parameter count and the 16-bit share, from the `safetensors` metadata |

Mistral's own repos ship `params.json` instead of `config.json`; that layout is read too.

Each repo is read at most once per session and kept in an in-memory cache.

`config.json` is read defensively, because the fields are not always where you would expect:

- Multimodal repos nest the language model config under `text_config`
- `head_dim` is frequently absent and is derived from `hidden_size ÷ num_attention_heads`
- A missing `num_key_value_heads` means plain MHA, so it falls back to `num_attention_heads`
- `quantization_config` sets the weight precision automatically when present, reading the
  per-group `num_bits` that compressed-tensors nests rather than the top level
- `kv_lora_rank` + `qk_rope_head_dim` mark a latent-attention model and size its real cache
- The attention layers are counted from `layer_types`, Nemotron's `hybrid_override_pattern`
  or Qwen3 Next's `full_attention_interval`, whichever the config publishes
- A `sliding_window` is ignored when `use_sliding_window` is false, which is how Qwen2.5 and
  Phi-4 Mini declare a window they do not use

Every field stays editable, so a failed lookup never blocks you.

### When the formula does not apply

The app raises a note instead of silently returning a wrong number:

- **Mixture of Experts** — every expert weight occupies memory, not just the active ones
- **Latent attention (MLA)** — modelled, including the copy each GPU keeps of it
- **Hybrid stacks** — only the attention layers hold a KV cache; the Mamba or linear-attention
  state is small, fixed, and not counted here
- **Sliding-window attention** — modelled per layer, since most such models window only some

### Gated repos and CORS

The browser calls `huggingface.co` directly, which works for public repos. Two cases need a
proxy:

- The repo is gated or private and needs a token
- Your network blocks the cross-origin request

`vite.config.js` includes a `/hf` dev proxy for the second case — set `VITE_HF_BASE=/hf`.
For gated repos in production, put a small proxy in front that injects the token server-side,
and point `VITE_HF_BASE` at it. Never ship a Hugging Face token to the browser.

## Treat the output as an estimate

Verify against reality before you provision anything. vLLM prints `GPU KV cache size` at
startup once profiling finishes, which tells you exactly how many tokens of cache survived.
In a running cluster, `vllm:gpu_cache_usage_perc` and `vllm:num_preemptions_total` show
whether the sizing held under real traffic — preemptions mean the cache is under pressure.

The gap between the estimate and the observed value is a stable calibration factor for that
runtime and accelerator. Measure it once, apply it afterwards.

## Layout

```
scripts/
  redhat-minimums.json  the published vRAM table, transcribed from the PDF
  validate-redhat.mjs   checks every preset against it
src/
  lib/
    constants.js     accelerators, precisions, presets
    vram.js          the calculation, as a pure function
    huggingface.js   repo parsing, config extraction, fetch cache
    useTheme.js      light/dark, persisted, set before first paint
  components/
    ModelPanel.jsx   repo lookup and model fields
    MemoryMap.jsx    the allocation bar
    Fields.jsx       shared form controls
  App.jsx
  index.css          both themes as CSS custom properties
```

`vram.js` has no React and no I/O, so it can be lifted into a CLI or a notebook unchanged.

## License

MIT
