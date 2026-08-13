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
weights      = params × bytes_per_param
kv_per_token = 2 × layers × kv_heads × head_dim × kv_bytes
kv_total     = kv_per_token × context × concurrency
overhead     = 0.8 GiB per device + activation_pct × (weights + kv)
usable        = gpu_capacity × gpu_memory_utilization
```

The factor of 2 in the KV term covers the key and the value tensor. `kv_heads` means
`num_key_value_heads`, not `num_attention_heads` — grouped-query attention models cache far
less than the query head count suggests, and mixing these up inflates the estimate several
times over.

Accelerator capacities are the values the driver reports, not the marketing number: an
"80GB" A100 has about 79.2 GiB.

## Hugging Face lookup

Nothing is fetched on page load. When you press **Load**, two requests go out:

| Request | Supplies |
| --- | --- |
| `/{repo}/resolve/main/config.json` | layers, KV heads, head dim, quantization, trained context |
| `/api/models/{repo}` | parameter count, from the `safetensors` metadata |

Each repo is read at most once per session and kept in an in-memory cache.

`config.json` is read defensively, because the fields are not always where you would expect:

- Multimodal repos nest the language model config under `text_config`
- `head_dim` is frequently absent and is derived from `hidden_size ÷ num_attention_heads`
- A missing `num_key_value_heads` means plain MHA, so it falls back to `num_attention_heads`
- `quantization_config` sets the weight precision automatically when present

Every field stays editable, so a failed lookup never blocks you.

### When the formula does not apply

The app raises a note instead of silently returning a wrong number:

- **Mixture of Experts** — every expert weight occupies memory, not just the active ones
- **Latent attention (MLA)**, as in DeepSeek V2/V3 — the compressed cache is much smaller
- **Sliding-window attention** — the cache is capped by the window, not by max context

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
