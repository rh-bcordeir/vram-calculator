import React, { useMemo, useState } from "react";
import { GPUS, PRECISIONS, KV_DTYPES, fmt } from "./lib/constants.js";
import { computeVram, vllmFlags } from "./lib/vram.js";
import { useTheme } from "./lib/useTheme.js";
import ModelPanel from "./components/ModelPanel.jsx";
import MemoryMap from "./components/MemoryMap.jsx";
import { NumberField, SliderField, SelectField, TableRow } from "./components/Fields.jsx";

export default function App() {
  const [theme, toggleTheme] = useTheme();

  const [model, setModel] = useState({ params: 8.17, layers: 40, kvHeads: 8, headDim: 128 });
  const [precision, setPrecision] = useState("fp16");
  const [kvDtype, setKvDtype] = useState("auto");

  const [context, setContext] = useState(8192);
  const [concurrency, setConcurrency] = useState(8);

  const [gpuId, setGpuId] = useState("l40s");
  const [gpuCount, setGpuCount] = useState(1);
  const [utilization, setUtilization] = useState(0.9);
  const [overheadPct, setOverheadPct] = useState(12);

  const gpu = GPUS.find((g) => g.id === gpuId);
  const weightBytes = PRECISIONS.find((p) => p.id === precision).bytes;
  const kvBytes =
    kvDtype === "auto"
      ? Math.max(weightBytes, 1) // the cache never goes below 8 bits
      : KV_DTYPES.find((k) => k.id === kvDtype).bytes;

  const result = useMemo(
    () =>
      computeVram({
        ...model,
        weightBytes,
        kvBytes,
        context,
        concurrency,
        gpuGib: gpu.gib,
        gpuCount,
        utilization,
        overheadPct,
      }),
    [model, weightBytes, kvBytes, context, concurrency, gpu, gpuCount, utilization, overheadPct]
  );

  const flags = vllmFlags({ context, concurrency, utilization, gpuCount, kvBytes });

  return (
    <div className="root">
      <header className="head">
        <div>
          <p className="eyebrow">GPU sizing · vLLM</p>
          <h1>
            How much VRAM<span className="dot">.</span>
          </h1>
          <p className="sub">
            Weights, KV cache and overhead against the memory the server can actually use.
          </p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
        >
          <span className={theme === "light" ? "on" : ""}>Light</span>
          <span className={theme === "dark" ? "on" : ""}>Dark</span>
        </button>
      </header>

      <div className="grid">
        <section className="col">
          <ModelPanel
            model={model}
            setModel={setModel}
            precision={precision}
            setPrecision={setPrecision}
            kvDtype={kvDtype}
            setKvDtype={setKvDtype}
          />

          <fieldset className="card">
            <legend>Workload</legend>
            <SliderField
              label="Max context"
              value={context}
              onChange={setContext}
              min={512}
              max={131072}
              step={512}
              suffix=" tokens"
            />
            <SliderField
              label="Concurrent requests"
              value={concurrency}
              onChange={setConcurrency}
              min={1}
              max={256}
              step={1}
            />
            <p className="hint">
              Context × concurrency is the worst case: every sequence sitting at full length at the
              same moment.
            </p>
          </fieldset>

          <fieldset className="card">
            <legend>Hardware</legend>
            <div className="row-2">
              <SelectField
                label="Accelerator"
                value={gpuId}
                onChange={setGpuId}
                options={GPUS.map((g) => ({ id: g.id, label: `${g.name} · ${fmt(g.gib, 0)} GiB` }))}
              />
              <NumberField label="Count" value={gpuCount} onChange={setGpuCount} int min={1} max={8} />
            </div>
            <SliderField
              label="gpu-memory-utilization"
              value={utilization}
              onChange={setUtilization}
              min={0.5}
              max={0.98}
              step={0.01}
              places={2}
            />
            <SliderField
              label="Activation overhead"
              value={overheadPct}
              onChange={setOverheadPct}
              min={5}
              max={30}
              step={1}
              suffix="%"
            />
          </fieldset>
        </section>

        <section className="col">
          <div className={result.fits ? "verdict ok" : "verdict bad"}>
            <div className="verdict-num">
              <strong>{fmt(result.total)}</strong>
              <span>GiB</span>
            </div>
            <div>
              <p className="verdict-title">{result.fits ? "Fits" : "Does not fit"}</p>
              <p className="verdict-body">
                {result.fits
                  ? `${fmt(result.slack)} GiB left of the ${fmt(result.usable)} GiB usable on ${gpuCount}× ${gpu.name}.`
                  : `${fmt(-result.slack)} GiB short. ${gpuCount}× ${gpu.name} gives ${fmt(result.usable)} GiB usable.`}
              </p>
            </div>
          </div>

          <MemoryMap result={result} />

          <div className="table">
            <TableRow label="KV cache per token" value={`${fmt(result.kvPerTokenKiB, 0)} KiB`} />
            <TableRow label="KV cache per request" value={`${fmt(result.kvPerRequestGib, 2)} GiB`} />
            <TableRow
              label="Max concurrency at this context"
              value={`${result.maxConcurrency}`}
              warn={result.maxConcurrency < concurrency}
            />
            <TableRow
              label="Max context at this concurrency"
              value={`${result.maxContext.toLocaleString("en-US")} tokens`}
              warn={result.maxContext < context}
            />
          </div>

          {result.weightsAloneTooBig && (
            <p className="alarm">
              The weights alone exceed usable memory. Add GPUs, quantize the weights, or move to a
              larger accelerator — trimming context or concurrency will not help.
            </p>
          )}

          <div className="flags">
            <p className="flags-title">Matching vLLM arguments</p>
            <pre>{flags.join(" \\\n")}</pre>
          </div>

          <p className="foot">
            A planning estimate. Confirm it against the <code>GPU KV cache size</code> vLLM prints at
            startup, or the <code>vllm:gpu_cache_usage_perc</code> metric. Latent attention (MLA) and
            sliding-window models cache far less than this formula assumes.
          </p>
        </section>
      </div>
    </div>
  );
}
