import React, { useMemo, useState } from "react";
import { GPUS, PRECISIONS, KV_DTYPES, fmt } from "./lib/constants.js";
import { computeVram, vllmFlags } from "./lib/vram.js";
import { useTheme } from "./lib/useTheme.js";
import ModelPanel from "./components/ModelPanel.jsx";
import MemoryMap from "./components/MemoryMap.jsx";
import { NumberField, SliderField, SelectField, TableRow } from "./components/Fields.jsx";
import { caption, card, code, hint, legend } from "./lib/ui.js";

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
    <div className="px-5 pt-6.5 pb-11">
      <header className="mx-auto mb-5.5 flex max-w-[70rem] flex-wrap items-start justify-between gap-5">
        <div>
          <p className={`${caption} mb-1.5 tracking-[0.16em]`}>GPU sizing · vLLM</p>
          <h1 className="m-0 font-display text-[clamp(2rem,5.5vw,3.25rem)] leading-[0.95] font-black tracking-[-0.03em]">
            How much VRAM<span className="text-bad">.</span>
          </h1>
          <p className="mt-2.5 max-w-[44ch] text-field text-muted">
            Weights, KV cache and overhead against the memory the server can actually use.
          </p>
        </div>
        <button
          type="button"
          className="flex cursor-pointer gap-0.5 rounded-ctl border border-line bg-sunken p-[3px]"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
        >
          {["light", "dark"].map((t) => (
            <span
              key={t}
              className={`rounded-[1px] px-3 py-1.5 font-mono text-label tracking-[0.08em] uppercase ${
                theme === t ? "bg-ink font-bold text-panel" : "text-muted"
              }`}
            >
              {t}
            </span>
          ))}
        </button>
      </header>

      <div className="mx-auto grid max-w-[70rem] grid-cols-1 items-start gap-5 min-[921px]:grid-cols-[minmax(0,1fr)_minmax(0,1.02fr)]">
        <section className="flex flex-col gap-4">
          <ModelPanel
            model={model}
            setModel={setModel}
            precision={precision}
            setPrecision={setPrecision}
            kvDtype={kvDtype}
            setKvDtype={setKvDtype}
          />

          <fieldset className={card}>
            <legend className={legend}>Workload</legend>
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
            <p className={hint}>
              Context × concurrency is the worst case: every sequence sitting at full length at the
              same moment.
            </p>
          </fieldset>

          <fieldset className={card}>
            <legend className={legend}>Hardware</legend>
            <div className="grid grid-cols-[2fr_1fr] gap-3">
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

        <section className="flex flex-col gap-4">
          <div
            className={`flex items-center gap-4.5 rounded-panel border border-line border-l-[5px] bg-panel px-5 py-4.5 ${
              result.fits ? "border-l-good" : "border-l-bad"
            }`}
          >
            <div className="flex items-baseline gap-1.25">
              <strong className="font-display text-hero leading-none font-black tracking-[-0.035em]">
                {fmt(result.total)}
              </strong>
              <span className="font-mono text-row text-muted">GiB</span>
            </div>
            <div>
              <p
                className={`mb-0.75 font-mono text-label font-bold tracking-[0.14em] uppercase ${
                  result.fits ? "text-good" : "text-bad"
                }`}
              >
                {result.fits ? "Fits" : "Does not fit"}
              </p>
              <p className="text-row leading-[1.5] text-muted">
                {result.fits
                  ? `${fmt(result.slack)} GiB left of the ${fmt(result.usable)} GiB usable on ${gpuCount}× ${gpu.name}.`
                  : `${fmt(-result.slack)} GiB short. ${gpuCount}× ${gpu.name} gives ${fmt(result.usable)} GiB usable.`}
              </p>
            </div>
          </div>

          <MemoryMap result={result} />

          <div className="rounded-panel border border-line bg-panel">
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
            <p className="border-l-[3px] border-l-bad bg-alarm-bg px-3.5 py-3 text-row leading-[1.55] text-alarm-ink">
              The weights alone exceed usable memory. Add GPUs, quantize the weights, or move to a
              larger accelerator — trimming context or concurrency will not help.
            </p>
          )}

          <div className="rounded-panel bg-code-bg px-4.5 pt-3.5 pb-4">
            <p className={`${caption} mb-2.25 tracking-[0.14em] text-code-label`}>
              Matching vLLM arguments
            </p>
            <pre className="m-0 font-mono text-note leading-[1.7] break-words whitespace-pre-wrap text-code-ink">
              {flags.join(" \\\n")}
            </pre>
          </div>

          <p className="text-note leading-[1.6] text-muted">
            A planning estimate. Confirm it against the <code className={code}>GPU KV cache size</code> vLLM prints at
            startup, or the <code className={code}>vllm:gpu_cache_usage_perc</code> metric. Latent attention (MLA) and
            sliding-window models cache far less than this formula assumes.
          </p>
        </section>
      </div>
    </div>
  );
}
