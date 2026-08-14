import React, { useMemo, useState } from "react";
import {
  GPUS,
  PRECISIONS,
  KV_DTYPES,
  AWS_INSTANCES,
  AWS_INSTANCE_LIST,
  TP_SIZES,
  tpSizesFor,
  fmt,
} from "./lib/constants.js";
import { computeVram, vllmFlags } from "./lib/vram.js";
import { useTheme } from "./lib/useTheme.js";
import Masthead from "./components/Masthead.jsx";
import Sidebar from "./components/Sidebar.jsx";
import PageHeader from "./components/PageHeader.jsx";
import Alert from "./components/Alert.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import MemoryMap from "./components/MemoryMap.jsx";
import { SliderField, SelectField, TableRow } from "./components/Fields.jsx";
import { caption, card, code, control, fieldLabel, fieldWrap, hint, legend } from "./lib/ui.js";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [navOpen, setNavOpen] = useState(false);

  const [model, setModel] = useState({
    params: 8.17,
    layers: 40,
    kvHeads: 8,
    headDim: 128,
    hidden: 4096,
    keptParams: 0.4,
  });
  const [precision, setPrecision] = useState("fp16");
  const [kvDtype, setKvDtype] = useState("auto");

  const [context, setContext] = useState(8192);
  const [concurrency, setConcurrency] = useState(8);

  const [gpuId, setGpuId] = useState("l40s");
  const [gpuCount, setGpuCount] = useState(1);
  const [instance, setInstance] = useState("");
  const [utilization, setUtilization] = useState(0.9);
  // vLLM V1 chunks prefill by default and caps a forward pass at this many
  // tokens, which is what bounds activation memory — not max-model-len.
  const [batchedTokens, setBatchedTokens] = useState(2048);

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
        batchedTokens,
        gpuGib: gpu.gib,
        gpuCount,
        utilization,
      }),
    [model, weightBytes, kvBytes, context, concurrency, gpu, gpuCount, utilization, batchedTokens]
  );

  const flags = vllmFlags({ context, concurrency, utilization, gpuCount, kvBytes, batchedTokens });

  /** A count that does not divide the KV heads is not a tight fit — vLLM exits
   *  on startup. Memory arithmetic says nothing about it, so it is checked
   *  separately and allowed to override the verdict. */
  const tpSizes = tpSizesFor(model.kvHeads);
  const tpValid = tpSizes.includes(gpuCount);

  /** Picking an instance is a shortcut: it fills the two fields below it, which
   *  stay editable. Editing either one clears the instance, since the pair no
   *  longer describes it. */
  function applyInstance(id) {
    setInstance(id);
    const i = AWS_INSTANCE_LIST.find((x) => x.id === id);
    if (!i) return;

    setGpuId(i.gpu);
    setGpuCount(i.count);
  }

  return (
    <div className="min-h-full">
      <Masthead
        theme={theme}
        toggleTheme={toggleTheme}
        onToggleNav={() => setNavOpen((o) => !o)}
      />

      <div
        className="grid items-start"
        style={{ gridTemplateColumns: navOpen ? "15rem minmax(0,1fr)" : "0 minmax(0,1fr)" }}
      >
        <Sidebar open={navOpen} />

        <main className="px-8 pt-8 pb-16">
        <div className="mx-auto max-w-[73.75rem]">
          <PageHeader
            title="VRAM calculator"
            description="Size weights, KV cache and overhead against the memory vLLM is actually allowed to use."
          />

          <div className="grid grid-cols-[repeat(auto-fit,minmax(27rem,1fr))] items-start gap-6">
            <section className="flex flex-col gap-6">
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
                  Context × concurrency is the worst case: every sequence sitting at full length at
                  the same moment.
                </p>
              </fieldset>

              <fieldset className={card}>
                <legend className={legend}>Hardware</legend>

                <label className={fieldWrap}>
                  <span className={`${fieldLabel} block`}>AWS EC2 instance</span>
                  <select
                    className={control}
                    value={instance}
                    onChange={(e) => applyInstance(e.target.value)}
                  >
                    <option value="">Choose an instance…</option>
                    {AWS_INSTANCES.map((f) => (
                      <optgroup key={f.label} label={f.label}>
                        {f.instances.map((i) => (
                          <option key={i.id} value={i.id}>
                            {i.name} · {i.count}× {GPUS.find((g) => g.id === i.gpu).name}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </label>

                <div className="grid grid-cols-[2fr_1fr] gap-3">
                  <SelectField
                    label="Accelerator"
                    value={gpuId}
                    onChange={(v) => {
                      setGpuId(v);
                      setInstance("");
                    }}
                    options={GPUS.map((g) => ({
                      id: g.id,
                      label: `${g.name} · ${fmt(g.gib, 0)} GiB`,
                    }))}
                  />
                  <label className={fieldWrap}>
                    <span className={`${fieldLabel} block`}>Count</span>
                    <select
                      className={control}
                      value={gpuCount}
                      onChange={(e) => {
                        setGpuCount(Number(e.target.value));
                        setInstance("");
                      }}
                    >
                      {TP_SIZES.map((n) => (
                        <option key={n} value={n} disabled={!tpSizes.includes(n)}>
                          {n}
                          {tpSizes.includes(n) ? "" : " —"}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                {!tpValid && (
                  <div className="mt-4">
                    <Alert state="danger" heading="vLLM will not start">
                      {gpuCount} does not divide the {model.kvHeads} KV heads this model has, so the
                      attention cannot be split evenly across the GPUs. Use {tpSizes.join(" or ")}{" "}
                      instead — the memory below is only reachable at those counts.
                    </Alert>
                  </div>
                )}

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
                  label="Max batched tokens"
                  value={batchedTokens}
                  onChange={setBatchedTokens}
                  min={512}
                  max={16384}
                  step={512}
                  suffix=" tokens"
                />
                <p className={hint}>
                  Count is GPUs in <b>one</b> machine — it becomes{" "}
                  <code className={code}>--tensor-parallel-size</code>. Tokens one forward pass may
                  process is the only thing activation memory scales with, model width aside; vLLM
                  chunks prefill by default and caps it at <b>2,048</b> however long the context is.
                </p>
              </fieldset>
            </section>

            <section className="flex flex-col gap-6">
              <div
                className={`flex items-center gap-5 rounded-panel border border-line border-l-5 bg-panel px-5 py-5 ${
                  result.fits && tpValid ? "border-l-good" : "border-l-bad"
                }`}
              >
                <div className="flex items-baseline gap-1.5">
                  <strong className="font-display text-hero leading-none font-bold tracking-[-0.02em]">
                    {fmt(result.total)}
                  </strong>
                  <span className="font-mono text-row text-muted">GiB</span>
                </div>
                <div>
                  <p
                    className={`mb-1 font-mono text-label font-bold tracking-[0.14em] uppercase ${
                      result.fits && tpValid ? "text-good" : "text-bad"
                    }`}
                  >
                    {!tpValid ? "Will not start" : result.fits ? "Fits" : "Does not fit"}
                  </p>
                  <p className="text-row leading-[1.5] text-muted">
                    {!tpValid
                      ? `Tensor parallelism cannot split ${model.kvHeads} KV heads across ${gpuCount} GPUs, whatever the memory says.`
                      : result.fits
                        ? `${fmt(result.slack)} GiB left of the ${fmt(result.usable)} GiB usable on ${gpuCount}× ${gpu.name}.`
                        : `${fmt(-result.slack)} GiB short. ${gpuCount}× ${gpu.name} gives ${fmt(result.usable)} GiB usable.`}
                  </p>
                </div>
              </div>

              <MemoryMap result={result} />

              <div className="rounded-panel border border-line bg-panel">
                <TableRow label="KV cache per token" value={`${fmt(result.kvPerTokenKiB, 0)} KiB`} />
                <TableRow
                  label="KV cache per request"
                  value={`${fmt(result.kvPerRequestGib, 2)} GiB`}
                />
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
                <Alert state="danger" heading="Weights alone exceed usable memory">
                  Add GPUs, quantize the weights, or move to a larger accelerator — trimming context
                  or concurrency will not help.
                </Alert>
              )}

              <div className="rounded-panel bg-code-bg px-5 pt-4 pb-4.5">
                <p className={`${caption} mb-2.5 tracking-[0.14em] text-code-label`}>
                  Matching vLLM arguments
                </p>
                <pre className="m-0 font-mono text-note leading-[1.7] break-words whitespace-pre-wrap text-code-ink">
                  {flags.join(" \\\n")}
                </pre>
              </div>

              <p className="text-note leading-[1.6] text-muted">
                A planning estimate. Confirm it against the{" "}
                <code className={code}>GPU KV cache size</code> vLLM prints at startup, or the{" "}
                <code className={code}>vllm:gpu_cache_usage_perc</code> metric. Latent attention
                (MLA) and sliding-window models cache far less than this formula assumes.
              </p>
            </section>
          </div>
          </div>
        </main>
      </div>
    </div>
  );
}
