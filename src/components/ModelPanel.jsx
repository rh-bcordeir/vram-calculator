import React, { useState } from "react";
import { fetchModel, parseRepoId } from "../lib/huggingface.js";
import { PRECISIONS, KV_DTYPES, PRESETS, PRESET_GROUPS, keptFor } from "../lib/constants.js";
import { NumberField, SelectField } from "./Fields.jsx";
import { card, control, fieldWrap, fieldLabel, focusRing, hint, legend } from "../lib/ui.js";

/** Every field a preset fills in, for one weight precision. */
function modelFromPreset(p, precision) {
  return {
    params: p.params,
    layers: p.layers,
    kvLayers: p.kvLayers ?? p.layers,
    kvHeads: p.kvHeads,
    headDim: p.headDim,
    kvLatent: p.kvLatent ?? null,
    swaLayers: p.swaLayers ?? 0,
    swaWindow: p.swaWindow ?? 0,
    hidden: p.hidden,
    keptParams: keptFor(p, precision),
    keptBytes: p.keptBytes ?? 2,
  };
}

const STATUS_COLOR = {
  loading: "text-muted",
  ok: "text-good",
  error: "text-bad",
};

export default function ModelPanel({ model, setModel, precision, setPrecision, kvDtype, setKvDtype }) {
  const [url, setUrl] = useState("");
  const [preset, setPreset] = useState("");
  const [status, setStatus] = useState({ state: "idle" });
  const [source, setSource] = useState(null);
  const [notes, setNotes] = useState([]);

  async function load() {
    const repoId = parseRepoId(url);
    if (!repoId) {
      setStatus({
        state: "error",
        msg: "Enter a repo as owner/name, or paste a full huggingface.co model URL.",
      });
      return;
    }

    setStatus({ state: "loading", msg: `Reading ${repoId}…` });
    try {
      const { model: m, cached } = await fetchModel(repoId);
      setModel({
        params: m.params ?? model.params,
        layers: m.layers,
        kvLayers: m.kvLayers,
        kvHeads: m.kvHeads,
        headDim: m.headDim,
        kvLatent: m.kvLatent,
        swaLayers: m.swaLayers,
        swaWindow: m.swaWindow,
        hidden: m.hidden,
        keptParams: m.keptParams,
        keptBytes: m.keptBytes,
      });
      setPrecision(m.precision);
      setNotes(m.notes);
      // The fields no longer describe whatever preset was picked before.
      setPreset("");
      setSource({ label: repoId, cached, maxLen: m.maxLen });
      setStatus({ state: "ok", msg: cached ? `Loaded ${repoId} from session cache.` : `Loaded ${repoId}.` });
    } catch (e) {
      setStatus({
        state: "error",
        msg:
          e instanceof TypeError
            ? "Could not reach huggingface.co from this page. Enter the values by hand."
            : e.message,
      });
    }
  }

  function applyPreset(id) {
    setPreset(id);
    const p = PRESETS.find((m) => m.id === id);
    if (!p) return;

    // A preset that does not ship at the current precision switches to one it
    // does; leaving it would show the very number the restriction exists to
    // prevent.
    const next = p.precisions && !p.precisions.includes(precision) ? p.precisions[0] : precision;
    if (next !== precision) setPrecision(next);

    setModel(modelFromPreset(p, next));
    setNotes(p.notes ?? []);
    setSource({ label: p.name, cached: false });
    setStatus({ state: "idle" });
  }

  /** The 16-bit share is measured per checkpoint, so it moves with precision. */
  function choosePrecision(id) {
    setPrecision(id);
    const p = PRESETS.find((m) => m.id === preset);
    if (p) setModel(modelFromPreset(p, id));
  }

  /** Presets may narrow the precisions they can be trusted at. A repo loaded
   *  by hand carries no such restriction. */
  const allowed = PRESETS.find((m) => m.id === preset)?.precisions ?? null;
  const allows = (id) => !allowed || allowed.includes(id);

  const field = (key) => (v) => setModel({ ...model, [key]: v });

  return (
    <fieldset className={card}>
      <legend className={legend}>Model</legend>

      <label className={fieldWrap}>
        <span className={`${fieldLabel} block`}>Hugging Face repo</span>
        <div className="flex gap-1.75">
          <input
            type="text"
            className={control}
            placeholder="ibm-granite/granite-3.3-8b-instruct"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && load()}
            spellCheck={false}
          />
          <button
            type="button"
            className={`shrink-0 cursor-pointer rounded-ctl border border-ink bg-ink px-4.5 py-2 text-row font-bold text-panel disabled:cursor-progress disabled:opacity-55 ${focusRing}`}
            onClick={load}
            disabled={status.state === "loading"}
          >
            {status.state === "loading" ? "Reading…" : "Load"}
          </button>
        </div>
      </label>

      <label className={fieldWrap}>
        <span className={`${fieldLabel} block`}>Red Hat AI validated model</span>
        <select className={control} value={preset} onChange={(e) => applyPreset(e.target.value)}>
          <option value="">Choose a model…</option>
          {PRESET_GROUPS.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.models.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>

      {status.state !== "idle" && (
        <p className={`mt-2.25 font-mono text-note ${STATUS_COLOR[status.state]}`}>{status.msg}</p>
      )}

      {source && (
        <p className="mt-1.75 text-note text-muted">
          Fields below came from <b>{source.label}</b>
          {source.cached && " (cached)"}
          {source.maxLen && ` · trained context ${source.maxLen.toLocaleString("en-US")}`}
        </p>
      )}

      {notes.map((n, i) => (
        <p
          key={i}
          className="mt-2.25 border-l-[3px] border-l-note-line bg-note-bg px-2.75 py-2 text-note leading-[1.5] text-note-ink"
        >
          {n}
        </p>
      ))}

      <div className="grid grid-cols-2 gap-2.5 min-[521px]:grid-cols-4">
        <NumberField label="Params (B)" value={model.params} onChange={field("params")} step={0.01} />
        <NumberField label="Layers" value={model.layers} onChange={field("layers")} int min={1} />
        <NumberField
          label="Attention layers"
          value={model.kvLayers ?? model.layers}
          onChange={field("kvLayers")}
          int
          min={1}
        />
        <NumberField label="KV heads" value={model.kvHeads} onChange={field("kvHeads")} int min={1} />
        <NumberField label="Head dim" value={model.headDim} onChange={field("headDim")} int min={1} />
        <NumberField label="Hidden size" value={model.hidden} onChange={field("hidden")} int min={1} />
      </div>

      <p className={hint}>
        Every field stays editable. Loading a repo only fills them in — nothing is fetched until you
        press Load, and each repo is read once per session. <b>Attention layers</b> is the count that
        actually holds a KV cache: the same as Layers on a plain transformer, a fraction of it on a
        hybrid Mamba or linear-attention stack.
      </p>

      <label className={fieldWrap}>
        <span className={`${fieldLabel} block`}>Weight precision</span>
        <div className="flex flex-wrap gap-1.5">
          {PRECISIONS.map((p) => (
            <button
              key={p.id}
              type="button"
              disabled={!allows(p.id)}
              title={
                allows(p.id)
                  ? undefined
                  : `Not offered for this model — see the note above.`
              }
              className={`min-w-23 flex-1 rounded-ctl border px-1.5 py-2 font-mono text-note disabled:cursor-not-allowed disabled:line-through disabled:opacity-40 ${focusRing} ${
                precision === p.id
                  ? "border-ink bg-ink font-bold text-panel"
                  : "border-line bg-sunken text-muted"
              } ${allows(p.id) ? "cursor-pointer" : ""}`}
              onClick={() => choosePrecision(p.id)}
            >
              {p.name}
            </button>
          ))}
        </div>
      </label>

      {/* Only shown for quantized weights, because at FP16 it changes nothing:
          these params are already stored at the same width as everything else. */}
      {PRECISIONS.find((p) => p.id === precision).bytes < 2 && (
        <>
          <NumberField
            label="Params (B) left at 16-bit"
            value={model.keptParams}
            onChange={field("keptParams")}
            step={0.01}
            max={model.params}
          />
          <p className={hint}>
            Quantization never covers the whole model. The embedding table and output head always
            stay at 16 bits, and MoE routers, shared experts and vision towers usually do too — 6%
            of Llama 4 Scout, 12% of Kimi K2. Presets carry the figure measured from Red Hat's own
            checkpoint; loading a repo reads it from the file list. Set it to 0 to see the
            uncorrected figure.
          </p>
        </>
      )}

      <SelectField
        label="KV cache precision"
        value={kvDtype}
        onChange={setKvDtype}
        options={KV_DTYPES}
      />
    </fieldset>
  );
}
