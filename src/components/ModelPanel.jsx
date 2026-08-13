import React, { useState } from "react";
import { fetchModel, parseRepoId } from "../lib/huggingface.js";
import { PRECISIONS, KV_DTYPES, PRESETS } from "../lib/constants.js";
import { NumberField, SelectField } from "./Fields.jsx";
import { card, control, fieldWrap, fieldLabel, focusRing, hint, legend } from "../lib/ui.js";

const STATUS_COLOR = {
  loading: "text-muted",
  ok: "text-good",
  error: "text-bad",
};

export default function ModelPanel({ model, setModel, precision, setPrecision, kvDtype, setKvDtype }) {
  const [url, setUrl] = useState("");
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
        kvHeads: m.kvHeads,
        headDim: m.headDim,
      });
      setPrecision(m.precision);
      setNotes(m.notes);
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

  function applyPreset(p) {
    setModel({ params: p.params, layers: p.layers, kvHeads: p.kvHeads, headDim: p.headDim });
    setNotes([]);
    setSource({ label: p.name, cached: false });
    setStatus({ state: "idle" });
  }

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

      <div className="mt-3.5 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`cursor-pointer rounded-full border border-line bg-sunken px-2.5 py-1.25 text-label text-muted hover:border-muted hover:text-ink ${focusRing}`}
            onClick={() => applyPreset(p)}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-2.5 min-[521px]:grid-cols-4">
        <NumberField label="Params (B)" value={model.params} onChange={field("params")} step={0.01} />
        <NumberField label="Layers" value={model.layers} onChange={field("layers")} int min={1} />
        <NumberField label="KV heads" value={model.kvHeads} onChange={field("kvHeads")} int min={1} />
        <NumberField label="Head dim" value={model.headDim} onChange={field("headDim")} int min={1} />
      </div>

      <p className={hint}>
        Every field stays editable. Loading a repo only fills them in — nothing is fetched until you
        press Load, and each repo is read once per session.
      </p>

      <label className={fieldWrap}>
        <span className={`${fieldLabel} block`}>Weight precision</span>
        <div className="flex flex-wrap gap-1.5">
          {PRECISIONS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`min-w-23 flex-1 cursor-pointer rounded-ctl border px-1.5 py-2 font-mono text-note ${focusRing} ${
                precision === p.id
                  ? "border-ink bg-ink font-bold text-panel"
                  : "border-line bg-sunken text-muted"
              }`}
              onClick={() => setPrecision(p.id)}
            >
              {p.name}
            </button>
          ))}
        </div>
      </label>

      <SelectField
        label="KV cache precision"
        value={kvDtype}
        onChange={setKvDtype}
        options={KV_DTYPES}
      />
    </fieldset>
  );
}
