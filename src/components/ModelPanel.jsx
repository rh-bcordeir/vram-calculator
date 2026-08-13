import React, { useState } from "react";
import { fetchModel, parseRepoId } from "../lib/huggingface.js";
import { PRECISIONS, KV_DTYPES, PRESETS } from "../lib/constants.js";
import { NumberField, SelectField } from "./Fields.jsx";

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
    <fieldset className="card">
      <legend>Model</legend>

      <label className="field">
        <span>Hugging Face repo</span>
        <div className="lookup">
          <input
            type="text"
            placeholder="ibm-granite/granite-3.3-8b-instruct"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && load()}
            spellCheck={false}
          />
          <button type="button" onClick={load} disabled={status.state === "loading"}>
            {status.state === "loading" ? "Reading…" : "Load"}
          </button>
        </div>
      </label>

      {status.state !== "idle" && <p className={`status ${status.state}`}>{status.msg}</p>}

      {source && (
        <p className="source">
          Fields below came from <b>{source.label}</b>
          {source.cached && " (cached)"}
          {source.maxLen && ` · trained context ${source.maxLen.toLocaleString("en-US")}`}
        </p>
      )}

      {notes.map((n, i) => (
        <p className="note" key={i}>
          {n}
        </p>
      ))}

      <div className="presets">
        {PRESETS.map((p) => (
          <button key={p.id} type="button" onClick={() => applyPreset(p)}>
            {p.name}
          </button>
        ))}
      </div>

      <div className="row-4">
        <NumberField label="Params (B)" value={model.params} onChange={field("params")} step={0.01} />
        <NumberField label="Layers" value={model.layers} onChange={field("layers")} int min={1} />
        <NumberField label="KV heads" value={model.kvHeads} onChange={field("kvHeads")} int min={1} />
        <NumberField label="Head dim" value={model.headDim} onChange={field("headDim")} int min={1} />
      </div>

      <p className="hint">
        Every field stays editable. Loading a repo only fills them in — nothing is fetched until you
        press Load, and each repo is read once per session.
      </p>

      <label className="field">
        <span>Weight precision</span>
        <div className="pills">
          {PRECISIONS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={precision === p.id ? "pill on" : "pill"}
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
