import React from "react";
import { fmt } from "../lib/constants.js";
import { control, fieldWrap, fieldLabel } from "../lib/ui.js";

export function NumberField({ label, value, onChange, step = 1, int, min = 0, max = 1000000 }) {
  return (
    <label className={fieldWrap}>
      <span className={`${fieldLabel} block`}>{label}</span>
      <input
        type="number"
        className={control}
        value={value}
        min={min}
        max={max}
        step={int ? 1 : step}
        onChange={(e) => {
          const v = int ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
          if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
        }}
      />
    </label>
  );
}

export function SliderField({ label, value, onChange, min, max, step, suffix = "", places = 0 }) {
  return (
    <label className={fieldWrap}>
      <span className={`${fieldLabel} flex items-baseline justify-between`}>
        {label}
        <b className="font-mono text-row text-ink">
          {places ? fmt(value, places) : value.toLocaleString("en-US")}
          {suffix}
        </b>
      </span>
      <input
        type="range"
        className="mt-0.5 w-full accent-kv"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function SelectField({ label, value, onChange, options }) {
  return (
    <label className={fieldWrap}>
      <span className={`${fieldLabel} block`}>{label}</span>
      <select className={control} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.label ?? o.name}
          </option>
        ))}
      </select>
    </label>
  );
}

export function TableRow({ label, value, warn }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-b-bg px-4.5 py-2.75 text-row last:border-b-0">
      <span className="text-muted">{label}</span>
      <b className={`font-mono ${warn ? "text-bad" : ""}`}>{value}</b>
    </div>
  );
}
