import React from "react";
import { fmt } from "../lib/constants.js";

export function NumberField({ label, value, onChange, step = 1, int, min = 0, max = 1000000 }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
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
    <label className="field">
      <span className="field-head">
        {label}
        <b>
          {places ? fmt(value, places) : value.toLocaleString("en-US")}
          {suffix}
        </b>
      </span>
      <input
        type="range"
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
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
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
    <div className={warn ? "table-row warn" : "table-row"}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}
