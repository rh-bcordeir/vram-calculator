import React from "react";
import { fmt } from "../lib/constants.js";

/**
 * Segments stack left to right against a fixed axis. When the total exceeds
 * capacity the axis grows, so the usable-memory marker visibly falls behind
 * the bar instead of the bar silently clipping.
 */
export default function MemoryMap({ result }) {
  const scale = Math.max(result.capacity, result.total);
  const pct = (v) => `${(v / scale) * 100}%`;

  const segments = [
    { name: "Weights", value: result.weights, color: "var(--c-weights)" },
    { name: "KV cache", value: result.kv, color: "var(--c-kv)" },
    { name: "Overhead", value: result.overhead, color: "var(--c-over)" },
  ];

  return (
    <div className="map">
      <div className="map-head">
        <span>Memory map</span>
        <span>0 → {fmt(scale, 0)} GiB</span>
      </div>

      <div className="track">
        {segments.map((s) => (
          <div
            key={s.name}
            className="seg"
            style={{ width: pct(s.value), background: s.color }}
            title={`${s.name}: ${fmt(s.value)} GiB`}
          />
        ))}
        {result.fits && (
          <div className="seg free" style={{ width: pct(result.usable - result.total) }} />
        )}

        <div className="mark" style={{ left: pct(result.usable) }}>
          <span>usable {fmt(result.usable, 0)}</span>
        </div>
        {result.capacity !== result.usable && (
          <div className="mark phys" style={{ left: pct(result.capacity) }}>
            <span>physical</span>
          </div>
        )}
      </div>

      <ul className="key">
        {segments.map((s) => (
          <li key={s.name}>
            <i style={{ background: s.color }} />
            <span className="key-name">{s.name}</span>
            <span className="key-val">{fmt(s.value)} GiB</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
