import React from "react";
import { fmt } from "../lib/constants.js";
import { caption } from "../lib/ui.js";

/** Label centred on its mark. Near the ends of the track it is anchored
 *  inside instead, so it cannot hang off the edge. */
function labelPos(value, scale) {
  const p = (value / scale) * 100;
  if (p < 10) return "left-0";
  if (p > 90) return "left-auto right-0";
  return "left-1/2 -translate-x-1/2";
}

const markLabel =
  "absolute pb-[3px] font-mono text-mark tracking-[0.08em] whitespace-nowrap uppercase";

/**
 * Segments stack left to right against a fixed axis. When the total exceeds
 * capacity the axis grows, so the usable-memory marker visibly falls behind
 * the bar instead of the bar silently clipping.
 */
export default function MemoryMap({ result }) {
  const scale = Math.max(result.capacity, result.total);
  const pct = (v) => `${(v / scale) * 100}%`;

  const segments = [
    { name: "Weights", value: result.weights, color: "bg-weights" },
    { name: "KV cache", value: result.kv, color: "bg-kv" },
    { name: "Overhead", value: result.overhead, color: "bg-over" },
  ];

  return (
    <div className="rounded-panel border border-line bg-panel px-4.5 pt-4 pb-3.5">
      {/* The bottom margin leaves room for the two rows of mark labels. */}
      <div className={`${caption} mb-10 flex justify-between`}>
        <span>Memory map</span>
        <span>0 → {fmt(scale, 0)} GiB</span>
      </div>

      <div className="hatched relative flex h-11.5 border border-line">
        {segments.map((s) => (
          <div
            key={s.name}
            className={`h-full min-w-0 motion-safe:transition-[width] motion-safe:duration-[180ms] ${s.color}`}
            style={{ width: pct(s.value) }}
            title={`${s.name}: ${fmt(s.value)} GiB`}
          />
        ))}
        {result.fits && (
          <div
            className="h-full min-w-0 bg-free motion-safe:transition-[width] motion-safe:duration-[180ms]"
            style={{ width: pct(result.usable - result.total) }}
          />
        )}

        {/* Usable takes the upper label row and physical the lower one, so the
            two never collide however close together the marks land. */}
        <div className="absolute -top-2 -bottom-2 w-0.5 bg-bad" style={{ left: pct(result.usable) }}>
          <span
            className={`${markLabel} bottom-[calc(100%+13px)] text-bad ${labelPos(result.usable, scale)}`}
          >
            usable {fmt(result.usable, 0)}
          </span>
        </div>
        {result.capacity !== result.usable && (
          <div
            className="absolute -top-2 -bottom-2 w-0.5 bg-muted opacity-60"
            style={{ left: pct(result.capacity) }}
          >
            <span
              className={`${markLabel} bottom-full text-muted ${labelPos(result.capacity, scale)}`}
            >
              physical {fmt(result.capacity, 0)}
            </span>
          </div>
        )}
      </div>

      <ul className="mt-4 flex list-none flex-wrap gap-x-5.5 gap-y-1.5 p-0">
        {segments.map((s) => (
          <li key={s.name} className="flex items-center gap-1.75 text-note">
            <i className={`size-2.75 rounded-[1px] ${s.color}`} />
            <span className="text-muted">{s.name}</span>
            <span className="font-mono font-bold">{fmt(s.value)} GiB</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
