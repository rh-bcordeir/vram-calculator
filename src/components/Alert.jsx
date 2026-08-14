import React from "react";

/**
 * RHDS-style status alert: 2px status rule on top, status icon, heading,
 * body. Danger uses red-orange — never brand red.
 */
const STATE = {
  danger: { line: "border-t-bad", bg: "bg-alarm-bg", icon: "text-bad" },
  caution: { line: "border-t-note-line", bg: "bg-note-bg", icon: "text-note-line" },
};

export default function Alert({ state = "danger", heading, children }) {
  const s = STATE[state] ?? STATE.danger;
  return (
    <div
      role="status"
      className={`grid grid-cols-[min-content_1fr] gap-x-2.5 gap-y-1 border-t-2 px-4 py-3.5 text-row leading-[1.55] text-alarm-ink ${s.line} ${s.bg}`}
    >
      <svg
        width="20"
        height="20"
        viewBox="0 0 32 32"
        fill="currentColor"
        aria-hidden="true"
        className={`row-span-2 ${s.icon}`}
      >
        <path d="M16 1C7.729 1 1 7.729 1 16s6.729 15 15 15 15-6.729 15-15S24.271 1 16 1Zm-1.5 8a1.5 1.5 0 1 1 3 0v7a1.5 1.5 0 1 1-3 0V9ZM16 25.001a2 2 0 1 1-.001-3.999A2 2 0 0 1 16 25.001Z" />
      </svg>
      {heading ? <span className="font-medium">{heading}</span> : null}
      <div className="col-start-2">{children}</div>
    </div>
  );
}
