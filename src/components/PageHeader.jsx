import React from "react";

/** Page title row: back arrow, title, one-line description. */
export default function PageHeader({ title, description, onBack }) {
  return (
    <div className="mb-7 flex items-start gap-3.5">
      <div>
        <h1 className="m-0 font-display text-[1.75rem] font-medium tracking-[-0.01em]">{title}</h1>
        <p className="mt-1.5 max-w-[70ch] text-field text-muted">{description}</p>
      </div>
    </div>
  );
}
