import React from "react";

/**
 * Application sidenav. One item for now — the active item carries the 3px
 * brand-red rule, which is how RHDS marks the current entry. Collapsing is
 * done by the grid column in App, so this only has to hide its own overflow.
 */
const ITEMS = [{ id: "vram", label: "VRAM calculator" }];

export default function Sidebar({ open, current = "vram" }) {
  return (
    <nav
      aria-label="Sections"
      className={`min-h-[calc(100vh-3.75rem)] overflow-hidden bg-[#0f0f0f] py-4 ${
        open ? "border-r border-r-[#292929]" : ""
      }`}
    >
      {ITEMS.map((item) => {
        const active = item.id === current;
        return (
          <a
            key={item.id}
            href="#"
            aria-current={active ? "page" : undefined}
            className={`block border-l-3 px-5 py-2.25 text-field ${
              active
                ? "border-l-brand bg-[#1f1f1f] text-white"
                : "border-l-transparent text-[#c7c7c7] hover:bg-[#1f1f1f]"
            }`}
          >
            {item.label}
          </a>
        );
      })}
    </nav>
  );
}
