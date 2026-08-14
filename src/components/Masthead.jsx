import React from "react";

// Imported (not referenced from public/) so Vite rewrites the URL against the
// relative base and the logo resolves when served from a subpath.
import logoRedHat from "../assets/logo-redhat-reversed.png";

/**
 * Red Hat application masthead: fixed black bar, hamburger for the sidenav,
 * reversed logo, product name, and the account slot on the right. Always
 * dark, whatever the page theme is — that is how Red Hat consoles look.
 */
export default function Masthead({ theme, toggleTheme, onToggleNav, user = "bcordeir" }) {
  return (
    <header className="sticky top-0 z-20 flex h-15 items-center gap-4 border-b border-b-[#292929] bg-black px-6">
      <button
        type="button"
        onClick={onToggleNav}
        aria-label="Toggle navigation"
        className="flex cursor-pointer p-1.5 text-white"
      >
        {/* @rhds/icons — ui/menu-bars */}
        <svg width="18" height="18" viewBox="0 0 32 32" fill="currentColor" aria-hidden="true">
          <path d="M31 16a1 1 0 0 1-1 1H2a1 1 0 1 1 0-2h28a1 1 0 0 1 1 1Zm-1 9H2a1 1 0 1 0 0 2h28a1 1 0 1 0 0-2ZM2 7h28a1 1 0 1 0 0-2H2a1 1 0 1 0 0 2Z" />
        </svg>
      </button>

      <img src={logoRedHat} alt="Red Hat" className="block h-6.5 w-auto" />
      <div className="ml-0.5 flex flex-col leading-tight">
        <span className="font-display text-[0.9375rem] font-medium text-white">Red Hat AI</span>
        <span className="text-label text-[#a3a3a3]">VRAM calculator</span>
      </div>

      <div className="flex-1" />

      <button
        type="button"
        onClick={toggleTheme}
        className="cursor-pointer rounded-ctl border border-[#383838] px-2.5 py-1.5 font-mono text-label tracking-[0.08em] text-[#e0e0e0] uppercase hover:border-[#707070]"
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      >
        {theme === "dark" ? "Light" : "Dark"}
      </button>
      <span className="text-row text-[#e0e0e0]">{user}</span>
    </header>
  );
}
