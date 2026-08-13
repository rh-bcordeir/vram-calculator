/**
 * Class strings for the elements that repeat across panels. Anything used
 * once lives inline on the element instead — this file is only for the
 * shapes that would otherwise drift apart.
 */

export const focusRing =
  "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-kv";

/** A bordered panel. Applied to <fieldset>, whose default styling Tailwind's
 *  preflight strips, so the border is re-declared here. */
export const card = "m-0 rounded-panel border border-line bg-panel px-4.5 pt-3.5 pb-4.5";

export const legend =
  "pr-2 font-mono text-label font-bold tracking-[0.14em] uppercase text-bad";

/** Uppercase mono caption, used for legends, eyebrows and axis labels. */
export const caption = "font-mono text-label tracking-[0.12em] uppercase text-muted";

export const fieldWrap = "mt-3.5 block";

/** Caller supplies the display mode — some labels put a live value on the
 *  right, and mixing `block` with `flex` here would be a coin toss over
 *  which utility wins in the generated stylesheet. */
export const fieldLabel = "mb-1.25 text-note font-bold text-muted";

export const control = `w-full rounded-ctl border border-line bg-sunken px-2.25 py-2 font-mono
  text-field text-ink placeholder:text-muted placeholder:opacity-65 ${focusRing}`;

export const hint = "mt-3 text-note leading-[1.55] text-muted";

/** Inline <code> inside hints and the footnote. */
export const code =
  "rounded-ctl border border-line bg-sunken px-1 py-px font-mono text-label";
