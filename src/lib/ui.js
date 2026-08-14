/**
 * Class strings for the elements that repeat across panels, in the
 * Red Hat Design System's field and card treatment.
 */

export const focusRing =
  "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent";

/** A bordered panel. Applied to <fieldset>, whose default styling Tailwind's
 *  preflight strips, so the border is re-declared here. */
export const card = "m-0 rounded-panel border border-line bg-panel px-5 pt-1 pb-5.5";

/** Section legend: mono, uppercase, brand red as an accent. */
export const legend =
  "px-2 font-mono text-label font-bold tracking-[0.14em] uppercase text-brand";

export const caption = "font-mono text-label tracking-[0.12em] uppercase text-muted";

export const fieldWrap = "mt-4 block";

export const fieldLabel = "mb-1.5 text-note font-bold text-muted";

export const control = `w-full rounded-ctl border border-line bg-sunken px-2.5 py-2.25 font-sans
  text-field text-ink placeholder:text-muted placeholder:opacity-65 ${focusRing}`;

/** Primary action: interactive blue, never brand red inside an app. */
export const buttonPrimary = `shrink-0 cursor-pointer rounded-ctl border border-accent bg-accent
  px-5 py-2.25 text-field text-white hover:border-[#004d99] hover:bg-[#004d99]
  disabled:cursor-progress disabled:opacity-55 ${focusRing}`;

export const hint = "mt-4 text-note leading-[1.55] text-muted";

export const code =
  "rounded-ctl border border-line bg-sunken px-1 py-px font-mono text-label";
