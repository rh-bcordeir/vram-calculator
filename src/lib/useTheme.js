import { useCallback, useEffect, useState } from "react";

const KEY = "vram-theme";

function initial() {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") || "light";
}

/** Theme lives on <html> so the inline script in index.html can set it early. */
export function useTheme() {
  const [theme, setTheme] = useState(initial);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // Private mode or blocked storage: the theme just won't persist.
    }
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);

  return [theme, toggle];
}
