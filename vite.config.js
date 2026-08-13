import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Relative base so the build works from a subpath, e.g. GitHub Pages
  // project sites served at /<repo>/.
  base: "./",
  server: {
    port: 5173,
    // Optional escape hatch: if huggingface.co CORS is blocked on your
    // network, point VITE_HF_BASE at "/hf" and requests go through here.
    proxy: {
      "/hf": {
        target: "https://huggingface.co",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/hf/, ""),
      },
    },
  },
});
