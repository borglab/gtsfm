import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/api": "http://127.0.0.1:5173",
      "/data": "http://127.0.0.1:5173",
      "/static": "http://127.0.0.1:5173",
    },
  },
  build: {
    outDir: fileURLToPath(new URL("../static", import.meta.url)),
    emptyOutDir: false,
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL("./src/main.tsx", import.meta.url)),
      output: {
        entryFileNames: "studio.js",
        assetFileNames: (assetInfo) => assetInfo.names?.some((name) => name.endsWith(".css")) ? "studio.css" : "assets/[name]-[hash][extname]",
      },
    },
  },
});
