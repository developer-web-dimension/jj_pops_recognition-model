import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    commonjsOptions: {
      include: [/onnxruntime-web/, /node_modules/],
    },
  },
  server: {
    host: true,               
    port: 5173,
    allowedHosts: true,       
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
