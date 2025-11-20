import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    commonjsOptions: {
      include: [/onnxruntime-web/, /node_modules/],
    }
  }
});
