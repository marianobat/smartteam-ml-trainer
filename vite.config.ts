import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? (process.env.VERCEL ? "/" : "/smartteam-ml-trainer/"),
  plugins: [react()],
  resolve: {
    alias: {
      // @tensorflow-models/speech-commands usa util.promisify de Node;
      // lo resolvemos a un shim local apto para navegador.
      util: fileURLToPath(new URL("./src/shims/util.ts", import.meta.url)),
    },
  },
});
