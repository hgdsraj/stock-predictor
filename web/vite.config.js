import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
// Vite config for the stock-predictor frontend.
// - Dev server proxies /api/* (well, we use absolute URLs via VITE_API_BASE,
//   so we keep the proxy simple for unproxied paths).
// - Production build emits to dist/ which the FastAPI app statics-mounts.
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 5173,
        proxy: {
            // Forward backend routes to the local FastAPI server during dev.
            "^/(healthz|tickers|predictions|runs|backtest|jobs)": {
                target: "http://127.0.0.1:8000",
                changeOrigin: true,
            },
        },
    },
    build: {
        outDir: "dist",
        sourcemap: true,
    },
});
