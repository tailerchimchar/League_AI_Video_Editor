import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 8080,
    proxy: {
      // Forward /api/* to the FastAPI backend — avoids CORS entirely.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // Prevent proxy from hanging on large file uploads
        timeout: 120000,       // 2 min timeout for the overall request
        proxyTimeout: 120000,  // 2 min timeout for the proxy connection
        configure: (proxy) => {
          // Handle proxy errors gracefully instead of hanging
          proxy.on("error", (err, _req, res) => {
            console.error("[proxy] error:", err.message);
            if (res && "writeHead" in res && !res.headersSent) {
              (res as any).writeHead(502, { "Content-Type": "application/json" });
              (res as any).end(JSON.stringify({ error_code: "PROXY_ERROR", message: "API server not ready. Try again." }));
            }
          });
        },
      },
    },
  },
});
