import { defineConfig } from 'vite';

// Note: WASM files should be manually copied to public/wasm/
// Run: cp node_modules/onnxruntime-web/dist/*.wasm public/wasm/

export default defineConfig({
  server: {
    port: 5173,
    host: true
    // Removed COEP/COOP headers as they might cause issues with WASM loading
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  publicDir: 'public'
});
