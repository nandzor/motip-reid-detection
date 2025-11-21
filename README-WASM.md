# WASM Configuration untuk ONNX.js

## Setup WASM Files

File WASM sudah di-copy ke `public/wasm/` untuk backup, tapi ONNX.js akan menggunakan file dari `node_modules/onnxruntime-web/dist/` secara default di development mode.

## Jika Ada Error WASM

Jika masih ada error WASM loading, coba:

1. **Pastikan file WASM tersedia:**
   ```bash
   ls node_modules/onnxruntime-web/dist/*.wasm
   ```

2. **Copy ke public (jika diperlukan):**
   ```bash
   cp node_modules/onnxruntime-web/dist/*.wasm public/wasm/
   ```

3. **Restart dev server:**
   ```bash
   npm run dev
   ```

4. **Clear browser cache** dan hard refresh (Ctrl+Shift+R)

## File WASM yang Tersedia

- `ort-wasm-simd-threaded.wasm` - Main WASM file
- `ort-wasm-simd-threaded.asyncify.wasm` - Async version
- `ort-wasm-simd-threaded.jsep.wasm` - JSEP version

ONNX.js akan secara otomatis memilih file yang sesuai dengan konfigurasi browser.
