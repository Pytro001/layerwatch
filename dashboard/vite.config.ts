import { defineConfig } from 'vite'

export default defineConfig({
  server: { port: 5304, strictPort: true, proxy: { '/api': 'http://localhost:8304' } },
  preview: { port: 5304, strictPort: true },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1200 },
})
