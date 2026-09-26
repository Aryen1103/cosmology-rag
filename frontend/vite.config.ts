import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend (uvicorn app.main:app --port 8000) serves everything under /api, including figure images.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
