import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: parseInt(process.env.MIROFISH_FRONTEND_PORT || '3000', 10),
    open: true,
    proxy: {
      '/api': {
        target: process.env.MIROFISH_BACKEND_URL || 'http://localhost:5001',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
