import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const snakeApiTarget = env.VITE_SNAKE_API_URL || 'http://127.0.0.1:8770'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/snake': {
          target: snakeApiTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
