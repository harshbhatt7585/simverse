import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const snakeLiveTarget = env.VITE_SNAKE_LIVE_URL || 'http://127.0.0.1:8766'
  const snakeWebTarget = env.VITE_SNAKE_WEB_URL || snakeLiveTarget
  const mazeLiveTarget = env.VITE_MAZE_LIVE_URL || 'http://127.0.0.1:8765'
  const battleReplayTarget = env.VITE_BATTLE_REPLAY_URL || 'http://127.0.0.1:8866'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/snake-live': {
          target: snakeLiveTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/snake-live/, ''),
        },
        '/snake-web': {
          target: snakeWebTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/snake-web/, ''),
        },
        '/maze-live': {
          target: mazeLiveTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/maze-live/, ''),
        },
        '/battle-replay': {
          target: battleReplayTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/battle-replay/, ''),
        },
      },
    },
  }
})
