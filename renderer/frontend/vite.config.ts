import { promises as fs } from 'node:fs'
import path from 'node:path'

import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

type ReplayPayload = {
  episode?: number
  steps?: number
  frames?: unknown[]
  [key: string]: unknown
}

async function pathExists(candidate: string): Promise<boolean> {
  try {
    const stat = await fs.stat(candidate)
    return stat.isFile()
  } catch {
    return false
  }
}

async function resolveReplayFile(dirValue: string): Promise<string | null> {
  const candidate = path.resolve(dirValue)
  if (await pathExists(candidate)) {
    return candidate
  }

  try {
    const stat = await fs.stat(candidate)
    if (!stat.isDirectory()) {
      return null
    }
  } catch {
    return null
  }

  const singleReplay = path.join(candidate, 'replay.json')
  if (await pathExists(singleReplay)) {
    return singleReplay
  }

  const entries = await fs.readdir(candidate, { withFileTypes: true })
  const jsonFiles = entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.json'))
    .map((entry) => entry.name)
    .sort()

  if (jsonFiles.length === 0) {
    return null
  }

  return path.join(candidate, jsonFiles[jsonFiles.length - 1])
}

async function readReplayPayload(filePath: string): Promise<ReplayPayload> {
  const raw = await fs.readFile(filePath, 'utf-8')
  return JSON.parse(raw) as ReplayPayload
}

function localReplayFallbackPlugin(): Plugin {
  return {
    name: 'simverse-local-replay-fallback',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url) {
          next()
          return
        }

        const url = new URL(req.url, 'http://localhost')
        const gameRoute = /^\/(snake|maze|battle-grid)\/replays?\/?$/.test(url.pathname)
        const replayRoute = /^\/(snake|maze|battle-grid)\/replay\/?$/.test(url.pathname)
        if (!gameRoute && !replayRoute) {
          next()
          return
        }

        const dir = url.searchParams.get('dir')
        if (!dir) {
          next()
          return
        }

        try {
          const replayFile = await resolveReplayFile(dir)
          if (!replayFile) {
            next()
            return
          }

          const replayName = path.basename(replayFile)
          const payload = await readReplayPayload(replayFile)

          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          if (replayRoute) {
            const replayId = path.basename(replayFile, '.json')
            res.end(
              JSON.stringify({
                id: replayId,
                name: replayName,
                data: payload,
              }),
            )
            return
          }

          res.end(
            JSON.stringify({
              episodes: [
                {
                  id: path.basename(replayFile, '.json'),
                  name: replayName,
                },
              ],
            }),
          )
        } catch (error) {
          next(error as Error)
        }
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const defaultApiTarget = 'http://127.0.0.1:8770'
  const snakeApiTarget = env.VITE_SNAKE_API_URL || defaultApiTarget
  const mazeApiTarget = env.VITE_MAZE_API_URL || env.VITE_SNAKE_API_URL || defaultApiTarget
  const battleGridApiTarget =
    env.VITE_BATTLE_GRID_API_URL || env.VITE_MAZE_API_URL || env.VITE_SNAKE_API_URL || defaultApiTarget

  return {
    plugins: [localReplayFallbackPlugin(), react()],
    server: {
      proxy: {
        '/snake': {
          target: snakeApiTarget,
          changeOrigin: true,
        },
        '/maze': {
          target: mazeApiTarget,
          changeOrigin: true,
        },
        '/battle-grid': {
          target: battleGridApiTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
