import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type ViewMode = 'snake' | 'maze' | 'battle'

type Position = { x: number; y: number }

type MetaPayload = {
  title?: string
  env?: string
  width?: number
  height?: number
  channels?: number
  replay_files?: string[]
  replay_count?: number
  snapshot_url?: string
  fps?: number
}

type EventPayload = {
  type?: string
  data?: unknown
}

type GenericFrame = {
  step?: number
  episode?: unknown
  observation?: unknown
  rewards?: unknown
  info?: Record<string, unknown>
  done?: boolean
  _replay_file_index?: unknown
  _replay_file_name?: unknown
}

type EpisodePayload = {
  metadata?: Record<string, unknown>
  frames?: GenericFrame[]
}

type BattleReward = {
  0: number
  1: number
}

type BattleDims = {
  width: number
  height: number
  cell: number
}

const SNAKE_COLORS = {
  bg: '#0e141b',
  floor: '#eef3f8',
  wall: '#3c4e62',
  food: '#d93f47',
  head: '#23924c',
  body: '#5acb85',
  grid: 'rgba(0,0,0,0.08)',
} as const

const MAZE_COLORS = {
  bg: '#0f1218',
  floor: '#eef2f8',
  wall: '#39465e',
  goal0: '#4b87ff',
  goal1: '#ff8c5a',
  agent0: '#1f61d4',
  agent1: '#d4622a',
  grid: 'rgba(0,0,0,0.08)',
} as const

const BATTLE_COLORS = {
  bg: '#111722',
  grid: '#f0f4fb',
  line: 'rgba(22,38,56,0.2)',
  agent0: '#2f74e6',
  agent1: '#e26a34',
  dead: '#7f8794',
} as const

function firstScalar(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (Array.isArray(value) && value.length > 0) {
    return firstScalar(value[0], fallback)
  }
  return fallback
}

function parseNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

function parseReward(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (Array.isArray(value)) {
    let total = 0
    let found = false
    for (const row of value) {
      if (!row || typeof row !== 'object') {
        continue
      }
      const rewardVal = (row as Record<string, unknown>).reward
      if (typeof rewardVal === 'number' && Number.isFinite(rewardVal)) {
        total += rewardVal
        found = true
      }
    }
    return found ? total : 0
  }
  if (value && typeof value === 'object') {
    const rewardVal = (value as Record<string, unknown>).reward
    if (typeof rewardVal === 'number' && Number.isFinite(rewardVal)) {
      return rewardVal
    }
  }
  return 0
}

function parseBattleReward(value: unknown): BattleReward {
  const out: BattleReward = { 0: 0, 1: 0 }
  if (!Array.isArray(value)) {
    return out
  }
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') {
      continue
    }
    const row = entry as Record<string, unknown>
    const id = Number(row.agent_id)
    const reward = Number(row.reward)
    if ((id === 0 || id === 1) && Number.isFinite(reward)) {
      out[id] = reward
    }
  }
  return out
}

function as2DLayer(value: unknown): number[][] {
  if (!Array.isArray(value)) {
    return []
  }
  const out: number[][] = []
  for (const row of value) {
    if (!Array.isArray(row)) {
      continue
    }
    out.push(row.map((cell) => parseNumber(cell, 0)))
  }
  return out
}

function as3DObservation(value: unknown): number[][][] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map((layer) => as2DLayer(layer))
}

function findAgentPos(layer: number[][]): Position | null {
  for (let y = 0; y < layer.length; y += 1) {
    const row = layer[y]
    for (let x = 0; x < row.length; x += 1) {
      if (Number(row[x]) > 0.5) {
        return { x, y }
      }
    }
  }
  return null
}

function resolveUrl(baseUrl: string, maybeRelative: string): string {
  if (maybeRelative.startsWith('http://') || maybeRelative.startsWith('https://')) {
    return maybeRelative
  }
  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  const rel = maybeRelative.startsWith('/') ? maybeRelative : `/${maybeRelative}`
  return `${base}${rel}`
}

function SnakeViewer({ baseUrl, active }: { baseUrl: string; active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameBufferRef = useRef<GenericFrame[]>([])
  const episodeStartIndicesRef = useRef<number[]>([])
  const episodeIdsRef = useRef<number[]>([])
  const replayFilesRef = useRef<string[]>([])
  const replayFileFirstFrameRef = useRef<Map<number, number>>(new Map())
  const currentIndexRef = useRef<number>(-1)
  const currentReplayFileIndexRef = useRef<number>(-1)
  const dimsRef = useRef<{ width: number; height: number } | null>(null)
  const followLiveRef = useRef<boolean>(true)

  const [title, setTitle] = useState('Snake Live')
  const [status, setStatus] = useState('Waiting for frames...')
  const [error, setError] = useState('')
  const [isPlaying, setIsPlaying] = useState(true)
  const [followLive, setFollowLive] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [reconnectToken, setReconnectToken] = useState(0)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    followLiveRef.current = followLive
  }, [followLive])

  const bump = useCallback(() => {
    setTick((value) => value + 1)
  }, [])

  const drawFrame = useCallback((frame: GenericFrame) => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const obs = as3DObservation(frame.observation)
    if (obs.length < 4) {
      return
    }

    const walls = obs[0]
    const food = obs[1]
    const head = obs[2]
    const body = obs[3]

    const height = walls.length
    const width = height > 0 ? walls[0].length : 0
    if (!width || !height) {
      return
    }

    if (!dimsRef.current || dimsRef.current.width !== width || dimsRef.current.height !== height) {
      const maxSize = 760
      const cellSize = Math.max(10, Math.floor(Math.min(34, maxSize / Math.max(width, height))))
      canvas.width = width * cellSize
      canvas.height = height * cellSize
      dimsRef.current = { width, height }
    }

    const cellSize = Math.floor(canvas.width / width)

    let headX = -1
    let headY = -1
    let foodX = -1
    let foodY = -1
    let bodyCount = 0

    ctx.fillStyle = SNAKE_COLORS.bg
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        ctx.fillStyle = walls[y]?.[x] > 0.5 ? SNAKE_COLORS.wall : SNAKE_COLORS.floor
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize)
        ctx.strokeStyle = SNAKE_COLORS.grid
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize)

        if (food[y]?.[x] > 0.5) {
          foodX = x
          foodY = y
        }
        if (body[y]?.[x] > 0.5) {
          bodyCount += 1
        }
        if (head[y]?.[x] > 0.5) {
          headX = x
          headY = y
        }
      }
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (body[y]?.[x] > 0.5) {
          ctx.fillStyle = SNAKE_COLORS.body
          ctx.fillRect(x * cellSize + 2, y * cellSize + 2, cellSize - 4, cellSize - 4)
        }
      }
    }

    if (headX >= 0 && headY >= 0) {
      ctx.fillStyle = SNAKE_COLORS.head
      ctx.fillRect(headX * cellSize + 2, headY * cellSize + 2, cellSize - 4, cellSize - 4)
    }
    if (foodX >= 0 && foodY >= 0) {
      ctx.fillStyle = SNAKE_COLORS.food
      ctx.fillRect(foodX * cellSize + 2, foodY * cellSize + 2, cellSize - 4, cellSize - 4)
    }

    const info = frame.info ?? {}
    const reward = parseReward(frame.rewards)
    const episode = firstScalar(frame.episode, 0)
    const score = firstScalar(info.score, 0)
    const steps = firstScalar(info.steps, 0)
    const term = firstScalar(info.termination_reason, 0)
    const done = frame.done ? 'yes' : 'no'

    setStatus(
      [
        `episode: ${episode}`,
        `step: ${frame.step ?? '?'}`,
        `done: ${done}`,
        `term: ${term}`,
        `reward: ${reward.toFixed(3)}`,
        `score: ${score}`,
        `steps: ${steps}`,
        `length: ${bodyCount + (headX >= 0 ? 1 : 0)}`,
        `head: (${headX}, ${headY})`,
        `food: (${foodX}, ${foodY})`,
      ].join('\n'),
    )

    currentReplayFileIndexRef.current = firstScalar(frame._replay_file_index, -1)
  }, [])

  const renderIndex = useCallback(
    (index: number) => {
      const frameBuffer = frameBufferRef.current
      if (frameBuffer.length === 0) {
        return
      }
      const clamped = Math.min(frameBuffer.length - 1, Math.max(0, index))
      currentIndexRef.current = clamped
      const frame = frameBuffer[clamped]
      drawFrame(frame)
      bump()
    },
    [bump, drawFrame],
  )

  const seekTo = useCallback(
    (index: number) => {
      setFollowLive(false)
      setIsPlaying(false)
      renderIndex(index)
    },
    [renderIndex],
  )

  const seekToEpisodePos = useCallback(
    (episodePos: number) => {
      const replayFiles = replayFilesRef.current
      if (replayFiles.length > 0) {
        const clamped = Math.min(replayFiles.length - 1, Math.max(0, episodePos))
        const startFrame = replayFileFirstFrameRef.current.get(clamped)
        if (startFrame !== undefined) {
          seekTo(startFrame)
          return
        }
        for (let index = clamped; index >= 0; index -= 1) {
          const fallbackFrame = replayFileFirstFrameRef.current.get(index)
          if (fallbackFrame !== undefined) {
            seekTo(fallbackFrame)
            return
          }
        }
        return
      }

      const starts = episodeStartIndicesRef.current
      if (starts.length === 0) {
        return
      }
      const clamped = Math.min(starts.length - 1, Math.max(0, episodePos))
      seekTo(starts[clamped])
    },
    [seekTo],
  )

  const loadSnapshot = useCallback(
    async (snapshotUrl: string) => {
      try {
        const response = await fetch(resolveUrl(baseUrl, snapshotUrl), { cache: 'no-store' })
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const payload = (await response.json()) as { frames?: unknown; replay_files?: unknown }

        const frames = Array.isArray(payload.frames) ? (payload.frames as GenericFrame[]) : []
        const replayFiles = Array.isArray(payload.replay_files)
          ? payload.replay_files.map((entry) => String(entry))
          : []

        frameBufferRef.current = frames
        replayFilesRef.current = replayFiles
        episodeStartIndicesRef.current = []
        episodeIdsRef.current = []
        replayFileFirstFrameRef.current = new Map()
        currentReplayFileIndexRef.current = -1
        currentIndexRef.current = -1

        for (let index = 0; index < frames.length; index += 1) {
          const frame = frames[index]
          const replayIndex = firstScalar(frame?._replay_file_index, -1)
          if (replayIndex >= 0 && !replayFileFirstFrameRef.current.has(replayIndex)) {
            replayFileFirstFrameRef.current.set(replayIndex, index)
          }

          const episode = firstScalar(frame?.episode, 0)
          const episodes = episodeIdsRef.current
          if (episodes.length === 0 || episodes[episodes.length - 1] !== episode) {
            episodes.push(episode)
            episodeStartIndicesRef.current.push(index)
          }
        }

        setFollowLive(false)
        setIsPlaying(false)
        if (frameBufferRef.current.length > 0) {
          renderIndex(0)
        } else {
          bump()
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(`Failed to load snapshot: ${message}`)
      }
    },
    [baseUrl, bump, renderIndex],
  )

  useEffect(() => {
    if (!active) {
      return
    }

    setError('')
    setStatus('Connecting...')

    const source = new EventSource(resolveUrl(baseUrl, '/events'))

    source.onmessage = (event) => {
      let payload: EventPayload
      try {
        payload = JSON.parse(event.data) as EventPayload
      } catch {
        return
      }

      if (payload.type === 'meta') {
        const data = (payload.data ?? {}) as MetaPayload
        setTitle(data.title ?? 'Snake Live')
        if (Array.isArray(data.replay_files)) {
          replayFilesRef.current = data.replay_files.map((entry) => String(entry))
        }
        if (typeof data.snapshot_url === 'string' && data.snapshot_url.length > 0) {
          void loadSnapshot(data.snapshot_url)
        }
        bump()
        return
      }

      if (payload.type === 'frame') {
        const frame = (payload.data ?? {}) as GenericFrame
        const buffer = frameBufferRef.current
        buffer.push(frame)

        const replayIndex = firstScalar(frame._replay_file_index, -1)
        if (replayIndex >= 0 && !replayFileFirstFrameRef.current.has(replayIndex)) {
          replayFileFirstFrameRef.current.set(replayIndex, buffer.length - 1)
        }

        const episode = firstScalar(frame.episode, 0)
        const episodeIds = episodeIdsRef.current
        if (episodeIds.length === 0 || episodeIds[episodeIds.length - 1] !== episode) {
          episodeIds.push(episode)
          episodeStartIndicesRef.current.push(buffer.length - 1)
        }

        if (buffer.length > 8000) {
          buffer.shift()

          episodeStartIndicesRef.current = episodeStartIndicesRef.current.map((value) =>
            Math.max(0, value - 1),
          )

          const remapped = new Map<number, number>()
          for (const [key, value] of replayFileFirstFrameRef.current.entries()) {
            remapped.set(key, Math.max(0, value - 1))
          }
          replayFileFirstFrameRef.current = remapped

          while (
            episodeStartIndicesRef.current.length > 1 &&
            episodeStartIndicesRef.current[1] === 0
          ) {
            episodeStartIndicesRef.current.shift()
            episodeIdsRef.current.shift()
          }

          if (!followLiveRef.current) {
            currentIndexRef.current = Math.max(0, currentIndexRef.current - 1)
          }
        }

        if (followLiveRef.current) {
          renderIndex(buffer.length - 1)
        } else {
          bump()
        }
      }
    }

    source.onerror = () => {
      setStatus('Connection lost. Check backend and reconnect.')
    }

    return () => {
      source.close()
    }
  }, [active, baseUrl, bump, loadSnapshot, reconnectToken, renderIndex])

  useEffect(() => {
    if (!active) {
      return
    }

    const interval = window.setInterval(() => {
      if (!isPlaying || frameBufferRef.current.length === 0) {
        return
      }
      if (followLive) {
        renderIndex(frameBufferRef.current.length - 1)
        return
      }

      let episodePos = 0
      if (replayFilesRef.current.length > 0) {
        episodePos = Math.max(0, currentReplayFileIndexRef.current)
      } else {
        const starts = episodeStartIndicesRef.current
        for (let index = 0; index < starts.length; index += 1) {
          if (starts[index] <= Math.max(0, currentIndexRef.current)) {
            episodePos = index
          } else {
            break
          }
        }
      }

      const maxEpisodes =
        replayFilesRef.current.length > 0
          ? replayFilesRef.current.length
          : episodeStartIndicesRef.current.length
      const nextEpisode = Math.min(maxEpisodes - 1, episodePos + 1)
      if (nextEpisode === episodePos) {
        setIsPlaying(false)
        return
      }
      seekToEpisodePos(nextEpisode)
    }, Math.max(25, Math.floor(1000 / (18 * speed))))

    return () => {
      window.clearInterval(interval)
    }
  }, [active, followLive, isPlaying, renderIndex, seekToEpisodePos, speed])

  const controls = useMemo(() => {
    const usingReplayFiles = replayFilesRef.current.length > 0
    const max = usingReplayFiles
      ? Math.max(0, replayFilesRef.current.length - 1)
      : Math.max(0, episodeStartIndicesRef.current.length - 1)

    let episodePos = 0
    if (usingReplayFiles) {
      episodePos = Math.max(0, currentReplayFileIndexRef.current)
    } else {
      for (let index = 0; index < episodeStartIndicesRef.current.length; index += 1) {
        if (episodeStartIndicesRef.current[index] <= Math.max(0, currentIndexRef.current)) {
          episodePos = index
        } else {
          break
        }
      }
    }

    let episodeMeta = 'episode 0/0'
    if (usingReplayFiles) {
      const fileName = replayFilesRef.current[episodePos] ?? '?'
      episodeMeta =
        `episode ${Math.max(0, episodePos + 1)}/${Math.max(1, replayFilesRef.current.length)} ` +
        `(${fileName})`
    } else {
      const episodeId = episodeIdsRef.current[episodePos] ?? 0
      episodeMeta =
        `episode ${Math.max(0, episodePos + 1)}/${Math.max(1, episodeIdsRef.current.length)} ` +
        `(id=${episodeId})`
    }

    return {
      scrubMax: max,
      scrubValue: Math.min(max, Math.max(0, episodePos)),
      episodeMeta,
    }
  }, [tick])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="main-canvas" />
      </div>
      <aside className="panel">
        <h2>{title}</h2>

        <div className="control-row">
          <button
            type="button"
            onClick={() => {
              let episodePos = 0
              if (replayFilesRef.current.length > 0) {
                episodePos = Math.max(0, currentReplayFileIndexRef.current)
              } else {
                const starts = episodeStartIndicesRef.current
                for (let index = 0; index < starts.length; index += 1) {
                  if (starts[index] <= Math.max(0, currentIndexRef.current)) {
                    episodePos = index
                  } else {
                    break
                  }
                }
              }
              seekToEpisodePos(episodePos - 1)
            }}
          >
            Prev Ep
          </button>
          <button
            type="button"
            onClick={() => {
              setIsPlaying((value) => !value)
            }}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => {
              let episodePos = 0
              if (replayFilesRef.current.length > 0) {
                episodePos = Math.max(0, currentReplayFileIndexRef.current)
              } else {
                const starts = episodeStartIndicesRef.current
                for (let index = 0; index < starts.length; index += 1) {
                  if (starts[index] <= Math.max(0, currentIndexRef.current)) {
                    episodePos = index
                  } else {
                    break
                  }
                }
              }
              seekToEpisodePos(episodePos + 1)
            }}
          >
            Next Ep
          </button>
          <button
            type="button"
            onClick={() => {
              setFollowLive((value) => {
                const next = !value
                if (next && frameBufferRef.current.length > 0) {
                  renderIndex(frameBufferRef.current.length - 1)
                }
                return next
              })
            }}
          >
            {followLive ? 'Live: On' : 'Live: Off'}
          </button>
        </div>

        <label className="inline-label" htmlFor="snake-scrub">
          Episode
        </label>
        <input
          id="snake-scrub"
          type="range"
          min={0}
          max={controls.scrubMax}
          step={1}
          value={controls.scrubValue}
          onChange={(event) => {
            seekToEpisodePos(parseInt(event.target.value, 10) || 0)
          }}
        />
        <p className="subtle">{controls.episodeMeta}</p>

        <label className="inline-label" htmlFor="snake-speed">
          Speed
        </label>
        <select
          id="snake-speed"
          value={String(speed)}
          onChange={(event) => {
            setSpeed(parseNumber(event.target.value, 1))
          }}
        >
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>

        <button
          type="button"
          className="full"
          onClick={() => {
            setReconnectToken((value) => value + 1)
          }}
        >
          Reconnect
        </button>

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

function MazeViewer({ baseUrl, active }: { baseUrl: string; active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dimsRef = useRef<{ width: number; height: number } | null>(null)

  const [title, setTitle] = useState('Maze Race Live')
  const [status, setStatus] = useState('Waiting for frames...')
  const [error, setError] = useState('')
  const [reconnectToken, setReconnectToken] = useState(0)

  const drawFrame = useCallback((frame: GenericFrame) => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const obs = as3DObservation(frame.observation)
    if (obs.length < 5) {
      return
    }

    const walls = obs[0]
    const goal0 = obs[1]
    const goal1 = obs[2]
    const agent0 = obs[3]
    const agent1 = obs[4]

    const height = walls.length
    const width = height > 0 ? walls[0].length : 0
    if (!width || !height) {
      return
    }

    if (!dimsRef.current || dimsRef.current.width !== width || dimsRef.current.height !== height) {
      const maxSize = 720
      const cellSize = Math.max(10, Math.floor(Math.min(32, maxSize / Math.max(width, height))))
      canvas.width = width * cellSize
      canvas.height = height * cellSize
      dimsRef.current = { width, height }
    }

    const cellSize = Math.floor(canvas.width / width)

    ctx.fillStyle = MAZE_COLORS.bg
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    let a0x = 0
    let a0y = 0
    let a1x = 0
    let a1y = 0

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let color: string = walls[y]?.[x] > 0.5 ? MAZE_COLORS.wall : MAZE_COLORS.floor
        if (goal0[y]?.[x] > 0.5) {
          color = MAZE_COLORS.goal0
        } else if (goal1[y]?.[x] > 0.5) {
          color = MAZE_COLORS.goal1
        }
        ctx.fillStyle = color
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize)
        ctx.strokeStyle = MAZE_COLORS.grid
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize)

        if (agent0[y]?.[x] > 0.5) {
          a0x = x
          a0y = y
        }
        if (agent1[y]?.[x] > 0.5) {
          a1x = x
          a1y = y
        }
      }
    }

    const radius = Math.max(5, Math.floor(cellSize * 0.35))

    ctx.fillStyle = MAZE_COLORS.agent0
    ctx.beginPath()
    ctx.arc(a0x * cellSize + cellSize / 2, a0y * cellSize + cellSize / 2, radius, 0, Math.PI * 2)
    ctx.fill()

    ctx.fillStyle = MAZE_COLORS.agent1
    ctx.beginPath()
    ctx.arc(a1x * cellSize + cellSize / 2, a1y * cellSize + cellSize / 2, radius, 0, Math.PI * 2)
    ctx.fill()

    const info = frame.info ?? {}
    const winnerValue = (info.winner as unknown) ?? '?'
    const winner = Array.isArray(winnerValue) ? winnerValue[0] : winnerValue

    setStatus(
      [
        `step: ${frame.step ?? '?'}`,
        `done: ${frame.done ? 'yes' : 'no'}`,
        `winner: ${String(winner)}`,
        `reward: ${JSON.stringify(frame.rewards ?? {})}`,
      ].join('\n'),
    )
  }, [])

  useEffect(() => {
    if (!active) {
      return
    }

    setError('')
    setStatus('Connecting...')

    const source = new EventSource(resolveUrl(baseUrl, '/events'))

    source.onmessage = (event) => {
      let payload: EventPayload
      try {
        payload = JSON.parse(event.data) as EventPayload
      } catch {
        return
      }

      if (payload.type === 'meta') {
        const data = (payload.data ?? {}) as MetaPayload
        setTitle(data.title ?? 'Maze Race Live')
        return
      }

      if (payload.type === 'frame') {
        drawFrame((payload.data ?? {}) as GenericFrame)
      }
    }

    source.onerror = () => {
      setStatus('Connection lost. Check backend and reconnect.')
    }

    return () => {
      source.close()
    }
  }, [active, baseUrl, drawFrame, reconnectToken])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="main-canvas" />
      </div>
      <aside className="panel">
        <h2>{title}</h2>
        <button
          type="button"
          className="full"
          onClick={() => {
            setReconnectToken((value) => value + 1)
          }}
        >
          Reconnect
        </button>
        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

function BattleViewer({ baseUrl, active }: { baseUrl: string; active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const timerRef = useRef<number | null>(null)

  const [episodes, setEpisodes] = useState<string[]>([])
  const [selectedEpisode, setSelectedEpisode] = useState('')
  const [frames, setFrames] = useState<GenericFrame[]>([])
  const [metadata, setMetadata] = useState<Record<string, unknown>>({})
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [status, setStatus] = useState('Loading episodes...')
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)
  const [dims, setDims] = useState<BattleDims>({ width: 13, height: 13, cell: 28 })

  const stopTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const resizeCanvas = useCallback((width: number, height: number) => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const maxSize = 760
    const cell = Math.max(10, Math.floor(Math.min(34, maxSize / Math.max(width, height))))
    canvas.width = width * cell
    canvas.height = height * cell
    setDims({ width, height, cell })
  }, [])

  const drawFrame = useCallback(
    (index: number) => {
      const canvas = canvasRef.current
      if (!canvas) {
        return
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        return
      }
      if (frames.length === 0) {
        return
      }

      const frame = frames[Math.max(0, Math.min(frames.length - 1, index))]
      const obs = as3DObservation(frame.observation)
      if (obs.length < 5) {
        return
      }

      const a0 = obs[0]
      const a1 = obs[1]
      const hp0Map = obs[2]
      const hp1Map = obs[3]
      const stepMap = obs[4]

      const height = a0.length
      const width = height > 0 ? a0[0].length : 0
      if (!width || !height) {
        return
      }

      if (dims.width !== width || dims.height !== height) {
        resizeCanvas(width, height)
      }

      const cell = Math.floor(canvas.width / width)

      ctx.fillStyle = BATTLE_COLORS.bg
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          ctx.fillStyle = BATTLE_COLORS.grid
          ctx.fillRect(x * cell, y * cell, cell, cell)
          ctx.strokeStyle = BATTLE_COLORS.line
          ctx.strokeRect(x * cell, y * cell, cell, cell)
        }
      }

      const pos0 = findAgentPos(a0)
      const pos1 = findAgentPos(a1)

      const info = frame.info ?? {}
      const health = Array.isArray(info.health) ? info.health : [null, null]
      const hp0Norm = hp0Map?.[0]?.[0] ?? 0
      const hp1Norm = hp1Map?.[0]?.[0] ?? 0
      const maxHealth = parseNumber(metadata.max_health, 3) || 3
      const hp0 = health[0] == null ? Math.round(hp0Norm * maxHealth) : parseNumber(health[0], 0)
      const hp1 = health[1] == null ? Math.round(hp1Norm * maxHealth) : parseNumber(health[1], 0)
      const rewards = parseBattleReward(frame.rewards)

      const radius = Math.max(5, Math.floor(cell * 0.34))

      if (pos0) {
        ctx.fillStyle = hp0 > 0 ? BATTLE_COLORS.agent0 : BATTLE_COLORS.dead
        ctx.beginPath()
        ctx.arc(pos0.x * cell + cell / 2, pos0.y * cell + cell / 2, radius, 0, Math.PI * 2)
        ctx.fill()
      }
      if (pos1) {
        ctx.fillStyle = hp1 > 0 ? BATTLE_COLORS.agent1 : BATTLE_COLORS.dead
        ctx.beginPath()
        ctx.arc(pos1.x * cell + cell / 2, pos1.y * cell + cell / 2, radius, 0, Math.PI * 2)
        ctx.fill()
      }

      const winner = parseNumber(info.winner, -1)
      const winnerText =
        winner === 0 ? 'agent0' : winner === 1 ? 'agent1' : winner === -2 ? 'draw' : 'running'

      const step = parseNumber(frame.step, 0)
      const progress = parseNumber(stepMap?.[0]?.[0], 0)

      setStatus(
        [
          `episode: ${selectedEpisode}`,
          `frame: ${index + 1}/${frames.length} step: ${step} progress: ${progress.toFixed(3)}`,
          `winner: ${winnerText} done: ${String(Boolean(frame.done))}`,
          `hp: agent0=${hp0} agent1=${hp1}`,
          `reward: agent0=${rewards[0].toFixed(3)} agent1=${rewards[1].toFixed(3)}`,
          'keys: space=play/pause, left/right=frame',
        ].join('\n'),
      )
    },
    [dims.height, dims.width, frames, metadata.max_health, resizeCanvas, selectedEpisode],
  )

  const loadEpisode = useCallback(
    async (episodeName: string) => {
      if (!episodeName) {
        return
      }
      setError('')
      try {
        const response = await fetch(
          resolveUrl(baseUrl, `/api/episode?name=${encodeURIComponent(episodeName)}`),
          { cache: 'no-store' },
        )
        if (!response.ok) {
          throw new Error(`Unable to load ${episodeName} (${response.status})`)
        }

        const data = (await response.json()) as EpisodePayload
        const nextFrames = Array.isArray(data.frames) ? data.frames : []
        setFrames(nextFrames)
        setMetadata(data.metadata ?? {})
        setFrameIndex(0)
        setPlaying(false)

        if (nextFrames.length === 0) {
          setStatus(`No frames in ${episodeName}`)
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    },
    [baseUrl],
  )

  useEffect(() => {
    if (!active) {
      return
    }

    void (async () => {
      try {
        const response = await fetch(resolveUrl(baseUrl, '/api/episodes'), { cache: 'no-store' })
        if (!response.ok) {
          throw new Error(`Unable to list episodes (${response.status})`)
        }

        const data = (await response.json()) as { episodes?: unknown }
        const names = Array.isArray(data.episodes) ? data.episodes.map((value) => String(value)) : []
        setEpisodes(names)

        if (names.length === 0) {
          setSelectedEpisode('')
          setFrames([])
          setStatus('No replay JSON files found.')
          return
        }

        const first = names[0]
        setSelectedEpisode(first)
        await loadEpisode(first)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }, [active, baseUrl, loadEpisode, refreshToken])

  useEffect(() => {
    drawFrame(frameIndex)
  }, [drawFrame, frameIndex])

  useEffect(() => {
    stopTimer()
    if (!playing || !active || frames.length === 0) {
      return
    }

    const fps = 18 * speed
    const intervalMs = Math.max(16, Math.floor(1000 / fps))
    timerRef.current = window.setInterval(() => {
      setFrameIndex((current) => {
        const next = current + 1
        if (next >= frames.length) {
          setPlaying(false)
          return Math.max(frames.length - 1, 0)
        }
        return next
      })
    }, intervalMs)

    return () => {
      stopTimer()
    }
  }, [active, frames.length, playing, speed, stopTimer])

  useEffect(() => {
    if (!active) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === ' ') {
        event.preventDefault()
        if (frames.length > 0) {
          setPlaying((value) => !value)
        }
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        setFrameIndex((value) => Math.max(0, value - 1))
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        setFrameIndex((value) => Math.min(frames.length - 1, value + 1))
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [active, frames.length])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="main-canvas" />
      </div>
      <aside className="panel">
        <h2>Battle Grid Replay</h2>

        <label className="inline-label" htmlFor="battle-episode">
          Episode
        </label>
        <select
          id="battle-episode"
          value={selectedEpisode}
          onChange={(event) => {
            const next = event.target.value
            setSelectedEpisode(next)
            void loadEpisode(next)
          }}
          disabled={episodes.length === 0}
        >
          {episodes.map((episode) => (
            <option key={episode} value={episode}>
              {episode}
            </option>
          ))}
        </select>

        <div className="control-row compact">
          <button
            type="button"
            onClick={() => {
              setFrameIndex((value) => Math.max(0, value - 1))
            }}
            disabled={frames.length === 0}
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() => {
              if (frames.length > 0) {
                setPlaying((value) => !value)
              }
            }}
            disabled={frames.length === 0}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => {
              setFrameIndex((value) => Math.min(frames.length - 1, value + 1))
            }}
            disabled={frames.length === 0}
          >
            Next
          </button>
        </div>

        <label className="inline-label" htmlFor="battle-speed">
          Speed: {speed.toFixed(2)}x
        </label>
        <input
          id="battle-speed"
          type="range"
          min={0.25}
          max={4}
          step={0.25}
          value={speed}
          onChange={(event) => {
            setSpeed(parseNumber(event.target.value, 1))
          }}
        />

        <label className="inline-label" htmlFor="battle-seek">
          Frame
        </label>
        <input
          id="battle-seek"
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          step={1}
          value={Math.max(0, Math.min(frameIndex, Math.max(frames.length - 1, 0)))}
          onChange={(event) => {
            setFrameIndex(parseInt(event.target.value, 10) || 0)
          }}
          disabled={frames.length === 0}
        />

        <button
          type="button"
          className="full"
          onClick={() => {
            setRefreshToken((value) => value + 1)
          }}
        >
          Reload Episodes
        </button>

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

function App() {
  const [mode, setMode] = useState<ViewMode>('snake')
  const [snakeUrl, setSnakeUrl] = useState('/snake-web')
  const [mazeUrl, setMazeUrl] = useState('/maze-live')
  const [battleUrl, setBattleUrl] = useState('/battle-replay')

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Simverse Renderer</h1>
          <p>React frontend for live env rendering and replay playback</p>
        </div>

        <div className="header-controls">
          <label htmlFor="mode-select">Mode</label>
          <select
            id="mode-select"
            value={mode}
            onChange={(event) => {
              setMode(event.target.value as ViewMode)
            }}
          >
            <option value="snake">Snake Live / Web Render</option>
            <option value="maze">Maze Race Live</option>
            <option value="battle">Battle Grid Replay Web</option>
          </select>

          {mode === 'snake' ? (
            <input
              value={snakeUrl}
              onChange={(event) => {
                setSnakeUrl(event.target.value)
              }}
              placeholder="Snake backend base URL"
            />
          ) : null}

          {mode === 'maze' ? (
            <input
              value={mazeUrl}
              onChange={(event) => {
                setMazeUrl(event.target.value)
              }}
              placeholder="Maze backend base URL"
            />
          ) : null}

          {mode === 'battle' ? (
            <input
              value={battleUrl}
              onChange={(event) => {
                setBattleUrl(event.target.value)
              }}
              placeholder="Battle backend base URL"
            />
          ) : null}
        </div>
      </header>

      {mode === 'snake' ? <SnakeViewer baseUrl={snakeUrl} active /> : null}
      {mode === 'maze' ? <MazeViewer baseUrl={mazeUrl} active /> : null}
      {mode === 'battle' ? <BattleViewer baseUrl={battleUrl} active /> : null}

      <footer className="footnote">
        <p>
          Defaults are Vite proxy paths. Use absolute URLs only if your backend serves CORS headers.
        </p>
      </footer>
    </main>
  )
}

export default App
