import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { EventPayload, GenericFrame, MetaPayload } from '../types'
import { SNAKE_COLORS, as3DObservation, firstScalar, parseNumber, parseReward, resolveUrl } from '../utils'

type SnakeViewerProps = {
  baseUrl: string
}

function SnakeViewer({ baseUrl }: SnakeViewerProps) {
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
      drawFrame(frameBuffer[clamped])
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
  }, [baseUrl, bump, loadSnapshot, reconnectToken, renderIndex])

  useEffect(() => {
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
  }, [followLive, isPlaying, renderIndex, seekToEpisodePos, speed])

  const controls = useMemo(() => {
    // This forces controls recomputation when we mutate ref-based playback state.
    const renderTick = tick
    void renderTick

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

  const getCurrentEpisodePosition = useCallback(() => {
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
    return episodePos
  }, [])

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
              seekToEpisodePos(getCurrentEpisodePosition() - 1)
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
              seekToEpisodePos(getCurrentEpisodePosition() + 1)
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

export default SnakeViewer
