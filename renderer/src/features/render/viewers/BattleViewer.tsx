import { useCallback, useEffect, useRef, useState } from 'react'

import type { BattleDims, EpisodePayload, GenericFrame } from '../types'
import {
  BATTLE_COLORS,
  as3DObservation,
  findAgentPos,
  parseBattleReward,
  parseNumber,
  resolveUrl,
} from '../utils'

type BattleViewerProps = {
  baseUrl: string
}

function BattleViewer({ baseUrl }: BattleViewerProps) {
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
  }, [baseUrl, loadEpisode, refreshToken])

  useEffect(() => {
    drawFrame(frameIndex)
  }, [drawFrame, frameIndex])

  useEffect(() => {
    stopTimer()
    if (!playing || frames.length === 0) {
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
  }, [frames.length, playing, speed, stopTimer])

  useEffect(() => {
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
  }, [frames.length])

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

export default BattleViewer
