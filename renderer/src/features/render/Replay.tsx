import { useEffect, useMemo, useState } from 'react'

import SnakeRenderer from './SnakeRenderer'
import type { GenericFrame, SnakeReplayEpisode, SnakeReplaysResponse } from './types'
import { firstScalar, parseNumber, parseReward, resolveUrl } from './utils'

type ReplayProps = {
  baseUrl: string
}

function Replay({ baseUrl }: ReplayProps) {
  const [episodes, setEpisodes] = useState<SnakeReplayEpisode[]>([])
  const [selectedName, setSelectedName] = useState('')
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [status, setStatus] = useState('Loading replays...')
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)

  const selectedEpisode = useMemo(
    () => episodes.find((episode) => episode.name === selectedName) ?? null,
    [episodes, selectedName],
  )
  const frames = selectedEpisode?.data.frames ?? []
  const currentFrame: GenericFrame | null =
    frames.length > 0 ? frames[Math.max(0, Math.min(frameIndex, frames.length - 1))] : null

  useEffect(() => {
    void (async () => {
      setError('')
      setStatus('Loading replays...')
      try {
        const response = await fetch(resolveUrl(baseUrl, '/replays/'), { cache: 'no-store' })
        if (!response.ok) {
          throw new Error(`Unable to load replays (${response.status})`)
        }

        const payload = (await response.json()) as SnakeReplaysResponse
        const nextEpisodes = Array.isArray(payload.episodes)
          ? payload.episodes.filter((episode) => episode && typeof episode.name === 'string')
          : []

        setEpisodes(nextEpisodes)
        if (nextEpisodes.length === 0) {
          setSelectedName('')
          setFrameIndex(0)
          setStatus('No replay JSON files found.')
          return
        }

        setSelectedName(nextEpisodes[0].name)
        setFrameIndex(0)
        setStatus(`Loaded ${nextEpisodes.length} replay files.`)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }, [baseUrl, refreshToken])

  useEffect(() => {
    if (frameIndex >= frames.length) {
      setFrameIndex(Math.max(0, frames.length - 1))
    }
  }, [frameIndex, frames.length])

  useEffect(() => {
    if (!playing || frames.length === 0) {
      return
    }

    const fps = 18 * speed
    const intervalMs = Math.max(16, Math.floor(1000 / fps))
    const timer = window.setInterval(() => {
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
      window.clearInterval(timer)
    }
  }, [frames.length, playing, speed])

  useEffect(() => {
    if (!currentFrame) {
      return
    }

    const info = currentFrame.info ?? {}
    const reward = parseReward(currentFrame.rewards)
    const episode = firstScalar(currentFrame.episode, 0)
    const score = firstScalar(info.score, 0)
    const steps = firstScalar(info.steps, 0)
    const term = firstScalar(info.termination_reason, 0)
    const done = currentFrame.done ? 'yes' : 'no'

    setStatus(
      [
        `file: ${selectedName}`,
        `episode: ${episode}`,
        `frame: ${frameIndex + 1}/${frames.length}`,
        `step: ${currentFrame.step ?? '?'}`,
        `done: ${done}`,
        `term: ${term}`,
        `reward: ${reward.toFixed(3)}`,
        `score: ${score}`,
        `steps: ${steps}`,
      ].join('\n'),
    )
  }, [currentFrame, frameIndex, frames.length, selectedName])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <SnakeRenderer frame={currentFrame} />
      </div>
      <aside className="panel">
        <h2>Replay</h2>

        <label className="inline-label" htmlFor="replay-episode">
          Replay File
        </label>
        <select
          id="replay-episode"
          value={selectedName}
          onChange={(event) => {
            setSelectedName(event.target.value)
            setFrameIndex(0)
            setPlaying(false)
          }}
          disabled={episodes.length === 0}
        >
          {episodes.map((episode) => (
            <option key={episode.name} value={episode.name}>
              {episode.name}
            </option>
          ))}
        </select>

        <div className="control-row compact">
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setFrameIndex((value) => Math.max(0, value - 1))
            }}
          >
            Prev
          </button>
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setPlaying((value) => !value)
            }}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setFrameIndex((value) => Math.min(frames.length - 1, value + 1))
            }}
          >
            Next
          </button>
        </div>

        <label className="inline-label" htmlFor="replay-speed">
          Speed: {speed.toFixed(2)}x
        </label>
        <input
          id="replay-speed"
          type="range"
          min={0.25}
          max={4}
          step={0.25}
          value={speed}
          onChange={(event) => {
            setSpeed(parseNumber(event.target.value, 1))
          }}
        />

        <label className="inline-label" htmlFor="replay-seek">
          Frame
        </label>
        <input
          id="replay-seek"
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
          Reload
        </button>

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

export default Replay
