import { useEffect, useMemo, useRef, useState } from 'react'

import GameRenderer from './GameRenderer'
import type {
  GenericFrame,
  RenderGame,
  ReplayDetail,
  ReplaySummary,
  ReplaysResponse,
  ViewMode,
} from './types'
import { firstScalar, parseNumber, parseReward, resolveUrl } from './utils'

type ReplayProps = {
  game: RenderGame
  onGameChange: (game: RenderGame) => void
  viewMode: ViewMode
  onViewModeChange: (mode: ViewMode) => void
  replayDir: string
  onReplayDirChange: (nextDir: string) => void
  baseUrl: string
}

function Replay({
  game,
  onGameChange,
  viewMode,
  onViewModeChange,
  replayDir,
  onReplayDirChange,
  baseUrl,
}: ReplayProps) {
  type ReplayUpdateEvent = {
    replay_count?: number
    latest_replay_id?: string | null
    latest_frame_index?: number
  }

  const [episodes, setEpisodes] = useState<ReplaySummary[]>([])
  const [selectedReplayId, setSelectedReplayId] = useState('')
  const [selectedReplay, setSelectedReplay] = useState<ReplayDetail | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [status, setStatus] = useState('Loading replays...')
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)
  const [liveUpdateToken, setLiveUpdateToken] = useState(0)
  const [followLatest, setFollowLatest] = useState(viewMode === 'live')
  const lastReplayUpdateRef = useRef<ReplayUpdateEvent | null>(null)

  const selectedReplayName = useMemo(() => {
    if (selectedReplay?.name) {
      return selectedReplay.name
    }
    return episodes.find((episode) => episode.id === selectedReplayId)?.name ?? ''
  }, [episodes, selectedReplay, selectedReplayId])
  const selectedReplayIndex = useMemo(
    () => episodes.findIndex((episode) => episode.id === selectedReplayId),
    [episodes, selectedReplayId],
  )

  const frames = Array.isArray(selectedReplay?.data?.frames) ? selectedReplay.data.frames : []
  const currentFrame: GenericFrame | null =
    frames.length > 0 ? frames[Math.max(0, Math.min(frameIndex, frames.length - 1))] : null
  const [pendingReplayDir, setPendingReplayDir] = useState(replayDir)

  useEffect(() => {
    setPendingReplayDir(replayDir)
  }, [replayDir])

  const resolveApiPath = (path: string): string => {
    if (!replayDir.trim()) {
      return path
    }
    const searchParams = new URLSearchParams()
    searchParams.set('dir', replayDir.trim())
    const separator = path.includes('?') ? '&' : '?'
    return `${path}${separator}${searchParams.toString()}`
  }

  useEffect(() => {
    if (viewMode === 'live') {
      setFollowLatest(true)
      setPlaying(false)
      setRefreshToken((value) => value + 1)
      setLiveUpdateToken((value) => value + 1)
    }
  }, [viewMode])

  useEffect(() => {
    if (viewMode !== 'live') {
      return
    }
    const eventsUrl = resolveUrl(baseUrl, resolveApiPath('/replays/events'))
    const eventSource = new EventSource(eventsUrl)
    const onReplayUpdate = (event: MessageEvent) => {
      let payload: ReplayUpdateEvent = {}
      try {
        payload = JSON.parse(event.data) as ReplayUpdateEvent
      } catch (_err) {
        return
      }
      const previous = lastReplayUpdateRef.current
      lastReplayUpdateRef.current = payload
      const replayCount = typeof payload.replay_count === 'number' ? payload.replay_count : 0
      const latestReplayId =
        typeof payload.latest_replay_id === 'string' ? payload.latest_replay_id : ''
      const latestFrameIndex =
        typeof payload.latest_frame_index === 'number' ? payload.latest_frame_index : -1
      const replayChanged =
        !previous ||
        previous.replay_count !== replayCount ||
        previous.latest_replay_id !== latestReplayId
      const frameChanged = !previous || previous.latest_frame_index !== latestFrameIndex
      if (replayChanged) {
        setRefreshToken((value) => value + 1)
      }
      if (!latestReplayId) {
        return
      }
      setFollowLatest(true)
      setSelectedReplayId(latestReplayId)
      if (replayChanged || frameChanged) {
        setLiveUpdateToken((value) => value + 1)
      }
    }
    eventSource.addEventListener('replay_update', onReplayUpdate)
    eventSource.onerror = () => {
      // 20s polling remains active as fallback when SSE temporarily drops.
    }
    return () => {
      eventSource.removeEventListener('replay_update', onReplayUpdate)
      eventSource.close()
    }
  }, [baseUrl, replayDir, viewMode])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setRefreshToken((value) => value + 1)
    }, 20_000)

    return () => {
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    void (async () => {
      setError('')
      setStatus(followLatest ? 'Watching replay directory for new episodes...' : 'Loading replays...')
      try {
        const response = await fetch(resolveUrl(baseUrl, resolveApiPath('/replays/')), {
          cache: 'no-store',
        })
        if (!response.ok) {
          throw new Error(`Unable to load replays (${response.status})`)
        }

        const payload = (await response.json()) as ReplaysResponse
        const nextEpisodes = Array.isArray(payload.episodes)
          ? payload.episodes
              .map((episode) => {
                if (!episode || typeof episode.name !== 'string') {
                  return null
                }
                const fallbackId = episode.name.replace(/\.json$/i, '')
                const id =
                  typeof episode.id === 'string' && episode.id.length > 0 ? episode.id : fallbackId
                return { id, name: episode.name } satisfies ReplaySummary
              })
              .filter((episode): episode is ReplaySummary => episode !== null)
          : []

        setEpisodes(nextEpisodes)
        if (nextEpisodes.length === 0) {
          setSelectedReplayId('')
          setSelectedReplay(null)
          setFrameIndex(0)
          setStatus('No replay JSON files found yet.')
          return
        }

        const latestReplayId = nextEpisodes[nextEpisodes.length - 1]?.id ?? ''
        setSelectedReplayId((currentId) => {
          if (followLatest) {
            return latestReplayId
          }
          if (nextEpisodes.some((episode) => episode.id === currentId)) {
            return currentId
          }
          return latestReplayId
        })
        setStatus(
          followLatest
            ? `Watching ${nextEpisodes.length} replay files. Latest: ${latestReplayId}`
            : `Loaded ${nextEpisodes.length} replay files.`,
        )
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }, [baseUrl, followLatest, refreshToken, replayDir])

  const loadSelectedReplay = (targetReplayId: string, pinToLatestFrame: boolean) => {
    void (async () => {
      setError('')
      setStatus(`Loading replay ${targetReplayId}...`)
      try {
        const response = await fetch(
          resolveUrl(baseUrl, resolveApiPath(`/replays/${encodeURIComponent(targetReplayId)}`)),
          {
            cache: 'no-store',
          },
        )
        if (!response.ok) {
          throw new Error(`Unable to load replay ${targetReplayId} (${response.status})`)
        }

        const payload = (await response.json()) as ReplayDetail
        if (!payload || typeof payload.name !== 'string' || typeof payload.id !== 'string') {
          throw new Error('Invalid replay response format')
        }

        setSelectedReplay(payload)
        if (pinToLatestFrame) {
          const nextFrames = Array.isArray(payload.data?.frames) ? payload.data.frames : []
          setFrameIndex(Math.max(nextFrames.length - 1, 0))
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }

  useEffect(() => {
    if (!selectedReplayId) {
      setSelectedReplay(null)
      return
    }
    loadSelectedReplay(selectedReplayId, viewMode === 'live')
    if (viewMode !== 'live') {
      setFrameIndex(0)
    }
  }, [baseUrl, replayDir, selectedReplayId, viewMode])

  useEffect(() => {
    if (viewMode !== 'live' || !selectedReplayId) {
      return
    }
    loadSelectedReplay(selectedReplayId, true)
  }, [baseUrl, liveUpdateToken, replayDir, selectedReplayId, viewMode])

  const selectEpisodeByOffset = (offset: number, autoPlay = false) => {
    if (episodes.length === 0 || selectedReplayIndex < 0) {
      return
    }
    const nextIndex = Math.max(0, Math.min(episodes.length - 1, selectedReplayIndex + offset))
    const nextEpisode = episodes[nextIndex]
    if (!nextEpisode) {
      return
    }
    setFollowLatest(nextIndex === episodes.length - 1)
    setSelectedReplayId(nextEpisode.id)
    setFrameIndex(0)
    setPlaying(autoPlay)
  }

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
          const nextEpisode = episodes[selectedReplayIndex + 1]
          if (nextEpisode) {
            setSelectedReplayId(nextEpisode.id)
            return 0
          }
          setPlaying(false)
          return Math.max(frames.length - 1, 0)
        }
        return next
      })
    }, intervalMs)

    return () => {
      window.clearInterval(timer)
    }
  }, [episodes, frames.length, playing, selectedReplayIndex, speed])

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
        `game: ${game}`,
        `dir: ${replayDir || '(default)'}`,
        `file: ${selectedReplayName}`,
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
  }, [currentFrame, frameIndex, frames.length, game, replayDir, selectedReplayName])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <GameRenderer game={game} frame={currentFrame} />
      </div>
      <aside className="panel">
        <label className="inline-label" htmlFor="replay-game">
          Game
        </label>
        <div className="control-row compact">
          <button
            type="button"
            className={viewMode === 'live' ? 'active-mode' : ''}
            onClick={() => {
              onViewModeChange('live')
            }}
          >
            Live
          </button>
          <button
            type="button"
            className={viewMode === 'replay' ? 'active-mode' : ''}
            onClick={() => {
              onViewModeChange('replay')
            }}
          >
            Replay
          </button>
          <button
            type="button"
            onClick={() => {
              setRefreshToken((value) => value + 1)
              if (viewMode === 'live') {
                setLiveUpdateToken((value) => value + 1)
              }
            }}
          >
            Sync now
          </button>
        </div>
        <label className="inline-label" htmlFor="replay-dir">
          Replay Directory (optional)
        </label>
        <input
          id="replay-dir"
          type="text"
          placeholder="recordings/snake or /abs/path/to/replays"
          value={pendingReplayDir}
          onChange={(event) => {
            setPendingReplayDir(event.target.value)
          }}
        />
        <button
          type="button"
          onClick={() => {
            onReplayDirChange(pendingReplayDir.trim())
            setSelectedReplayId('')
            setSelectedReplay(null)
            setFrameIndex(0)
            setPlaying(false)
            setFollowLatest(true)
            setRefreshToken((value) => value + 1)
          }}
        >
          Use Directory
        </button>
        <select
          id="replay-game"
          value={game}
          onChange={(event) => {
            onGameChange(event.target.value as RenderGame)
            setSelectedReplayId('')
            setSelectedReplay(null)
            setFrameIndex(0)
            setPlaying(false)
            setFollowLatest(true)
            setRefreshToken((value) => value + 1)
          }}
        >
          <option value="snake">Snake</option>
          <option value="maze">Maze Race</option>
          <option value="battle-grid">Battle Grid</option>
        </select>

        <label className="inline-label" htmlFor="replay-episode">
          Replay File
        </label>
        <select
          id="replay-episode"
          value={selectedReplayId}
          onChange={(event) => {
            const nextReplayId = event.target.value
            const latestReplayId = episodes[episodes.length - 1]?.id ?? ''
            setFollowLatest(nextReplayId === latestReplayId)
            setSelectedReplayId(event.target.value)
            setFrameIndex(0)
            setPlaying(false)
          }}
          disabled={episodes.length === 0 || viewMode === 'live'}
        >
          {episodes.map((episode) => (
            <option key={episode.id} value={episode.id}>
              {episode.name}
            </option>
          ))}
        </select>

        <div className="control-row compact">
          <button
            type="button"
            disabled={episodes.length === 0 || viewMode === 'live'}
            onClick={() => {
              selectEpisodeByOffset(-1, false)
            }}
          >
            Prev
          </button>
          <button
            type="button"
            disabled={frames.length === 0 || viewMode === 'live'}
            onClick={() => {
              setPlaying((value) => !value)
            }}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            disabled={episodes.length === 0 || viewMode === 'live'}
            onClick={() => {
              selectEpisodeByOffset(1, true)
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
          disabled={viewMode === 'live'}
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
          disabled={frames.length === 0 || viewMode === 'live'}
        />

        <button
          type="button"
          className="full"
          onClick={() => {
            setFollowLatest(true)
            setRefreshToken((value) => value + 1)
          }}
        >
          {followLatest ? 'Following latest' : 'Follow latest'}
        </button>

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

export default Replay
