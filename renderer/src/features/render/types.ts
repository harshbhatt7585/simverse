export type ViewMode = 'live' | 'replay'

export type GenericFrame = {
  step?: number
  episode?: unknown
  observation?: unknown
  rewards?: unknown
  info?: Record<string, unknown>
  done?: boolean
}

export type SnakeReplayFile = {
  episode?: number
  steps?: number
  frames?: GenericFrame[]
  [key: string]: unknown
}

export type SnakeReplayEpisode = {
  name: string
  data: SnakeReplayFile
}

export type SnakeReplaysResponse = {
  episodes: SnakeReplayEpisode[]
}
