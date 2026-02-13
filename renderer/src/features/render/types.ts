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

export type SnakeReplaySummary = {
  id: string
  name: string
}

export type SnakeReplaysResponse = {
  episodes: SnakeReplaySummary[]
}

export type SnakeReplayDetail = {
  id: string
  name: string
  data: SnakeReplayFile
}
