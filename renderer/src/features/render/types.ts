export type ViewMode = 'snake' | 'maze' | 'battle'

export type Position = { x: number; y: number }

export type MetaPayload = {
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

export type EventPayload = {
  type?: string
  data?: unknown
}

export type GenericFrame = {
  step?: number
  episode?: unknown
  observation?: unknown
  rewards?: unknown
  info?: Record<string, unknown>
  done?: boolean
  _replay_file_index?: unknown
  _replay_file_name?: unknown
}

export type EpisodePayload = {
  metadata?: Record<string, unknown>
  frames?: GenericFrame[]
}

export type BattleReward = {
  0: number
  1: number
}

export type BattleDims = {
  width: number
  height: number
  cell: number
}
