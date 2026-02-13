import type { BattleReward, Position } from './types'

export const SNAKE_COLORS = {
  bg: '#0e141b',
  floor: '#eef3f8',
  wall: '#3c4e62',
  food: '#d93f47',
  head: '#23924c',
  body: '#5acb85',
  grid: 'rgba(0,0,0,0.08)',
} as const

export const MAZE_COLORS = {
  bg: '#0f1218',
  floor: '#eef2f8',
  wall: '#39465e',
  goal0: '#4b87ff',
  goal1: '#ff8c5a',
  agent0: '#1f61d4',
  agent1: '#d4622a',
  grid: 'rgba(0,0,0,0.08)',
} as const

export const BATTLE_COLORS = {
  bg: '#111722',
  grid: '#f0f4fb',
  line: 'rgba(22,38,56,0.2)',
  agent0: '#2f74e6',
  agent1: '#e26a34',
  dead: '#7f8794',
} as const

export function firstScalar(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (Array.isArray(value) && value.length > 0) {
    return firstScalar(value[0], fallback)
  }
  return fallback
}

export function parseNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

export function parseReward(value: unknown): number {
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

export function parseBattleReward(value: unknown): BattleReward {
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

export function as2DLayer(value: unknown): number[][] {
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

export function as3DObservation(value: unknown): number[][][] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map((layer) => as2DLayer(layer))
}

export function findAgentPos(layer: number[][]): Position | null {
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

export function resolveUrl(baseUrl: string, maybeRelative: string): string {
  if (maybeRelative.startsWith('http://') || maybeRelative.startsWith('https://')) {
    return maybeRelative
  }
  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  const rel = maybeRelative.startsWith('/') ? maybeRelative : `/${maybeRelative}`
  return `${base}${rel}`
}
