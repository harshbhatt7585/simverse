import { useCallback, useEffect, useRef, useState } from 'react'

import type { EventPayload, GenericFrame, MetaPayload } from '../types'
import { MAZE_COLORS, as3DObservation, resolveUrl } from '../utils'

type MazeViewerProps = {
  baseUrl: string
}

function MazeViewer({ baseUrl }: MazeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dimsRef = useRef<{ width: number; height: number } | null>(null)

  const [title, setTitle] = useState('Maze Race Live')
  const [status, setStatus] = useState('Connecting...')
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
  }, [baseUrl, drawFrame, reconnectToken])

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
      </aside>
    </div>
  )
}

export default MazeViewer
