import { useEffect, useRef } from 'react'

import type { GenericFrame } from './types'
import { SNAKE_COLORS, as3DObservation } from './utils'

type SnakeRendererProps = {
  frame: GenericFrame | null
}

function SnakeRenderer({ frame }: SnakeRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dimsRef = useRef<{ width: number; height: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !frame) {
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

    ctx.fillStyle = SNAKE_COLORS.bg
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    let headX = -1
    let headY = -1
    let foodX = -1
    let foodY = -1

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
  }, [frame])

  return <canvas ref={canvasRef} className="main-canvas" />
}

export default SnakeRenderer
