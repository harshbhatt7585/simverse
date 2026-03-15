import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import Replay from '../features/render/Replay'
import type { RenderGame, ViewMode } from '../features/render/types'

function RenderPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialGame = useMemo<RenderGame>(() => {
    const requestedGame = searchParams.get('game')
    if (requestedGame === 'maze' || requestedGame === 'battle-grid' || requestedGame === 'snake') {
      return requestedGame
    }
    return 'snake'
  }, [searchParams])
  const initialMode = useMemo<ViewMode>(() => {
    const requestedMode = searchParams.get('mode')
    if (requestedMode === 'live' || requestedMode === 'replay') {
      return requestedMode
    }
    return 'replay'
  }, [searchParams])
  const initialReplayDir = useMemo(() => searchParams.get('dir') ?? '', [searchParams])
  const [game, setGame] = useState<RenderGame>(initialGame)
  const [mode, setMode] = useState<ViewMode>(initialMode)
  const [replayDir, setReplayDir] = useState(initialReplayDir)

  useEffect(() => {
    setGame(initialGame)
  }, [initialGame])
  useEffect(() => {
    setMode(initialMode)
  }, [initialMode])
  useEffect(() => {
    setReplayDir(initialReplayDir)
  }, [initialReplayDir])

  const handleGameChange = (nextGame: RenderGame) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', nextGame)
    nextParams.set('mode', mode)
    if (replayDir) {
      nextParams.set('dir', replayDir)
    } else {
      nextParams.delete('dir')
    }
    setSearchParams(nextParams, { replace: true })
    setGame(nextGame)
  }

  const handleModeChange = (nextMode: ViewMode) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', game)
    nextParams.set('mode', nextMode)
    if (replayDir) {
      nextParams.set('dir', replayDir)
    } else {
      nextParams.delete('dir')
    }
    setSearchParams(nextParams, { replace: true })
    setMode(nextMode)
  }

  const handleReplayDirChange = (nextDir: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', game)
    nextParams.set('mode', mode)
    if (nextDir) {
      nextParams.set('dir', nextDir)
    } else {
      nextParams.delete('dir')
    }
    setSearchParams(nextParams, { replace: true })
    setReplayDir(nextDir)
  }

  return (
    <main className="app-shell">
      <Replay
        game={game}
        onGameChange={handleGameChange}
        viewMode={mode}
        onViewModeChange={handleModeChange}
        replayDir={replayDir}
        onReplayDirChange={handleReplayDirChange}
        baseUrl={`/${game}`}
      />
    </main>
  )
}

export default RenderPage
