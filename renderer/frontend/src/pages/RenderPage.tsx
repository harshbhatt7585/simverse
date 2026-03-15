import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import Replay from '../features/render/Replay'
import type { RenderGame } from '../features/render/types'

function RenderPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialGame = useMemo<RenderGame>(() => {
    const requestedGame = searchParams.get('game')
    if (requestedGame === 'maze' || requestedGame === 'battle-grid' || requestedGame === 'snake') {
      return requestedGame
    }
    return 'snake'
  }, [searchParams])
  const initialReplayDir = useMemo(() => searchParams.get('dir') ?? '', [searchParams])
  const [game, setGame] = useState<RenderGame>(initialGame)
  const [replayDir, setReplayDir] = useState(initialReplayDir)

  useEffect(() => {
    setGame(initialGame)
  }, [initialGame])
  useEffect(() => {
    setReplayDir(initialReplayDir)
  }, [initialReplayDir])

  const handleGameChange = (nextGame: RenderGame) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', nextGame)
    if (replayDir) {
      nextParams.set('dir', replayDir)
    } else {
      nextParams.delete('dir')
    }
    setSearchParams(nextParams, { replace: true })
    setGame(nextGame)
  }

  const handleReplayDirChange = (nextDir: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', game)
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
        replayDir={replayDir}
        onReplayDirChange={handleReplayDirChange}
        baseUrl={`/${game}`}
      />
    </main>
  )
}

export default RenderPage
