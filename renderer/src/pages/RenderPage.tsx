import { useState } from 'react'

import type { ViewMode } from '../features/render/types'
import BattleViewer from '../features/render/viewers/BattleViewer'
import MazeViewer from '../features/render/viewers/MazeViewer'
import SnakeViewer from '../features/render/viewers/SnakeViewer'

function RenderPage() {
  const [mode, setMode] = useState<ViewMode>('snake')
  const [snakeUrl, setSnakeUrl] = useState('/snake-web')
  const [mazeUrl, setMazeUrl] = useState('/maze-live')
  const [battleUrl, setBattleUrl] = useState('/battle-replay')

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Simverse Renderer</h1>
          <p>React frontend for live env rendering and replay playback</p>
        </div>

        <div className="header-controls">
          <label htmlFor="mode-select">Mode</label>
          <select
            id="mode-select"
            value={mode}
            onChange={(event) => {
              setMode(event.target.value as ViewMode)
            }}
          >
            <option value="snake">Snake Live / Web Render</option>
            <option value="maze">Maze Race Live</option>
            <option value="battle">Battle Grid Replay Web</option>
          </select>

          {mode === 'snake' ? (
            <input
              value={snakeUrl}
              onChange={(event) => {
                setSnakeUrl(event.target.value)
              }}
              placeholder="Snake backend base URL"
            />
          ) : null}

          {mode === 'maze' ? (
            <input
              value={mazeUrl}
              onChange={(event) => {
                setMazeUrl(event.target.value)
              }}
              placeholder="Maze backend base URL"
            />
          ) : null}

          {mode === 'battle' ? (
            <input
              value={battleUrl}
              onChange={(event) => {
                setBattleUrl(event.target.value)
              }}
              placeholder="Battle backend base URL"
            />
          ) : null}
        </div>
      </header>

      {mode === 'snake' ? <SnakeViewer baseUrl={snakeUrl} /> : null}
      {mode === 'maze' ? <MazeViewer baseUrl={mazeUrl} /> : null}
      {mode === 'battle' ? <BattleViewer baseUrl={battleUrl} /> : null}

      <footer className="footnote">
        <p>Defaults are Vite proxy paths. Use absolute URLs if your backend serves CORS headers.</p>
      </footer>
    </main>
  )
}

export default RenderPage
