import { useState } from 'react'

import type { ViewMode } from '../features/render/types'
import Live from '../features/render/Live'
import Replay from '../features/render/Replay'

function RenderPage() {
  const [mode, setMode] = useState<ViewMode>('replay')
  const [replayApiUrl, setReplayApiUrl] = useState('/snake')

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
            <option value="live">Live</option>
            <option value="replay">Replay</option>
          </select>

          {mode === 'replay' ? (
            <input
              value={replayApiUrl}
              onChange={(event) => {
                setReplayApiUrl(event.target.value)
              }}
              placeholder="Replay API base URL (example: /snake)"
            />
          ) : null}
        </div>
      </header>

      {mode === 'live' ? <Live /> : null}
      {mode === 'replay' ? <Replay baseUrl={replayApiUrl} /> : null}

      <footer className="footnote">
        <p>Replay uses {`/replays`} and {`/replays/{id}`} under the provided base URL.</p>
      </footer>
    </main>
  )
}

export default RenderPage
