import { Link } from 'react-router-dom'

function HomePage() {
  return (
    <main className="landing-shell">
      <section className="landing-hero">
        <p className="landing-kicker">Simulation Platform</p>
        <h1>Simverse</h1>
        <p className="landing-lead">
          Unified environment playback for reinforcement learning runs. Review episodes, inspect
          frames, and compare game behavior from one renderer.
        </p>

        <div className="landing-actions">
          <Link className="landing-button primary" to="/render">
            Open Renderer
          </Link>
          <a className="landing-button ghost" href="#games">
            View Supported Games
          </a>
        </div>
      </section>

      <section className="landing-grid" id="games">
        <article className="landing-card">
          <h2>Snake</h2>
          <p>Replay endpoint: `/snake/replays`</p>
          <p>Live stream: `/snake/events`</p>
        </article>
        <article className="landing-card">
          <h2>Maze Runner</h2>
          <p>Replay endpoint: `/maze/replays`</p>
          <p>Live stream: `/maze/events`</p>
        </article>
      </section>
    </main>
  )
}

export default HomePage
