export default function Home() {
  return (
    <main className="landing">
      <section className="hero">
        <p className="eyebrow">Simulation Platform</p>
        <h1>Simverse</h1>
        <p className="lead">
          A unified surface for environment playback, episode review, and live stream inspection
          across multiple RL games.
        </p>
        <div className="hero-actions">
          <a className="btn btn-primary" href="#games">
            Explore Games
          </a>
          <a className="btn btn-secondary" href="#platform">
            Platform Details
          </a>
        </div>
      </section>

      <section className="stats" id="platform">
        <article className="stat-card">
          <h2>Single API</h2>
          <p>Run one server and route by namespace (`/snake`, `/maze`) for replay and live data.</p>
        </article>
        <article className="stat-card">
          <h2>Shared Renderer</h2>
          <p>One centralized canvas renderer with per-game adapters for fast feature expansion.</p>
        </article>
        <article className="stat-card">
          <h2>Replay + Live</h2>
          <p>Switch from offline episode playback to SSE-based live streams without changing tools.</p>
        </article>
      </section>

      <section className="games" id="games">
        <article className="game-card">
          <h3>Snake</h3>
          <p>Replay: `/snake/replays`</p>
          <p>Live: `/snake/events`</p>
        </article>
        <article className="game-card">
          <h3>Maze Runner</h3>
          <p>Replay: `/maze/replays`</p>
          <p>Live: `/maze/events`</p>
        </article>
      </section>
    </main>
  );
}
