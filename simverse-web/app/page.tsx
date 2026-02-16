export default function Home() {
  return (
    <main className="site-shell">
      <header className="topbar reveal-up">
        <p className="brand-mark">SIMVERSE</p>
        <nav className="topbar-links" aria-label="Primary">
          <a href="#features">Features</a>
          <a href="#catalog">Environments</a>
          <a href="#launch">Launch</a>
        </nav>
      </header>

      <section className="hero-grid reveal-up delay-1" id="launch">
        <div className="hero-copy">
          <p className="hero-kicker">Simulation infrastructure for RL teams</p>
          <h1 className="hero-title">
            Design training worlds in plain language and run them at cloud scale.
          </h1>
          <p className="hero-subtitle">
            Go from prompt to reproducible environments with built-in metrics,
            rollout logs, and experiment lineage.
          </p>

          <form className="hero-search" action="#" method="get">
            <input
              className="hero-search-input"
              type="search"
              name="q"
              placeholder="Try: multi-agent traffic junction with sparse rewards"
              aria-label="Search"
            />
            <button className="hero-search-button" type="submit">
              Generate
            </button>
          </form>
        </div>

        <aside className="hero-card" aria-label="Latest run metrics">
          <h2>Latest Experiment</h2>
          <p className="hero-card-title">battle_grid / curriculum_v4</p>
          <div className="metric-row">
            <span>Episodes</span>
            <strong>1.2M</strong>
          </div>
          <div className="metric-row">
            <span>Success Rate</span>
            <strong>84.7%</strong>
          </div>
          <div className="metric-row">
            <span>Throughput</span>
            <strong>7.8k steps/s</strong>
          </div>
          <button type="button" className="secondary-button">
            Open dashboard
          </button>
        </aside>
      </section>

      <section className="stats-row reveal-up delay-2" aria-label="Key stats">
        <article>
          <p>Active Runs</p>
          <h3>342</h3>
        </article>
        <article>
          <p>Environment Templates</p>
          <h3>91</h3>
        </article>
        <article>
          <p>Cloud Regions</p>
          <h3>12</h3>
        </article>
      </section>

      <section className="feature-grid reveal-up delay-3" id="features">
        <article className="feature-card">
          <h4>Prompt-to-Scene Compiler</h4>
          <p>
            Turn natural language into parameterized maps, entities, and reward
            logic.
          </p>
        </article>
        <article className="feature-card">
          <h4>Parallel Rollout Engine</h4>
          <p>
            Run synchronized training batches across CPU/GPU nodes with one
            config.
          </p>
        </article>
        <article className="feature-card">
          <h4>Replay + Debug Lens</h4>
          <p>
            Inspect trajectories frame-by-frame and trace policy decisions back
            to state inputs.
          </p>
        </article>
      </section>
    </main>
  );
}
