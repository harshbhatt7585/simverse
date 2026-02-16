export default function Home() {
  return (
    <main className="minimal-shell">
      <h1 className="minimal-title">Build RL environments with Simverse</h1>
      <input
        className="minimal-input"
        type="text"
        placeholder="Describe your environment..."
        aria-label="Environment description"
      />
    </main>
  );
}
