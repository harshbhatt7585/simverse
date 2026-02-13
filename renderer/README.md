# Simverse Renderer (React)

This app is the frontend for Simverse live rendering and replay playback.

## Modes
- `Snake Live / Web Render`
  - Uses backend SSE endpoint: `/events`
  - Supports replay snapshots from `/snapshot` when running `simverse.envs.snake.web_render`
- `Maze Race Live`
  - Uses backend SSE endpoint: `/events`
- `Battle Grid Replay Web`
  - Uses `/api/episodes` and `/api/episode?name=...`

## Run
```bash
cd renderer
npm run dev
```

## Dev Proxy Defaults
Vite proxies these local paths to your backend services:
- `/snake-web` -> `http://127.0.0.1:8766`
- `/snake-live` -> `http://127.0.0.1:8766`
- `/maze-live` -> `http://127.0.0.1:8765`
- `/battle-replay` -> `http://127.0.0.1:8866`

Override targets with env vars if needed:
- `VITE_SNAKE_WEB_URL`
- `VITE_SNAKE_LIVE_URL`
- `VITE_MAZE_LIVE_URL`
- `VITE_BATTLE_REPLAY_URL`

Example:
```bash
VITE_SNAKE_WEB_URL=http://127.0.0.1:9000 npm run dev
```

## Backend examples
```bash
python -m simverse.envs.snake.web_render --replay-dir recordings/snake --host 127.0.0.1 --port 8766
python -m simverse.envs.maze_race.train --render-server on --render-port 8765
python -m simverse.envs.battle_grid.replay_web --replay-dir recordings/battle_grid --host 127.0.0.1 --port 8866
```
