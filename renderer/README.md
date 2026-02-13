# Simverse Renderer (React)

This app is the frontend for Simverse snake replay playback.

## Modes
- `Live`
  - Placeholder only for now.
- `Replay`
  - Fetches all replay files from `/snake/replays/`.
  - Uses `SnakeRenderer` to display frames.

## Run
```bash
cd renderer
npm run dev
```

Open `http://localhost:5173/render`.

## Dev Proxy Defaults
Vite proxies:
- `/snake` -> `http://127.0.0.1:8770`

Override target with:
- `VITE_SNAKE_API_URL`

Example:
```bash
VITE_SNAKE_API_URL=http://127.0.0.1:9000 npm run dev
```

## Backend API
```bash
SNAKE_REPLAY_DIR=recordings/snake SNAKE_RENDER_HOST=127.0.0.1 SNAKE_RENDER_PORT=8770 python -m server.snake.render
```
