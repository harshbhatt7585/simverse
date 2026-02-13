# Server

Server-side modules used by the frontend live/replay UIs live here.

Current modules:
- `server/snake/server.py` - unified snake backend (serves `GET /snake/replays/`).
- `server/snake/replays_api.py` - backward-compatible alias to `server/snake/server.py`.
- `server/snake/web_render.py` - backward-compatible alias to `server/snake/server.py`.
- `server/snake/live_server.py` - snake SSE/live frame server for training/live mode.
