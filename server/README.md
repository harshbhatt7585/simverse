# Server

Server-side modules used by the frontend live/replay UIs live here.

Single server entrypoint:
- `server/main.py` - central FastAPI app.

Routers:
- `server/snake/router.py` - snake endpoints mounted under `/snake`:
  - `GET /snake/`
  - `GET /snake/snapshot`
  - `GET /snake/events`
  - `GET /snake/replays`
  - `GET /snake/replays/{replay_id}`

Compatibility:
- Use `server/main.py` as the single server entrypoint.
