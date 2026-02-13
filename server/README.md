# Server

Server-side modules used by the frontend live/replay UIs live here.

Single server entrypoint:
- `server/main.py` - central FastAPI app.

Routers:
- `server/snake/router.py` - snake endpoints mounted under `/snake`:
  - `GET /snake/`
  - `GET /snake/snapshot`
  - `GET /snake/events`
  - `POST /snake/live/meta`
  - `POST /snake/live/frame`
  - `GET /snake/replays`
  - `GET /snake/replays/{replay_id}`

Shared live stream service:
- `server/live/stream.py` - centralized in-memory live publish/subscribe broker used by routers.

Compatibility:
- Use `server/main.py` as the single server entrypoint.
