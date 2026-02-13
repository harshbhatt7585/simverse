# Server

Server-side modules used by the frontend live/replay UIs live here.

Current module:
- `server/snake/render.py` - FastAPI server with:
  - `GET /`
  - `GET /snapshot`
  - `GET /events`
  - `GET /replays` and `GET /snake/replays`
  - `GET /replays/{replay_id}` and `GET /snake/replays/{replay_id}`
