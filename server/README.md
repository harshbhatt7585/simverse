# Server

Server-side modules used by the frontend live/replay UIs live here.

Current modules:
- `server/snake/live_server.py` - SSE/live frame server for snake rendering.
- `server/snake/web_render.py` - snake replay web server entrypoint.
- `server/snake/replays_api.py` - serves all snake replay JSON files via `GET /snake/replays/`.
