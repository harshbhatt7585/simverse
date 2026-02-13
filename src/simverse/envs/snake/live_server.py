from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Snake Live</title>
    <style>
      :root { color-scheme: dark; }
      body {
        margin: 0;
        padding: 22px;
        background: radial-gradient(circle at top, #1b2530, #0b1118);
        color: #e6edf3;
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        display: flex;
        gap: 24px;
        align-items: flex-start;
      }
      canvas {
        background: #0f141b;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
      }
      #panel {
        width: 320px;
        background: rgba(18, 24, 32, 0.85);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 14px 16px;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
      }
      #title { font-size: 20px; font-weight: 700; margin-bottom: 10px; color: #9dd2ff; }
      #status { font-size: 14px; line-height: 1.45; white-space: pre-line; }
    </style>
  </head>
  <body>
    <canvas id="grid"></canvas>
    <div id="panel">
      <div id="title">Snake Live</div>
      <div id="status">Waiting for frames...</div>
    </div>
    <script>
      const canvas = document.getElementById("grid");
      const ctx = canvas.getContext("2d");
      const status = document.getElementById("status");

      const colors = {
        bg: "#0e141b",
        floor: "#eef3f8",
        wall: "#3c4e62",
        food: "#d93f47",
        head: "#23924c",
        body: "#5acb85",
        grid: "rgba(0,0,0,0.08)",
      };

      let cellSize = 24;
      let lastDims = null;

      function resizeCanvas(width, height) {
        const maxSize = 760;
        cellSize = Math.max(10, Math.floor(Math.min(34, maxSize / Math.max(width, height))));
        canvas.width = width * cellSize;
        canvas.height = height * cellSize;
        lastDims = { width, height };
      }

      function parseReward(value) {
        if (typeof value === "number") return value;
        if (Array.isArray(value)) {
          let total = 0.0;
          for (const row of value) {
            if (row && typeof row.reward === "number") total += row.reward;
          }
          return total;
        }
        if (value && typeof value.reward === "number") return value.reward;
        return 0.0;
      }

      function firstScalar(value, fallback = 0) {
        if (typeof value === "number") return value;
        if (Array.isArray(value) && value.length) return firstScalar(value[0], fallback);
        return fallback;
      }

      function drawFrame(frame) {
        if (!frame || !frame.observation || frame.observation.length < 4) return;
        const obs = frame.observation;
        const walls = obs[0];
        const food = obs[1];
        const head = obs[2];
        const body = obs[3];
        const height = walls.length || 0;
        const width = height ? walls[0].length : 0;
        if (!width || !height) return;
        if (!lastDims || lastDims.width !== width || lastDims.height !== height) {
          resizeCanvas(width, height);
        }

        let headX = -1;
        let headY = -1;
        let foodX = -1;
        let foodY = -1;
        let bodyCount = 0;

        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            if (walls[y][x] > 0.5) {
              ctx.fillStyle = colors.wall;
            } else {
              ctx.fillStyle = colors.floor;
            }
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            ctx.strokeStyle = colors.grid;
            ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);

            if (food[y][x] > 0.5) { foodX = x; foodY = y; }
            if (body[y][x] > 0.5) bodyCount += 1;
            if (head[y][x] > 0.5) { headX = x; headY = y; }
          }
        }

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            if (body[y][x] > 0.5) {
              ctx.fillStyle = colors.body;
              ctx.fillRect(x * cellSize + 2, y * cellSize + 2, cellSize - 4, cellSize - 4);
            }
          }
        }
        if (headX >= 0) {
          ctx.fillStyle = colors.head;
          ctx.fillRect(headX * cellSize + 2, headY * cellSize + 2, cellSize - 4, cellSize - 4);
        }
        if (foodX >= 0) {
          ctx.fillStyle = colors.food;
          ctx.fillRect(foodX * cellSize + 2, foodY * cellSize + 2, cellSize - 4, cellSize - 4);
        }

        const info = frame.info || {};
        const reward = parseReward(frame.rewards);
        const episode = firstScalar(frame.episode, 0);
        const score = firstScalar(info.score, 0);
        const steps = firstScalar(info.steps, 0);
        const term = firstScalar(info.termination_reason, 0);
        const done = frame.done ? "yes" : "no";
        const lines = [
          `episode: ${episode}`,
          `step: ${frame.step ?? "?"}`,
          `done: ${done}`,
          `term: ${term}`,
          `reward: ${reward.toFixed(3)}`,
          `score: ${score}`,
          `steps: ${steps}`,
          `length: ${bodyCount + (headX >= 0 ? 1 : 0)}`,
          `head: (${headX}, ${headY})`,
          `food: (${foodX}, ${foodY})`,
        ];
        status.textContent = lines.join("\\n");
      }

      const source = new EventSource("/events");
      source.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "meta") {
          document.getElementById("title").textContent = payload.data?.title || "Snake Live";
          return;
        }
        if (payload.type === "frame") {
          drawFrame(payload.data);
        }
      };
    </script>
  </body>
</html>
"""


class LiveRenderServer:
    def __init__(
        self,
        output_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8766,
        title: str = "Snake Live",
        frame_stride: int = 1,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.title = title
        self.output_path = Path(output_path)
        self.frame_stride = max(int(frame_stride), 1)
        self._clients: set[queue.Queue[bytes]] = set()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._file = None
        self._meta_payload: Optional[bytes] = None

    def start(self) -> None:
        if self._server is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        self._running.set()
        server_state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def handle(self) -> None:
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    return

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/events"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    client_queue: queue.Queue[bytes] = queue.Queue(maxsize=5)
                    server_state._register_client(client_queue)
                    try:
                        if server_state._meta_payload is not None:
                            self._send_event(server_state._meta_payload)
                        while server_state._running.is_set():
                            try:
                                payload = client_queue.get(timeout=1.0)
                            except queue.Empty:
                                self._send_keepalive()
                                continue
                            self._send_event(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        server_state._unregister_client(client_queue)
                    return
                self.send_response(404)
                self.end_headers()

            def _send_event(self, payload: bytes) -> None:
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()

            def _send_keepalive(self) -> None:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _register_client(self, client_queue: queue.Queue[bytes]) -> None:
        with self._lock:
            self._clients.add(client_queue)

    def _unregister_client(self, client_queue: queue.Queue[bytes]) -> None:
        with self._lock:
            self._clients.discard(client_queue)

    def _broadcast(self, payload: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        for client_queue in clients:
            try:
                client_queue.put_nowait(payload)
            except queue.Full:
                pass

    def push_meta(self, meta: Dict[str, Any]) -> None:
        payload = json.dumps({"type": "meta", "data": meta}, separators=(",", ":")).encode("utf-8")
        self._meta_payload = payload
        self._broadcast(payload)

    def push_frame(self, frame: Dict[str, Any]) -> None:
        step = int(frame.get("step", 0))
        if step % self.frame_stride != 0:
            return
        if self._file is not None:
            self._file.write(json.dumps(frame) + "\n")
            self._file.flush()
        payload = json.dumps({"type": "frame", "data": frame}, separators=(",", ":")).encode(
            "utf-8"
        )
        self._broadcast(payload)

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
