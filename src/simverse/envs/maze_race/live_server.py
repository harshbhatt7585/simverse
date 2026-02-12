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
    <title>Maze Race Live</title>
    <style>
      :root {
        color-scheme: dark;
      }
      body {
        margin: 0;
        padding: 24px;
        background: radial-gradient(circle at top, #1f2430, #0c0f14);
        color: #e6edf3;
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        display: flex;
        gap: 32px;
        align-items: flex-start;
      }
      #panel {
        max-width: 320px;
        background: rgba(18, 22, 29, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
      }
      #title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
      }
      #status {
        font-size: 14px;
        line-height: 1.4;
        opacity: 0.9;
        white-space: pre-line;
      }
      canvas {
        background: #10131a;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
      }
    </style>
  </head>
  <body>
    <canvas id="maze"></canvas>
    <div id="panel">
      <div id="title">Maze Race Live</div>
      <div id="status">Waiting for frames...</div>
    </div>
    <script>
      const canvas = document.getElementById("maze");
      const ctx = canvas.getContext("2d");
      const status = document.getElementById("status");

      const colors = {
        bg: "#0f1218",
        floor: "#eef2f8",
        wall: "#39465e",
        goal0: "#4b87ff",
        goal1: "#ff8c5a",
        agent0: "#1f61d4",
        agent1: "#d4622a",
        grid: "rgba(0,0,0,0.08)",
      };

      let cellSize = 24;
      let lastDims = null;

      function resizeCanvas(width, height) {
        const maxSize = 720;
        cellSize = Math.max(10, Math.floor(Math.min(32, maxSize / Math.max(width, height))));
        canvas.width = width * cellSize;
        canvas.height = height * cellSize;
        lastDims = { width, height };
      }

      function drawFrame(frame) {
        if (!frame || !frame.observation || frame.observation.length < 5) {
          return;
        }
        const obs = frame.observation;
        const walls = obs[0];
        const goal0 = obs[1];
        const goal1 = obs[2];
        const agent0 = obs[3];
        const agent1 = obs[4];

        const height = walls.length || 0;
        const width = height ? walls[0].length : 0;
        if (!width || !height) {
          return;
        }
        if (!lastDims || lastDims.width !== width || lastDims.height !== height) {
          resizeCanvas(width, height);
        }

        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let a0x = 0;
        let a0y = 0;
        let a1x = 0;
        let a1y = 0;

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            let color = walls[y][x] > 0.5 ? colors.wall : colors.floor;
            if (goal0[y][x] > 0.5) {
              color = colors.goal0;
            } else if (goal1[y][x] > 0.5) {
              color = colors.goal1;
            }
            ctx.fillStyle = color;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            ctx.strokeStyle = colors.grid;
            ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);

            if (agent0[y][x] > 0.5) {
              a0x = x;
              a0y = y;
            }
            if (agent1[y][x] > 0.5) {
              a1x = x;
              a1y = y;
            }
          }
        }

        const radius = Math.max(5, Math.floor(cellSize * 0.35));
        ctx.fillStyle = colors.agent0;
        ctx.beginPath();
        ctx.arc(
          a0x * cellSize + cellSize / 2,
          a0y * cellSize + cellSize / 2,
          radius,
          0,
          Math.PI * 2
        );
        ctx.fill();

        ctx.fillStyle = colors.agent1;
        ctx.beginPath();
        ctx.arc(
          a1x * cellSize + cellSize / 2,
          a1y * cellSize + cellSize / 2,
          radius,
          0,
          Math.PI * 2
        );
        ctx.fill();

        const info = frame.info || {};
        const rewards = frame.rewards || {};
        const winner = info.winner === undefined ? "?" : info.winner;
        const lines = [
          `step: ${frame.step ?? "?"}`,
          `done: ${frame.done ? "yes" : "no"}`,
          `winner: ${winner}`,
          `reward: ${JSON.stringify(rewards)}`,
        ];
        status.textContent = lines.join("\\n");
      }

      const source = new EventSource("/events");
      source.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "meta") {
          const title = payload.data?.title || "Maze Race Live";
          document.getElementById("title").textContent = title;
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
        port: int = 8765,
        title: str = "Maze Race Live",
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

            def handle(self) -> None:  # noqa: D401
                """Handle a single connection; ignore resets from closed clients."""
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
                self.wfile.write(b"data: " + payload + b"\\n\\n")
                self.wfile.flush()

            def _send_keepalive(self) -> None:
                self.wfile.write(b": ping\\n\\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def push_meta(self, metadata: Dict[str, Any]) -> None:
        payload = {"type": "meta", "data": metadata}
        payload_json = json.dumps(payload).encode("utf-8")
        self._meta_payload = payload_json
        self._write_line(payload_json)
        self._broadcast(payload_json)

    def push_frame(self, frame: Dict[str, Any]) -> None:
        step = int(frame.get("step", 0))
        if step % self.frame_stride != 0:
            return
        payload = {"type": "frame", "data": frame}
        payload_json = json.dumps(payload).encode("utf-8")
        self._write_line(payload_json)
        self._broadcast(payload_json)

    def _register_client(self, client_queue: queue.Queue[bytes]) -> None:
        with self._lock:
            self._clients.add(client_queue)

    def _unregister_client(self, client_queue: queue.Queue[bytes]) -> None:
        with self._lock:
            self._clients.discard(client_queue)

    def _broadcast(self, payload: bytes) -> None:
        with self._lock:
            for client_queue in list(self._clients):
                if client_queue.full():
                    try:
                        client_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    continue

    def _write_line(self, payload: bytes) -> None:
        if self._file is None:
            return
        self._file.write(payload.decode("utf-8") + "\n")
        self._file.flush()

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "LiveRenderServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
