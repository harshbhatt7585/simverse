from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Battle Grid Replays</title>
    <style>
      :root {
        color-scheme: dark;
      }
      body {
        margin: 0;
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #1f2936, #0f141a);
        color: #e6edf3;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(320px, 780px) 360px;
        gap: 20px;
        padding: 20px;
      }
      canvas {
        width: 100%;
        max-width: 780px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
        background: #101722;
      }
      .panel {
        background: rgba(16, 22, 30, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
      }
      .title {
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 10px;
      }
      .controls {
        display: grid;
        gap: 10px;
      }
      .row {
        display: flex;
        gap: 8px;
        align-items: center;
      }
      button,
      select,
      input[type="range"] {
        font: inherit;
      }
      button,
      select {
        background: #1c2735;
        color: #ecf2fb;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 8px 10px;
      }
      button:hover {
        background: #253244;
      }
      #status {
        white-space: pre-line;
        line-height: 1.5;
        font-size: 14px;
        opacity: 0.95;
        margin-top: 8px;
      }
      #error {
        color: #ff8f8f;
        margin-top: 8px;
      }
      @media (max-width: 1100px) {
        .layout {
          grid-template-columns: 1fr;
        }
        canvas {
          max-width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <canvas id="arena"></canvas>
      <div class="panel">
        <h1 class="title">Battle Grid Replay</h1>
        <div class="controls">
          <div class="row">
            <label for="episode">Episode</label>
            <select id="episode"></select>
          </div>
          <div class="row">
            <button id="prev">Prev</button>
            <button id="play">Play</button>
            <button id="next">Next</button>
          </div>
          <div class="row">
            <label for="speed">Speed</label>
            <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1" />
            <span id="speedLabel">1.00x</span>
          </div>
          <div class="row">
            <label for="seek">Frame</label>
            <input id="seek" type="range" min="0" max="0" step="1" value="0" style="flex:1" />
          </div>
        </div>
        <div id="status">Loading episodes...</div>
        <div id="error"></div>
      </div>
    </div>
    <script>
      const canvas = document.getElementById("arena");
      const ctx = canvas.getContext("2d");
      const episodeSelect = document.getElementById("episode");
      const playBtn = document.getElementById("play");
      const prevBtn = document.getElementById("prev");
      const nextBtn = document.getElementById("next");
      const speedInput = document.getElementById("speed");
      const speedLabel = document.getElementById("speedLabel");
      const seekInput = document.getElementById("seek");
      const statusEl = document.getElementById("status");
      const errorEl = document.getElementById("error");

      const colors = {
        bg: "#111722",
        grid: "#f0f4fb",
        line: "rgba(22,38,56,0.2)",
        agent0: "#2f74e6",
        agent1: "#e26a34",
        dead: "#7f8794",
      };

      let episodes = [];
      let frames = [];
      let metadata = {};
      let frameIndex = 0;
      let playing = false;
      let speed = 1.0;
      let timerId = null;
      let dims = { width: 13, height: 13, cell: 28 };

      function setError(msg) {
        errorEl.textContent = msg || "";
      }

      function setStatus(text) {
        statusEl.textContent = text;
      }

      function parseReward(frameRewards) {
        const out = { 0: 0, 1: 0 };
        if (!Array.isArray(frameRewards)) return out;
        for (const item of frameRewards) {
          if (!item) continue;
          const id = Number(item.agent_id);
          if (id === 0 || id === 1) out[id] = Number(item.reward) || 0;
        }
        return out;
      }

      function findAgentPos(layer) {
        if (!Array.isArray(layer)) return null;
        for (let y = 0; y < layer.length; y++) {
          for (let x = 0; x < layer[y].length; x++) {
            if (Number(layer[y][x]) > 0.5) return { x, y };
          }
        }
        return null;
      }

      function resizeCanvas(width, height) {
        const maxSize = 760;
        const cell = Math.max(10, Math.floor(Math.min(34, maxSize / Math.max(width, height))));
        dims = { width, height, cell };
        canvas.width = width * cell;
        canvas.height = height * cell;
      }

      function drawFrame() {
        if (!frames.length) return;
        const frame = frames[frameIndex];
        const obs = frame.observation;
        if (!Array.isArray(obs) || obs.length < 5) return;
        const a0 = obs[0];
        const a1 = obs[1];
        const hp0Map = obs[2];
        const hp1Map = obs[3];
        const stepMap = obs[4];
        const height = Array.isArray(a0) ? a0.length : 0;
        const width = height && Array.isArray(a0[0]) ? a0[0].length : 0;
        if (!width || !height) return;

        if (dims.width !== width || dims.height !== height) {
          resizeCanvas(width, height);
        }

        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            ctx.fillStyle = colors.grid;
            ctx.fillRect(x * dims.cell, y * dims.cell, dims.cell, dims.cell);
            ctx.strokeStyle = colors.line;
            ctx.strokeRect(x * dims.cell, y * dims.cell, dims.cell, dims.cell);
          }
        }

        const pos0 = findAgentPos(a0);
        const pos1 = findAgentPos(a1);
        const info = frame.info || {};
        const hp = Array.isArray(info.health) ? info.health : [null, null];
        const hp0Norm = hp0Map?.[0]?.[0] ?? 0;
        const hp1Norm = hp1Map?.[0]?.[0] ?? 0;
        const maxHealth = Number(metadata.max_health || 3) || 3;
        const hp0 = hp[0] == null ? Math.round(hp0Norm * maxHealth) : hp[0];
        const hp1 = hp[1] == null ? Math.round(hp1Norm * maxHealth) : hp[1];
        const r = parseReward(frame.rewards);

        const radius = Math.max(5, Math.floor(dims.cell * 0.34));
        if (pos0) {
          ctx.fillStyle = hp0 > 0 ? colors.agent0 : colors.dead;
          ctx.beginPath();
          ctx.arc(
            pos0.x * dims.cell + dims.cell / 2,
            pos0.y * dims.cell + dims.cell / 2,
            radius,
            0,
            Math.PI * 2
          );
          ctx.fill();
        }
        if (pos1) {
          ctx.fillStyle = hp1 > 0 ? colors.agent1 : colors.dead;
          ctx.beginPath();
          ctx.arc(
            pos1.x * dims.cell + dims.cell / 2,
            pos1.y * dims.cell + dims.cell / 2,
            radius,
            0,
            Math.PI * 2
          );
          ctx.fill();
        }

        const winner = info.winner;
        const winnerText =
          winner === 0 ? "agent0" : winner === 1 ? "agent1" : winner === -2 ? "draw" : "running";
        const step = Number(frame.step || 0);
        const progress = Number(stepMap?.[0]?.[0] ?? 0);
        const status = [
          `episode: ${episodeSelect.value}`,
          `frame: ${frameIndex + 1}/${frames.length} ` +
            `step: ${step} progress: ${progress.toFixed(3)}`,
          `winner: ${winnerText}    done: ${Boolean(frame.done)}`,
          `hp: agent0=${hp0} agent1=${hp1}`,
          `reward: agent0=${r[0].toFixed(3)} agent1=${r[1].toFixed(3)}`,
          "keys: space=play/pause, left/right=frame",
        ];
        setStatus(status.join("\\n"));
        seekInput.value = String(frameIndex);
      }

      function stopTimer() {
        if (timerId !== null) {
          window.clearInterval(timerId);
          timerId = null;
        }
      }

      function startTimer() {
        stopTimer();
        const fps = 18 * speed;
        const intervalMs = Math.max(16, Math.floor(1000 / fps));
        timerId = window.setInterval(() => {
          if (!playing) return;
          frameIndex += 1;
          if (frameIndex >= frames.length) {
            frameIndex = frames.length - 1;
            playing = false;
            playBtn.textContent = "Play";
            stopTimer();
          }
          drawFrame();
        }, intervalMs);
      }

      function setPlaying(next) {
        playing = next;
        playBtn.textContent = playing ? "Pause" : "Play";
        if (playing) startTimer();
        else stopTimer();
      }

      async function loadEpisodes() {
        const res = await fetch("/api/episodes");
        if (!res.ok) throw new Error(`Unable to list episodes (${res.status})`);
        const data = await res.json();
        episodes = Array.isArray(data.episodes) ? data.episodes : [];
        episodeSelect.innerHTML = "";
        for (const name of episodes) {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          episodeSelect.appendChild(opt);
        }
        if (!episodes.length) {
          setStatus("No replay JSON files found.");
          return false;
        }
        return true;
      }

      async function loadEpisode(name) {
        setError("");
        const url = `/api/episode?name=${encodeURIComponent(name)}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Unable to load ${name} (${res.status})`);
        const data = await res.json();
        frames = Array.isArray(data.frames) ? data.frames : [];
        metadata = data.metadata || {};
        frameIndex = 0;
        seekInput.min = "0";
        seekInput.max = String(Math.max(frames.length - 1, 0));
        seekInput.value = "0";
        setPlaying(false);
        if (frames.length) drawFrame();
        else setStatus(`No frames in ${name}`);
      }

      async function bootstrap() {
        try {
          const ok = await loadEpisodes();
          if (!ok) return;
          await loadEpisode(episodeSelect.value);
        } catch (err) {
          setError(String(err));
        }
      }

      playBtn.addEventListener("click", () => {
        if (!frames.length) return;
        setPlaying(!playing);
      });
      prevBtn.addEventListener("click", () => {
        if (!frames.length) return;
        frameIndex = Math.max(0, frameIndex - 1);
        drawFrame();
      });
      nextBtn.addEventListener("click", () => {
        if (!frames.length) return;
        frameIndex = Math.min(frames.length - 1, frameIndex + 1);
        drawFrame();
      });
      speedInput.addEventListener("input", () => {
        speed = Number(speedInput.value) || 1;
        speedLabel.textContent = `${speed.toFixed(2)}x`;
        if (playing) startTimer();
      });
      seekInput.addEventListener("input", () => {
        if (!frames.length) return;
        frameIndex = Math.max(0, Math.min(frames.length - 1, Number(seekInput.value) || 0));
        drawFrame();
      });
      episodeSelect.addEventListener("change", async () => {
        try {
          await loadEpisode(episodeSelect.value);
        } catch (err) {
          setError(String(err));
        }
      });
      window.addEventListener("keydown", (event) => {
        if (event.key === " ") {
          event.preventDefault();
          if (frames.length) setPlaying(!playing);
        } else if (event.key === "ArrowLeft") {
          event.preventDefault();
          if (frames.length) {
            frameIndex = Math.max(0, frameIndex - 1);
            drawFrame();
          }
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          if (frames.length) {
            frameIndex = Math.min(frames.length - 1, frameIndex + 1);
            drawFrame();
          }
        }
      });

      bootstrap();
    </script>
  </body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _default_replay_dir() -> str:
    candidates = [
        Path("recording/battle"),
        Path("recordings/battle"),
        Path("recordings/battle_grid"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "recordings/battle_grid"


class BattleGridReplayServer:
    def __init__(
        self,
        replay_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 8866,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None

    def _list_episode_files(self) -> list[str]:
        if not self.replay_dir.exists():
            return []
        files = []
        for path in sorted(self.replay_dir.glob("*.json")):
            if path.name.startswith("episode_"):
                files.append(path.name)
        return files

    def _safe_episode_path(self, file_name: str) -> Path | None:
        candidate = (self.replay_dir / file_name).resolve()
        replay_root = self.replay_dir.resolve()
        if replay_root not in candidate.parents and candidate != replay_root:
            return None
        if candidate.suffix != ".json" or not candidate.is_file():
            return None
        return candidate

    def start(self) -> None:
        server_state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/api/episodes":
                    _json_response(self, 200, {"episodes": server_state._list_episode_files()})
                    return

                if path == "/api/episode":
                    qs = parse_qs(parsed.query)
                    file_name = (qs.get("name") or [""])[0]
                    if not file_name:
                        _json_response(self, 400, {"error": "missing episode name"})
                        return
                    replay_path = server_state._safe_episode_path(file_name)
                    if replay_path is None:
                        _json_response(self, 404, {"error": "episode not found"})
                        return
                    try:
                        payload = json.loads(replay_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        _json_response(self, 500, {"error": "invalid json"})
                        return
                    _json_response(self, 200, payload)
                    return

                _json_response(self, 404, {"error": "not found"})

        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"Battle Grid replay web: http://{self.host}:{self.port}")
        print(f"Replay dir: {self.replay_dir}")
        self._server.serve_forever()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Battle Grid replay JSON files in browser")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=_default_replay_dir(),
        help="Directory containing episode_*.json files",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8866)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = BattleGridReplayServer(replay_dir=args.replay_dir, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
