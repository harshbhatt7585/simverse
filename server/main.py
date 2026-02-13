from __future__ import annotations

import os

from fastapi import FastAPI

from server.snake.router import router as snake_router

app = FastAPI(title="Simverse API")
app.include_router(snake_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "simverse-api", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SIMVERSE_API_HOST", os.getenv("SNAKE_RENDER_HOST", "127.0.0.1"))
    port = int(os.getenv("SIMVERSE_API_PORT", os.getenv("SNAKE_RENDER_PORT", "8770")))
    uvicorn.run("server.main:app", host=host, port=port, reload=False)
