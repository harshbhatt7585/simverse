from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/snapshot")
def get_snapshot():
    return {"message": "Hello, World!"}


@app.get("/events")
def get_events():
    return {"message": "Hello, World!"}


@app.get("/replays")
def get_replays():
    return {"message": "Hello, World!"}


@app.get("/replays/{replay_id}")
def get_replay(replay_id: str):
    return {"message": "Hello, World!"}
