from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from dqa import core, retrieval

app = FastAPI(title="Data Quality Investigation Assistant", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.exception_handler(ValueError)
async def invalid_input(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class Investigation(BaseModel):
    current: str
    baseline: str
    question: str = Field(min_length=1, max_length=2000)
    generate: bool = False


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/datasets")
def datasets():
    return core.datasets()


@app.get("/api/profile")
def profile(dataset: str):
    return core.profile(dataset)


@app.get("/api/compare")
def compare(current: str, baseline: str):
    return core.compare(current, baseline)


@app.get("/api/anomalies")
def anomalies(current: str, baseline: str):
    return core.anomalies(current, baseline)


@app.get("/api/search")
def search(query: str):
    return retrieval.search(query)


@app.post("/api/investigate")
def investigate(body: Investigation):
    return retrieval.investigate(**body.model_dump())

