"""FastAPI sentiment inference endpoint."""

from fastapi import FastAPI

app = FastAPI(title="Cross-lingual Sentiment API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
