"""Pydantic schemas for corpus rows, predictions, and API payloads."""

from datetime import datetime

from pydantic import BaseModel, Field


class CorpusRow(BaseModel):
    """Normalized row in a processed study corpus."""

    text: str
    lang: str
    source: str
    topic: str
    community: str = ""
    timestamp: datetime | None = None
    post_id: str = ""


class PredictionRow(BaseModel):
    """Model inference output for one text."""

    text: str
    lang: str
    topic: str
    sentiment_label: int = Field(ge=0, le=1)
    sentiment_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime | None = None
