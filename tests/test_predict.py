"""Tests for batch inference."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from models.predict import run_inference


def test_run_inference_writes_prediction_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pl.DataFrame(
        {
            "text": ["I love this product", "Terrible experience"],
            "lang": ["en", "en"],
            "topic": ["apple", "apple"],
        }
    ).write_parquet(input_path)

    mock_predictions = [
        {"sentiment_label": 1, "sentiment_score": 0.9, "confidence": 0.9},
        {"sentiment_label": 0, "sentiment_score": 0.1, "confidence": 0.85},
    ]

    with patch("models.predict.predict_texts", return_value=mock_predictions):
        run_inference(
            model_path=tmp_path / "model",
            input_path=input_path,
            output_path=output_path,
        )

    result = pl.read_parquet(output_path)
    assert "sentiment_label" in result.columns
    assert result.height == 2
    assert result["sentiment_label"].to_list() == [1, 0]
