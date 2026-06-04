"""Batch inference over study corpora → parquet output."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_SEQUENCE_LENGTH
from data.loaders import load_corpus_frame
from data.preprocessing import truncate_to_max_tokens

logger = logging.getLogger(__name__)


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def predict_texts(
    texts: list[str],
    model_path: Path,
    *,
    batch_size: int = 32,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[dict[str, float | int]]:
    """Run sentiment classification on a list of texts.

    Args:
        texts: Input strings to classify.
        model_path: Directory containing a fine-tuned checkpoint.
        batch_size: Inference batch size.
        max_length: Token truncation limit.

    Returns:
        List of dicts with ``sentiment_label``, ``sentiment_score``, ``confidence``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    truncated = [
        truncate_to_max_tokens(text, tokenizer, max_length=max_length) for text in texts
    ]
    outputs: list[dict[str, float | int]] = []

    for batch in _batched(truncated, batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for probs in probabilities:
            label = int(probs.argmax())
            confidence = float(probs.max())
            sentiment_score = float(probs[1])
            outputs.append(
                {
                    "sentiment_label": label,
                    "sentiment_score": sentiment_score,
                    "confidence": confidence,
                }
            )
    return outputs


def run_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int = 32,
    text_col: str = "text",
) -> Path:
    """Run sentiment classification on a corpus and write parquet results.

    Args:
        model_path: Path to fine-tuned checkpoint directory.
        input_path: Input corpus parquet with ``text``, ``lang``, ``topic``.
        output_path: Destination parquet with prediction columns.
        batch_size: Inference batch size.
        text_col: Name of the text column in the input frame.

    Returns:
        Path to written output file.
    """
    frame = load_corpus_frame(input_path)
    if frame.is_empty():
        raise ValueError(f"Input corpus is empty: {input_path}")

    missing = {text_col, "lang", "topic"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    texts = frame[text_col].to_list()
    predictions = predict_texts(texts, model_path, batch_size=batch_size)

    result = frame.with_columns(
        pl.Series("sentiment_label", [row["sentiment_label"] for row in predictions]),
        pl.Series("sentiment_score", [row["sentiment_score"] for row in predictions]),
        pl.Series("confidence", [row["confidence"] for row in predictions]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    logger.info("Wrote %d predictions to %s", result.height, output_path)
    return output_path
