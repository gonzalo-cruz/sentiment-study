"""Exploratory analysis helpers for baseline sentiment corpora."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
from datasets import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from config import MAX_SEQUENCE_LENGTH


def build_eda_frame(
    dataset: Dataset,
    text_column: str,
    label_column: str,
    language: str,
) -> pl.DataFrame:
    """Build a Polars frame with text, label, and length features for plotting.

    Args:
        dataset: Source HuggingFace Dataset.
        text_column: Column containing input text.
        label_column: Column containing labels.
        language: ISO language code used to pick length semantics in summaries.

    Returns:
        Polars DataFrame with ``text``, ``label``, ``char_len``, and ``word_len``.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = {text_column, label_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    texts = ["" if value is None else str(value) for value in dataset[text_column]]
    labels = [str(value) for value in dataset[label_column]]

    return pl.DataFrame(
        {
            "text": texts,
            "label": labels,
            "char_len": [len(text) for text in texts],
            "word_len": [len(text.split()) for text in texts],
            "language": [language] * len(texts),
        }
    )


def label_distribution(df: pl.DataFrame, label_column: str = "label") -> pl.DataFrame:
    """Compute absolute and relative label counts.

    Args:
        df: Input frame containing a label column.
        label_column: Name of the label field.

    Returns:
        Polars DataFrame sorted by descending count.
    """
    total = df.height
    return (
        df.group_by(label_column)
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / total).round(4).alias("ratio"),
            (pl.col("count") / total * 100).round(1).alias("pct"),
        )
        .sort("count", descending=True)
    )


def duplicate_rate(texts: list[str]) -> float:
    """Return the fraction of duplicate non-empty texts in a corpus.

    Args:
        texts: Iterable of raw text strings.

    Returns:
        Duplicate rate in ``[0, 1]``.
    """
    normalized = [text.strip().lower() for text in texts if text.strip()]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicates / len(normalized), 4)


def _quantile(values: list[int], q: float) -> float:
    """Return a single quantile from a non-empty integer list."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def summarize_corpus(
    df: pl.DataFrame,
    language: str,
    char_threshold: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, object]:
    """Aggregate corpus-level EDA metrics for one dataset sample.

    Args:
        df: Frame produced by ``build_eda_frame``.
        language: ISO language code for the corpus.
        char_threshold: Character-length threshold used as a microblog proxy
            before token-level chunking is applied in preprocessing.

    Returns:
        Dictionary of summary statistics suitable for tabular display.
    """
    char_lengths = df["char_len"].to_list()
    word_lengths = df["word_len"].to_list()
    texts = df["text"].to_list()
    labels = df["label"].to_list()
    label_counts = dict(sorted(Counter(labels).items()))

    primary_length = "char_len" if language == "zh" else "word_len"
    primary_values = char_lengths if language == "zh" else word_lengths
    over_threshold = sum(1 for value in char_lengths if value > char_threshold)

    positive_ratio = _positive_ratio(labels)

    return {
        "language": language,
        "n_rows": df.height,
        "n_unique_labels": len(set(labels)),
        "label_counts": label_counts,
        "positive_ratio": positive_ratio,
        "label_skew": round(abs(positive_ratio - 0.5), 4) if positive_ratio is not None else None,
        "char_len_mean": round(sum(char_lengths) / len(char_lengths), 1),
        "char_len_p50": round(_quantile(char_lengths, 0.5), 1),
        "char_len_p90": round(_quantile(char_lengths, 0.9), 1),
        "char_len_p95": round(_quantile(char_lengths, 0.95), 1),
        "char_len_max": max(char_lengths),
        "word_len_mean": round(sum(word_lengths) / len(word_lengths), 1),
        "word_len_p50": round(_quantile(word_lengths, 0.5), 1),
        "word_len_p90": round(_quantile(word_lengths, 0.9), 1),
        "word_len_p95": round(_quantile(word_lengths, 0.95), 1),
        "primary_length_metric": primary_length,
        "primary_len_p95": round(_quantile(primary_values, 0.95), 1),
        "pct_over_char_threshold": round(over_threshold / df.height * 100, 2),
        "empty_texts": sum(1 for text in texts if not text.strip()),
        "duplicate_rate": duplicate_rate(texts),
        "is_binary": len(set(labels)) == 2,
    }


def _positive_ratio(labels: list[object]) -> float | None:
    """Estimate positive-class ratio for common label encodings."""
    if not labels:
        return None

    unique = set(labels)
    if unique <= {0, 1}:
        return round(sum(1 for label in labels if label == 1) / len(labels), 4)
    if unique <= {"0", "1"}:
        return round(sum(1 for label in labels if label == "1") / len(labels), 4)

    string_map = {str(label).lower() for label in unique}
    if string_map <= {"negative", "positive"}:
        return round(
            sum(1 for label in labels if str(label).lower() == "positive") / len(labels),
            4,
        )
    if string_map <= {"neg", "pos"}:
        return round(
            sum(1 for label in labels if str(label).lower() == "pos") / len(labels),
            4,
        )

    numeric_labels = sorted(unique)
    if len(numeric_labels) == 2 and all(isinstance(label, (int, float)) for label in numeric_labels):
        return round(
            sum(1 for label in labels if label == max(numeric_labels)) / len(labels),
            4,
        )

    return None


def token_length_stats(
    dataset: Dataset,
    text_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 500,
) -> dict[str, float | int]:
    """Estimate XLM-R token lengths on a capped sample.

    Args:
        dataset: HuggingFace Dataset containing text rows.
        text_column: Text field to tokenize.
        tokenizer: Loaded XLM-R tokenizer.
        max_samples: Maximum rows to tokenize for speed.

    Returns:
        Dictionary with token-length quantiles and share above ``MAX_SEQUENCE_LENGTH``.
    """
    sample_size = min(len(dataset), max_samples)
    texts = dataset.select(range(sample_size))[text_column]
    token_lengths = [
        len(tokenizer.encode(text if text is not None else "", add_special_tokens=True))
        for text in texts
    ]
    over_limit = sum(1 for length in token_lengths if length > MAX_SEQUENCE_LENGTH)

    return {
        "token_sample_size": sample_size,
        "token_len_mean": round(sum(token_lengths) / len(token_lengths), 1),
        "token_len_p50": round(_quantile(token_lengths, 0.5), 1),
        "token_len_p95": round(_quantile(token_lengths, 0.95), 1),
        "token_len_max": max(token_lengths),
        "pct_over_max_tokens": round(over_limit / len(token_lengths) * 100, 2),
    }


def build_corpus_summary_row(
    name: str,
    role: str,
    language: str,
    hf_id: str,
    frame: pl.DataFrame,
) -> dict[str, object]:
    """Build one display-ready summary row for the corpus summary table.

    Args:
        name: Corpus identifier (e.g. ``weibo_senti_100k``).
        role: ``train`` or ``validation``.
        language: ISO language code.
        hf_id: HuggingFace dataset ID used for loading.
        frame: EDA frame from ``build_eda_frame``.

    Returns:
        Flat dictionary aligned with the notebook summary table columns.
    """
    stats = summarize_corpus(frame, language=language)
    return {
        "name": name,
        "role": role,
        "language": language,
        "hf_id": hf_id,
        "n_rows": stats["n_rows"],
        "is_binary": stats["is_binary"],
        "positive_ratio": stats["positive_ratio"],
        "label_skew": stats["label_skew"],
        "char_len_p95": stats["char_len_p95"],
        "word_len_p95": stats["word_len_p95"],
        "pct_over_char_256": stats["pct_over_char_threshold"],
        "duplicate_rate": stats["duplicate_rate"],
        "empty_texts": stats["empty_texts"],
        "label_counts": str(stats["label_counts"]),
    }


def corpus_summaries_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Convert corpus summary dicts into a display-ready Polars table.

    Args:
        rows: Output rows from ``summarize_corpus`` with ``name`` and metadata.

    Returns:
        Polars DataFrame sorted by role then language.
    """
    return pl.DataFrame(rows).sort(["role", "language", "name"])
