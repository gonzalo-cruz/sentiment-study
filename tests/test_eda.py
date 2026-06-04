"""Tests for baseline EDA helpers."""

import polars as pl
from datasets import Dataset

from data.eda import (
    build_corpus_summary_row,
    build_eda_frame,
    duplicate_rate,
    label_distribution,
    summarize_corpus,
)


def test_build_eda_frame_adds_length_columns() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["hello world", "短文本"],
            "label": [1, 0],
        }
    )
    frame = build_eda_frame(dataset, "text", "label", language="en")

    assert frame.columns == ["text", "label", "char_len", "word_len", "language"]
    assert frame["char_len"].to_list() == [11, 3]
    assert frame["label"].to_list() == ["1", "0"]


def test_label_distribution_includes_ratio() -> None:
    frame = build_eda_frame(
        Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 1]}),
        "text",
        "label",
        language="en",
    )
    distribution = label_distribution(frame)

    assert distribution.height == 2
    assert distribution.filter(pl.col("label") == "1")["pct"][0] == 66.7


def test_build_eda_frame_unifies_label_dtype_for_concat() -> None:
    int_frame = build_eda_frame(
        Dataset.from_dict({"text": ["a"], "label": [1]}),
        "text",
        "label",
        language="en",
    )
    str_frame = build_eda_frame(
        Dataset.from_dict({"text": ["b"], "label": ["negative"]}),
        "text",
        "label",
        language="en",
    )
    combined = pl.concat([int_frame, str_frame])
    assert combined["label"].dtype == pl.Utf8


def test_summarize_corpus_flags_binary_balance() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["a" * 300, "b" * 50, "c" * 20],
            "label": [1, 0, 1],
        }
    )
    frame = build_eda_frame(dataset, "text", "label", language="en")
    summary = summarize_corpus(frame, language="en", char_threshold=256)

    assert summary["is_binary"] is True
    assert summary["positive_ratio"] == 0.6667
    assert summary["pct_over_char_threshold"] == 33.33


def test_build_corpus_summary_row_includes_length_columns() -> None:
    frame = build_eda_frame(
        Dataset.from_dict({"text": ["one two three", "four five"], "label": [1, 0]}),
        "text",
        "label",
        language="en",
    )
    row = build_corpus_summary_row(
        name="test",
        role="train",
        language="en",
        hf_id="example/test",
        frame=frame,
    )

    assert row["word_len_p95"] is not None
    assert row["char_len_p95"] is not None
    assert row["is_binary"] is True

