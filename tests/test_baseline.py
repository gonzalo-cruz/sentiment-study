"""Tests for baseline dataset profiling helpers."""

from datasets import Dataset

from data.baseline import BaselineDatasetSpec, _cap_dataset, _resolve_split, profile_dataset, sample_rows


def test_profile_dataset_returns_expected_stats() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["good", "bad", ""],
            "label": [1, 0, 1],
        }
    )
    profile = profile_dataset(dataset, text_column="text", label_column="label")

    assert profile["n_rows"] == 3
    assert profile["label_counts"] == {0: 1, 1: 2}
    assert profile["empty_texts"] == 1
    assert profile["char_len_max"] == 4


def test_sample_rows_limits_output() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
        }
    )
    sample = sample_rows(dataset, "text", "label", n=2)

    assert sample.height == 2
    assert sample.columns == ["text", "label"]


def test_resolve_split_applies_inspection_cap() -> None:
    spec = BaselineDatasetSpec(
        name="sentiment140",
        role="train",
        language="en",
        hf_id="example/sentiment140",
        split="train",
        text_column="text",
        label_column="label",
        plan_id="sentiment140",
        train_subsample=100_000,
    )
    assert _resolve_split(spec, max_rows=2_000) == "train[:2000]"
    assert _resolve_split(spec, max_rows=None) == "train[:100000]"


def test_cap_dataset_shuffled_sample_spans_both_labels() -> None:
    dataset = Dataset.from_dict(
        {
            "text": [f"t{i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        }
    )
    capped = _cap_dataset(dataset, max_rows=20, seed=42)
    labels = set(capped["label"])

    assert len(labels) == 2
