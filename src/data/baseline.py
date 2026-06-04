"""Baseline dataset loaders and profiling for Phase 1 inspection."""

from collections import Counter
from dataclasses import dataclass, replace

import polars as pl
from datasets import Dataset, load_dataset

from config import (
    INSPECTION_SAMPLE_SIZE,
    INSPECTION_SHUFFLE_SEED,
    SENTIMENT140_TRAIN_SUBSAMPLE,
)


@dataclass(frozen=True)
class BaselineDatasetSpec:
    """Metadata for a baseline training or validation corpus."""

    name: str
    role: str
    language: str
    hf_id: str
    split: str
    text_column: str
    label_column: str
    plan_id: str
    hf_config: str | None = None
    train_subsample: int | None = None
    load_note: str = ""


BASELINE_DATASETS: tuple[BaselineDatasetSpec, ...] = (
    BaselineDatasetSpec(
        name="weibo_senti_100k",
        role="train",
        language="zh",
        hf_id="dirtycomputer/weibo_senti_100k",
        split="train",
        text_column="review",
        label_column="label",
        plan_id="dirtycomputer/weibo_senti_100k",
    ),
    BaselineDatasetSpec(
        name="sentiment140",
        role="train",
        language="en",
        hf_id="Roh2014/sentiment140_10k_tweets",
        split="train",
        text_column="clean_tweet",
        label_column="sentiment",
        plan_id="sentiment140",
        train_subsample=SENTIMENT140_TRAIN_SUBSAMPLE,
        load_note=(
            "Phase 1 uses 10k-row clean mirror for inspection. "
            "Phase 2 training subsamples 100k from adilbekovich/Sentiment140Twitter. "
            "Spanish is not in training — XLM-R transfers zero-shot; validate manually."
        ),
    ),
    BaselineDatasetSpec(
        name="ChnSentiCorp",
        role="validation",
        language="zh",
        hf_id="lansinuote/ChnSentiCorp",
        split="validation",
        text_column="text",
        label_column="label",
        plan_id="seamew/ChnSentiCorp",
        load_note=(
            "seamew/ChnSentiCorp uses a deprecated loading script; "
            "parquet mirror used instead."
        ),
    ),
    BaselineDatasetSpec(
        name="sst2",
        role="validation",
        language="en",
        hf_id="nyu-mll/glue",
        split="validation",
        text_column="sentence",
        label_column="label",
        plan_id="sst2",
        hf_config="sst2",
    ),
)


def training_specs() -> tuple[BaselineDatasetSpec, ...]:
    """Return baseline training corpus specs (ZH + EN)."""
    return tuple(spec for spec in BASELINE_DATASETS if spec.role == "train")


def validation_specs() -> tuple[BaselineDatasetSpec, ...]:
    """Return baseline validation corpus specs (ZH + EN)."""
    return tuple(spec for spec in BASELINE_DATASETS if spec.role == "validation")


def _resolve_split(spec: BaselineDatasetSpec, max_rows: int | None) -> str:
    """Build a split string, optionally capped by row count."""
    if max_rows is None:
        if spec.train_subsample is None:
            return spec.split
        return f"{spec.split}[:{spec.train_subsample}]"
    return f"{spec.split}[:{max_rows}]"


def _load_full_split(spec: BaselineDatasetSpec) -> Dataset:
    """Load an entire split without row caps."""
    if spec.hf_config is None:
        return load_dataset(spec.hf_id, split=spec.split)
    return load_dataset(spec.hf_id, spec.hf_config, split=spec.split)


def _cap_dataset(dataset: Dataset, max_rows: int, seed: int) -> Dataset:
    """Shuffle then take up to ``max_rows`` so inspection samples cover all labels."""
    shuffled = dataset.shuffle(seed=seed)
    if len(shuffled) <= max_rows:
        return shuffled
    return shuffled.select(range(max_rows))


def load_baseline_dataset(
    spec: BaselineDatasetSpec,
    max_rows: int | None = INSPECTION_SAMPLE_SIZE,
) -> Dataset:
    """Load one baseline dataset according to its HuggingFace specification.

    Args:
        spec: Baseline dataset metadata entry.
        max_rows: Cap rows loaded into memory. Defaults to ``INSPECTION_SAMPLE_SIZE``.
            Pass ``None`` to use each spec's full split (or ``train_subsample``).
            When capped, the full split is shuffled first so ordered corpora
            (e.g. weibo_senti_100k) still yield both labels in the sample.

    Returns:
        Loaded HuggingFace Dataset split.
    """
    if max_rows is None:
        split = _resolve_split(spec, max_rows=None)
        if spec.hf_config is None:
            return load_dataset(spec.hf_id, split=split)
        return load_dataset(spec.hf_id, spec.hf_config, split=split)

    return _cap_dataset(_load_full_split(spec), max_rows, INSPECTION_SHUFFLE_SEED)


def apply_row_cap(spec: BaselineDatasetSpec, max_rows: int | None) -> BaselineDatasetSpec:
    """Return a spec annotated with the effective row cap for display in notebooks."""
    if max_rows is None:
        cap = spec.train_subsample
    else:
        cap = max_rows
    note = spec.load_note
    if max_rows is not None:
        cap_note = f"Inspection cap: {max_rows:,} rows."
        note = f"{note} {cap_note}".strip() if note else cap_note
    return replace(spec, load_note=note)


def profile_dataset(
    dataset: Dataset,
    text_column: str,
    label_column: str,
) -> dict[str, object]:
    """Compute schema and distribution summary for a text classification corpus.

    Args:
        dataset: HuggingFace Dataset to summarize.
        text_column: Column containing input text.
        label_column: Column containing sentiment labels.

    Returns:
        Dictionary with row count, columns, label counts, and text-length stats.

    Raises:
        ValueError: If required columns are missing from the dataset.
    """
    missing = {text_column, label_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    texts = dataset[text_column]
    labels = dataset[label_column]
    char_lengths = [len(text) if text is not None else 0 for text in texts]
    word_lengths = [
        len(text.split()) if text is not None else 0 for text in texts
    ]

    return {
        "n_rows": len(dataset),
        "columns": dataset.column_names,
        "label_counts": dict(sorted(Counter(labels).items())),
        "n_unique_labels": len(set(labels)),
        "char_len_min": min(char_lengths) if char_lengths else 0,
        "char_len_max": max(char_lengths) if char_lengths else 0,
        "char_len_mean": round(sum(char_lengths) / len(char_lengths), 1)
        if char_lengths
        else 0.0,
        "word_len_mean": round(sum(word_lengths) / len(word_lengths), 1)
        if word_lengths
        else 0.0,
        "empty_texts": sum(1 for text in texts if not text or not str(text).strip()),
    }


def profiles_to_frame(profiles: list[dict[str, object]]) -> pl.DataFrame:
    """Convert profile dictionaries into a Polars summary table.

    Args:
        profiles: Output rows from ``profile_dataset``, each with a ``name`` key.

    Returns:
        Polars DataFrame with one row per dataset profile.
    """
    return pl.DataFrame(profiles)


def sample_rows(
    dataset: Dataset,
    text_column: str,
    label_column: str,
    n: int = 3,
) -> pl.DataFrame:
    """Return the first n rows as a Polars DataFrame for manual inspection.

    Args:
        dataset: Source dataset.
        text_column: Text field to display.
        label_column: Label field to display.
        n: Number of sample rows.

    Returns:
        Polars DataFrame with text and label columns only.
    """
    rows = dataset.select(range(min(n, len(dataset))))
    return pl.DataFrame(
        {
            text_column: rows[text_column],
            label_column: rows[label_column],
        }
    )
