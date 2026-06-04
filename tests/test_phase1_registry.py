"""Tests for Phase 1 baseline registry and configuration alignment."""

import pytest

from config import TRAINING_DATASETS, VALIDATION_DATASETS
from data.baseline import (
    BASELINE_DATASETS,
    profile_dataset,
    training_specs,
    validation_specs,
)
from datasets import Dataset


def test_baseline_registry_has_four_corpora() -> None:
    assert len(BASELINE_DATASETS) == 4
    names = {spec.name for spec in BASELINE_DATASETS}
    assert names == {"weibo_senti_100k", "sentiment140", "ChnSentiCorp", "sst2"}


def test_no_spanish_training_corpus() -> None:
    languages = {spec.language for spec in BASELINE_DATASETS}
    assert "es" not in languages
    assert set(TRAINING_DATASETS.keys()) == {"zh", "en"}


def test_training_and_validation_specs_partition_registry() -> None:
    train = training_specs()
    val = validation_specs()
    assert len(train) == 2
    assert len(val) == 2
    assert {spec.name for spec in train} == {"weibo_senti_100k", "sentiment140"}
    assert {spec.name for spec in val} == {"ChnSentiCorp", "sst2"}


def test_validation_datasets_match_config() -> None:
    val_ids = {spec.hf_id for spec in validation_specs()}
    assert VALIDATION_DATASETS["zh"] in val_ids
    assert VALIDATION_DATASETS["en"] in val_ids


def test_profile_dataset_raises_on_missing_columns() -> None:
    dataset = Dataset.from_dict({"text": ["hello"], "label": [1]})
    with pytest.raises(ValueError, match="Missing columns"):
        profile_dataset(dataset, text_column="missing", label_column="label")
