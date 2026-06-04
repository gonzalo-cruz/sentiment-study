"""Tests for preprocessing utilities."""

import polars as pl
from transformers import AutoTokenizer

from config import BASE_MODEL_ID, MAX_SEQUENCE_LENGTH
from data.preprocessing import (
    chunk_text,
    clean_weibo_text,
    deduplicate_texts,
    filter_language,
    harmonize_label,
    normalize_social_text,
    preprocess_study_frame,
    truncate_to_max_tokens,
)


def test_clean_weibo_text_strips_moderation_marker() -> None:
    raw = "很好用的手机 该微博因被投诉违反社区公约"
    assert clean_weibo_text(raw) == "很好用的手机"


def test_chunk_text_splits_long_input() -> None:
    text = "a" * 300
    chunks = chunk_text(text, max_length=256)
    assert len(chunks) == 2
    assert len(chunks[0]) == 256
    assert len(chunks[1]) == 44


def test_chunk_text_empty_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_chunk_text_short_returns_single_chunk() -> None:
    assert chunk_text("hello") == ["hello"]


def test_normalize_social_text_strips_urls_and_whitespace() -> None:
    raw = "  hello   world  https://example.com  "
    assert normalize_social_text(raw) == "hello world"


def test_harmonize_label_maps_string_labels() -> None:
    assert harmonize_label("positive") == 1
    assert harmonize_label("negative") == 0
    assert harmonize_label(1) == 1
    assert harmonize_label("unknown") is None


def test_deduplicate_texts_keeps_first_occurrence() -> None:
    frame = pl.DataFrame({"text": ["Hello", "hello", "other"]})
    deduped = deduplicate_texts(frame)
    assert deduped.height == 2


def test_preprocess_study_frame_drops_empty_rows() -> None:
    frame = pl.DataFrame(
        {
            "text": ["valid post", "   ", "another"],
            "lang": ["en", "en", "en"],
            "source": ["reddit", "reddit", "reddit"],
            "topic": ["apple", "apple", "apple"],
        }
    )
    processed = preprocess_study_frame(frame, lang="en", language_filter=False)
    assert processed.height == 2


def test_truncate_to_max_tokens_shortens_long_text() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    long_text = "word " * 400
    truncated = truncate_to_max_tokens(long_text, tokenizer, max_length=32)
    token_count = len(tokenizer.encode(truncated, add_special_tokens=False))
    assert token_count <= 32


def test_filter_language_keeps_matching_rows() -> None:
    frame = pl.DataFrame({"text": ["This is an English sentence.", "Esto es español."]})
    filtered = filter_language(frame, expected_lang="en")
    assert filtered.height >= 1
