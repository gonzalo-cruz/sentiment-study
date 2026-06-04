"""Tests for corpus loaders and parquet I/O."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from data.loaders import (
    RedditCollectionSpec,
    _submission_to_row,
    collect_subreddit_posts,
    default_processed_corpus_path,
    default_raw_reddit_path,
    load_corpus_frame,
    save_corpus_frame,
)
from data.preprocessing import harmonize_label


def test_harmonize_label_in_loaders_path() -> None:
    assert harmonize_label("pos") == 1


def test_save_and_load_corpus_frame_roundtrip(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "text": ["hello"],
            "lang": ["en"],
            "source": ["reddit"],
            "topic": ["apple"],
            "community": ["apple"],
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "post_id": ["abc"],
        }
    )
    path = tmp_path / "corpus.parquet"
    save_corpus_frame(frame, path)
    loaded = load_corpus_frame(path)
    assert loaded.height == 1
    assert loaded["text"][0] == "hello"


def test_submission_to_row_combines_title_and_body() -> None:
    submission = MagicMock()
    submission.title = "Great phone"
    submission.selftext = "Battery lasts all day."
    submission.created_utc = 1_700_000_000
    submission.id = "post123"

    row = _submission_to_row(
        submission,
        RedditCollectionSpec(subreddit="apple", language="en", topic="apple"),
    )
    assert "Great phone" in row["text"]
    assert row["lang"] == "en"
    assert row["community"] == "apple"


def test_collect_subreddit_posts_with_mock_reddit() -> None:
    submission = MagicMock()
    submission.title = "Test"
    submission.selftext = "Body"
    submission.created_utc = 1_700_000_000
    submission.id = "x1"

    reddit = MagicMock()
    reddit.subreddit.return_value.new.return_value = [submission]

    spec = RedditCollectionSpec(subreddit="apple", language="en", topic="apple", limit=1)
    frame = collect_subreddit_posts(reddit, spec)
    assert frame.height == 1


def test_default_paths_follow_language_code() -> None:
    assert default_raw_reddit_path("en").name == "reddit_en.parquet"
    assert default_processed_corpus_path("es").name == "study_es.parquet"
