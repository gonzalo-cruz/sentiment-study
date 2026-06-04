"""HuggingFace baseline loading, Reddit collection, and parquet persistence."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import praw
from datasets import Dataset, concatenate_datasets, load_dataset

from config import (
    PROCESSED_CORPUS_DIR,
    RAW_DATA_DIR,
    REDDIT_CLIENT_ID_ENV,
    REDDIT_CLIENT_SECRET_ENV,
    REDDIT_EN_SUBREDDITS,
    REDDIT_ES_SUBREDDITS,
    REDDIT_USER_AGENT_ENV,
    SENTIMENT140_TRAIN_SUBSAMPLE,
    SUBREDDIT_TOPICS,
    TRAIN_SEED,
    TRAINING_DATASETS,
    VALIDATION_DATASETS,
)
from data.preprocessing import clean_weibo_text, harmonize_label, normalize_social_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedditCollectionSpec:
    """Parameters for scraping one subreddit's recent submissions."""

    subreddit: str
    language: str
    topic: str
    limit: int = 500
    sort: str = "new"


def create_reddit_client() -> praw.Reddit:
    """Build an authenticated PRAW client from environment variables.

    Returns:
        Configured Reddit API client.

    Raises:
        ValueError: If required Reddit credentials are missing.
    """
    client_id = os.environ.get(REDDIT_CLIENT_ID_ENV)
    client_secret = os.environ.get(REDDIT_CLIENT_SECRET_ENV)
    user_agent = os.environ.get(REDDIT_USER_AGENT_ENV, "sentiment-study/0.1")

    missing = [
        name
        for name, value in (
            (REDDIT_CLIENT_ID_ENV, client_id),
            (REDDIT_CLIENT_SECRET_ENV, client_secret),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing Reddit credentials in environment: {', '.join(missing)}"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def save_corpus_frame(df: pl.DataFrame, output_path: Path) -> Path:
    """Persist a corpus Polars frame to parquet.

    Args:
        df: Corpus rows to save.
        output_path: Destination ``.parquet`` path.

    Returns:
        Path to the written parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return output_path


def load_corpus_frame(input_path: Path) -> pl.DataFrame:
    """Load a corpus parquet file into Polars.

    Args:
        input_path: Source parquet path.

    Returns:
        Loaded Polars DataFrame.
    """
    return pl.read_parquet(input_path)


def _submission_to_row(submission: praw.models.Submission, spec: RedditCollectionSpec) -> dict[str, object]:
    """Convert one Reddit submission into a normalized corpus row dict."""
    body = normalize_social_text(submission.selftext or "")
    title = normalize_social_text(submission.title or "")
    text = f"{title}. {body}".strip(". ").strip() if body else title
    created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
    return {
        "text": text,
        "lang": spec.language,
        "source": "reddit",
        "topic": spec.topic,
        "community": spec.subreddit,
        "timestamp": created,
        "post_id": submission.id,
    }


def collect_subreddit_posts(
    reddit: praw.Reddit,
    spec: RedditCollectionSpec,
) -> pl.DataFrame:
    """Collect recent submissions from one subreddit.

    Args:
        reddit: Authenticated PRAW client.
        spec: Collection parameters for the target subreddit.

    Returns:
        Polars frame of raw Reddit rows before language filtering.
    """
    subreddit = reddit.subreddit(spec.subreddit)
    iterator = subreddit.new(limit=spec.limit)
    rows = [_submission_to_row(post, spec) for post in iterator]
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "text": pl.Utf8,
            "lang": pl.Utf8,
            "source": pl.Utf8,
            "topic": pl.Utf8,
            "community": pl.Utf8,
            "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
            "post_id": pl.Utf8,
        }
    )


def collect_reddit_corpus(
    language: str,
    *,
    limit_per_subreddit: int = 500,
    reddit: praw.Reddit | None = None,
) -> pl.DataFrame:
    """Collect Reddit submissions for EN or peninsular ES study corpora.

    Args:
        language: ``en`` or ``es``.
        limit_per_subreddit: Max submissions per subreddit.
        reddit: Optional pre-built PRAW client.

    Returns:
        Concatenated Polars frame across configured subreddits.

    Raises:
        ValueError: If ``language`` is not ``en`` or ``es``.
    """
    if language == "en":
        subreddits = REDDIT_EN_SUBREDDITS
    elif language == "es":
        subreddits = REDDIT_ES_SUBREDDITS
    else:
        raise ValueError(f"Unsupported Reddit language: {language!r}")

    client = reddit or create_reddit_client()
    frames: list[pl.DataFrame] = []
    for name in subreddits:
        spec = RedditCollectionSpec(
            subreddit=name,
            language=language,
            topic=SUBREDDIT_TOPICS.get(name, "general"),
            limit=limit_per_subreddit,
        )
        logger.info("Collecting r/%s (%s)", name, language)
        frames.append(collect_subreddit_posts(client, spec))

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def _map_training_rows(dataset: Dataset, text_col: str, label_col: str, language: str) -> Dataset:
    """Project a HF dataset into unified ``text``, ``label``, ``language`` columns."""

    def _transform_row(example: dict[str, object]) -> dict[str, object]:
        text = normalize_social_text(str(example[text_col]) if example[text_col] is not None else "")
        if language == "zh":
            text = clean_weibo_text(text)
        mapped = harmonize_label(example[label_col])
        if not text.strip() or mapped is None:
            return {"text": "", "label": -1, "language": language}
        return {"text": text, "label": mapped, "language": language}

    mapped = dataset.map(
        _transform_row,
        remove_columns=dataset.column_names,
        desc=f"harmonize-{language}",
    )
    return mapped.filter(lambda row: row["label"] != -1)


def load_training_dataset(
    *,
    en_subsample: int = SENTIMENT140_TRAIN_SUBSAMPLE,
    zh_max_rows: int | None = None,
    seed: int = TRAIN_SEED,
) -> Dataset:
    """Load and harmonize the mixed ZH/EN training corpus.

    Args:
        en_subsample: Maximum English rows from Sentiment140.
        zh_max_rows: Optional cap on Chinese rows (``None`` uses full weibo split).
        seed: Shuffle seed for the combined dataset.

    Returns:
        Shuffled HuggingFace Dataset with ``text``, ``label``, ``language``.
    """
    zh_split = "train" if zh_max_rows is None else f"train[:{zh_max_rows}]"
    en_split = f"train[:{en_subsample}]"

    zh_raw = load_dataset(TRAINING_DATASETS["zh"], split=zh_split)
    en_raw = load_dataset(TRAINING_DATASETS["en"], split=en_split)

    zh_mapped = _map_training_rows(zh_raw, "review", "label", "zh")
    en_mapped = _map_training_rows(en_raw, "text", "label", "en")

    combined = concatenate_datasets([zh_mapped, en_mapped])
    return combined.shuffle(seed=seed)


def load_validation_datasets() -> dict[str, Dataset]:
    """Load ZH and EN validation corpora keyed by language code.

    Returns:
        Dictionary mapping ``zh`` and ``en`` to harmonized validation datasets.
    """
    zh_raw = load_dataset(VALIDATION_DATASETS["zh"], split="validation")
    en_raw = load_dataset(VALIDATION_DATASETS["en"], "sst2", split="validation")

    return {
        "zh": _map_training_rows(zh_raw, "text", "label", "zh"),
        "en": _map_training_rows(en_raw, "sentence", "label", "en"),
    }


def load_weibo_study_corpus(max_rows: int | None = None) -> pl.DataFrame:
    """Build a Chinese study corpus from the weibo training pool.

    Args:
        max_rows: Optional row cap for smoke tests.

    Returns:
        Polars frame ready for preprocessing and inference.
    """
    split = "train" if max_rows is None else f"train[:{max_rows}]"
    dataset = load_dataset(TRAINING_DATASETS["zh"], split=split)
    rows = []
    for review in dataset["review"]:
        text = clean_weibo_text(normalize_social_text(str(review)))
        if text:
            rows.append(
                {
                    "text": text,
                    "lang": "zh",
                    "source": "weibo",
                    "topic": "general",
                    "community": "weibo_senti_100k",
                    "timestamp": None,
                    "post_id": "",
                }
            )
    return pl.DataFrame(rows)


def default_raw_reddit_path(language: str) -> Path:
    """Return the default raw Reddit parquet path for a language code."""
    return RAW_DATA_DIR / "reddit" / f"reddit_{language}.parquet"


def default_processed_corpus_path(language: str) -> Path:
    """Return the default processed study corpus path for a language code."""
    return PROCESSED_CORPUS_DIR / f"study_{language}.parquet"
