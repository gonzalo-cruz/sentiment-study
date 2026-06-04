"""Collect Reddit EN/ES study corpora via PRAW and save raw parquet."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.loaders import collect_reddit_corpus, default_raw_reddit_path, save_corpus_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for Reddit collection."""
    parser = argparse.ArgumentParser(description="Collect Reddit study corpora.")
    parser.add_argument(
        "--language",
        choices=["en", "es"],
        required=True,
        help="Target language corpus to collect.",
    )
    parser.add_argument(
        "--limit-per-subreddit",
        type=int,
        default=500,
        help="Maximum submissions per subreddit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: data/raw/reddit/reddit_{lang}.parquet).",
    )
    args = parser.parse_args()

    output_path = args.output or default_raw_reddit_path(args.language)
    frame = collect_reddit_corpus(
        args.language,
        limit_per_subreddit=args.limit_per_subreddit,
    )
    save_corpus_frame(frame, output_path)
    logger.info("Saved %d rows to %s", frame.height, output_path)


if __name__ == "__main__":
    main()
