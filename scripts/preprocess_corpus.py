"""Preprocess raw study corpora into cleaned parquet files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PROCESSED_CORPUS_DIR
from data.loaders import load_corpus_frame, save_corpus_frame
from data.preprocessing import preprocess_study_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for study corpus preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess a raw study corpus.")
    parser.add_argument("--input", type=Path, required=True, help="Raw corpus parquet.")
    parser.add_argument("--lang", choices=["en", "es", "zh"], required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Processed output path (default: data/processed/study/{input stem}_processed.parquet).",
    )
    parser.add_argument(
        "--skip-language-filter",
        action="store_true",
        help="Skip langdetect filtering (useful for short smoke samples).",
    )
    args = parser.parse_args()

    output_path = args.output or (PROCESSED_CORPUS_DIR / f"{args.input.stem}_processed.parquet")
    frame = load_corpus_frame(args.input)
    processed = preprocess_study_frame(
        frame,
        lang=args.lang,
        apply_weibo_clean=(args.lang == "zh"),
        language_filter=not args.skip_language_filter,
    )
    save_corpus_frame(processed, output_path)
    logger.info("Wrote %d processed rows to %s", processed.height, output_path)


if __name__ == "__main__":
    main()
