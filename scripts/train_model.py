"""Fine-tune XLM-R on mixed ZH/EN baseline corpora."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401 — adds src/ to sys.path

from config import DEFAULT_CHECKPOINT_DIR, SENTIMENT140_TRAIN_SUBSAMPLE, TRAIN_EPOCHS, TRAIN_LEARNING_RATE
from models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R sentiment classifier.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint output directory.",
    )
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_LEARNING_RATE)
    parser.add_argument(
        "--en-subsample",
        type=int,
        default=SENTIMENT140_TRAIN_SUBSAMPLE,
        help="Maximum Sentiment140 training rows.",
    )
    parser.add_argument(
        "--zh-max-rows",
        type=int,
        default=None,
        help="Optional cap on Weibo training rows.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step cap for smoke tests.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    checkpoint = train_model(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        en_subsample=args.en_subsample,
        zh_max_rows=args.zh_max_rows,
        max_steps=args.max_steps,
        run_name=args.run_name,
    )
    logger.info("Best checkpoint saved to %s", checkpoint)


if __name__ == "__main__":
    main()
