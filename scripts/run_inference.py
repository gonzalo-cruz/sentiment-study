"""Preprocess study corpora and run batch sentiment inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401 — adds src/ to sys.path

from config import DEFAULT_CHECKPOINT_DIR, INFERENCE_OUTPUT_DIR
from models.predict import run_inference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for corpus preprocessing and inference."""
    parser = argparse.ArgumentParser(description="Run batch inference on a study corpus.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Processed corpus parquet with text/lang/topic columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prediction parquet path (default: data/outputs/predictions/{input stem}.parquet).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "best",
        help="Fine-tuned checkpoint directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output_path = args.output or (
        INFERENCE_OUTPUT_DIR / "predictions" / f"{args.input.stem}_predictions.parquet"
    )
    run_inference(
        model_path=args.model_path,
        input_path=args.input,
        output_path=output_path,
        batch_size=args.batch_size,
    )
    logger.info("Predictions written to %s", output_path)


if __name__ == "__main__":
    main()
