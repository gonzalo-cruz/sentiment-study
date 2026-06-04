"""End-to-end Phase 2 smoke test on tiny samples.

By default writes artifacts into the project tree (see config paths below).
Use ``--ephemeral`` to run in a temp directory that is deleted on exit.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401 — adds src/ to sys.path

from config import (
    DEFAULT_CHECKPOINT_DIR,
    INFERENCE_OUTPUT_DIR,
    PROCESSED_CORPUS_DIR,
)
from data.loaders import load_weibo_study_corpus, save_corpus_frame
from data.preprocessing import harmonize_label, preprocess_study_frame
from models.predict import run_inference
from models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SMOKE_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_DIR.parent / "smoke"
SMOKE_PROCESSED_PATH = PROCESSED_CORPUS_DIR / "smoke_zh.parquet"
SMOKE_PREDICTIONS_PATH = INFERENCE_OUTPUT_DIR / "predictions" / "smoke_zh_predictions.parquet"


def run_smoke_test(*, ephemeral: bool = False) -> dict[str, Path]:
    """Train on tiny data, preprocess a ZH sample, and run inference.

    Args:
        ephemeral: When True, use a temp directory (nothing persisted).

    Returns:
        Dict with ``checkpoint``, ``processed``, and ``predictions`` paths.
    """
    assert harmonize_label("positive") == 1
    assert harmonize_label("negative") == 0

    if ephemeral:
        temp_ctx: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory()
        base = Path(temp_ctx.name)
        checkpoint_dir = base / "checkpoint"
        processed_path = base / "study_zh.parquet"
        output_path = base / "predictions.parquet"
    else:
        temp_ctx = None
        checkpoint_dir = SMOKE_CHECKPOINT_DIR
        processed_path = SMOKE_PROCESSED_PATH
        output_path = SMOKE_PREDICTIONS_PATH
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Smoke training (max_steps=5)...")
        checkpoint = train_model(
            output_dir=checkpoint_dir,
            en_subsample=40,
            zh_max_rows=40,
            max_steps=5,
            batch_size=4,
            run_name="phase2-smoke",
        )

        raw_zh = load_weibo_study_corpus(max_rows=20)
        processed = preprocess_study_frame(
            raw_zh,
            lang="zh",
            language_filter=False,
        )
        save_corpus_frame(processed, processed_path)

        run_inference(
            model_path=checkpoint,
            input_path=processed_path,
            output_path=output_path,
            batch_size=4,
        )

        artifacts = {
            "checkpoint": checkpoint,
            "processed": processed_path,
            "predictions": output_path,
        }
        logger.info("Smoke test passed.")
        for name, path in artifacts.items():
            logger.info("  %s → %s", name, path)
        return artifacts
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def main() -> None:
    """CLI entry point for the Phase 2 smoke test."""
    parser = argparse.ArgumentParser(description="Phase 2 end-to-end smoke test.")
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Use a temp directory; artifacts are deleted when the script exits.",
    )
    args = parser.parse_args()
    run_smoke_test(ephemeral=args.ephemeral)


if __name__ == "__main__":
    main()
