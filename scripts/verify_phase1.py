"""Verify Phase 1 setup: directories, offline model cache, and optional dataset smoke loads."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    BASE_MODEL_ID,
    DATA_DIR,
    GOLD_DATA_DIR,
    MODELS_DIR,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from data.baseline import BASELINE_DATASETS, load_baseline_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_DIRS = (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    GOLD_DATA_DIR,
    OUTPUT_DATA_DIR,
    MODELS_DIR,
)


def verify_directories() -> list[str]:
    """Check that expected artifact directories exist.

    Returns:
        List of error messages; empty when all directories are present.
    """
    errors: list[str] = []
    for directory in REQUIRED_DIRS:
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")
    return errors


def verify_model_cache(model_id: str = BASE_MODEL_ID) -> list[str]:
    """Confirm XLM-R tokenizer and weights load from the local HuggingFace cache.

    Returns:
        List of error messages; empty when the model loads offline.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    errors: list[str] = []
    try:
        from transformers import AutoModel, AutoTokenizer

        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        AutoModel.from_pretrained(model_id, local_files_only=True)
        logger.info("Offline cache OK: %s", model_id)
    except OSError as exc:
        errors.append(
            f"Model not cached offline ({model_id}). Run: python scripts/cache_xlmr.py"
        )
        errors.append(str(exc))
    return errors


def smoke_load_datasets(max_rows: int = 10) -> list[str]:
    """Load a tiny sample from each baseline corpus to confirm HF availability.

    Args:
        max_rows: Row cap per dataset for the smoke test.

    Returns:
        List of error messages; empty when all datasets load.
    """
    errors: list[str] = []
    for spec in BASELINE_DATASETS:
        try:
            dataset = load_baseline_dataset(spec, max_rows=max_rows)
            logger.info(
                "Loaded %s (%s, %s): %d rows",
                spec.name,
                spec.language,
                spec.role,
                len(dataset),
            )
        except Exception as exc:
            errors.append(f"Failed to load {spec.name} ({spec.hf_id}): {exc}")
    return errors


def run_verification(*, online: bool = False, smoke_rows: int = 10) -> int:
    """Run all Phase 1 verification checks.

    Args:
        online: When True, smoke-load baseline datasets from HuggingFace.
        smoke_rows: Row cap per dataset when ``online`` is True.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    all_errors: list[str] = []

    all_errors.extend(verify_directories())
    all_errors.extend(verify_model_cache())

    if online:
        all_errors.extend(smoke_load_datasets(max_rows=smoke_rows))

    if all_errors:
        for message in all_errors:
            logger.error(message)
        return 1

    logger.info("Phase 1 verification passed.")
    return 0


def main() -> None:
    """CLI entry point for Phase 1 verification."""
    parser = argparse.ArgumentParser(description="Verify Phase 1 project setup.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Smoke-load baseline datasets from HuggingFace (requires network).",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=10,
        help="Rows to load per dataset when --online is set.",
    )
    args = parser.parse_args()
    raise SystemExit(run_verification(online=args.online, smoke_rows=args.smoke_rows))


if __name__ == "__main__":
    main()
