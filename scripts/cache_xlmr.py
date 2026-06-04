"""Download and cache xlm-roberta-base tokenizer and weights for offline use."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import BASE_MODEL_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cache_xlmr(model_id: str = BASE_MODEL_ID) -> None:
    """Fetch tokenizer and model weights into the local HuggingFace cache.

    Args:
        model_id: HuggingFace hub model identifier.
    """
    logger.info("Downloading tokenizer: %s", model_id)
    AutoTokenizer.from_pretrained(model_id)

    logger.info("Downloading model weights: %s", model_id)
    AutoModel.from_pretrained(model_id)

    logger.info("Cached successfully: %s", model_id)


def verify_cache(model_id: str = BASE_MODEL_ID) -> bool:
    """Confirm the model loads from local cache only.

    Args:
        model_id: HuggingFace hub model identifier.

    Returns:
        True when tokenizer and weights load offline.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        AutoModel.from_pretrained(model_id, local_files_only=True)
        logger.info("Offline verification OK: %s", model_id)
        return True
    except OSError as exc:
        logger.error("Offline verification failed: %s", exc)
        return False


def main() -> None:
    """CLI entry point for caching or verifying XLM-R weights."""
    parser = argparse.ArgumentParser(description="Cache or verify xlm-roberta-base.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the model is available offline; do not download.",
    )
    parser.add_argument(
        "--model-id",
        default=BASE_MODEL_ID,
        help="HuggingFace model ID to cache or verify.",
    )
    args = parser.parse_args()

    if args.verify_only:
        raise SystemExit(0 if verify_cache(args.model_id) else 1)

    cache_xlmr(args.model_id)
    if not verify_cache(args.model_id):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
