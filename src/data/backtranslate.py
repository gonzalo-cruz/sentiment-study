"""Isolated back-translation pipeline (VRAM-safe sequential MT)."""

from pathlib import Path


def backtranslate_batch(
    texts: list[str],
    source_lang: str,
    model_cache_dir: Path | None = None,
) -> list[str]:
    """Translate a batch of texts to English via Helsinki-NLP opus-mt models.

    Args:
        texts: Source texts to translate.
        source_lang: ISO language code (``zh`` or ``es``).
        model_cache_dir: Optional local path for cached MT weights.

    Returns:
        English translations aligned with input order.

    Raises:
        ValueError: If source_lang is not supported.
        NotImplementedError: Pipeline not yet implemented (Phase 3).
    """
    if source_lang not in {"zh", "es"}:
        raise ValueError(f"Unsupported source_lang: {source_lang!r}")
    raise NotImplementedError("Back-translation pipeline — Phase 3")
