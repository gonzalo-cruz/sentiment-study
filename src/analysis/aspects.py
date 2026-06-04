"""Aspect-based sentiment profiling (Phase 4)."""

from pathlib import Path


def compute_aspect_sentiment(
    df_path: Path,
    aspects: list[str],
) -> dict[str, dict[str, float]]:
    """Profile sentiment per aspect per language per topic.

    Args:
        df_path: Parquet path with aspect-tagged posts.
        aspects: Aspect labels (e.g. price, privacy, ecosystem).

    Returns:
        Nested dict: aspect → language → mean sentiment score.

    Raises:
        NotImplementedError: ABSA pipeline — Phase 4.
    """
    raise NotImplementedError("Aspect-based analysis — Phase 4")
