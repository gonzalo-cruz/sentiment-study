"""Sentiment ratio and distribution divergence metrics."""

import polars as pl


def compute_sentiment_ratios(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Compute positive sentiment ratio grouped by language and topic.

    Args:
        df: DataFrame with ``sentiment_label`` and grouping columns.
        group_cols: Columns to group by (e.g. ``lang``, ``topic``).

    Returns:
        Aggregated DataFrame with positive ratio per group.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"sentiment_label", *group_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return (
        df.group_by(group_cols)
        .agg(
            (pl.col("sentiment_label") == 1).mean().alias("positive_ratio"),
            pl.len().alias("count"),
        )
        .sort(group_cols)
    )
