"""Text preprocessing: normalization, Weibo cleaning, dedup, and language filtering."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl
from langdetect import LangDetectException, detect

from config import MAX_SEQUENCE_LENGTH

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

WEIBO_MODERATION_PATTERN = re.compile(
    r"\s*该微博因被投诉违反.*",
    flags=re.DOTALL,
)
URL_PATTERN = re.compile(r"https?://\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_weibo_text(text: str) -> str:
    """Strip Weibo platform moderation artifacts from text.

    Args:
        text: Raw Weibo post content.

    Returns:
        Cleaned text with moderation boilerplate removed.
    """
    return WEIBO_MODERATION_PATTERN.sub("", text).strip()


def normalize_social_text(text: str) -> str:
    """Normalize whitespace and strip platform noise from social text.

    Args:
        text: Raw post or comment body.

    Returns:
        Cleaned single-line text suitable for classification.
    """
    if text is None:
        return ""
    cleaned = str(text).replace("\x00", "").strip()
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def chunk_text(text: str, max_length: int = 256) -> list[str]:
    """Split long text into character-bounded chunks for model input.

    Args:
        text: Input text (e.g. Reddit comment).
        max_length: Maximum characters per chunk.

    Returns:
        List of text chunks; single-element list if text fits in one chunk.
    """
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def truncate_to_max_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> str:
    """Truncate text to fit within a tokenizer's max token budget.

    Args:
        text: Input text.
        tokenizer: Loaded HuggingFace tokenizer.
        max_length: Maximum token count excluding special tokens.

    Returns:
        Decoded truncated text.
    """
    if not text:
        return ""
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def detect_language(text: str) -> str | None:
    """Detect ISO 639-1 language code for a text snippet.

    Args:
        text: Input text.

    Returns:
        Language code or ``None`` when detection fails.
    """
    if not text or not text.strip():
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def deduplicate_texts(df: pl.DataFrame, text_col: str = "text") -> pl.DataFrame:
    """Remove duplicate rows by normalized lowercase text.

    Args:
        df: Input frame containing a text column.
        text_col: Name of the text field.

    Returns:
        Deduplicated frame preserving first occurrence order.
    """
    if df.is_empty():
        return df
    keyed = df.with_columns(
        pl.col(text_col)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("_norm_key")
    )
    return keyed.unique(subset="_norm_key", keep="first").drop("_norm_key")


def filter_language(
    df: pl.DataFrame,
    expected_lang: str,
    text_col: str = "text",
) -> pl.DataFrame:
    """Keep rows whose detected language matches ``expected_lang``.

    Args:
        df: Input frame containing text rows.
        expected_lang: ISO 639-1 code (``en``, ``es``, ``zh``).
        text_col: Name of the text field.

    Returns:
        Filtered frame containing only language-matched rows.
    """
    if df.is_empty():
        return df

    langs = [detect_language(text) for text in df[text_col].to_list()]
    mask = [lang == expected_lang for lang in langs]
    return df.filter(pl.Series(name="_lang_match", values=mask))


def preprocess_study_frame(
    df: pl.DataFrame,
    lang: str,
    *,
    text_col: str = "text",
    apply_weibo_clean: bool = False,
    dedupe: bool = True,
    language_filter: bool = True,
) -> pl.DataFrame:
    """Run the standard study-corpus preprocessing pipeline on a Polars frame.

    Args:
        df: Raw corpus frame with a text column.
        lang: Expected language code for optional filtering.
        text_col: Name of the text field.
        apply_weibo_clean: Apply Weibo moderation regex when ``lang`` is ``zh``.
        dedupe: Remove duplicate texts when True.
        language_filter: Apply langdetect filtering when True.

    Returns:
        Cleaned frame with normalized text.
    """
    if df.is_empty():
        return df

    texts = df[text_col].to_list()
    normalized: list[str] = []
    for text in texts:
        value = normalize_social_text(text if text is not None else "")
        if apply_weibo_clean or lang == "zh":
            value = clean_weibo_text(value)
        normalized.append(value)

    processed = df.with_columns(pl.Series(name=text_col, values=normalized))
    processed = processed.filter(pl.col(text_col).str.strip_chars().str.len_chars() > 0)

    if dedupe:
        processed = deduplicate_texts(processed, text_col=text_col)
    if language_filter:
        processed = filter_language(processed, expected_lang=lang, text_col=text_col)
    return processed


def harmonize_label(raw_label: object) -> int | None:
    """Map heterogeneous sentiment labels to binary ``{0: negative, 1: positive}``.

    Args:
        raw_label: Source label value from a training corpus.

    Returns:
        ``0``, ``1``, or ``None`` when the label cannot be mapped.
    """
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return int(raw_label)
    if isinstance(raw_label, (int, float)):
        value = int(raw_label)
        if value in (0, 1):
            return value
        return None

    normalized = str(raw_label).strip().lower()
    negative = {"0", "negative", "neg", "bad"}
    positive = {"1", "positive", "pos", "good"}
    if normalized in negative:
        return 0
    if normalized in positive:
        return 1
    return None
