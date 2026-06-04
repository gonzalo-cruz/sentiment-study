"""Edge-case tests for EDA helpers."""

from datasets import Dataset

from data.eda import duplicate_rate, summarize_corpus, build_eda_frame


def test_duplicate_rate_empty_list() -> None:
    assert duplicate_rate([]) == 0.0


def test_duplicate_rate_all_unique() -> None:
    assert duplicate_rate(["a", "b", "c"]) == 0.0


def test_duplicate_rate_counts_repeats() -> None:
    assert duplicate_rate(["a", "a", "b", "b", "b"]) == 0.6


def test_summarize_corpus_uses_char_len_for_chinese() -> None:
    frame = build_eda_frame(
        Dataset.from_dict({"text": ["中文测试文本"], "label": [1]}),
        "text",
        "label",
        language="zh",
    )
    summary = summarize_corpus(frame, language="zh")
    assert summary["primary_length_metric"] == "char_len"


def test_summarize_corpus_uses_word_len_for_english() -> None:
    frame = build_eda_frame(
        Dataset.from_dict({"text": ["hello world"], "label": [1]}),
        "text",
        "label",
        language="en",
    )
    summary = summarize_corpus(frame, language="en")
    assert summary["primary_length_metric"] == "word_len"
