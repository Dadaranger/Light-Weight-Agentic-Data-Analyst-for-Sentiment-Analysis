"""BERTopic wrapper. Mirrors Day 3's setup for multilingual sentiment data.

The course's defaults: multilingual MiniLM embeddings, UMAP→HDBSCAN, c-TF-IDF
on already-tokenized strings. We expose those as a single `fit_topics` call.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def fit_topics(
    embed_corpus: list[str],
    ctfidf_corpus: list[str],
    embeddings: np.ndarray | None = None,
    *,
    min_topic_size: int = 15,
    nr_topics: str | int | None = "auto",
    language: str = "chinese",
) -> tuple[Any, list[int]]:
    """Fit BERTopic and return (model, topic_assignments).

    `embed_corpus` is the text used for sentence embeddings (preserves context).
    `ctfidf_corpus` is the already-tokenized text used for keyword extraction.
    Pre-computed `embeddings` can be passed to skip re-encoding.
    """
    # Lazy imports — heavy deps shouldn't load just from `ada.tools` import
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(analyzer="word", min_df=2, max_features=5000)
    model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer,
        language=language,
        calculate_probabilities=False,
        verbose=False,
    )
    if embeddings is not None:
        topics, _ = model.fit_transform(ctfidf_corpus, embeddings=embeddings)
    else:
        topics, _ = model.fit_transform(ctfidf_corpus)
    return model, list(topics)


def top_keywords(model: Any, topic_id: int, n: int = 10) -> list[str]:
    """Get top N keywords for a topic (by c-TF-IDF score)."""
    if topic_id == -1:
        return []
    pairs = model.get_topic(topic_id) or []
    return [w for w, _score in pairs[:n]]


def auto_label(keywords: list[str], topic_id: int, max_words: int = 3) -> str:
    """Generate a starter label from top keywords. Human is expected to refine."""
    if topic_id == -1:
        return "未歸類貼文（離群值）"
    head = "·".join(keywords[:max_words]) if keywords else "?"
    return f"T{topic_id:02d}：{head}（待標記）"
