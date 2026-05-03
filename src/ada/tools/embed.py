"""Sentence embeddings for topic modeling.

Wraps sentence-transformers with caching. The default model
(`paraphrase-multilingual-MiniLM-L12-v2`, ~420 MB) is downloaded on first use
and cached in `~/.cache/torch/sentence_transformers/`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from ada.config import settings


@lru_cache(maxsize=2)
def get_embedder(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name or settings.embed_model)


def encode(texts: list[str], model_name: str | None = None, batch_size: int = 32) -> "np.ndarray":
    """Encode a list of strings to embeddings."""
    return get_embedder(model_name).encode(
        texts, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True,
    )
