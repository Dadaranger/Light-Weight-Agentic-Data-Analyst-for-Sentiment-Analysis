"""Node registry — maps `Stage` enum to the node callable.

Each node has signature `(state: GraphState) -> dict` and returns a state patch.
Nodes are registered here as they're implemented (vertical-slice style).
"""
from __future__ import annotations

from typing import Callable

from ada.state import GraphState, Stage

NodeFn = Callable[[GraphState], dict]


def _build_registry() -> dict[Stage, NodeFn]:
    from ada.nodes.clean import clean_node
    from ada.nodes.eda import eda_node
    from ada.nodes.ingest import ingest_node
    from ada.nodes.preprocess import preprocess_node
    from ada.nodes.reshape import reshape_node
    from ada.nodes.schema_infer import schema_infer_node
    from ada.nodes.sentiment import sentiment_node
    from ada.nodes.topic import topic_node
    return {
        Stage.INGEST: ingest_node,
        Stage.SCHEMA_INFER: schema_infer_node,
        Stage.RESHAPE: reshape_node,
        Stage.EDA: eda_node,
        Stage.CLEAN: clean_node,
        Stage.PREPROCESS: preprocess_node,
        Stage.SENTIMENT: sentiment_node,
        Stage.TOPIC: topic_node,
    }


NODE_REGISTRY: dict[Stage, NodeFn] = _build_registry()
