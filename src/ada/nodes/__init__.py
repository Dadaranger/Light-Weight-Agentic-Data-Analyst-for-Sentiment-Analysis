"""Node registry — maps `Stage` enum to the node callable.

Each node has signature `(state: GraphState) -> dict` and returns a state patch.
Nodes are registered here as they're implemented (vertical-slice style).
"""
from __future__ import annotations

from typing import Callable

from ada.state import GraphState, Stage

NodeFn = Callable[[GraphState], dict]


def _slice1_registry() -> dict[Stage, NodeFn]:
    from ada.nodes.ingest import ingest_node
    from ada.nodes.schema_infer import schema_infer_node
    from ada.nodes.reshape import reshape_node
    return {
        Stage.INGEST: ingest_node,
        Stage.SCHEMA_INFER: schema_infer_node,
        Stage.RESHAPE: reshape_node,
    }


NODE_REGISTRY: dict[Stage, NodeFn] = _slice1_registry()
