"""LangGraph assembly.

Topology:

    START
      │
      ▼
   planner ──► (decision dispatcher)
      │
      ├── RUN_NODE     ─► subgraph for that stage ─► back to planner
      ├── ASK_HUMAN    ─► interrupt() ───────────── (resumes to planner)
      ├── REPLAN       ─► record new plan ───────── back to planner
      └── FINISH       ─► END

Each subgraph reads file paths and a confirmed schema from state, writes a
`StageArtifact`, and appends `AuditEntry` records.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ada.nodes import NODE_REGISTRY
from ada.nodes.human import integrate_response
from ada.nodes.planner import planner_node
from ada.state import GraphState, PlannerAction, Stage


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def planner_router(state: GraphState) -> str:
    decision = state.last_decision
    if decision is None:
        return "planner"
    if decision.action == PlannerAction.FINISH:
        return END
    if decision.action == PlannerAction.ASK_HUMAN:
        return "human_gate"
    if decision.action == PlannerAction.REPLAN:
        return "planner"
    if decision.action == PlannerAction.RUN_NODE:
        assert decision.next_stage is not None, "RUN_NODE requires next_stage"
        if decision.next_stage not in NODE_REGISTRY:
            # Unimplemented stage → finish gracefully so the planner doesn't loop.
            return END
        return f"stage_{decision.next_stage.value}"
    raise ValueError(f"Unknown planner action: {decision.action}")


# ─────────────────────────────────────────────────────────────────────────────
# Human gate
# ─────────────────────────────────────────────────────────────────────────────

def human_gate_node(state: GraphState) -> dict:
    decision = state.last_decision
    assert decision is not None and decision.question is not None, "ASK_HUMAN missing question"
    q = decision.question

    response = interrupt({
        "question_id": q.question_id,
        "stage": q.stage.value,
        "type": q.question_type.value,
        "prompt": q.prompt,
        "payload": q.payload,
        "proposal": q.proposal,
        "why_asking": q.why_asking,
    })

    return integrate_response(state, q, response)


# ─────────────────────────────────────────────────────────────────────────────
# Stage adapter
# ─────────────────────────────────────────────────────────────────────────────

def make_stage_node(stage: Stage):
    def _node(state: GraphState) -> dict:
        impl = NODE_REGISTRY.get(stage)
        if impl is None:
            raise NotImplementedError(f"Stage {stage.value} not implemented yet")
        return impl(state)
    _node.__name__ = f"stage_{stage.value}"
    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("human_gate", human_gate_node)
    for stage in Stage:
        g.add_node(f"stage_{stage.value}", make_stage_node(stage))

    g.add_edge(START, "planner")

    edge_map = {f"stage_{s.value}": f"stage_{s.value}" for s in Stage}
    edge_map["human_gate"] = "human_gate"
    edge_map["planner"] = "planner"
    edge_map[END] = END
    g.add_conditional_edges("planner", planner_router, edge_map)
    g.add_edge("human_gate", "planner")
    for stage in Stage:
        g.add_edge(f"stage_{stage.value}", "planner")

    return g


def compile_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_graph().compile(checkpointer=checkpointer)
