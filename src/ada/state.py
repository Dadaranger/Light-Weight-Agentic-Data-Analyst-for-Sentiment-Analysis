"""GraphState and all nested Pydantic models for the LangGraph agent.

Design rules:
- DataFrames never live in state. State holds file paths, schemas, summary stats,
  audit entries, and LLM-facing text.
- Every transformation is recorded in `audit_log`. Silent mutations are forbidden.
- HITL questions and answers are typed and persisted so runs are replayable.
- Memory updates require human approval — pending updates land in
  `pending_memory_diffs` and only flush after a `MemoryUpdate` HITL turn.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Stage(str, Enum):
    INGEST = "ingest"
    SCHEMA_INFER = "schema_infer"
    RESHAPE = "reshape"
    EDA = "eda"
    CLEAN = "clean"
    PREPROCESS = "preprocess"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    NARRATIVE = "narrative"
    AMPLIFICATION = "amplification"
    BRIEF = "brief"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class QuestionType(str, Enum):
    CONFIRM = "confirm"        # low-effort yes/no/correct
    OPEN = "open"              # free-form domain knowledge
    CALIBRATE = "calibrate"    # label N examples to tune a model
    LABEL = "label"            # name a topic / cluster
    JUDGE = "judge"            # set confidence / refine framing
    MEMORY_DIFF = "memory_diff"  # approve a proposed memory update


class PlannerAction(str, Enum):
    RUN_NODE = "run_node"
    ASK_HUMAN = "ask_human"
    REPLAN = "replan"
    FINISH = "finish"


# ─────────────────────────────────────────────────────────────────────────────
# Schema discovery + reshape
# ─────────────────────────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float
    unique_pct: float
    sample_values: list[str] = Field(max_length=5)


class DatasetSchema(BaseModel):
    """Canonical schema downstream nodes rely on. Only `id_col`, `text_col`,
    and `language` are required; the rest are optional capabilities that
    enable specific analyses (timestamp → temporal phases; engagement → proxy 01).
    """
    id_col: str
    text_col: str
    language: str  # ISO 639-1 / BCP-47, or "auto"
    timestamp_col: str | None = None
    author_col: str | None = None
    engagement_col: str | None = None
    platform_col: str | None = None
    extra_dims: dict[str, str] = Field(default_factory=dict)

    def capabilities(self) -> set[str]:
        caps = {"text", "id"}
        if self.timestamp_col: caps.add("temporal")
        if self.author_col: caps.add("author_profile")
        if self.engagement_col: caps.add("amplification")
        if self.platform_col: caps.add("cross_platform")
        return caps


class ReshapeOp(BaseModel):
    """One step in a reshape recipe. Executable by the deterministic reshape engine."""
    op: Literal[
        "rename", "concat", "split", "parse_datetime", "set_timezone",
        "aggregate", "explode", "melt", "filter", "drop_columns",
    ]
    source: list[str]
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""  # why the agent proposed this


class ReshapeRecipe(BaseModel):
    ops: list[ReshapeOp] = Field(default_factory=list)
    proposed_by: Literal["agent", "human", "memory"] = "agent"
    approved: bool = False

    def is_trivial(self) -> bool:
        """True if the recipe is just renames — auto-applicable without HITL."""
        return all(op.op == "rename" for op in self.ops)


# ─────────────────────────────────────────────────────────────────────────────
# Audit + chain of custody
# ─────────────────────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    timestamp: datetime
    stage: Stage
    action: str
    affected_rows: int | None = None
    reason: str
    artifact_path: str | None = None  # str, not Path — for msgpack serialization
    artifact_hash: str | None = None


class StageArtifact(BaseModel):
    """One stage's outputs. Files only — no DataFrames.
    Path-typed fields are stored as strings to keep state msgpack-serializable.
    """
    stage: Stage
    parquet_path: str | None = None
    parquet_hash: str | None = None
    figure_paths: list[str] = Field(default_factory=list)
    summary_stats: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Human-in-the-loop
# ─────────────────────────────────────────────────────────────────────────────

class HumanQuestion(BaseModel):
    question_id: str
    stage: Stage
    question_type: QuestionType
    prompt: str                  # the human-facing question text
    payload: dict[str, Any] = Field(default_factory=dict)  # data to render (samples, charts, tables)
    proposal: dict[str, Any] | None = None                  # agent's draft answer
    why_asking: str              # the uncertainty signal that triggered the question
    blocks_stage: bool = True    # does the graph wait, or proceed with a default?


class HumanResponse(BaseModel):
    question_id: str
    response: dict[str, Any]
    timestamp: datetime
    persist_to_memory: bool = False  # human chose to save this insight


# ─────────────────────────────────────────────────────────────────────────────
# Domain memory
# ─────────────────────────────────────────────────────────────────────────────

class PlatformProfile(BaseModel):
    name: str
    demographic: str = ""
    text_norm: str = ""           # e.g. "long-form discussion", "forwarded chains"
    coordination_baseline: str = ""  # what looks coordinated here vs. what's normal


class SarcasmPattern(BaseModel):
    pattern: str                  # regex or template
    polarity_flip: Literal["positive_to_negative", "negative_to_positive", "ambiguous"]
    example: str
    confidence: float = 0.7


class NarrativePriors(BaseModel):
    default_actors: list[str] = Field(default_factory=list)
    default_victims: list[str] = Field(default_factory=list)
    default_framings: list[str] = Field(default_factory=list)


class AnalysisThresholds(BaseModel):
    bot_quarantine_pct: float = 10.0
    outlier_rerun_pct: float = 25.0
    burst_sigma: float = 3.0
    duplication_alarm_pct: float = 20.0
    min_topic_size: int = 15


class DomainKnowledge(BaseModel):
    """Persisted per project (`projects/<name>/domain.yaml`).
    Loaded into planner context at startup, updated only via approved HITL diffs.
    """
    domain: str
    language: str
    platforms: dict[str, PlatformProfile] = Field(default_factory=dict)
    sarcasm_patterns: list[SarcasmPattern] = Field(default_factory=list)
    narrative_priors: NarrativePriors = Field(default_factory=NarrativePriors)
    thresholds: AnalysisThresholds = Field(default_factory=AnalysisThresholds)
    reshape_recipes: dict[str, ReshapeRecipe] = Field(default_factory=dict)  # keyed by source signature
    notes: list[str] = Field(default_factory=list)


class MemoryDiff(BaseModel):
    """A pending change to DomainKnowledge, awaiting human approval."""
    diff_id: str
    path: str                     # JSONPath into DomainKnowledge
    before: Any
    after: Any
    source_question_id: str | None = None
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# Findings + brief
# ─────────────────────────────────────────────────────────────────────────────

class BLUFFinding(BaseModel):
    """One finding in the integrated brief. Mirrors Day 4's `BLUF_FINDINGS` shape."""
    title: str
    bluf: str
    evidence: str
    confidence: Confidence
    confidence_reason: str
    limitations: list[str] = Field(min_length=2)  # at least two, per Day 4 rule
    recommendation: str
    related_topics: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Planner output
# ─────────────────────────────────────────────────────────────────────────────

class PlannerDecision(BaseModel):
    """Constrained JSON output from the planner LLM each turn."""
    action: PlannerAction
    next_stage: Stage | None = None       # required if action == RUN_NODE
    question: HumanQuestion | None = None  # required if action == ASK_HUMAN
    new_plan: list[Stage] | None = None    # required if action == REPLAN
    reasoning: str                         # short, ≤ 2 sentences


class ExecutionPlan(BaseModel):
    stages: list[Stage]
    created_at: datetime
    revision: int = 0
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level GraphState
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(BaseModel):
    """The single source of truth threaded through every LangGraph node.

    Categorized for readability — flat at the wire level so LangGraph's
    add_messages-style reducers can target individual fields if needed.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Identity
    run_id: str
    project_name: str             # name of the per-domain working dir
    started_at: datetime
    user_initial_prompt: str = ""  # what the user said when launching the run

    # Inputs
    raw_file_path: str
    raw_file_hash: str = ""

    # Schema discovery
    raw_columns: list[ColumnProfile] = Field(default_factory=list)
    proposed_schema: DatasetSchema | None = None
    confirmed_schema: DatasetSchema | None = None
    reshape_recipe: ReshapeRecipe | None = None
    canonical_data_path: str | None = None
    canonical_hash: str = ""

    # Domain memory (loaded at start, mutated only via approved diffs)
    domain_knowledge: DomainKnowledge
    pending_memory_diffs: list[MemoryDiff] = Field(default_factory=list)

    # Per-stage artifacts
    artifacts: dict[Stage, StageArtifact] = Field(default_factory=dict)

    # Audit log — append-only
    audit_log: list[AuditEntry] = Field(default_factory=list)

    # HITL queues
    pending_questions: list[HumanQuestion] = Field(default_factory=list)
    answered_questions: list[tuple[HumanQuestion, HumanResponse]] = Field(default_factory=list)

    # Planner
    plan: ExecutionPlan | None = None
    current_stage: Stage | None = None
    completed_stages: list[Stage] = Field(default_factory=list)
    last_decision: PlannerDecision | None = None

    # Findings — built up incrementally as topic + amplification land
    findings: list[BLUFFinding] = Field(default_factory=list)

    # Output
    brief_path: str | None = None

    # Convenience accessors -----------------------------------------------------

    def stage_done(self, stage: Stage) -> bool:
        return stage in self.completed_stages

    def latest_artifact(self, stage: Stage) -> StageArtifact | None:
        return self.artifacts.get(stage)

    def has_blocking_questions(self) -> bool:
        return any(q.blocks_stage for q in self.pending_questions)
