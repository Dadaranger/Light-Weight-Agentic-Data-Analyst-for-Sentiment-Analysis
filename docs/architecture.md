# Architecture

This document captures the design decisions behind the project. The README explains *what* exists; this explains *why*.

## Stage pipeline

Modeled directly on the Week 2 sentiment course's five-day workflow. Each stage is a LangGraph subgraph that consumes a parquet path + schema and produces a new artifact + audit entries.

```
ingest → schema_infer → reshape → eda → clean
       → preprocess → sentiment → topic
       → narrative → amplification → brief
```

Stages with optional capabilities self-disable when the schema lacks the required column (e.g. `amplification` proxy 03 needs `timestamp_col`).

## Planner pattern: hybrid plan-then-execute

Pure supervisor (LLM picks next node every step) is too expensive on a 7B model and produces unpredictable runs. Pure DAG is too rigid — Day 1 EDA findings legitimately change downstream choices (bot %, outlier %).

The compromise: **plan once, replan at three checkpoints** (after EDA, after sentiment, after topic). The planner is otherwise a router. `PlannerDecision` is constrained JSON with four actions: `RUN_NODE`, `ASK_HUMAN`, `REPLAN`, `FINISH`.

## State design

The `GraphState` Pydantic model in `src/ada/state.py` is the single source of truth. Two rules:

1. **DataFrames never live in state.** Only file paths, schemas, summary stats, and LLM-facing text. Inter-stage handoff is parquet on disk.
2. **State is append-only where possible.** `audit_log`, `completed_stages`, `findings` only grow. Mutations are state diffs returned by nodes.

This makes checkpointing cheap and resume-from-interrupt reliable.

## Where the LLM thinks

| LLM | Code |
|---|---|
| Schema inference from column names + samples | SHA-256, dtype enforcement, parsing |
| Reshape recipe proposal | Recipe execution |
| Language → tokenizer routing | Tokenization itself |
| Topic cluster naming (analyst labels) | UMAP / HDBSCAN / c-TF-IDF |
| Narrative 6-element extraction | Topic statistics |
| Confidence assessment + BLUF prose | Burst detection (3σ), engagement %s |
| Question generation for HITL | Threshold checks |

Rule of thumb: if removing the LLM would make the step less *judgmental* but not less *correct*, it belongs in code.

## HITL — five interrupt points

| Stage | Question type | What's shown |
|---|---|---|
| `schema_infer` | confirm | column profiles + proposed mapping |
| `reshape` (non-trivial only) | confirm | recipe ops + before/after sample |
| `eda` (when memory is thin) | open | platform mix, time window, language profile |
| `sentiment` (when rules + transformer disagree > 15%) | calibrate | 10 disputed posts |
| `topic` | label | top keywords + 8 representative posts per cluster |
| `narrative` (borderline confidence) | judge | draft BLUF + evidence |
| any stage | memory_diff | proposed YAML change, before/after |

Driven by the Q&A from planning: no question cap, all questions ask "I propose X — correct me", never "what should I do?".

## Memory — per-domain, conflict-surfaced

`projects/<name>/domain.yaml` holds:
- `platforms`: per-platform demographics + coordination baselines
- `sarcasm_patterns`: regex + polarity flip + example
- `narrative_priors`: default actors / victims / framings
- `thresholds`: per-domain overrides for things like `bot_quarantine_pct`
- `reshape_recipes`: keyed by source signature (column hash) — auto-applied on matching uploads

A shared `src/ada/memory/seeds/language_norms.yaml` provides cross-cutting language facts (Jieba dict extras, platform norms for zh-TW, etc.). Seeds are loaded once and merged into a project's `DomainKnowledge` only when the project's `language` matches.

Updates always go through the `MemoryDiff` flow: a node proposes, the planner emits an `ASK_HUMAN` of type `memory_diff`, the human approves or edits, and only then does `apply_diff` mutate the YAML. **Conflicts surface as a diff with both versions** — silent overwrites are forbidden.

Warm-start runs show what was skipped: when the planner uses a memory-derived default to bypass a question, it logs `"skipped Q-X: domain memory says Y"` in `reasoning`.

## Reshape: agent reshapes, doesn't reject

Three layers of input flexibility:

1. **File format**: CSV / XLSX / JSON / JSONL / Parquet / SQLite — handled by code in `tools/loader.py` (TBD).
2. **Column renaming**: the LLM maps user columns to the canonical `DatasetSchema`.
3. **Structural reshaping**: the LLM proposes a `ReshapeRecipe` (sequence of typed `ReshapeOp`s); code executes deterministically.

The chain-of-custody hash is computed on the **reshaped parquet**, with the original file hash + recipe both recorded — reproducibility preserved.

Fail-loud cases:
- No plausible text column → reject with diagnostic
- > 50% rows fail to parse after reshape → reject
- No tokenizer for detected language → continue but flag for downstream limitation

## Local LLM stack

| Role | Model | Why |
|---|---|---|
| Planner / writer | Qwen 2.5 7B-Instruct | Strong zh-TW + multilingual + JSON-mode + function-calling |
| Cheap labeler | Qwen 2.5 3B-Instruct | Per-cluster labels, sentiment disambiguation |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Already used in the Day 3 course material |
| Topic | BERTopic + UMAP + HDBSCAN | Direct port from Day 3 |

Served via Ollama. JSON-mode is used for everything except the brief writer (free-form text).

## What's deliberately NOT in scope

- **Network graph analysis**: would need account-level relationships (followers, retweets). The four amplification *proxies* substitute.
- **Cross-platform identity resolution**: out of scope.
- **Real-time streaming**: batch only.
- **Generic AutoML**: we are explicitly text-bearing-tabular for sentiment/topic/narrative. Other dataset types should fail fast at schema inference.

## Vertical-slice build order (next steps)

1. `ingest` + `schema_infer` + `reshape` — get a CSV through to canonical parquet with HITL on schema confirmation
2. `eda` + `clean` — port the Day 1 audit log + three-stream split
3. `preprocess` + `sentiment` — three-tier baseline from Day 2
4. `topic` + HITL labeling loop — Day 3
5. `narrative` + `amplification` + `brief` — Day 4

Each slice is end-to-end runnable on the typhoon-Krathon dataset before moving on.
