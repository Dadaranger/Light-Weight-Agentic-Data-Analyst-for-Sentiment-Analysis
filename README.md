# Agentic Data Analyst for Sentiment Analysis

A LangGraph-based agent that runs an end-to-end sentiment / topic / narrative analysis on text-bearing tabular datasets in any language, with a human-in-the-loop for cultural and domain nuance.

The agent is modeled on a five-stage analyst workflow:

1. **Ingest** — schema inference, optional reshape, chain-of-custody hashing, audit log
2. **EDA + Clean** — profiling, dedup, three-stream split (organic / quarantine / bots)
3. **Preprocess + Sentiment baseline** — language-aware tokenization, three-tier sentiment (rules → lexicon → transformer)
4. **Topic + Narrative** — BERTopic with multilingual embeddings, six-element narrative framework, amplification proxies
5. **Brief** — BLUF-format integrated analytic brief with confidence levels and limitations

Each stage is a LangGraph subgraph. A central planner decides which stage runs next, when to interrupt for human input, and when to update domain memory.

## Design principles

- **The LLM plans and writes; code does the analytics.** Stats, hashing, viz, parsing — deterministic. Schema inference, narrative labels, confidence assessments, brief prose — LLM.
- **DataFrames live on disk, not in graph state.** State carries file paths, schemas, audit entries, and LLM-facing summaries. Inter-stage handoff is parquet.
- **Every transformation is auditable.** Reshape recipes, cleaning decisions, label assignments — all logged with reason and replayable.
- **Human input is structured.** The agent always proposes a draft and asks "correct me", never "what should I do?" Question types are typed (`confirm`, `open`, `calibrate`, `label`, `judge`).
- **Domain knowledge persists per project.** Sarcasm patterns, platform demographics, narrative priors, reshape recipes — stored in `domain.yaml`, loaded into planner context, updated only with human approval.

## Local LLM stack

- **Planner / writer:** Qwen 2.5 7B-Instruct via Ollama (strong zh-TW + multilingual + function-calling)
- **Cheap router / labeler:** Qwen 2.5 3B
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- **Topic model:** BERTopic + UMAP + HDBSCAN

## Repository layout

```
src/ada/
  config.py              # Settings (LLM endpoints, paths, thresholds)
  state.py               # Pydantic models for the LangGraph state
  graph.py               # Graph assembly (planner + subgraphs)
  cli.py                 # Entry point

  llm/
    client.py            # Ollama wrapper
    prompts/
      planner.md         # Supervisor prompt template
      schema_inference.md
      question_generator.md
      brief_writer.md

  nodes/                 # Per-stage node implementations
  tools/                 # Deterministic helpers (hashing, tokenize, stats, viz)
  memory/
    store.py             # domain.yaml read/write
    seeds/               # Bundled language norms

projects/                # Per-domain working dirs (data, runs, domain.yaml)
docs/
  architecture.md        # Detailed design notes
```

## Setup

Project uses [uv](https://docs.astral.sh/uv/) for dependency management. All Python deps install into a project-local `.venv` — nothing pollutes your system Python.

**One-time:**
```
setup.bat         # Windows
./setup.sh        # Bash / WSL
```
This creates `.venv` (Python 3.13), installs all runtime + dev dependencies, and registers the `ada` package in editable mode. ~1.5 GB total (torch + sentence-transformers are the bulk).

If you don't have uv yet:
```
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`requirements.txt` is a locked snapshot — useful for reproducing the exact environment elsewhere (`uv pip install -r requirements.txt`).

## Running

**UI (recommended):**
```
run_ui.bat        # Windows
./run_ui.sh       # Bash / WSL
```
Opens the Streamlit app at http://localhost:8501. Four pages: 啟動分析 (upload + start) → 進度追蹤 (live audit log) → 人工確認 (HITL forms — schema, topic labels, sentiment calibration) → 分析報告 (BLUF brief in Traditional Chinese).

**CLI (headless / scripting):**
```
run_cli.bat run path\to\data.csv --project mydomain --auto-confirm   # Windows
./run_cli.sh run path/to/data.csv --project mydomain --auto-confirm  # Bash
```

## Status

All five vertical slices implemented and tested end-to-end on the typhoon-Krathon dataset. UI in Traditional Chinese. Brief renders in zh-TW or English depending on the dataset's declared language.

## Provenance

Built on the Week 2 Sentiment Analysis course materials (Day 1–5 instructor demos). The course's Traditional Chinese typhoon-Krathon dataset is the reference test case.
