"""Per-project domain knowledge persistence.

Layout:
    projects/<name>/
      domain.yaml          # DomainKnowledge serialized
      data/                # raw uploads
      artifacts/           # parquet + figures per stage
      runs/<run_id>/       # checkpoint + audit log per run

Memory updates ALWAYS go through `apply_diff` after human approval — never
edit `domain.yaml` directly from a node.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ada.config import settings
from ada.state import DomainKnowledge, MemoryDiff


def _domain_yaml_path(project_name: str) -> Path:
    return settings.project_path(project_name) / "domain.yaml"


def load_domain(project_name: str) -> DomainKnowledge:
    """Load `domain.yaml` for a project. Returns a default-initialized
    DomainKnowledge if the file doesn't exist (first run on this domain).
    """
    path = _domain_yaml_path(project_name)
    if not path.exists():
        return DomainKnowledge(domain=project_name, language="auto")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DomainKnowledge.model_validate(data)


def save_domain(project_name: str, dk: DomainKnowledge) -> None:
    path = _domain_yaml_path(project_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dk.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def apply_diff(dk: DomainKnowledge, diff: MemoryDiff) -> DomainKnowledge:
    """Apply an approved MemoryDiff. Path is dotted (e.g. `thresholds.bot_quarantine_pct`)."""
    data = dk.model_dump()
    keys = diff.path.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = diff.after
    return DomainKnowledge.model_validate(data)
