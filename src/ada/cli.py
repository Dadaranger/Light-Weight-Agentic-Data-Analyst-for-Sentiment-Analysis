"""Entry point — `ada run <file>` kicks off a pipeline.

  ada run <file> --project <name> [--prompt "..."] [--auto-confirm]
  ada resume <run_id> --response '<json>'
  ada inspect <run_id>
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import typer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ada.config import settings
from ada.graph import compile_graph
from ada.memory.store import load_domain
from ada.state import GraphState, PlannerAction

app = typer.Typer(help="Agentic Data Analyst for Sentiment Analysis.")
console = Console()

# Single in-process checkpointer so `run` and `resume` share state during testing.
# For a production deploy, use SqliteSaver pointed at projects/<name>/runs/<run_id>.db.
_CHECKPOINTER = MemorySaver()


def _print_decision(decision):
    if decision is None:
        return
    action = decision.action.value if hasattr(decision.action, "value") else decision.action
    console.print(f"  [dim]→[/dim] [bold]{action}[/bold] {decision.reasoning}")


def _print_question(payload: dict):
    console.print(Panel.fit(
        f"[bold yellow]{payload['prompt']}[/bold yellow]\n\n"
        f"[dim]why:[/dim] {payload['why_asking']}",
        title=f"HITL · {payload['stage']} · {payload['type']}",
        border_style="yellow",
    ))
    console.print("[dim]question_id:[/dim]", payload["question_id"])
    console.print("[dim]proposal:[/dim]")
    console.print_json(data=payload["proposal"])


def _stream(graph, input_, config):
    """Stream graph events; print decisions; surface interrupts."""
    interrupted = False
    for event in graph.stream(input_, config=config, stream_mode="values"):
        decision = event.get("last_decision") if isinstance(event, dict) else None
        _print_decision(decision)

    # Check final state for interrupts
    snapshot = graph.get_state(config)
    if snapshot.next:  # graph is paused (interrupt or pending node)
        # When interrupted, snapshot.tasks contains the pending interrupt payload
        for task in snapshot.tasks:
            if task.interrupts:
                interrupted = True
                payload = task.interrupts[0].value
                console.print()
                _print_question(payload)
                console.print()
                console.print(
                    f"[dim]Resume with:[/dim]\n"
                    f"  [cyan]ada resume {config['configurable']['thread_id']} "
                    f"--response '<json>'[/cyan]"
                )
    if not interrupted and snapshot.next:
        console.print(f"[dim]Graph paused at:[/dim] {snapshot.next}")


@app.command()
def run(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Dataset to analyze."),
    project: str = typer.Option(settings.default_project, help="Per-domain working dir name."),
    prompt: str = typer.Option("", help="Optional context about the dataset / analysis goal."),
    auto_confirm: bool = typer.Option(False, "--auto-confirm",
                                      help="Auto-accept agent's proposals at every HITL stop."),
):
    """Run the pipeline on a dataset."""
    run_id = uuid4().hex[:12]
    settings.project_path(project).mkdir(parents=True, exist_ok=True)
    state = GraphState(
        run_id=run_id,
        project_name=project,
        started_at=datetime.now(timezone.utc),
        user_initial_prompt=prompt,
        raw_file_path=file.resolve(),
        domain_knowledge=load_domain(project),
    )
    graph = compile_graph(checkpointer=_CHECKPOINTER)
    config = {"configurable": {"thread_id": run_id}}
    console.print(f"[bold cyan]Run {run_id}[/bold cyan] — project: {project}")

    _stream(graph, state, config)

    if auto_confirm:
        # Loop: keep approving every interrupt until the graph finishes
        while True:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                break
            interrupt_payload = None
            for task in snapshot.tasks:
                if task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
            if interrupt_payload is None:
                break
            response = interrupt_payload.get("proposal") or {"approved": True}
            console.print(f"[dim]auto-approving {interrupt_payload['question_id']}[/dim]")
            _stream(graph, Command(resume=response), config)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume."),
    response: str = typer.Option(..., help="JSON response payload."),
):
    """Resume an interrupted run by providing a human response."""
    graph = compile_graph(checkpointer=_CHECKPOINTER)
    config = {"configurable": {"thread_id": run_id}}
    payload = json.loads(response)
    _stream(graph, Command(resume=payload), config)


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run ID to inspect."),
):
    """Print the latest snapshot of a run: completed stages + audit tail."""
    graph = compile_graph(checkpointer=_CHECKPOINTER)
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    values = snapshot.values
    if not values:
        console.print(f"[red]No state found for run {run_id}[/red]")
        return

    completed = values.get("completed_stages", [])
    audit = values.get("audit_log", [])

    console.print(f"[bold]Run {run_id}[/bold]")
    console.print(f"Completed stages: {[s.value if hasattr(s, 'value') else s for s in completed]}")
    console.print(f"Pending: {snapshot.next}")

    table = Table(title="Audit log (tail)")
    table.add_column("Time"); table.add_column("Stage"); table.add_column("Action"); table.add_column("Rows")
    for e in audit[-10:]:
        ts = e.timestamp if hasattr(e, "timestamp") else e["timestamp"]
        stg = e.stage.value if hasattr(e, "stage") and hasattr(e.stage, "value") else str(e.stage if hasattr(e, "stage") else e["stage"])
        act = e.action if hasattr(e, "action") else e["action"]
        rows = e.affected_rows if hasattr(e, "affected_rows") else e.get("affected_rows")
        table.add_row(str(ts)[:19], stg, act, str(rows or ""))
    console.print(table)


if __name__ == "__main__":
    app()
