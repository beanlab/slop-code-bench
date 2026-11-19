"""Workflow snapshots and shared run provenance for reporting."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class WorkflowProvenance(BaseModel):
    """Identity of the workflow bundle, independent of its source location."""

    name: str
    source_dir: str
    entrypoint: str
    args: list[str] = Field(default_factory=list)
    content_hash: str
    myteam_version: str | None = None
    backend: str = "pi"
    pi_version: str | None = None


def copy_workflow(source: Path, destination: Path) -> None:
    """Copy source resources, excluding interpreter and VCS housekeeping."""
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )


def workflow_provenance(agent: dict[str, Any]) -> WorkflowProvenance | None:
    """Hash workflow paths and bytes deterministically, not modification times."""
    if agent.get("type") != "myteam":
        return None
    root = (
        Path(agent.get("workflow_dir", "configs/workflows/myteam/example"))
        .expanduser()
        .resolve()
    )
    if not root.is_dir():
        raise ValueError(f"Workflow directory does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".git"} for part in relative.parts):
            continue
        if not path.is_file() or path.suffix == ".pyc":
            continue
        name = relative.as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return WorkflowProvenance(
        name=root.name,
        source_dir=str(root),
        entrypoint=agent.get("workflow_entrypoint", "solve.py"),
        args=agent.get("workflow_args", []),
        content_hash=digest.hexdigest(),
        myteam_version=agent.get("version"),
        backend=agent.get("backend", "pi"),
        pi_version=agent.get("pi_version"),
    )


def run_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Read saved metadata without requiring the original workflow to exist.

    Legacy myteam runs only have a workflow directory reference. Do not
    invent a content hash for those runs by reading today's source files.
    """
    task_template = Path(config.get("prompt_path") or "unknown").stem
    agent = config.get("agent") or {}
    workflow = config.get("workflow")
    if agent.get("type") == "myteam" and not workflow:
        workflow = {
            "name": Path(agent.get("workflow_dir") or "example").name,
            "entrypoint": agent.get("workflow_entrypoint", "solve.py"),
            "args": agent.get("workflow_args", []),
            "content_hash": None,
        }
    identity = task_template
    if workflow:
        # Include invocation settings as well as source in grouping identity.
        invocation = json.dumps(
            [
                workflow.get("entrypoint"),
                workflow.get("args", []),
                task_template,
            ],
            sort_keys=True,
        )
        suffix = hashlib.sha256(invocation.encode()).hexdigest()[:8]
        content_hash = workflow.get("content_hash")
        identity = f"{workflow['name']}@{content_hash or 'unrecorded'}:{suffix}"
    return {
        "task_template": task_template,
        "workflow": workflow,
        "execution_kind": "workflow" if workflow else "prompt",
        "execution_identity": identity,
    }
