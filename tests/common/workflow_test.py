"""Workflow identity is content based and survives source removal."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from slop_code.agent_runner.agents.myteam import MyTeamConfig
from slop_code.common.workflow import copy_workflow
from slop_code.common.workflow import run_provenance
from slop_code.common.workflow import workflow_provenance
from slop_code.dashboard.data import load_config_metadata
from slop_code.entrypoints.commands.combine_results import _load_run_metadata
from slop_code.entrypoints.commands.consolidate_runs import flatten_result
from slop_code.entrypoints.commands.run_agent import _prepare_run_artifacts
from slop_code.entrypoints.commands.run_agent import _validate_resume_config
from slop_code.entrypoints.commands.variance import _load_group_key
from slop_code.entrypoints.config.loader import _build_interpolation_context
from slop_code.execution import LocalEnvironmentSpec
from slop_code.metrics.summary import compute_run_summary
from slop_code.problem_catalog import CatalogManifest


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "design_implement_review"
    root.mkdir()
    (root / "solve.py").write_text("# workflow\n")
    (root / "prompts").mkdir()
    (root / "prompts" / "design.txt").write_text("Design the task")
    agent = MyTeamConfig(
        workflow_dir=root,
        cost_limits={"cost_limit": 0, "net_cost_limit": 0},
    ).model_dump(mode="json")
    provenance = workflow_provenance(agent)
    assert provenance is not None
    return {
        "agent": agent,
        "workflow": provenance.model_dump(mode="json"),
        "prompt_path": "configs/prompts/just-solve.jinja",
        "thinking": "low",
        "model": {"name": "test-model", "provider": "anthropic"},
    }


def test_hash_covers_resources_not_location_or_cache(
    config: dict[str, Any], tmp_path: Path
) -> None:
    agent = config["agent"]
    source = Path(agent["workflow_dir"])
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "solve.pyc").write_bytes(b"cache")
    copied = tmp_path / "copied"
    copy_workflow(source, copied)
    assert not (copied / "__pycache__").exists()
    before = workflow_provenance(agent)
    assert before is not None
    assert before.content_hash == config["workflow"]["content_hash"]
    moved = workflow_provenance({**agent, "workflow_dir": copied})
    assert moved is not None
    assert moved.content_hash == before.content_hash
    (copied / "prompts" / "design.txt").write_text("New instructions")
    changed = workflow_provenance({**agent, "workflow_dir": copied})
    assert changed is not None
    assert changed.content_hash != before.content_hash


def test_reporting_uses_saved_workflow_without_source(
    config: dict[str, Any], tmp_path: Path
) -> None:
    shutil.rmtree(config["agent"]["workflow_dir"])
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    summary = compute_run_summary(config, [], expected_checkpoints=1)
    assert summary.prompt is None
    assert summary.task_template == "just-solve"
    assert summary.workflow == config["workflow"]
    assert summary.execution_kind == "workflow"
    flattened = flatten_result(summary.model_dump(), "test", "run", "today")
    assert flattened["workflow_hash"] == config["workflow"]["content_hash"]
    assert flattened["workflow_name"] == "design_implement_review"
    assert flattened["execution_identity"] == summary.execution_identity
    dashboard = load_config_metadata(tmp_path)
    assert dashboard["prompt_template"] == summary.execution_identity
    assert dashboard["workflow"] == summary.workflow
    combined = _load_run_metadata(tmp_path, tmp_path.parent)
    assert combined["prompt_name"] is None
    assert combined["execution_identity"] == summary.execution_identity
    key = _load_group_key(tmp_path)
    assert key.workflow_identity == summary.execution_identity
    config["workflow"]["args"] = ["--review-twice"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    assert _load_group_key(tmp_path) != key


def test_legacy_metadata_is_not_reconstructed_from_current_sources(
    config: dict[str, Any],
) -> None:
    config.pop("workflow")
    legacy = run_provenance(config)
    assert legacy["execution_kind"] == "workflow"
    assert legacy["workflow"]["content_hash"] is None
    assert "unrecorded" in legacy["execution_identity"]
    config["agent"] = {"type": "pi"}
    normal = run_provenance(config)
    assert normal["workflow"] is None
    assert normal["execution_identity"] == "just-solve"


@pytest.mark.parametrize("change", ["content", "entrypoint", "args"])
def test_resume_rejects_workflow_changes(
    config: dict[str, Any], tmp_path: Path, change: str
) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    run_cfg = MagicMock()
    run_cfg.model_dump.return_value = config
    env = LocalEnvironmentSpec(name="test")
    assert _validate_resume_config(tmp_path, run_cfg, env) == []
    if change == "content":
        root = Path(config["agent"]["workflow_dir"])
        (root / "prompts" / "design.txt").write_text("Different prompt")
    elif change == "entrypoint":
        config["agent"]["workflow_entrypoint"] = "other.py"
    else:
        config["agent"]["workflow_args"] = ["--extra"]
    mismatches = _validate_resume_config(tmp_path, run_cfg, env)
    expected = "content_hash" if change == "content" else change
    assert any(field == f"workflow.{expected}" for field, _, _ in mismatches)


def test_run_workers_use_snapshot_and_keep_original_provenance(
    config: dict[str, Any], tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = MyTeamConfig.model_validate(config["agent"])
    run_cfg = MagicMock()
    run_cfg.workflow = workflow_provenance(config["agent"])
    run_cfg.model_dump.side_effect = lambda **_: {
        **config,
        "workflow": run_cfg.workflow.model_dump(mode="json"),
    }
    _prepare_run_artifacts(
        run_dir,
        LocalEnvironmentSpec(name="test"),
        agent,
        run_cfg,
        CatalogManifest(version="v1", commit="abc"),
    )
    original = Path(config["agent"]["workflow_dir"])
    assert agent.workflow_dir != original
    assert agent.workflow_dir.is_relative_to(run_dir)
    shutil.rmtree(original)
    assert (agent.workflow_dir / "prompts" / "design.txt").read_text() == (
        "Design the task"
    )
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["workflow"] == config["workflow"]
    assert saved["agent"]["workflow_dir"] == str(original)
    assert agent.workflow_metadata == run_cfg.workflow


def test_output_name_uses_workflow_instead_of_task_template(
    config: dict[str, Any],
) -> None:
    context = _build_interpolation_context(
        config["agent"], {}, Path(config["prompt_path"]), MagicMock(), "low"
    )
    assert context["prompt"] == "design_implement_review"
    assert context["workflow"] == "design_implement_review"
