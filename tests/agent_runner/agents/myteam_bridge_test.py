"""Behavioral tests for the portable myteam workflow helper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from slop_code.agent_runner.agents.myteam import scb_myteam


@dataclass
class FakeResult:
    session_id: str | None = "native-session"
    cost: float = 0.5
    input_tokens: int = 100
    exit_code: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exit_code": self.exit_code,
            "output": {"summary": "done"},
            "transcript": "agent display",
            "usage": [
                {
                    "model": "test-model",
                    "input_tokens": self.input_tokens,
                    "cached_input_tokens": 20,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 2,
                    "total_tokens": self.input_tokens + 10,
                    "estimated_cost": self.cost,
                }
            ],
        }


@dataclass
class NativeResult:
    session_id: str | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    exit_code: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "usage": self.usage,
            "exit_code": self.exit_code,
            "output": {"summary": "done"},
            "transcript": "native display",
        }


@pytest.fixture
def workflow_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payload = {
        "task": "Implement the checkpoint.",
        "model": "test-model",
        "reasoning": "high",
        "backend": "pi",
        "workflow": str(tmp_path / "workflow.py"),
        "workflow_args": ["--custom", "value"],
        "artifacts_dir": str(artifacts),
        "cost_limits": {"cost_limit": 0, "net_cost_limit": 0},
        "prior_cost": 0,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("SCB_MYTEAM_REQUEST", str(path))
    return payload


def _write_workflow_request(
    monkeypatch: pytest.MonkeyPatch, workflow_request: dict[str, Any]
) -> None:
    path = Path(scb_myteam.os.environ["SCB_MYTEAM_REQUEST"])
    path.write_text(json.dumps(workflow_request))


def _fake_myteam(
    monkeypatch: pytest.MonkeyPatch,
    results: list[FakeResult],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    module = ModuleType("myteam")

    def run_agent(**kwargs: Any) -> FakeResult:
        calls.append(kwargs)
        return results.pop(0)

    module.run_agent = run_agent
    monkeypatch.setitem(sys.modules, "myteam", module)
    return calls


def _records(workflow_request: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(workflow_request["artifacts_dir"]) / "sessions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_task_and_headless_defaults_reach_native_agent(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = FakeResult()
    calls = _fake_myteam(monkeypatch, [result])

    assert scb_myteam.load_request()["task"] == workflow_request["task"]
    assert scb_myteam.run_agent(prompt=workflow_request["task"]) is result

    options = calls[0].pop("extra_args")
    assert options[0] == "--session-dir"
    assert Path(options[1]).is_relative_to(
        Path(workflow_request["artifacts_dir"]) / "native_sessions"
    )
    assert calls == [
        {
            "prompt": workflow_request["task"],
            "agent": "pi",
            "model": "test-model",
            "reasoning": "high",
            "interactive": False,
            "fork": False,
        }
    ]
    record = _records(workflow_request)[0]
    assert record["session_index"] == 1
    assert record["result"]["transcript"] == "agent display"
    assert record["usage_delta"][0]["input_tokens"] == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interactive": True},
        {"agent": "codex"},
        {"fork": True},
        {"extra_args": ["--no-session"]},
        {"extra_args": ["--session-dir=/elsewhere"]},
        {"extra_args": ["--session", "elsewhere.jsonl"]},
    ],
)
def test_unsupported_sessions_fail_before_launch(
    workflow_request: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    calls = _fake_myteam(monkeypatch, [])
    with pytest.raises((ValueError, RuntimeError)):
        scb_myteam.run_agent(prompt=workflow_request["task"], **kwargs)
    assert calls == []


def test_resume_accounts_only_for_new_tokens_and_cost(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_myteam(
        monkeypatch,
        [FakeResult(), FakeResult(cost=0.8, input_tokens=160)],
    )
    scb_myteam.run_agent(prompt="First task")
    scb_myteam.run_agent(prompt="Continue", session_id="native-session")

    records = _records(workflow_request)
    assert records[1]["result"]["usage"][0]["input_tokens"] == 160
    delta = records[1]["usage_delta"][0]
    assert delta["input_tokens"] == 60
    assert delta["estimated_cost"] == pytest.approx(0.3)
    assert delta["cached_input_tokens"] == 0


def test_sessions_without_ids_are_charged_independently(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_myteam(
        monkeypatch,
        [FakeResult(session_id=None), FakeResult(session_id=None)],
    )
    scb_myteam.run_agent(prompt="One")
    scb_myteam.run_agent(prompt="Two")
    assert (
        sum(
            record["usage_delta"][0]["estimated_cost"]
            for record in _records(workflow_request)
        )
        == 1
    )


def test_failed_session_keeps_usage_and_transcript(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_myteam(monkeypatch, [FakeResult(exit_code=7)])
    with pytest.raises(RuntimeError):
        scb_myteam.run_agent(prompt="Fail")
    assert _records(workflow_request)[0]["result"]["exit_code"] == 7
    assert (
        _records(workflow_request)[0]["usage_delta"][0]["estimated_cost"] == 0.5
    )


def test_sigterm_after_structured_result_preserves_session_output(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = FakeResult(exit_code=143)
    _fake_myteam(monkeypatch, [result])

    assert scb_myteam.run_agent(prompt="Completed result") is result
    record = _records(workflow_request)[0]
    assert record["result"]["output"] == {"summary": "done"}
    events = [
        json.loads(line)
        for line in (
            Path(workflow_request["artifacts_dir"]) / "events.jsonl"
        ).read_text().splitlines()
    ]
    assert any(
        event["event"] == "session_result_preserved_after_sigterm"
        for event in events
    )


def test_budget_stops_next_session_and_keeps_completed_result(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_request["cost_limits"]["cost_limit"] = 0.4
    _write_workflow_request(monkeypatch, workflow_request)
    calls = _fake_myteam(monkeypatch, [FakeResult()])
    with pytest.raises(RuntimeError):
        scb_myteam.run_agent(prompt="Exceed budget")
    assert len(_records(workflow_request)) == 1
    with pytest.raises(RuntimeError):
        scb_myteam.run_agent(prompt="Must not launch")
    assert len(calls) == 1


def test_prior_checkpoint_cost_counts_toward_net_budget(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_request["cost_limits"]["net_cost_limit"] = 1
    workflow_request["prior_cost"] = 1
    _write_workflow_request(monkeypatch, workflow_request)
    calls = _fake_myteam(monkeypatch, [])
    with pytest.raises(RuntimeError):
        scb_myteam.run_agent(prompt="Must not launch")
    assert calls == []


def test_main_uses_supervised_workflow_launch(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    workflows = ModuleType("myteam.workflows")

    def start_workflow_cli(*args: str) -> None:
        calls.append(args)

    workflows.start_workflow_cli = start_workflow_cli
    monkeypatch.setitem(sys.modules, "myteam.workflows", workflows)
    scb_myteam.main()
    assert calls == [(workflow_request["workflow"], "--custom", "value")]


def test_running_session_is_visible_before_native_call_returns(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("myteam")

    def run_agent(**kwargs: Any) -> FakeResult:
        events_path = Path(workflow_request["artifacts_dir"]) / "events.jsonl"
        events = [
            json.loads(line) for line in events_path.read_text().splitlines()
        ]
        assert events[-1]["event"] == "session_started"
        return FakeResult()

    module.run_agent = run_agent
    monkeypatch.setitem(sys.modules, "myteam", module)
    scb_myteam.run_agent(prompt="Do not retain this prompt in lifecycle logs")
    events = (
        Path(workflow_request["artifacts_dir"]) / "events.jsonl"
    ).read_text()
    assert "session_finished" in events
    assert "Do not retain this prompt" not in events


def test_stalled_workflow_exits_with_retained_idle_diagnostic(
    workflow_request: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "fake_package" / "myteam"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "workflows.py").write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "def start_workflow_cli(*args):\n"
        "    request = json.loads(Path(os.environ['SCB_MYTEAM_REQUEST']).read_text())\n"
        "    native = Path(request['artifacts_dir']) / 'native_sessions'\n"
        "    native.mkdir()\n"
        "    (native / 'stalled.jsonl').write_text('{\"tool\": \"stalled bash\"}\\n')\n"
        "    time.sleep(60)\n"
    )
    workflow_request["idle_timeout"] = 0.2
    workflow_request["timeout"] = 20
    _write_workflow_request(monkeypatch, workflow_request)
    env = {**os.environ, "PYTHONPATH": str(package.parent)}
    completed = subprocess.run(  # noqa: S603 - executes the staged helper.
        [sys.executable, scb_myteam.__file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )
    assert completed.returncode != 0
    artifacts = Path(workflow_request["artifacts_dir"])
    assert "idle" in (artifacts / "idle_timeout.txt").read_text().lower()
    assert (artifacts / "events.jsonl").is_file()
    assert (
        "stalled bash"
        in (artifacts / "native_sessions" / "stalled.jsonl").read_text()
    )


def test_native_activity_resets_idle_deadline(
    workflow_request: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "fake_package" / "myteam"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "workflows.py").write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "def start_workflow_cli(*args):\n"
        "    request = json.loads(Path(os.environ['SCB_MYTEAM_REQUEST']).read_text())\n"
        "    native = Path(request['artifacts_dir']) / 'native_sessions'\n"
        "    native.mkdir()\n"
        "    with (native / 'active.jsonl').open('w') as stream:\n"
        "        for index in range(12):\n"
        "            stream.write(json.dumps({'step': index}) + '\\n')\n"
        "            stream.flush()\n"
        "            time.sleep(0.05)\n"
    )
    workflow_request["idle_timeout"] = 0.2
    workflow_request["timeout"] = 20
    _write_workflow_request(monkeypatch, workflow_request)
    completed = subprocess.run(  # noqa: S603 - executes the staged helper.
        [sys.executable, scb_myteam.__file__],
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifacts = Path(workflow_request["artifacts_dir"])
    assert not (artifacts / "idle_timeout.txt").exists()


def test_native_file_repairs_missing_metadata_and_resumes_known_file(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    native_files: list[Path] = []
    module = ModuleType("myteam")
    usage = [{"model": "native-model", "estimated_cost": 0.25}]

    def run_agent(**kwargs: Any) -> NativeResult:
        calls.append(kwargs)
        if len(calls) == 1:
            options = kwargs["extra_args"]
            native = Path(options[options.index("--session-dir") + 1])
            path = native / "native.jsonl"
            path.write_text(
                json.dumps({"type": "session", "id": "native-id"}) + "\n"
            )
            native_files.append(path)
        return NativeResult()

    parser = ModuleType("myteam.workflows.agents.pi")

    def get_usage_info(path: Path) -> list[dict[str, Any]]:
        assert path == native_files[0]
        return usage

    module.run_agent = run_agent
    parser.get_usage_info = get_usage_info
    monkeypatch.setitem(sys.modules, "myteam", module)
    monkeypatch.setitem(sys.modules, "myteam.workflows.agents.pi", parser)

    first = scb_myteam.run_agent(prompt="First")
    assert first.session_id == "native-id"
    assert first.usage == usage
    second = scb_myteam.run_agent(prompt="Continue", session_id="native-id")
    assert calls[1]["session_id"] == str(native_files[0])
    assert second.session_id == "native-id"
    assert _records(workflow_request)[0]["usage_delta"] == [
        {
            "model": "native-model",
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.25,
        }
    ]
    assert (
        _records(workflow_request)[1]["usage_delta"][0]["estimated_cost"] == 0
    )


def test_benchmark_pricing_replaces_pi_cost_per_message(
    workflow_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_request["pricing"] = {
        "input": 2.0,
        "output": 3.0,
        "cache_read": 0.2,
        "cache_write": 0,
        "prompt_tiers": [
            {
                "max_input_tokens": 1000,
                "input": 1.0,
                "output": 2.0,
                "cache_read": 0.1,
                "cache_write": 0,
            }
        ],
    }
    _write_workflow_request(monkeypatch, workflow_request)
    module = ModuleType("myteam")
    messages = [
        {"input": 400, "cacheRead": 500, "output": 100},
        {"input": 600, "cacheRead": 900, "output": 100},
    ]

    def run_agent(**kwargs: Any) -> FakeResult:
        options = kwargs["extra_args"]
        native = Path(options[options.index("--session-dir") + 1])
        (native / "native.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "message": {
                            "model": "test-model",
                            "usage": {**usage, "cost": {"total": 99}},
                        }
                    }
                )
                for usage in messages
            )
        )
        return FakeResult(cost=99)

    module.run_agent = run_agent
    monkeypatch.setitem(sys.modules, "myteam", module)

    scb_myteam.run_agent(prompt="Price me")

    # First message fits the 1000-token tier; the second uses base rates.
    expected = (400 * 1.0 + 500 * 0.1 + 100 * 2.0) / 1e6 + (
        600 * 2.0 + 900 * 0.2 + 100 * 3.0
    ) / 1e6
    delta = _records(workflow_request)[0]["usage_delta"][0]
    assert delta["estimated_cost"] == pytest.approx(expected)
