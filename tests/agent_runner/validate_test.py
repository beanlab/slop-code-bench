"""Tests for agent-callable validation."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from slop_code.agent_runner.validate import EXIT_BROKEN
from slop_code.agent_runner.validate import EXIT_FAIL
from slop_code.agent_runner.validate import EXIT_PASS
from slop_code.agent_runner.validate import PROMPT_ADDENDUM
from slop_code.agent_runner.validate import ValidateServer
from slop_code.agent_runner.validate import build_report
from slop_code.agent_runner.validate import collapse_cases
from slop_code.evaluation.report import CorrectnessResults
from slop_code.evaluation.report import GroupType
from slop_code.evaluation.report import TestResult as EvalTestResult


def _result(
    test_id: str,
    status: str,
    group: GroupType = GroupType.CORE,
    file_path: str = "tests/test_checkpoint_1.py",
) -> EvalTestResult:
    return EvalTestResult(
        id=test_id,
        checkpoint="checkpoint_1",
        group_type=group,
        status=status,
        duration_ms=1.0,
        file_path=file_path,
        failure_message="expected 3, got 4",
    )


def _results(
    tests: list[EvalTestResult], *, infrastructure_failure: bool = False
) -> CorrectnessResults:
    pass_counts: dict[GroupType, int] = {}
    total_counts: dict[GroupType, int] = {}
    for test in tests:
        total_counts[test.group_type] = total_counts.get(test.group_type, 0) + 1
        if test.status == "passed":
            pass_counts[test.group_type] = (
                pass_counts.get(test.group_type, 0) + 1
            )
    return CorrectnessResults(
        problem_name="demo",
        problem_version=1,
        checkpoint_name="checkpoint_1",
        checkpoint_version=1,
        duration=1.0,
        entrypoint="python main.py",
        tests=tests,
        pass_counts=pass_counts,
        total_counts=total_counts,
        pytest_exit_code=1,
        pytest_collected=len(tests),
        infrastructure_failure=infrastructure_failure,
    )


class TestCollapseCases:
    def test_groups_parametrized_cases(self):
        collapsed = collapse_cases(
            [
                _result("test_table[0-1-1]", "failed"),
                _result("test_table[1-0-1]", "passed"),
                _result("test_table[1-1-0]", "error"),
                _result("test_plain", "failed"),
                _result("test_ok", "passed"),
            ]
        )

        assert collapsed == [
            {"id": "test_table", "group": "Core", "failed": 2, "cases": 3},
            {"id": "test_plain", "group": "Core", "failed": 1, "cases": 1},
        ]

    def test_keeps_same_name_in_different_files_apart(self):
        collapsed = collapse_cases(
            [
                _result("test_x[a]", "failed", file_path="tests/test_a.py"),
                _result("test_x[a]", "failed", file_path="tests/test_b.py"),
            ]
        )

        assert len(collapsed) == 2

    def test_orders_core_first(self):
        collapsed = collapse_cases(
            [
                _result("test_old", "failed", GroupType.REGRESSION),
                _result("test_edge", "failed", GroupType.ERROR),
                _result("test_main", "failed", GroupType.CORE),
            ]
        )

        assert [entry["id"] for entry in collapsed] == [
            "test_main",
            "test_edge",
            "test_old",
        ]

    def test_keeps_class_prefix(self):
        collapsed = collapse_cases(
            [_result("TestGroup::test_method[1]", "failed")]
        )

        assert collapsed[0]["id"] == "TestGroup::test_method"


class TestBuildReport:
    def test_hides_case_ids_and_assertion_messages(self):
        results = _results(
            [
                _result("test_plain", "passed"),
                _result("test_table[0-1-1]", "failed"),
                _result("test_table[1-1-0]", "failed"),
                _result("test_table[1-0-1]", "passed"),
                _result("test_bad_input", "error", GroupType.ERROR),
                _result("test_skip", "skipped"),
            ]
        )

        report = build_report(results)

        assert report["exit_code"] == EXIT_FAIL
        assert report["passed"] == 2
        assert report["total"] == 6
        assert report["message"] == (
            "2/6 tests passed.\n"
            "core 2/5, error 0/1\n"
            "\n"
            "Failing tests:\n"
            "- [core] test_table: 2 of 3 cases failed\n"
            "- [error] test_bad_input"
        )
        assert "0-1-1" not in report["message"]
        assert "expected 3" not in report["message"]

    def test_log_keeps_raw_case_ids(self):
        report = build_report(_results([_result("test_t[0-1]", "failed")]))

        assert report["failing_cases"] == [
            {"id": "test_t[0-1]", "group": "Core"}
        ]

    def test_all_passed(self):
        report = build_report(_results([_result("test_plain", "passed")]))

        assert report["all_passed"] is True
        assert report["exit_code"] == EXIT_PASS
        assert "Failing tests" not in report["message"]

    def test_infrastructure_failure_never_passes(self):
        report = build_report(_results([], infrastructure_failure=True))

        assert report["all_passed"] is False
        assert report["exit_code"] == EXIT_FAIL
        assert report["message"].startswith("The test suite could not run")


class TestValidateServer:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)
        (workspace / ".venv" / "big.bin").write_text("x")
        (workspace / "main.py").write_text("print('hi')")
        return workspace

    def _server(self, tmp_path: Path, workspace: Path) -> ValidateServer:
        return ValidateServer(
            ipc_dir=tmp_path / "ipc",
            workspace_dir=workspace,
            ignore_globs={".venv/*"},
        )

    def _call(self, server: ValidateServer) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            ["sh", str(server.ipc_dir / "scb-validate")],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_round_trip(self, tmp_path, workspace):
        server = self._server(tmp_path, workspace)
        seen: list[set[str]] = []

        def evaluate(submission: Path) -> CorrectnessResults:
            seen.append(
                {p.relative_to(submission).as_posix() for p in submission.rglob("*")}
            )
            return _results(
                [_result("test_a", "passed"), _result("test_b", "failed")]
            )

        log_path = tmp_path / "calls.jsonl"
        server.start(
            evaluate=evaluate,
            log_path=log_path,
            usage=lambda: {"cost": 1.5, "steps": 7},
        )
        try:
            first = self._call(server)
            second = self._call(server)
        finally:
            server.stop()

        assert first.returncode == EXIT_FAIL
        assert "1/2 tests passed." in first.stdout
        assert "- [core] test_b" in first.stdout
        assert second.returncode == EXIT_FAIL

        assert seen == [{"main.py"}, {"main.py"}]
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert [r["attempt"] for r in records] == [1, 2]
        assert records[0]["usage"] == {"cost": 1.5, "steps": 7}
        assert records[0]["failing"][0]["id"] == "test_b"
        assert "message" not in records[0]
        assert list(server.responses_dir.iterdir()) == []

    def test_evaluator_error_is_reported(self, tmp_path, workspace):
        server = self._server(tmp_path, workspace)

        def evaluate(submission: Path) -> CorrectnessResults:
            raise RuntimeError("docker is down")

        log_path = tmp_path / "calls.jsonl"
        server.start(evaluate=evaluate, log_path=log_path, usage=dict)
        try:
            result = self._call(server)
        finally:
            server.stop()

        assert result.returncode == EXIT_BROKEN
        assert "not a problem with your code" in result.stdout
        record = json.loads(log_path.read_text())
        assert record["error"] == "RuntimeError: docker is down"

    def test_start_clears_stale_requests(self, tmp_path, workspace):
        server = self._server(tmp_path, workspace)
        (server.requests_dir / "old.req").touch()
        calls = threading.Event()

        def evaluate(submission: Path) -> CorrectnessResults:
            calls.set()
            return _results([])

        server.start(
            evaluate=evaluate, log_path=tmp_path / "calls.jsonl", usage=dict
        )
        server.stop()

        assert not calls.is_set()

    def test_mounts_ipc_dir_read_write(self, tmp_path, workspace):
        server = self._server(tmp_path, workspace)

        assert server.mounts == {
            str(tmp_path / "ipc"): {"bind": "/scb-ipc", "mode": "rw"}
        }

    def test_cleanup_removes_ipc_dir(self, tmp_path, workspace):
        server = self._server(tmp_path, workspace)
        server.cleanup()

        assert not server.ipc_dir.exists()


def test_prompt_addendum_names_command():
    assert "/scb-ipc/scb-validate" in PROMPT_ADDENDUM
    assert "description" not in PROMPT_ADDENDUM
    assert "you decide when you are done" in PROMPT_ADDENDUM
