"""Agent-callable validation.

The agent runs a script inside its container. The script drops a request file
into a mounted IPC directory and waits. A host thread copies the workspace,
runs the checkpoint's test suite on the copy, and writes back which tests
failed. Parametrized cases collapse into one line per test, because case IDs
often encode expected values. Assertion messages, docstrings, and test source
stay hidden.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from slop_code.evaluation.report import CorrectnessResults
from slop_code.evaluation.report import GroupType
from slop_code.evaluation.report import TestResult
from slop_code.execution.snapshot import _walk_candidates
from slop_code.logging import get_logger

logger = get_logger(__name__)

CONTAINER_IPC_DIR = "/scb-ipc"
VALIDATE_COMMAND = f"{CONTAINER_IPC_DIR}/scb-validate"
CALLS_FILENAME = "validate_calls.jsonl"
POLL_SECONDS = 1.0

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BROKEN = 2

FAILED_STATUSES = {"failed", "error"}

VALIDATE_DESCRIPTION = f"""

Run `{VALIDATE_COMMAND}` to check your work. It runs the hidden test suite \
against your workspace and reports the names of failing tests. A run can take several minutes. You may run it as often as you like, \
and you decide when you are done."""

# Writes a request, waits for the host's response, prints it, and exits with
# the host's exit code. The host writes `<id>.txt` before `<id>.done`.

SCRIPT = """#!/bin/sh
set -eu
dir=$(dirname "$0")
id="$(date +%s)-$$"
: > "$dir/requests/$id.tmp"
mv "$dir/requests/$id.tmp" "$dir/requests/$id.req"
echo "Running tests. This can take several minutes." >&2
while [ ! -f "$dir/responses/$id.done" ]; do sleep 1; done
cat "$dir/responses/$id.txt"
code=$(cat "$dir/responses/$id.done")
rm -f "$dir/responses/$id.txt" "$dir/responses/$id.done"
exit "$code"
"""

GROUP_ORDER = [
    GroupType.CORE,
    GroupType.FUNCTIONALITY,
    GroupType.ERROR,
    GroupType.REGRESSION,
]

Evaluator = Callable[[Path], CorrectnessResults]
UsageProbe = Callable[[], dict[str, Any]]


def collapse_cases(tests: Iterable[TestResult]) -> list[dict[str, Any]]:
    """Group parametrized cases under their test, in first-seen order."""
    collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for test in tests:
        name = test.id.split("[", 1)[0]
        key = (test.file_path, name, test.group_type.value)
        entry = collapsed.setdefault(
            key,
            {"id": name, "group": test.group_type.value, "failed": 0, "cases": 0},
        )
        entry["cases"] += 1
        if test.status in FAILED_STATUSES:
            entry["failed"] += 1
    failing = [entry for entry in collapsed.values() if entry["failed"]]
    rank = {group.value: i for i, group in enumerate(GROUP_ORDER)}
    return sorted(failing, key=lambda entry: rank[entry["group"]])


def build_report(results: CorrectnessResults) -> dict[str, Any]:
    """Summarize results into what the agent sees and what the log records."""
    failing_cases = [
        {"id": test.id, "group": test.group_type.value}
        for test in results.tests
        if test.status in FAILED_STATUSES
    ]
    failing = collapse_cases(results.tests)
    total = len(results.tests)
    passed = sum(1 for t in results.tests if t.status == "passed")
    all_passed = (
        total > 0 and not failing and not results.infrastructure_failure
    )
    report = {
        "all_passed": all_passed,
        "infrastructure_failure": results.infrastructure_failure,
        "passed": passed,
        "total": total,
        "pass_counts": {
            g.value: results.pass_counts.get(g, 0) for g in GROUP_ORDER
        },
        "total_counts": {
            g.value: results.total_counts.get(g, 0) for g in GROUP_ORDER
        },
        "failing": failing,
        "failing_cases": failing_cases,
    }
    report["message"] = format_message(report)
    report["exit_code"] = EXIT_PASS if all_passed else EXIT_FAIL
    return report


def format_message(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if report["infrastructure_failure"] or report["total"] == 0:
        lines.append(
            "The test suite could not run against your workspace. Check that "
            "the entry file exists and the program starts."
        )
    lines.append(f"{report['passed']}/{report['total']} tests passed.")

    group_parts = [
        f"{group.lower()} {report['pass_counts'][group]}/{count}"
        for group, count in report["total_counts"].items()
        if count
    ]
    if group_parts:
        lines.append(", ".join(group_parts))

    if report["failing"]:
        lines += ["", "Failing tests:"]
        for test in report["failing"]:
            line = f"- [{test['group'].lower()}] {test['id']}"
            if test["cases"] > 1:
                line += f": {test['failed']} of {test['cases']} cases failed"
            lines.append(line)
    return "\n".join(lines)


class ValidateServer:
    """Answers validate requests from the agent's container.

    One server lives for the whole run. `start` and `stop` bracket each
    checkpoint, so requests are always tested against the current checkpoint.
    """

    def __init__(
        self,
        ipc_dir: Path,
        workspace_dir: Path,
        ignore_globs: set[str],
        keep_globs: set[str] | None = None,
    ):
        self.ipc_dir = ipc_dir
        self.workspace_dir = workspace_dir
        self.ignore_globs = ignore_globs
        self.keep_globs = keep_globs or set()
        self.requests_dir = ipc_dir / "requests"
        self.responses_dir = ipc_dir / "responses"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        script = ipc_dir / "scb-validate"
        script.write_text(SCRIPT)
        script.chmod(0o755)

    @property
    def mounts(self) -> dict[str, dict[str, str]]:
        return {
            str(self.ipc_dir): {"bind": CONTAINER_IPC_DIR, "mode": "rw"}
        }

    def start(
        self,
        *,
        evaluate: Evaluator,
        log_path: Path,
        usage: UsageProbe,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("Validate server is already running")
        for stale in [
            *self.requests_dir.iterdir(),
            *self.responses_dir.iterdir(),
        ]:
            stale.unlink(missing_ok=True)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve,
            kwargs={
                "evaluate": evaluate,
                "log_path": log_path,
                "usage": usage,
            },
            name="validate-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop polling. Waits for a validation already in progress."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def cleanup(self) -> None:
        self.stop()
        shutil.rmtree(self.ipc_dir, ignore_errors=True)

    def _serve(
        self,
        *,
        evaluate: Evaluator,
        log_path: Path,
        usage: UsageProbe,
    ) -> None:
        started = time.monotonic()
        attempt = 0
        while not self._stop.is_set():
            for request in sorted(self.requests_dir.glob("*.req")):
                attempt += 1
                record = self._handle(
                    request,
                    attempt=attempt,
                    elapsed_s=time.monotonic() - started,
                    evaluate=evaluate,
                    usage=usage,
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            self._stop.wait(POLL_SECONDS)

    def _handle(
        self,
        request: Path,
        *,
        attempt: int,
        elapsed_s: float,
        evaluate: Evaluator,
        usage: UsageProbe,
    ) -> dict[str, Any]:
        request_id = request.stem
        request.unlink(missing_ok=True)
        record: dict[str, Any] = {
            "attempt": attempt,
            "requested_at": datetime.now().isoformat(),
            "elapsed_s": round(elapsed_s, 1),
            "usage": usage(),
        }
        logger.info("Running validate request", attempt=attempt)
        call_started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="scb-validate-") as tmp:
                submission = Path(tmp) / "submission"
                self._copy_workspace(submission)
                report = build_report(evaluate(submission))
        except Exception as e:  # noqa: BLE001
            logger.error("Validate request failed", exc_info=True)
            report = {
                "error": f"{type(e).__name__}: {e}",
                "message": "Validation could not run because of a harness "
                "error. This is not a problem with your code.",
                "exit_code": EXIT_BROKEN,
            }
        record["duration_s"] = round(time.monotonic() - call_started, 1)

        self._respond(request_id, report["message"], report["exit_code"])
        record.update(
            {k: v for k, v in report.items() if k not in {"message"}}
        )
        return record

    def _copy_workspace(self, dest: Path) -> None:
        """Copy the files a checkpoint snapshot would keep."""
        matched, _ = _walk_candidates(
            self.workspace_dir, self.ignore_globs, self.keep_globs
        )
        dest.mkdir(parents=True)
        for rel_path in matched:
            target = dest / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                self.workspace_dir / rel_path, target, follow_symlinks=False
            )

    def _respond(self, request_id: str, message: str, exit_code: int) -> None:
        text = self.responses_dir / f"{request_id}.txt"
        done = self.responses_dir / f"{request_id}.done"
        _write_atomic(text, message + "\n")
        _write_atomic(done, str(exit_code))


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)
