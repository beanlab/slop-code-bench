"""Cooperative API staged beside benchmark workflows.

Workflows import this module instead of calling myteam's run_agent directly.
Only the standard library is imported until a workflow starts a session.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn

USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "estimated_cost",
)


class WorkflowIdleTimeoutError(TimeoutError):
    """No workflow or native-session activity within the idle deadline."""


def _lifecycle(request: dict[str, Any], event: str, **fields: Any) -> None:
    """Persist progress without retaining prompts, arguments, or secrets."""
    artifacts = _artifacts_dir(request)
    with (artifacts / "events.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with (artifacts / "events.jsonl").open(
                "a", encoding="utf-8"
            ) as output:
                output.write(
                    json.dumps({"event": event, "time": time.time(), **fields})
                    + "\n"
                )
                output.flush()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class _IdleWatchdog:
    """Observe durable progress; the polling loop itself is not activity."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.artifacts = _artifacts_dir(request)
        self.timeout = float(request.get("idle_timeout", 0))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _signature(self) -> tuple[tuple[str, int, int], ...]:
        files = [self.artifacts / "events.jsonl"]
        native = self.artifacts / "native_sessions"
        if native.exists():
            files.extend(native.rglob("*.jsonl"))
        signature = []
        for path in files:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(signature))

    def _run(self) -> None:
        previous = self._signature()
        changed = time.monotonic()
        while not self._stop.wait(min(self.timeout / 4, 1)):
            current = self._signature()
            if current != previous:
                changed = time.monotonic()
                previous = current
            elif time.monotonic() - changed >= self.timeout:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def start(self) -> None:
        if self.timeout <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def load_request() -> dict[str, Any]:
    """Read the checkpoint request supplied by the benchmark adapter."""
    request_path = os.environ.get("SCB_MYTEAM_REQUEST")
    if not request_path:
        raise RuntimeError("SCB_MYTEAM_REQUEST is not configured.")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("The benchmark request must be a JSON object.")
    return request


def as_text(value: Any) -> str:
    """Render structured agent output as prompt text."""
    if value is None:
        return "No structured output was returned."
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def needs_changes(value: Any) -> bool:
    """Read a ``needs_changes`` flag that may arrive as a bool or a string."""
    if not isinstance(value, dict):
        return False
    requested = value.get("needs_changes", False)
    if isinstance(requested, str):
        return requested.strip().lower() in {"true", "yes", "1"}
    return bool(requested)


def _artifacts_dir(request: dict[str, Any]) -> Path:
    artifacts = Path(request["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def _fail(
    request: dict[str, Any],
    message: str,
    error_type: type[Exception] = RuntimeError,
) -> NoReturn:
    """Keep a safe diagnostic when myteam suppresses child display output."""
    (_artifacts_dir(request) / "error.txt").write_text(
        message + "\n", encoding="utf-8"
    )
    raise error_type(message)


@contextmanager
def _session_records(
    request: dict[str, Any],
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """Serialize updates across workflow processes, including nested ones."""
    artifacts = _artifacts_dir(request)
    records_path = artifacts / "sessions.jsonl"
    with (artifacts / "sessions.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = []
            if records_path.exists():
                records = [
                    json.loads(line)
                    for line in records_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            yield records_path, records
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _check_budget(
    request: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    cost = sum(
        float(usage.get("estimated_cost", 0))
        for record in records
        for usage in record["usage_delta"]
    )
    limits = request.get("cost_limits", {})
    checkpoint_limit = float(limits.get("cost_limit", 0))
    net_limit = float(limits.get("net_cost_limit", 0))
    if checkpoint_limit > 0 and cost >= checkpoint_limit:
        _fail(request, "The workflow checkpoint cost limit was reached.")
    if net_limit > 0 and cost + request.get("prior_cost", 0) >= net_limit:
        _fail(request, "The workflow cumulative cost limit was reached.")


def _group_usage(usage: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in usage:
        model = item.get("model", "")
        if model not in grouped:
            grouped[model] = {"model": model}
            grouped[model].update(dict.fromkeys(USAGE_FIELDS, 0))
        for field in USAGE_FIELDS:
            grouped[model][field] += item.get(field, 0)
    return grouped


def _usage_delta(
    result: dict[str, Any],
    records: list[dict[str, Any]],
    agent: str,
) -> list[dict[str, Any]]:
    """Deduplicate native totals when a workflow resumes an agent session."""
    previous: dict[str, dict[str, Any]] = {}
    session_id = result.get("session_id")
    if session_id is not None:
        for record in records:
            if (
                record["agent"] == agent
                and record["result"].get("session_id") == session_id
            ):
                previous.update(_group_usage(record["result"]["usage"]))
    deltas = []
    for model, usage in _group_usage(result["usage"]).items():
        baseline = previous.get(model, {})
        delta: dict[str, Any] = {"model": model}
        for field in USAGE_FIELDS:
            delta[field] = max(0, usage[field] - baseline.get(field, 0))
        deltas.append(delta)
    return deltas


def _session_header(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline())
        return header if isinstance(header, dict) else {}
    except (OSError, ValueError):
        return {}


def _native_session_options(
    request: dict[str, Any], kwargs: dict[str, Any]
) -> Path:
    """Keep native logs visible during a call, including before failure."""
    extra_args = list(kwargs.get("extra_args") or ())
    reserved = {"--no-session", "--session", "--session-dir"}
    if any(str(arg).split("=", 1)[0] in reserved for arg in extra_args):
        _fail(
            request,
            "Native PI session flags are managed by the benchmark; use "
            "run_agent(session_id=...) to resume a session.",
            ValueError,
        )
    native_root = _artifacts_dir(request) / "native_sessions"
    invocation = native_root / uuid.uuid4().hex
    invocation.mkdir(parents=True)
    native_path = invocation
    session_id = kwargs.get("session_id")
    if session_id is not None:
        for path in native_root.rglob("*.jsonl"):
            if (
                str(path) == str(session_id)
                or _session_header(path).get("id") == session_id
            ):
                kwargs["session_id"] = str(path)
                native_path = path
                break
    kwargs["extra_args"] = [*extra_args, "--session-dir", str(invocation)]
    return native_path


def _repair_native_metadata(
    request: dict[str, Any], result: Any, native_path: Path
) -> None:
    """Recover metadata without relying on a nonce in a system prompt.

    PI's native messages need not contain the system-prompt nonce used by
    myteam's session search. The benchmark knows the native directory because
    it supplied --session-dir through PI's supported CLI interface.
    """
    files = (
        [native_path]
        if native_path.is_file()
        else list(native_path.rglob("*.jsonl"))
    )
    if len(files) != 1:
        _lifecycle(request, "native_metadata_unavailable")
        return
    payload = result.to_jsonable()
    path = files[0]
    if payload.get("session_id") is None:
        session_id = _session_header(path).get("id")
        if isinstance(session_id, str):
            result.session_id = session_id
    if not payload.get("usage"):
        # Pinned myteam parser reads the known file without nonce discovery.
        from myteam.workflows.agents.pi import get_usage_info

        result.usage = get_usage_info(path) or []
        if not result.usage:
            _lifecycle(request, "native_usage_unavailable")


def run_agent(**kwargs: Any) -> Any:
    """Run a headless PI session and persist its result and usage.

    Budgets are checked between sessions. They cannot interrupt a running
    session at a precise token count or dollar amount.
    """
    request = load_request()
    backend = request.get("backend", "pi")
    if backend != "pi":
        _fail(request, "The myteam benchmark MVP supports only PI.", ValueError)
    if kwargs.get("interactive") not in (None, False):
        _fail(
            request,
            "Benchmark sessions must use interactive=False.",
            ValueError,
        )
    if kwargs.get("agent") not in (None, backend):
        _fail(
            request, "Benchmark sessions must use the PI backend.", ValueError
        )
    if kwargs.get("fork"):
        _fail(
            request,
            "Forked sessions are not supported by this MVP.",
            ValueError,
        )
    kwargs["agent"] = backend
    kwargs["interactive"] = False
    kwargs["fork"] = False
    if kwargs.get("model") is None:
        kwargs["model"] = request["model"]
    if kwargs.get("reasoning") is None and request.get("reasoning") is not None:
        kwargs["reasoning"] = request["reasoning"]

    with _session_records(request) as (_, records):
        _check_budget(request, records)

    native_path = _native_session_options(request, kwargs)

    from myteam import run_agent as myteam_run_agent

    _lifecycle(request, "session_started", agent=backend)
    try:
        result = myteam_run_agent(**kwargs)
    except BaseException as error:
        _lifecycle(
            request, "session_failed", exception_type=type(error).__name__
        )
        raise
    try:
        _repair_native_metadata(request, result, native_path)
    except Exception as error:  # noqa: BLE001 - preserve partial telemetry.
        _lifecycle(
            request,
            "native_metadata_unavailable",
            exception_type=type(error).__name__,
        )
    payload = result.to_jsonable()
    _lifecycle(request, "session_finished", exit_code=result.exit_code)
    with _session_records(request) as (records_path, records):
        record = {
            "session_index": len(records) + 1,
            "agent": backend,
            "result": payload,
            "usage_delta": _usage_delta(payload, records, backend),
        }
        with records_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
        records.append(record)
        _check_budget(request, records)
    if result.exit_code == 143 and payload.get("output") is not None:
        # myteam can terminate PI with SIGTERM after its result was submitted.
        # Preserve that completed result instead of treating the signal as a
        # failed session; the workflow can still consume its structured output.
        _lifecycle(request, "session_result_preserved_after_sigterm")
    elif result.exit_code != 0:
        _fail(request, f"PI session failed with exit code {result.exit_code}.")
    return result


def _interrupt_workflow(signum: int, frame: FrameType | None) -> None:
    del frame
    if signum == signal.SIGUSR1:
        raise WorkflowIdleTimeoutError("The workflow idle timeout was reached.")
    if signum == signal.SIGALRM:
        raise TimeoutError("The workflow timeout was reached.")
    raise RuntimeError("The workflow was terminated.")


def main() -> None:
    """Launch through myteam's supervisor, preserving its result protocol."""
    request = load_request()
    artifacts = _artifacts_dir(request)
    # A local benchmark may itself be invoked inside a managed myteam session.
    # Its workflow must have a fresh supervisor belonging to this benchmark.
    for name in (
        "MYTEAM_SUPERVISOR_SOCKET",
        "MYTEAM_WORKFLOW_INVOCATION_ID",
        "MYTEAM_AGENT_SESSION_RESULT_SOCKET",
        "MYTEAM_AGENT_SESSION_NONCE",
        "MYTEAM_WORKFLOW_INPUT_JSON",
    ):
        os.environ.pop(name, None)
    previous_alarm = signal.signal(signal.SIGALRM, _interrupt_workflow)
    previous_term = signal.signal(signal.SIGTERM, _interrupt_workflow)
    previous_idle = signal.signal(signal.SIGUSR1, _interrupt_workflow)
    watchdog = _IdleWatchdog(request)
    timeout = float(request.get("timeout", 0))
    if timeout > 0:
        signal.setitimer(signal.ITIMER_REAL, timeout)
    _lifecycle(request, "workflow_started")
    watchdog.start()
    try:
        from myteam.workflows import start_workflow_cli

        start_workflow_cli(
            request["workflow"], *request.get("workflow_args", [])
        )
    except SystemExit as error:
        if (
            error.code not in (None, 0)
            and not (artifacts / "error.txt").exists()
        ):
            (artifacts / "error.txt").write_text(
                "The myteam workflow exited unsuccessfully.\n",
                encoding="utf-8",
            )
        raise
    except Exception as error:  # noqa: BLE001 - sanitize the CLI boundary.
        # Exception messages can contain credentials or caller-supplied data.
        message = f"Workflow launch failed ({type(error).__name__}).\n"
        (artifacts / "error.txt").write_text(message, encoding="utf-8")
        if isinstance(error, WorkflowIdleTimeoutError):
            message = (
                "The workflow idle timeout was reached: no native-session "
                "or workflow progress was recorded. See events.jsonl and "
                "native_sessions for the last recorded activity.\n"
            )
            (artifacts / "idle_timeout.txt").write_text(
                message, encoding="utf-8"
            )
            (artifacts / "error.txt").write_text(message, encoding="utf-8")
        elif isinstance(error, TimeoutError):
            (artifacts / "timeout.txt").write_text(
                "The workflow timeout was reached.\n", encoding="utf-8"
            )
        print(message, file=sys.stderr, end="")
        raise SystemExit(1) from None
    finally:
        watchdog.close()
        _lifecycle(request, "workflow_stopped")
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGUSR1, previous_idle)


if __name__ == "__main__":
    main()
