"""Myteam adapter tests without provider API calls."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from slop_code.agent_runner.agents.myteam import MyTeamAgent
from slop_code.agent_runner.agents.myteam import MyTeamConfig
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.models import AgentSetupError
from slop_code.agent_runner.registry import build_agent_config
from slop_code.execution import DockerConfig
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution.local_streaming import LocalStreamingRuntime
from slop_code.execution.runtime import RuntimeEvent
from slop_code.execution.runtime import RuntimeResult


@pytest.fixture
def config(tmp_path: Path) -> MyTeamConfig:
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "solve.py").write_text('"""type: workflow"""\n')
    return MyTeamConfig(
        workflow_dir=workflow,
        python=sys.executable,
        timeout=10,
        cost_limits=AgentCostLimits(cost_limit=0, net_cost_limit=0),
    )


@pytest.fixture
def credential() -> ProviderCredential:
    return ProviderCredential(
        provider="anthropic",
        credential_type=CredentialType.ENV_VAR,
        value="test-api-key",
        source="TEST_KEY",
        destination_key="ANTHROPIC_API_KEY",
    )


def _agent(config: MyTeamConfig, credential: ProviderCredential) -> MyTeamAgent:
    return MyTeamAgent(
        config=config,
        model="anthropic/test-model",
        credential=credential,
        problem_name="test-problem",
        verbose=False,
    )


class FakeRuntime:
    def __init__(self, *, exit_code: int = 0, timed_out: bool = False) -> None:
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.cleaned = False
        self.request: dict[str, Any] = {}
        self.env: dict[str, str] = {}

    def stream(
        self, command: str, env: dict[str, str], timeout: float
    ) -> Iterator[RuntimeEvent]:
        self.env = env
        self.request = json.loads(Path(env["SCB_MYTEAM_REQUEST"]).read_text())
        record = {
            "session_index": 1,
            "agent": "pi",
            "result": {"transcript": "native transcript"},
            "usage_delta": [
                {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 2,
                    "estimated_cost": 0.25,
                }
            ],
        }
        artifacts = Path(self.request["artifacts_dir"])
        (artifacts / "sessions.jsonl").write_text(json.dumps(record) + "\n")
        yield RuntimeEvent(
            kind="finished",
            result=RuntimeResult(
                exit_code=self.exit_code,
                stdout="workflow result\n",
                stderr="workflow diagnostic\n",
                setup_stdout="",
                setup_stderr="",
                elapsed=0.1,
                timed_out=self.timed_out,
            ),
        )

    def cleanup(self) -> None:
        self.cleaned = True


@dataclass
class FakeSession:
    runtime: Any
    working_dir: Path
    spec: LocalEnvironmentSpec | DockerEnvironmentSpec
    spawn_kwargs: dict[str, Any] | None = None

    def spawn(self, **kwargs: Any) -> Any:
        self.spawn_kwargs = kwargs
        return self.runtime


@pytest.mark.parametrize("exit_code", [0, 1])
def test_artifacts_preserve_staged_workflow_after_source_changes(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    exit_code: int,
) -> None:
    (config.workflow_dir / "prompt.txt").write_text("Original instructions")
    agent = _agent(config, credential)
    agent.setup(
        FakeSession(
            FakeRuntime(exit_code=exit_code),
            tmp_path,
            LocalEnvironmentSpec(name="snapshot"),
        )
    )
    try:
        (config.workflow_dir / "prompt.txt").write_text("Changed instructions")
        (config.workflow_dir / "solve.py").unlink()
        if exit_code:
            with pytest.raises(AgentError):
                agent.run("Task")
        else:
            agent.run("Task")
        saved = tmp_path / "workflow-artifacts"
        agent.save_artifacts(saved)
        assert (saved / "workflow" / "solve.py").is_file()
        assert (saved / "workflow" / "prompt.txt").read_text() == (
            "Original instructions"
        )
        metadata = json.loads((saved / "workflow.json").read_text())
        assert metadata["entrypoint"] == "solve.py"
        assert metadata["source_dir"] == str(config.workflow_dir)
        assert len(metadata["content_hash"]) == 64
        agent.reset()
        second = tmp_path / "second-checkpoint"
        agent.save_artifacts(second)
        assert (second / "workflow.json").read_bytes() == (
            saved / "workflow.json"
        ).read_bytes()
    finally:
        agent.cleanup()


def test_yaml_config_loads_through_registry() -> None:
    path = Path("configs/agents/myteam.yaml")
    config = build_agent_config(yaml.safe_load(path.read_text()))
    assert isinstance(config, MyTeamConfig)


@pytest.mark.parametrize(
    "overrides",
    [
        {"workflow_entrypoint": "../escape.py"},
        {"timeout": 0},
        {"idle_timeout": -1},
        {
            "cost_limits": {
                "step_limit": 1,
                "cost_limit": 0,
                "net_cost_limit": 0,
            }
        },
    ],
)
def test_invalid_execution_settings_are_rejected(
    config: MyTeamConfig, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        MyTeamConfig.model_validate({**config.model_dump(), **overrides})


def test_file_credentials_fail_with_actionable_error(
    config: MyTeamConfig, credential: ProviderCredential
) -> None:
    credential = credential.model_copy(
        update={"credential_type": CredentialType.FILE}
    )
    with pytest.raises(AgentSetupError, match="API key"):
        _agent(config, credential)


def test_absolute_entrypoint_inside_workflow_root_is_rejected(
    config: MyTeamConfig,
) -> None:
    with pytest.raises(ValidationError, match="relative"):
        MyTeamConfig.model_validate(
            {
                **config.model_dump(),
                "workflow_entrypoint": str(config.workflow_dir / "solve.py"),
            }
        )


@pytest.mark.parametrize("idle_timeout", [0, 42])
def test_idle_timeout_is_forwarded_to_workflow_request(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    idle_timeout: int,
) -> None:
    configured = MyTeamConfig.model_validate(
        {**config.model_dump(), "idle_timeout": idle_timeout}
    )
    runtime = FakeRuntime()
    agent = _agent(configured, credential)
    agent.setup(
        FakeSession(runtime, tmp_path, LocalEnvironmentSpec(name="idle-limit"))
    )
    try:
        agent.run("Checkpoint task")
        assert runtime.request["idle_timeout"] == idle_timeout
    finally:
        agent.cleanup()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_output_is_saved_before_stream_completion_or_interruption(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    class InterruptedRuntime(FakeRuntime):
        def stream(
            self, command: str, env: dict[str, str], timeout: float
        ) -> Iterator[RuntimeEvent]:
            request = json.loads(Path(env["SCB_MYTEAM_REQUEST"]).read_text())
            artifacts = Path(request["artifacts_dir"])
            yield RuntimeEvent(kind="stdout", text="live workflow output\n")
            assert (artifacts / "stdout.log").read_text() == (
                "live workflow output\n"
            )
            yield RuntimeEvent(kind="stderr", text="live workflow diagnostic\n")
            assert (artifacts / "stderr.log").read_text() == (
                "live workflow diagnostic\n"
            )
            raise error_type("stream interrupted")

    agent = _agent(config, credential)
    agent.setup(
        FakeSession(
            InterruptedRuntime(),
            tmp_path,
            LocalEnvironmentSpec(name="live-artifacts"),
        )
    )
    try:
        with pytest.raises(error_type, match="stream interrupted"):
            agent.run("Checkpoint interrupted before completion")
        saved = tmp_path / "saved-live-artifacts"
        agent.save_artifacts(saved)
        assert (saved / "stdout.log").read_text() == "live workflow output\n"
        assert (saved / "stderr.log").read_text() == (
            "live workflow diagnostic\n"
        )
    finally:
        agent.cleanup()


def test_idle_timeout_failure_reports_inactivity_and_clears_on_retry(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    class IdleRuntime(FakeRuntime):
        def stream(
            self, command: str, env: dict[str, str], timeout: float
        ) -> Iterator[RuntimeEvent]:
            yield from super().stream(command, env, timeout)
            if self.exit_code:
                artifacts = Path(self.request["artifacts_dir"])
                (artifacts / "idle_timeout.txt").write_text("No activity\n")

    runtime = IdleRuntime(exit_code=1)
    agent = _agent(config, credential)
    agent.setup(
        FakeSession(runtime, tmp_path, LocalEnvironmentSpec(name="idle-error"))
    )
    try:
        with pytest.raises(AgentError, match="idle"):
            agent.run("Stalled checkpoint")
        runtime.exit_code = 0
        agent.retry()
        assert not (agent._artifacts_dir / "idle_timeout.txt").exists()
    finally:
        agent.cleanup()


def test_workflow_error_reason_is_reported_with_credential_redacted(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    class FailedRuntime(FakeRuntime):
        def stream(
            self, command: str, env: dict[str, str], timeout: float
        ) -> Iterator[RuntimeEvent]:
            yield from super().stream(command, env, timeout)
            artifacts = Path(self.request["artifacts_dir"])
            (artifacts / "error.txt").write_text(
                "Managed session failed: " + credential.value
            )

    agent = _agent(config, credential)
    agent.setup(
        FakeSession(
            FailedRuntime(exit_code=1),
            tmp_path,
            LocalEnvironmentSpec(name="failure-reason"),
        )
    )
    try:
        with pytest.raises(AgentError, match="Managed session failed") as error:
            agent.run("Failing checkpoint")
        assert credential.value not in str(error.value)
    finally:
        agent.cleanup()


def test_retry_preserves_original_checkpoint_task(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(exit_code=1)
    agent = _agent(config, credential)
    agent.setup(
        FakeSession(runtime, tmp_path, LocalEnvironmentSpec(name="retry"))
    )
    try:
        with pytest.raises(AgentError):
            agent.run("Original checkpoint instructions")
        runtime.exit_code = 0
        agent.retry()
        assert runtime.request["task"] == "Original checkpoint instructions"
        assert (agent._artifacts_dir / "stdout.log").read_text() == (
            "workflow result\nworkflow result\n"
        )
    finally:
        agent.cleanup()


def test_truncated_final_record_preserves_completed_session_accounting(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    class TruncatedRuntime(FakeRuntime):
        def stream(
            self, command: str, env: dict[str, str], timeout: float
        ) -> Iterator[RuntimeEvent]:
            yield from super().stream(command, env, timeout)
            records = Path(self.request["artifacts_dir"]) / "sessions.jsonl"
            with records.open("a") as output:
                output.write('{"session_index":')

    agent = _agent(config, credential)
    agent.setup(
        FakeSession(
            TruncatedRuntime(), tmp_path, LocalEnvironmentSpec(name="partial")
        )
    )
    try:
        with pytest.raises(AgentError, match="Invalid myteam"):
            agent.run("Checkpoint interrupted during record write")
        assert agent.usage.cost == 0.25
        assert agent.usage.net_tokens.input == 100
        assert agent.usage_by_agent["pi/1"].cost == 0.25
    finally:
        agent.cleanup()


def test_timeout_with_truncated_record_keeps_timeout_and_partial_usage(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    class TimedOutRuntime(FakeRuntime):
        def stream(
            self, command: str, env: dict[str, str], timeout: float
        ) -> Iterator[RuntimeEvent]:
            yield from super().stream(command, env, timeout)
            records = Path(self.request["artifacts_dir"]) / "sessions.jsonl"
            with records.open("a") as output:
                output.write('{"session_index":')

    agent = _agent(config, credential)
    agent.setup(
        FakeSession(
            TimedOutRuntime(timed_out=True),
            tmp_path,
            LocalEnvironmentSpec(name="timeout-partial"),
        )
    )
    try:
        with pytest.raises(AgentError, match="timed out"):
            agent.run("Checkpoint timed out during record write")
        assert agent.usage.cost == 0.25
        assert agent.usage.net_tokens.input == 100
        assert agent.usage_by_agent["pi/1"].cost == 0.25
    finally:
        agent.cleanup()


@pytest.mark.parametrize("failure", [None, "exit", "timeout"])
def test_usage_and_artifacts_survive_workflow_completion_or_failure(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    failure: str | None,
) -> None:
    runtime = FakeRuntime(
        exit_code=7 if failure == "exit" else 0,
        timed_out=failure == "timeout",
    )
    agent = _agent(config, credential)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent.setup(
        FakeSession(
            runtime, workspace, LocalEnvironmentSpec(name="myteam-test")
        )
    )
    try:
        if failure:
            with pytest.raises(AgentError):
                agent.run("checkpoint task")
        else:
            agent.run("checkpoint task")
        assert runtime.request["task"] == "checkpoint task"
        assert runtime.env["ANTHROPIC_API_KEY"] == "test-api-key"
        assert not Path(runtime.request["workflow"]).is_relative_to(workspace)
        assert not Path(runtime.env["HOME"]).is_relative_to(workspace)
        assert agent.usage.cost == 0.25
        assert agent.usage.net_tokens.input == 100
        assert agent.usage.net_tokens.cache_read == 20
        assert agent.usage_by_agent["pi/1"].cost == 0.25
        saved = tmp_path / "saved"
        agent.save_artifacts(saved)
        assert (saved / "stdout.log").read_text() == "workflow result\n"
        assert (saved / "stderr.log").read_text() == "workflow diagnostic\n"
        assert (saved / "sessions.jsonl").is_file()
        assert "test-api-key" not in (saved / "task.txt").read_text()
        solution = workspace / "solution.py"
        solution.write_text("print('solution')")
        agent.finish_checkpoint(reset_context=True)
        assert agent.prior_cost == 0.25
        assert agent.usage.cost == 0
        assert agent.usage_by_agent == {}
        assert solution.exists()
        assert not (agent._artifacts_dir / "sessions.jsonl").exists()
    finally:
        agent.cleanup()


def test_docker_mounts_keep_workflow_and_artifacts_outside_workspace(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    session = FakeSession(
        FakeRuntime(),
        tmp_path,
        DockerEnvironmentSpec(
            name="myteam-docker", docker=DockerConfig(image="test-image")
        ),
    )
    agent = _agent(config, credential)
    agent.setup(session)
    try:
        mounts = session.spawn_kwargs["mounts"]
        by_mode = {value["mode"]: (key, value) for key, value in mounts.items()}
        assert set(by_mode) == {"ro", "rw"}
        bundle, bundle_mount = by_mode["ro"]
        assert (Path(bundle) / "workflow/solve.py").is_file()
        assert (Path(bundle) / "scb_myteam.py").is_file()
        assert Path(bundle_mount["bind"]).is_absolute()
        state, _ = by_mode["rw"]
        assert (Path(state) / "artifacts").is_dir()
        assert (Path(state) / "home").is_dir()
        assert session.spawn_kwargs["disable_setup"] is True
        dockerfile = config.get_docker_file("test-base")
        assert "FROM test-base" in dockerfile
        assert f"myteam=={config.version}" in dockerfile
        assert (
            f"@earendil-works/pi-coding-agent@{config.pi_version}" in dockerfile
        )
    finally:
        agent.cleanup()
    assert session.runtime.cleaned


def _local_session(workspace: Path) -> FakeSession:
    workspace.mkdir()
    spec = LocalEnvironmentSpec(name="myteam-test")
    return FakeSession(
        LocalStreamingRuntime(environment=spec, working_dir=workspace),
        workspace,
        spec,
    )


def test_local_subprocess_delivers_task_and_collects_session_usage(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    fake_package = tmp_path / "fake-package"
    package = fake_package / "myteam"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "class Result:\n"
        "    exit_code = 0\n"
        "    output = {'summary': 'done'}\n"
        "    def to_jsonable(self):\n"
        "        return dict(exit_code=0, session_id='native-id', "
        "output=self.output, transcript='display', "
        "usage=[dict(model='test-model', input_tokens=50, "
        "estimated_cost=0.2)])\n"
        "def run_agent(**kwargs):\n"
        "    assert kwargs['interactive'] is False\n"
        "    Path('solution.txt').write_text(kwargs['prompt'])\n"
        "    return Result()\n"
    )
    (package / "workflows.py").write_text(
        "import runpy\n"
        "def start_workflow_cli(path, *args):\n"
        "    runpy.run_path(path, run_name='__main__')\n"
    )
    interpreter = tmp_path / "test-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(fake_package)!r})\n"
        "runpy.run_path(sys.argv[1], run_name='__main__')\n"
    )
    interpreter.chmod(0o755)
    config.python = str(interpreter)
    (config.workflow_dir / "solve.py").write_text(
        '"""type: workflow"""\n'
        "from scb_myteam import load_request, run_agent\n"
        "run_agent(prompt=load_request()['task'])\n"
        "print('workflow complete')\n"
    )
    session = _local_session(tmp_path / "workspace")
    agent = _agent(config, credential)
    agent.setup(session)
    try:
        agent.run("Build this exact solution.")
        assert (session.working_dir / "solution.txt").read_text() == (
            "Build this exact solution."
        )
        assert agent.usage.cost == 0.2
        assert agent.usage.net_tokens.input == 50
        assert sorted(path.name for path in session.working_dir.iterdir()) == [
            "solution.txt"
        ]
    finally:
        agent.cleanup()


def test_installed_myteam_runs_supervised_headless_workflow(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
) -> None:
    interpreter = os.environ.get("SCB_MYTEAM_TEST_PYTHON")
    if not interpreter:
        pytest.skip("Set SCB_MYTEAM_TEST_PYTHON for actual myteam tests")
    config.python = interpreter
    (config.workflow_dir / "solve.py").write_text(
        '"""type: workflow\ndescription: Offline supervisor smoke test\n"""\n'
        "from pathlib import Path\n"
        "from scb_myteam import load_request\n"
        "from myteam.workflows import report_workflow_result\n"
        "Path('solution.txt').write_text(load_request()['task'])\n"
        "report_workflow_result('supervisor returned result')\n"
    )
    session = _local_session(tmp_path / "workspace")
    agent = _agent(config, credential)
    agent.setup(session)
    try:
        try:
            agent.run("A real supervised checkpoint task.")
        except AgentError:
            diagnostic = (agent._artifacts_dir / "stderr.log").read_text()
            pytest.fail(f"Actual myteam workflow failed: {diagnostic}")
        assert (session.working_dir / "solution.txt").read_text() == (
            "A real supervised checkpoint task."
        )
        saved = tmp_path / "saved"
        agent.save_artifacts(saved)
        assert (
            "supervisor returned result" in (saved / "stdout.log").read_text()
        )
    finally:
        agent.cleanup()


@pytest.fixture
def installed_myteam_python() -> str:
    interpreter = os.environ.get("SCB_MYTEAM_TEST_PYTHON")
    if not interpreter:
        pytest.skip(
            "Set SCB_MYTEAM_TEST_PYTHON for installed myteam smoke tests"
        )
    return interpreter


def test_installed_myteam_nested_workflow_preserves_request_and_result(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    installed_myteam_python: str,
) -> None:
    config.python = installed_myteam_python
    (config.workflow_dir / "solve.py").write_text(
        '"""type: workflow\ndescription: Parent smoke workflow\n"""\n'
        "from pathlib import Path\n"
        "from myteam.workflows import start_workflow, report_workflow_result\n"
        "result = start_workflow(str(Path(__file__).with_name('child.py')))\n"
        "report_workflow_result('parent received: ' + result)\n"
    )
    (config.workflow_dir / "child.py").write_text(
        '"""type: workflow\ndescription: Nested smoke workflow\n"""\n'
        "from pathlib import Path\n"
        "from scb_myteam import load_request\n"
        "from myteam.workflows import report_workflow_result\n"
        "Path('nested.txt').write_text(load_request()['task'])\n"
        "report_workflow_result('child result')\n"
    )
    session = _local_session(tmp_path / "workspace")
    agent = _agent(config, credential)
    agent.setup(session)
    try:
        agent.run("Nested checkpoint task")
        assert (session.working_dir / "nested.txt").read_text() == (
            "Nested checkpoint task"
        )
        stdout = (agent._artifacts_dir / "stdout.log").read_text()
        assert "parent received: child result" in stdout
    finally:
        agent.cleanup()


def test_installed_myteam_timeout_terminates_workflow_process_tree(
    config: MyTeamConfig,
    credential: ProviderCredential,
    tmp_path: Path,
    installed_myteam_python: str,
) -> None:
    config.python = installed_myteam_python
    config.timeout = 2
    (config.workflow_dir / "solve.py").write_text(
        '"""type: workflow\ndescription: Timeout smoke workflow\n"""\n'
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    session = _local_session(tmp_path / "workspace")
    agent = _agent(config, credential)
    agent.setup(session)
    try:
        with pytest.raises(AgentError):
            agent.run("Timeout task")
        assert (agent._artifacts_dir / "timeout.txt").is_file()
        pid = int((session.working_dir / "child.pid").read_text())
        status = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 3
        while status.exists() and time.monotonic() < deadline:
            if status.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        assert not status.exists() or status.read_text().split()[2] == "Z"
    finally:
        agent.cleanup()
