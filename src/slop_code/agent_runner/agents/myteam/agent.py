"""Run cooperative myteam Python workflows in benchmark workspaces."""

from __future__ import annotations

import json
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from jinja2 import Template
from pydantic import Field
from pydantic import model_validator

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.models import AgentSetupError
from slop_code.agent_runner.models import AgentThreadUsage
from slop_code.agent_runner.registry import register_agent
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import ThinkingPreset
from slop_code.common.llms import TokenUsage
from slop_code.common.workflow import WorkflowProvenance
from slop_code.common.workflow import copy_workflow
from slop_code.common.workflow import workflow_provenance
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution import Session
from slop_code.execution import StreamingRuntime

_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GEMINI_API_KEY",
}
_CONTAINER_ROOT = Path("/opt/scb-myteam-run")


class MyTeamConfig(AgentConfigBase):
    """A pinned myteam installation and a cooperative workflow bundle."""

    type: Literal["myteam"] = "myteam"
    version: str = "0.3.21"
    pi_version: str = "0.74.0"
    python: str = "python"
    backend: Literal["pi"] = "pi"
    workflow_dir: Path = Path("configs/workflows/myteam/example")
    workflow_entrypoint: str = "solve.py"
    workflow_args: list[str] = Field(default_factory=list)
    workflow_metadata: WorkflowProvenance | None = None
    timeout: int = Field(default=7200, gt=0)
    idle_timeout: int = Field(default=600, ge=0)
    env: dict[str, str] = Field(default_factory=dict)
    docker_template: Path | None = Path(__file__).with_name("docker.j2")

    @model_validator(mode="after")
    def _validate_workflow(self) -> MyTeamConfig:
        if Path(self.workflow_entrypoint).is_absolute():
            raise ValueError("workflow_entrypoint must be a relative path")
        root = self.workflow_dir.expanduser().resolve()
        entrypoint = (root / self.workflow_entrypoint).resolve()
        if not entrypoint.is_relative_to(root):
            raise ValueError("workflow_entrypoint must stay in workflow_dir")
        if entrypoint.suffix != ".py":
            raise ValueError("myteam MVP requires a Python workflow")
        if not self.python.strip():
            raise ValueError("python must name an executable")
        if self.cost_limits.step_limit:
            raise ValueError(
                "myteam does not expose model steps; step_limit must be 0"
            )
        if not self.pi_version.strip():
            raise ValueError("pi_version must be non-empty")
        return self

    def get_docker_file(self, base_image: str) -> str | None:
        if self.docker_template is None:
            return None
        return Template(self.docker_template.read_text()).render(
            base_image=base_image,
            version=self.version,
            pi_version=self.pi_version,
        )

    def get_image(self, env_name: str) -> str:
        return f"myteam-{self.version}-pi-{self.pi_version}-{env_name}"


class MyTeamAgent(Agent):
    """Delegate each checkpoint to a fresh myteam workflow invocation."""

    def __init__(
        self,
        config: MyTeamConfig,
        model: str,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,  # noqa: FBT001 - matches the Agent interface.
        image: str | None = None,
        reasoning: str | None = None,
        pricing: APIPricing | None = None,
    ) -> None:
        super().__init__(
            agent_name="myteam",
            problem_name=problem_name,
            cost_limits=config.cost_limits,
            pricing=pricing,
            verbose=verbose,
        )
        if credential.credential_type != CredentialType.ENV_VAR:
            raise AgentSetupError(
                "myteam MVP supports API key credentials only; select an "
                "openai, anthropic, openrouter, or google API key provider"
            )
        if credential.provider not in _PROVIDER_KEYS:
            raise AgentSetupError(
                f"Unsupported myteam PI provider: {credential.provider}"
            )
        self.config = config
        self.model = model
        self.credential = credential
        self.reasoning = reasoning
        self._image = image
        self._runtime: StreamingRuntime | None = None
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._bundle_dir: Path | None = None
        self._artifacts_dir: Path | None = None
        self._request_path: Path | None = None
        self._runtime_root: Path | None = None
        self._last_task: str | None = None

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,  # noqa: FBT001 - matches the Agent interface.
        image: str | None,
        thinking_preset: ThinkingPreset | None = None,
        thinking_max_tokens: int | None = None,
    ) -> MyTeamAgent:
        if not isinstance(config, MyTeamConfig):
            raise TypeError("Expected MyTeamConfig")
        if thinking_max_tokens is not None:
            raise AgentSetupError(
                "myteam PI supports thinking presets, not token budgets"
            )
        reasoning = (
            "off"
            if thinking_preset in {"none", "disabled"}
            else thinking_preset
        )
        slug = model.get_model_slug(credential.provider)
        return cls(
            config=config,
            model=f"{credential.provider}/{slug}",
            credential=credential,
            problem_name=problem_name,
            verbose=verbose,
            image=image,
            reasoning=reasoning,
            pricing=model.pricing,
        )

    def setup(self, session: Session) -> None:
        if not isinstance(
            session.spec, (DockerEnvironmentSpec, LocalEnvironmentSpec)
        ):
            raise AgentSetupError("myteam supports local and Docker execution")
        workflow_root = self.config.workflow_dir.expanduser().resolve()
        entrypoint = (workflow_root / self.config.workflow_entrypoint).resolve()
        if not entrypoint.is_file():
            raise AgentSetupError(f"Workflow does not exist: {entrypoint}")
        if not entrypoint.is_relative_to(workflow_root):
            raise AgentSetupError("Workflow entrypoint escapes workflow_dir")
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="scb-myteam-")
        root = Path(self._tmp_dir.name)
        root.chmod(0o755)
        self._bundle_dir = root / "bundle"
        self._bundle_dir.mkdir()
        copy_workflow(workflow_root, self._bundle_dir / "workflow")
        provenance = workflow_provenance(
            {
                **self.config.model_dump(mode="json"),
                "workflow_dir": self._bundle_dir / "workflow",
            }
        )
        if provenance is None:
            raise AgentSetupError("Missing myteam workflow provenance")
        if self.config.workflow_metadata is not None:
            expected = self.config.workflow_metadata
            if provenance.content_hash != expected.content_hash:
                raise AgentSetupError("Workflow snapshot content has changed")
            provenance = expected.model_copy(deep=True)
        else:
            provenance.name = workflow_root.name
            provenance.source_dir = str(workflow_root)
        (self._bundle_dir / "workflow.json").write_text(
            provenance.model_dump_json(indent=2), encoding="utf-8"
        )
        shutil.copy2(
            Path(__file__).with_name("scb_myteam.py"),
            self._bundle_dir / "scb_myteam.py",
        )
        state = root / "state"
        state.mkdir(mode=0o777)
        state.chmod(0o777)
        self._artifacts_dir = state / "artifacts"
        self._request_path = state / "request.json"
        self._prepare_checkpoint()
        is_docker = isinstance(session.spec, DockerEnvironmentSpec)
        self._runtime_root = _CONTAINER_ROOT if is_docker else root
        mounts: dict[str, dict[str, str] | str] = {}
        if is_docker:
            mounts[str(self._bundle_dir)] = {
                "bind": str(_CONTAINER_ROOT / "bundle"),
                "mode": "ro",
            }
            mounts[str(state)] = {
                "bind": str(_CONTAINER_ROOT / "state"),
                "mode": "rw",
            }
        self._runtime = session.spawn(
            mounts=mounts, image=self._image, disable_setup=True
        )

    def _prepare_checkpoint(self) -> None:
        if self._artifacts_dir is None or self._request_path is None:
            raise AgentError("myteam has not been set up")
        state = self._request_path.parent
        for path in (self._artifacts_dir, state / "home"):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(mode=0o777)
            path.chmod(0o777)

    def run(self, task: str) -> None:
        if (
            self._runtime is None
            or self._runtime_root is None
            or self._request_path is None
            or self._artifacts_dir is None
        ):
            raise AgentError("myteam has not been set up")
        root = self._runtime_root
        self._last_task = task
        for marker in ("timeout.txt", "idle_timeout.txt", "error.txt"):
            (self._artifacts_dir / marker).unlink(missing_ok=True)
        request = {
            "task": task,
            "model": self.model,
            "reasoning": self.reasoning,
            "backend": self.config.backend,
            "workflow": str(
                root / "bundle/workflow" / self.config.workflow_entrypoint
            ),
            "workflow_args": self.config.workflow_args,
            "artifacts_dir": str(root / "state/artifacts"),
            "cost_limits": self.cost_limits.model_dump(),
            "pricing": self.pricing.model_dump() if self.pricing else None,
            "prior_cost": self.prior_cost,
            "timeout": self.config.timeout,
            "idle_timeout": self.config.idle_timeout,
        }
        self._request_path.write_text(json.dumps(request), encoding="utf-8")
        self._request_path.chmod(0o666)
        (self._artifacts_dir / "task.txt").write_text(task, encoding="utf-8")
        env = {
            **self.config.env,
            "HOME": str(root / "state/home"),
            "XDG_CONFIG_HOME": str(root / "state/home/.config"),
            "XDG_DATA_HOME": str(root / "state/home/.local/share"),
            "PI_CODING_AGENT_DIR": str(root / "state/home/.pi/agent"),
            "PYTHONPATH": str(root / "bundle"),
            "SCB_MYTEAM_REQUEST": str(root / "state/request.json"),
            "MYTEAM_SUPERVISOR_SOCKET": "",
            "MYTEAM_WORKFLOW_INVOCATION_ID": "",
            "MYTEAM_AGENT_SESSION_RESULT_SOCKET": "",
            "MYTEAM_AGENT_SESSION_NONCE": "",
            _PROVIDER_KEYS[self.credential.provider]: self.credential.value,
        }
        command = shlex.join(
            [self.config.python, str(root / "bundle/scb_myteam.py")]
        )
        result = None
        try:
            with (
                (self._artifacts_dir / "stdout.log").open(
                    "a", encoding="utf-8"
                ) as stdout,
                (self._artifacts_dir / "stderr.log").open(
                    "a", encoding="utf-8"
                ) as stderr,
            ):
                saw_stdout = False
                saw_stderr = False
                for event in self._runtime.stream(
                    command=command,
                    env=env,
                    timeout=self.config.timeout + 10,
                ):
                    if event.kind == "stdout" and event.text:
                        stdout.write(event.text)
                        stdout.flush()
                        saw_stdout = True
                    elif event.kind == "stderr" and event.text:
                        stderr.write(event.text)
                        stderr.flush()
                        saw_stderr = True
                    elif event.kind == "finished":
                        result = event.result
                if result is None:
                    raise AgentError(
                        "myteam runtime returned no completion event"
                    )
                # Some runtimes only emit aggregate output at completion.
                if not saw_stdout:
                    stdout.write(result.stdout)
                if not saw_stderr:
                    stderr.write(result.stderr)
            if (self._artifacts_dir / "idle_timeout.txt").exists():
                raise AgentError(
                    "myteam workflow exceeded its idle timeout "
                    f"({self.config.idle_timeout}s)"
                    f"{self._failure_detail()}; see agent artifacts"
                )
            if (
                result.timed_out
                or (self._artifacts_dir / "timeout.txt").exists()
            ):
                raise AgentError("myteam workflow timed out")
            if result.exit_code:
                raise AgentError(
                    "myteam workflow failed with exit code "
                    f"{result.exit_code}{self._failure_detail()}; "
                    "see agent artifacts"
                )
        finally:
            primary_error = sys.exception()
            try:
                self._sync_usage()
            except AgentError:
                if primary_error is None:
                    raise
                self.log.warning(
                    "Incomplete myteam usage artifact after workflow failure"
                )

    def _failure_detail(self) -> str:
        if self._artifacts_dir is None:
            return ""
        error = self._artifacts_dir / "error.txt"
        if not error.is_file():
            return ""
        detail = error.read_text(encoding="utf-8", errors="replace")
        if self.credential.value:
            detail = detail.replace(self.credential.value, "<REDACTED>")
        detail = " ".join(detail.split())[:1000]
        return f": {detail}" if detail else ""

    def _sync_usage(self) -> None:
        if self._artifacts_dir is None:
            return
        records = self._artifacts_dir / "sessions.jsonl"
        if not records.exists():
            return
        self.usage.cost = 0
        self.usage.net_tokens = TokenUsage()
        self.usage.current_tokens = TokenUsage()
        self.usage_by_agent = {}
        for line in records.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
                session_tokens = TokenUsage()
                session_cost = 0.0
                for usage in record["usage_delta"]:
                    session_cost += float(usage.get("estimated_cost", 0))
                    session_tokens += TokenUsage(
                        input=int(usage.get("input_tokens", 0)),
                        output=int(usage.get("output_tokens", 0)),
                        cache_read=int(usage.get("cached_input_tokens", 0)),
                        reasoning=int(usage.get("reasoning_output_tokens", 0)),
                    )
                key = f"pi/{record['session_index']}"
                self.usage_by_agent[key] = AgentThreadUsage(
                    cost=session_cost, tokens=session_tokens
                )
                self.usage.cost += session_cost
                self.usage.net_tokens += session_tokens
                self.usage.current_tokens = session_tokens
            except (ValueError, TypeError, KeyError) as exc:
                raise AgentError(
                    "Invalid myteam session usage artifact"
                ) from exc

    def retry(self) -> None:
        if self._last_task is None:
            raise AgentError("There is no myteam workflow invocation to retry")
        self.run(self._last_task)

    def reset(self) -> None:
        self._last_task = None
        if self._tmp_dir is not None:
            self._prepare_checkpoint()

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._artifacts_dir is not None:
            shutil.copytree(self._artifacts_dir, path, dirs_exist_ok=True)
        if self._bundle_dir is not None:
            shutil.copytree(
                self._bundle_dir / "workflow",
                path / "workflow",
                dirs_exist_ok=True,
            )
            shutil.copy2(
                self._bundle_dir / "workflow.json", path / "workflow.json"
            )

    def cleanup(self) -> None:
        try:
            if self._runtime is not None:
                self._runtime.cleanup()
        finally:
            self._runtime = None
            if self._tmp_dir is not None:
                self._tmp_dir.cleanup()
                self._tmp_dir = None


register_agent("myteam", MyTeamAgent)
