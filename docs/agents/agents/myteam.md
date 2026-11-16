# Myteam Python workflows

The `myteam` agent runs a complete Python workflow for each benchmark
checkpoint. The workflow can launch several PI sessions to build a solution in
the benchmark workspace. The benchmark evaluates the resulting files normally.
The default `configs/agents/myteam.yaml` uses a single-session example; the
`configs/agents/myteam_design_implement_review.yaml` config selects a simple
three-session design, implementation, and review workflow.

The initial integration targets myteam `0.3.21` and PI `0.74.0`.

## Configuration

```yaml
type: myteam
version: "0.3.21"
pi_version: "0.74.0"
python: python
workflow_dir: configs/workflows/myteam/example
workflow_entrypoint: solve.py
workflow_args: []
timeout: 7200
idle_timeout: 600
env: {}
cost_limits:
  cost_limit: 20
  net_cost_limit: 50
  step_limit: 0
```

`workflow_dir` is a host directory resolved from the command's working directory.
`workflow_entrypoint` must point inside that directory. Supporting files in the
directory are staged with the workflow, outside the solution workspace. Workflow
code runs with the solution workspace as its current directory, so relative
solution paths address the files that will be evaluated.

The `python` executable is used inside the selected execution environment. For
local execution it can be an absolute interpreter path, such as the Python in a
conda environment containing myteam. Local execution also requires `pi` on PATH;
set `env.PATH` when using a separately installed CLI. The Docker template installs
the pinned packages for container execution.

Example:

```bash
uv run slop-code run \
  --agent configs/agents/myteam.yaml \
  --model anthropic/sonnet-4.5 \
  --problem file_backup
```

Use the standard provider API-key setup. File-based authentication is not
supported by this initial adapter. Local and container execution use an isolated
home directory so personal myteam defaults and agent sessions are not reused.

## Writing a workflow

Use the staged `scb_myteam` helper for every managed session:

```python
from scb_myteam import load_request
from scb_myteam import run_agent

request = load_request()
result = run_agent(
    prompt="{{ task }}",
    input={"task": request["task"]},
    output={"summary": "Describe the implemented solution."},
)
```

The request includes `task`, `model`, and `reasoning`. The helper supplies the
selected model and reasoning to PI and forces `interactive=False`. Additional
PI `run_agent` options can be supplied explicitly. Session results have the normal
myteam interface, so later workflow stages can use `result.output` or resume
`result.session_id`. Forked sessions (`fork=True`) are unsupported in this MVP.

Workflows start through `myteam start`, preserving myteam's supervisor and nested
workflow behavior. Use `report_workflow_result(...)` to return caller-facing text;
ordinary printed output is display/logging. Nested workflows must also use the
helper to participate in benchmark usage accounting.

See [the single-session example](../../../configs/workflows/myteam/example/solve.py),
[the design/implement/review workflow](../../../configs/workflows/myteam/design_implement_review/solve.py),
and [the implementation plan](../myteam-mvp-plan.md).

## Design, implement, and review workflow

Run the included three-stage workflow by selecting its agent config:

```bash
uv run slop-code run \\
  --agent configs/agents/myteam_design_implement_review.yaml \\
  --model anthropic/sonnet-4.5 \\
  --problem file_backup
```

It starts with a read-only design stage, then runs implementation and
read-only review stages. When the review requests changes, its findings are
passed back to a new implementation session and the result is reviewed again.
The workflow allows up to three review rounds; if issues remain after the last
round, the report marks the work as not approved and includes the final
findings. Each stage counts as a session for usage and cost accounting.

## Accounting and artifacts

The helper records session results, including transcripts and raw usage, in
`sessions.jsonl`. The adapter aggregates usage and exposes per-session usage to
the benchmark. Resumed native sessions report cumulative usage; the integration
subtracts earlier recorded totals rather than charging the same tokens twice.

Checkpoint artifacts include available session records and captured process
output. Artifacts are retained when a workflow fails, times out, or is
interrupted. Captured stdout and stderr are written incrementally, so output
received before an interruption is retained.

PI writes native session JSONL files directly under `native_sessions/` while it
runs. These files retain the agent's messages and tool activity even when the
managed session has not returned a `SessionResult`. The integration uses the
native session header and usage records to recover session metadata when
myteam's normal session lookup cannot find it. Resumed session IDs resolve back
to these preserved files.

`events.jsonl` records workflow and session starts, finishes, and failures as
they occur. A `session_started` event without a matching completion event means
the session did not return; it does not establish that the benchmark is stuck
in final cleanup. `idle_timeout.txt` distinguishes an inactivity failure from
the overall checkpoint timeout, and `error.txt` contains its diagnostic.

Dollar budgets are checked before and after managed sessions. They do not stop
an individual running session at an exact dollar amount. `timeout` limits total
elapsed execution time for one checkpoint. `idle_timeout` limits the time a
workflow may make no recorded progress; its default is 600 seconds. Set it to
`0` to disable the inactivity limit for workloads with long, quiet operations.
Session lifecycle events and changes to `native_sessions/` files count as
progress. Console silence alone does not imply inactivity, and the watchdog's
own checks do not reset the inactivity deadline.
An inactivity timeout reports an error and closes the managed workflow rather
than leaving the benchmark waiting for the full checkpoint timeout.

Model/tool step limits are unsupported;
nonzero `step_limit` values are rejected. Usage and estimated costs depend on
myteam's native session parsers; missing telemetry must be investigated before
using cost results for comparisons. An empty usage record or zero displayed
cost does not establish that a provider charged nothing. Session files that
were not preserved by an earlier run cannot be reconstructed from its zero
usage totals.

Calling `myteam.run_agent` directly bypasses the helper's accounting and budget
checks. Workflows must use `scb_myteam.run_agent` for this MVP.

Run optional installed-package smoke tests by setting `SCB_MYTEAM_TEST_PYTHON`
to an interpreter containing myteam. These exercise the real supervisor and
nested workflows without contacting a model provider.

## Resuming an interrupted run

Use the saved run directory to resume with the same model and configuration:

```bash
uv run slop-code run --resume outputs/<model>/<run-directory>
```

Resume validates saved checkpoint artifacts and prompts before reusing them.
If Ctrl+C left an all-skipped summary, completed inference results and snapshots
can still establish the last completed checkpoint. Resume restores that
checkpoint's snapshot and runs the interrupted checkpoint again.
