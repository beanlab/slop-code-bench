# Myteam workflow MVP plan

Treat a complete Python myteam workflow as one benchmark agent invocation.
Keep checkpoint prompt preparation, workspace snapshots, evaluation, and
reporting in the existing benchmark runner.

1. Add a registered `myteam` config and agent adapter using `Session.spawn()`.
   Support local execution and a Docker image with pinned myteam and PI versions.
2. Stage a workflow directory and a standalone bridge outside the solution
   workspace. Pass checkpoint inputs through a JSON request file.
3. Launch workflows through `myteam start`. Provide a cooperative `run_agent`
   helper that forces headless PI sessions and records structured results.
4. Aggregate session usage, subtract resumed-session cumulative totals, and
   preserve transcripts and workflow output as checkpoint artifacts.
5. Add an example workflow, setup documentation, and tests using fake backends.
6. Retain progress evidence during execution and bound silent workflows with a
   configurable inactivity timeout, independently of the checkpoint timeout.
   Preserve native PI session files under checkpoint artifacts and use their
   activity and metadata even when myteam's session lookup is incomplete.

Acceptance criteria:

- YAML loading resolves `type: myteam` through the normal agent registry.
- The workflow receives the checkpoint task and edits the prepared workspace.
- Workflow resources and bridge files stay outside the submitted solution.
- Returned session usage appears in checkpoint totals and per-agent usage.
- Session failures, workflow failures, and timeouts produce benchmark errors
  while preserving available artifacts.
- Output received before interruption is retained, and a silent managed
  workflow reaches its inactivity deadline without requiring manual shutdown.
- Interrupted run summaries do not discard valid completed checkpoint artifacts
  when choosing the resume point.
- Checkpoint resets clear conversation/accounting state and retain solution files.
- Focused tests run without API calls or a myteam host dependency.

Initial limits:

- Workflows must launch sessions through the supplied `scb_myteam` helper.
- Only the PI backend and API-key credentials are supported initially.
- Dollar budgets are checked between sessions. A session may exceed its budget
  before it returns. Live dollar caps require additional myteam telemetry.
- Model/tool step limits are unsupported; configure `step_limit: 0`.
- Usage depends on locating and parsing the underlying session data. Missing
  telemetry must not be interpreted as evidence of a free provider request.
- A real provider-backed benchmark run and Docker build remain separate from
  the offline test suite.
