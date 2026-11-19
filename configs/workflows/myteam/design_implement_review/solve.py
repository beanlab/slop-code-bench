"""
type: workflow
description: Design, implement, and review a benchmark checkpoint in sequence.
usage: no arguments; checkpoint inputs are supplied by the benchmark adapter
"""

from __future__ import annotations

import json
from typing import Any

from myteam.workflows import report_workflow_result
from scb_myteam import load_request
from scb_myteam import run_agent


def _as_text(value: Any) -> str:
    if value is None:
        return "No structured output was returned."
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _needs_changes(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    requested = value.get("needs_changes", False)
    if isinstance(requested, str):
        return requested.strip().lower() in {"true", "yes", "1"}
    return bool(requested)

DESIGN_PROMPT = (
    "Design an approach for the benchmark task below. Inspect the "
    "workspace as needed, but do not edit files. Return a concise "
    "implementation plan.\n\nTask:\n{{ task }}"
)
IMPLEMENTATION_PROMPT = (
    "Implement the benchmark task, following the design plan. Make "
    "the required changes and run relevant tests.\n\nTask:\n{{ task }}"
    "\n\nDesign plan:\n{{ plan }}"
)
REVIEW_PROMPT = (
    "Review the implementation against the task and design plan. "
    "Inspect the changed files and test results. Do not edit files. "
    "Report any correctness issues or missing requirements; if none "
    "are found, say so. Set needs_changes to true only when the "
    "implementation agent should make further changes.\n\nTask:\n{{ task }}"
    "\n\nDesign plan:\n{{ plan }}"
    "\n\nImplementation summary:\n{{ implementation_summary }}"
)
IMPLEMENTATION_FIX_PROMPT = (
    "Address the review findings below. Inspect the current workspace, make "
    "the necessary changes, and run relevant tests.\n\nTask:\n{{ task }}"
    "\n\nDesign plan:\n{{ plan }}\n\nReview findings:\n{{ review }}"
)

def main() -> None:
    task = load_request()["task"]

    design = run_agent(
        prompt=DESIGN_PROMPT,
        input={"task": task},
        output={"plan": "A concise implementation plan."},
    )
    plan = _as_text(design.output)

    implementation = run_agent(
        prompt=IMPLEMENTATION_PROMPT,
        input={"task": task, "plan": plan},
        output={"summary": "Describe the implementation and validation."},
    )

    implementation_summary = _as_text(implementation.output)
    review = None
    for round_number in range(3):
        review = run_agent(
            prompt=REVIEW_PROMPT,
            input={
                "task": task,
                "plan": plan,
                "implementation_summary": implementation_summary,
            },
            output={
                "review": "Review findings, or state that none were found.",
                "needs_changes": "Boolean: whether the implementation agent must make changes.",
            },
        )
        if not _needs_changes(review.output) or round_number == 2:
            break

        implementation = run_agent(
            prompt=IMPLEMENTATION_FIX_PROMPT,
            input={
                "task": task,
                "plan": plan,
                "review": _as_text(review.output),
            },
            output={"summary": "Describe the changes and validation."},
            session_id=implementation.session_id,
        )
        implementation_summary = _as_text(implementation.output)

    report_workflow_result(
        json.dumps(
            {
                "design": design.output,
                "implementation": implementation.output,
                "review": review.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
