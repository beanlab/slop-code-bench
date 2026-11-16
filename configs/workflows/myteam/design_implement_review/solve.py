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


def main() -> None:
    task = load_request()["task"]

    design = run_agent(
        prompt=(
            "Design an approach for the benchmark task below. Inspect the "
            "workspace as needed, but do not edit files. Return a concise "
            "implementation plan.\n\nTask:\n{{ task }}"
        ),
        input={"task": task},
        output={"plan": "A concise implementation plan."},
    )
    plan = _as_text(design.output)

    implementation = run_agent(
        prompt=(
            "Implement the benchmark task, following the design plan. Make "
            "the required changes and run relevant tests.\n\nTask:\n{{ task }}"
            "\n\nDesign plan:\n{{ plan }}"
        ),
        input={"task": task, "plan": plan},
        output={"summary": "Describe the implementation and validation."},
    )

    implementation_summary = _as_text(implementation.output)
    review = run_agent(
        prompt=(
            "Review the implementation against the task and design plan. "
            "Inspect the changed files and test results. Do not edit files. "
            "Report any correctness issues or missing requirements; if none "
            "are found, say so.\n\nTask:\n{{ task }}"
            "\n\nDesign plan:\n{{ plan }}"
            "\n\nImplementation summary:\n{{ implementation_summary }}"
        ),
        input={
            "task": task,
            "plan": plan,
            "implementation_summary": implementation_summary,
        },
        output={"review": "Review findings, or state that none were found."},
    )

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
