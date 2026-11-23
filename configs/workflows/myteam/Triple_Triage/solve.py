"""
type: workflow
description: Plans, implements, and Triple Triages solutions to a benchmark checkpoint, fixing triaged actions in a loop.
usage: no arguments; checkpoint inputs are supplied by the benchmark adapter
"""

from __future__ import annotations

import json

from myteam.workflows import report_workflow_result
from scb_myteam import as_text
from scb_myteam import load_request
from scb_myteam import needs_changes
from scb_myteam import run_agent

# Each triage pass that requests changes gets one fix pass.

TRIAGE_ROUNDS = 2

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
SPEC_ALIGNMENT_PROMPT = (
    "Review the implementation of the spec. "
    "Inspect the changed files and test results. Do not edit files. "
    "Report any correctness issues or missing requirements; if none "
    "are found, say so.\n\nTask:\n{{ task }}"
    "\n\nImplementation summary:\n{{ implementation_summary }}"
)
EDGE_CASES_PROMPT = (
    "Review the implementation against the task, specifically considering "
    "what edge cases the implementation might have missed. "
    "Inspect the changed files and test results. Do not edit files. "
    "Report any unaddressed edge cases; if none "
    "are found, say so.\n\nTask:\n{{ task }}"
    "\n\nImplementation summary:\n{{ implementation_summary }}"
)
SIMPLICITY_PROMPT = (
    "Review the implementation's simplicity for accomplishing the task. "
    "Inspect the changed files and test results. Do not edit files. "
    "Report unnecessary complexity, code that is difficult to follow, or "
    "code that has outgrown its architecture; if none "
    "are found, say so.\n\nTask:\n{{ task }}"
    "\n\nImplementation summary:\n{{ implementation_summary }}"
)
TRIAGE_PROMPT = (
    "Review the three previous reviews. "
    "Inspect the changed files and test results. Do not edit files. "
    "Report which suggested actions are necessary for the longevity and "
    "functionality of the project. Some findings may already be addressed "
    "in the current workspace; include only actions that are still needed. "
    "Set needs_changes to true only when the implementation agent should "
    "make further changes."
    "\n\nSpec:\n{{ task }}"
    "\n\nReview for Spec Alignment:\n{{ review_spec_alignment }}"
    "\n\nReview for Edge Cases:\n{{ review_edge_cases }}"
    "\n\nReview for Simplicity:\n{{ review_simplicity }}"
    "\n\nImplementation summary:\n{{ implementation_summary }}"
)
FIX_PROMPT = (
    "Address the triaged actions below. Inspect the current workspace, make "
    "the necessary changes, and run relevant tests.\n\nTask:\n{{ task }}"
    "\n\nDesign plan:\n{{ plan }}\n\nTriaged actions:\n{{ triage_report }}"
)

REVIEW_SPECS = {
    "review_spec_alignment": (
        SPEC_ALIGNMENT_PROMPT,
        "Review findings related to spec alignment, or state that none were found.",
    ),
    "review_edge_cases": (
        EDGE_CASES_PROMPT,
        "Review findings related to edge cases, or state that none were found.",
    ),
    "review_simplicity": (
        SIMPLICITY_PROMPT,
        "Review findings related to code simplicity, or state that none were found.",
    ),
}


def main() -> None:
    task = load_request()["task"]

    design = run_agent(
        prompt=DESIGN_PROMPT,
        input={"task": task},
        output={"plan": "A concise implementation plan."},
    )
    plan = as_text(design.output)

    implementation = run_agent(
        prompt=IMPLEMENTATION_PROMPT,
        input={"task": task, "plan": plan},
        output={"summary": "Describe the implementation and validation."},
    )
    initial_implementation = implementation.output
    implementation_summary = as_text(implementation.output)

    reviews = {}
    for key, (prompt, description) in REVIEW_SPECS.items():
        review = run_agent(
            prompt=prompt,
            input={"task": task, "implementation_summary": implementation_summary},
            output={key: description},
        )
        reviews[key] = review.output
    review_texts = {key: as_text(output) for key, output in reviews.items()}

    # Triage starts cold each round. The summary accumulates fix summaries
    # so a fresh triage sees the whole implementation history.

    rounds = []
    for round_number in range(1, TRIAGE_ROUNDS + 1):
        triage = run_agent(
            prompt=TRIAGE_PROMPT,
            input={
                "task": task,
                **review_texts,
                "implementation_summary": implementation_summary,
            },
            output={
                "triage_report": (
                    "Prioritized actions drawn from the three reviews, "
                    "or state that none are needed."
                ),
                "needs_changes": (
                    "Boolean: whether the implementation agent must make changes."
                ),
            },
        )
        record = {"round": round_number, "triage": triage.output}
        rounds.append(record)
        if not needs_changes(triage.output):
            break

        implementation = run_agent(
            prompt=FIX_PROMPT,
            input={
                "task": task,
                "plan": plan,
                "triage_report": as_text(triage.output),
            },
            output={"summary": "Describe the changes and validation."},
            session_id=implementation.session_id,
        )
        record["fix"] = implementation.output
        implementation_summary += (
            f"\n\nFix round {round_number}:\n{as_text(implementation.output)}"
        )

    report_workflow_result(
        json.dumps(
            {
                "design": design.output,
                "implementation": initial_implementation,
                **reviews,
                "rounds": rounds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
