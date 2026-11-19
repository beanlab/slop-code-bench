"""
type: workflow
description: Solve the current benchmark checkpoint with one PI session.
usage: no arguments; checkpoint inputs are supplied by the benchmark adapter
"""

from __future__ import annotations

import json

from myteam.workflows import report_workflow_result
from scb_myteam import load_request
from scb_myteam import run_agent


def main() -> None:
    request = load_request()
    result = run_agent(
        prompt="{{ task }}",
        input={"task": request["task"]},
        output={"summary": "Describe the implemented solution and validation."},
    )
    if result.output is not None:
        report_workflow_result(json.dumps(result.output))


if __name__ == "__main__":
    main()
