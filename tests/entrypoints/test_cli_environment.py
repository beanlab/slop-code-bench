"""CLI environment initialization must precede runtime imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize("existing_key", [None, "shell-test-key"])
def test_dotenv_controls_tempfiles_before_command_imports(
    tmp_path: Path, existing_key: str | None
) -> None:
    temporary_root = tmp_path / "shared-temp"
    temporary_root.mkdir()
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        f"TMPDIR={temporary_root}\nOPENAI_API_KEY=dotenv-test-key\n"
    )
    environment = os.environ.copy()
    for name in ("TMPDIR", "TMP", "TEMP", "OPENAI_API_KEY"):
        environment.pop(name, None)
    if existing_key is not None:
        environment["OPENAI_API_KEY"] = existing_key
    script = dedent(
        """
        import json
        import os
        import sys
        import tempfile
        import dotenv

        original_loader = dotenv.load_dotenv
        dotenv.load_dotenv = lambda *args, **kwargs: (
            False if args or kwargs else original_loader(sys.argv[1])
        )

        import slop_code.entrypoints.cli

        print(json.dumps({
            "temporary_root": tempfile.gettempdir(),
            "key_matches": os.environ.get("OPENAI_API_KEY") == sys.argv[2],
        }))
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script.
        [
            sys.executable,
            "-c",
            script,
            str(dotenv_file),
            existing_key or "dotenv-test-key",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["temporary_root"] == str(temporary_root)
    assert payload["key_matches"]
