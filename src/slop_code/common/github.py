"""GitHub identity used to tag shared run outputs."""

from __future__ import annotations

import functools
import os
import re
import subprocess

GITHUB_USER_ENV = "SCB_GITHUB_USER"

# GitHub logins are alphanumeric plus single hyphens, so they never contain "_".
GITHUB_LOGIN_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_LOGIN_RE = re.compile(rf"^{GITHUB_LOGIN_PATTERN}$")


@functools.cache
def github_username() -> str:
    """Return the GitHub login from $SCB_GITHUB_USER or the gh CLI."""
    login = os.environ.get(GITHUB_USER_ENV, "").strip()
    if not login:
        try:
            result = subprocess.run(
                ["gh", "api", "user", "--jq", ".login"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            login = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Cannot determine GitHub username. Run `gh auth login` "
                f"or set {GITHUB_USER_ENV}."
            ) from exc

    if not _LOGIN_RE.match(login):
        raise RuntimeError(f"Invalid GitHub username: {login!r}")
    return login
