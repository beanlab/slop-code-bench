"""Sync finished run outputs with the shared S3 bucket.

Pulls every finished run from S3, then pushes your own finished runs.
Nothing is ever deleted on either side.

A run folder qualifies when it sits at <model>/<run>, its name follows
YYMMDD_<problem>_<agent>_<prompt>_<thinking>_<run#>_<github-user>, and it
contains result.json.
You only upload runs that end with your own GitHub username.

Untested against the live bucket; try --dry-run first.

Usage:
    uv run python scripts/sync_outputs.py [--dry-run] [--profile NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError

from slop_code.common.github import GITHUB_LOGIN_PATTERN
from slop_code.common.github import github_username

DEFAULT_BUCKET = "byu-cs401r-slopcoderuns"
DEFAULT_PREFIX = "outputs"
FINISHED_MARKER = "result.json"
EXCLUDED_DIRS = {".venv", "__pycache__", "node_modules", ".pytest_cache"}

RUN_NAME_RE = re.compile(
    rf"^\d{{6}}_.+_\d+_(?P<user>{GITHUB_LOGIN_PATTERN})$"
)


@dataclass(frozen=True)
class RemoteObject:
    size: int
    etag: str


def run_owner(run_name: str) -> str | None:
    match = RUN_NAME_RE.match(run_name)
    return match["user"] if match else None


def same_content(path: Path, remote: RemoteObject) -> bool:
    """Compare by size, plus MD5 when the ETag is a plain MD5."""
    if not path.is_file() or path.stat().st_size != remote.size:
        return False

    # Multipart ETags contain "-" and are not an MD5 of the file.
    if "-" in remote.etag:
        return True
    return (
        hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        == remote.etag
    )


def marker_last(keys: list[str]) -> list[str]:
    """Order keys so result.json transfers last and marks the run complete."""
    return sorted(keys, key=lambda k: k.endswith(f"/{FINISHED_MARKER}"))


def list_remote(s3, bucket: str, prefix: str) -> dict[str, RemoteObject]:
    objects = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = RemoteObject(
                size=obj["Size"], etag=obj["ETag"].strip('"')
            )
    return objects


def group_remote_runs(
    objects: dict[str, RemoteObject], prefix: str
) -> dict[str, list[str]]:
    """Map '<model>/<run>' to its keys, keeping only finished runs."""
    runs: dict[str, list[str]] = defaultdict(list)
    for key in objects:
        parts = key[len(prefix) + 1 :].split("/")
        if len(parts) < 3 or run_owner(parts[1]) is None:
            continue
        runs[f"{parts[0]}/{parts[1]}"].append(key)

    return {
        run: keys
        for run, keys in runs.items()
        if f"{prefix}/{run}/{FINISHED_MARKER}" in objects
    }


def local_runs(outputs_dir: Path, user: str) -> list[Path]:
    """Return your finished runs under outputs/<model>/<run>."""
    return sorted(
        run_dir
        for run_dir in outputs_dir.glob("*/*")
        if run_dir.is_dir()
        and run_owner(run_dir.name) == user
        and (run_dir / FINISHED_MARKER).is_file()
    )


def run_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and not EXCLUDED_DIRS.intersection(path.relative_to(run_dir).parts)
    ]


def transfer(batches: list[list[str]], copy_one, workers: int) -> None:
    """Copy each run's keys in parallel, holding result.json until the end."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for keys in batches:
            *body, last = keys
            list(pool.map(copy_one, body))
            copy_one(last)


def local_path(args, key: str) -> Path:
    return args.outputs_dir / key[len(args.prefix) + 1 :]


def pull(s3, args, objects) -> int:
    batches = []
    for run, keys in sorted(group_remote_runs(objects, args.prefix).items()):
        pending = [
            key
            for key in marker_last(keys)
            if not same_content(local_path(args, key), objects[key])
        ]
        if pending:
            print(f"pull {run}  ({len(pending)} files)")
            batches.append(pending)

    def download(key: str) -> None:
        dest = local_path(args, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(args.bucket, key, str(dest))

    if not args.dry_run:
        transfer(batches, download, args.workers)
    return sum(map(len, batches))


def push(s3, args, objects, user) -> int:
    batches = []
    for run_dir in local_runs(args.outputs_dir, user):
        keys = [
            f"{args.prefix}/{path.relative_to(args.outputs_dir).as_posix()}"
            for path in run_files(run_dir)
        ]
        pending = [
            key
            for key in marker_last(keys)
            if key not in objects
            or not same_content(local_path(args, key), objects[key])
        ]
        if pending:
            rel_run = run_dir.relative_to(args.outputs_dir).as_posix()
            print(f"push {rel_run}  ({len(pending)} files)")
            batches.append(pending)

    def upload(key: str) -> None:
        s3.upload_file(str(local_path(args, key)), args.bucket, key)

    if not args.dry_run:
        transfer(batches, upload, args.workers)
    return sum(map(len, batches))


def unclaimed_runs(outputs_dir: Path) -> list[Path]:
    return sorted(
        run_dir
        for run_dir in outputs_dir.glob("*/*")
        if run_dir.is_dir()
        and (run_dir / "config.yaml").is_file()
        and run_owner(run_dir.name) is None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bucket", default=os.environ.get("SCB_S3_BUCKET", DEFAULT_BUCKET)
    )
    parser.add_argument(
        "--prefix", default=os.environ.get("SCB_S3_PREFIX", DEFAULT_PREFIX)
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("SCB_AWS_PROFILE"),
        help="AWS profile with access to the bucket",
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would transfer without transferring",
    )
    parser.add_argument("--pull-only", action="store_true")
    parser.add_argument("--push-only", action="store_true")
    args = parser.parse_args()
    args.prefix = args.prefix.strip("/")
    return args


def main() -> int:
    args = parse_args()
    user = github_username()
    s3 = boto3.Session(profile_name=args.profile).client("s3")

    print(f"user {user}  bucket s3://{args.bucket}/{args.prefix}/")
    try:
        objects = list_remote(s3, args.bucket, args.prefix)
    except (BotoCoreError, ClientError) as exc:
        print(f"cannot list s3://{args.bucket}/{args.prefix}/: {exc}")
        print(
            "check --profile or SCB_AWS_PROFILE points at credentials "
            "with access to the bucket"
        )
        return 1

    pulled = 0 if args.push_only else pull(s3, args, objects)
    pushed = 0 if args.pull_only else push(s3, args, objects, user)

    verb = "would transfer" if args.dry_run else "transferred"
    print(f"{verb}: {pulled} files down, {pushed} files up")

    skipped = unclaimed_runs(args.outputs_dir)
    if skipped:
        print(
            f"\n{len(skipped)} local runs lack a username suffix "
            "and were not uploaded:"
        )
        for run_dir in skipped:
            print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
