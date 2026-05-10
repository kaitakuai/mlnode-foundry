"""Subprocess wrapper for the `cue` CLI binary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class CueError(RuntimeError):
    """Raised when cue CLI returns non-zero or produces invalid output."""


def cue_export(*paths: str | Path, cwd: Path | None = None) -> dict:
    """Run `cue export` on given paths, return parsed JSON."""
    cmd = ["cue", "export", *map(str, paths)]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as exc:
        raise CueError(
            f"cue export failed (exit {exc.returncode}):\n"
            f"stdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CueError(f"cue export produced invalid JSON: {exc}") from exc


def cue_vet(*paths: str | Path, cwd: Path | None = None) -> None:
    """Run `cue vet`. Raises CueError on validation failure."""
    cmd = ["cue", "vet", *map(str, paths)]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as exc:
        raise CueError(
            f"cue vet failed (exit {exc.returncode}):\n"
            f"stdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc
