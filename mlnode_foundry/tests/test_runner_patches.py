"""Every runner-patch must apply to the upstream runner.py we build on.

A patch that cannot find its anchor exits non-zero and Stage 4 fails the build
— but only after pulling a ~20 GB base image. These tests are the same check,
in seconds, against the vendored fixture. See fixtures/README.md on refreshing
it when a base digest moves.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO / "tools" / "runner-patches"
FIXTURE = Path(__file__).parent / "fixtures" / "upstream_runner.py.txt"


def _referenced_patches() -> list[str]:
    """Patch basenames that some profile actually uses (skips LEGACY ones)."""
    names = set()
    for cue in (REPO / "profiles").glob("*.cue"):
        names.update(re.findall(r'runner_patch:\s*"([^"]+)"', cue.read_text()))
    return sorted(names)


def _load(name: str, target: Path):
    spec = importlib.util.spec_from_file_location(name, PATCH_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FILE = str(target)
    return module


@pytest.mark.parametrize("name", _referenced_patches())
def test_patch_applies_to_upstream_runner(name: str, tmp_path: Path) -> None:
    target = tmp_path / "runner.py"
    target.write_text(FIXTURE.read_text())
    patch = _load(name, target)

    assert patch.main() == 0, f"{name} could not apply to upstream runner.py"

    patched = target.read_text()
    ast.parse(patched)
    assert "Kaitaku" in patched, f"{name} reported success but injected nothing"
    # Stage 4 can re-run a patch on an already-patched layer.
    assert patch.main() == 0, f"{name} is not idempotent"
    assert target.read_text() == patched


@pytest.mark.parametrize("name", _referenced_patches())
def test_patch_fails_loudly_on_unknown_runner(name: str, tmp_path: Path) -> None:
    """A refactor upstream must break the build, not ship an unconfigured image."""
    target = tmp_path / "runner.py"
    target.write_text("class VLLMRunner:\n    pass\n")
    assert _load(name, target).main() != 0


@pytest.mark.parametrize("name", _referenced_patches())
def test_patched_runner_passes_lint(name: str, tmp_path: Path) -> None:
    """The injected block is generated line by line — indentation is easy to get
    wrong in a way ast.parse accepts (a stray dedent still parses)."""
    target = tmp_path / "runner.py"
    target.write_text(FIXTURE.read_text())
    _load(name, target).main()
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "E9,F", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
