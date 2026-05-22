"""Smoke tests for `cue vet` integration."""

from __future__ import annotations

import pytest

from mlnode_foundry.cue import CueError
from mlnode_foundry.validate import validate_naming, validate_profile


def test_b200_kimi_int4_valid() -> None:
    validate_profile("b200-kimi-k2-6")  # should not raise


def test_h100_minimax_valid() -> None:
    validate_profile("h100-minimax-m2-7")


def test_naming_valid() -> None:
    validate_naming()


def test_nonexistent_profile_raises() -> None:
    with pytest.raises(FileNotFoundError):
        validate_profile("does-not-exist")


def test_invalid_profile_raises(tmp_path, monkeypatch) -> None:
    """Manually craft an invalid profile and verify cue vet rejects it."""
    from mlnode_foundry import validate as validate_module

    # Create a temp profiles dir with a broken profile (mode missing)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    schema = (validate_module.REPO_ROOT / "profiles" / "schema.cue").read_text()
    (profiles_dir / "schema.cue").write_text(schema)
    (profiles_dir / "broken.cue").write_text(
        """package profiles
broken: #BaseProfile & {
    // intentionally missing required fields (identity, runner_patch, etc.)
}
"""
    )
    monkeypatch.setattr(validate_module, "REPO_ROOT", tmp_path)
    with pytest.raises(CueError):
        validate_module.validate_profile("broken")
