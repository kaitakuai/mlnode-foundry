"""Profile validation via `cue vet`."""

from __future__ import annotations

from pathlib import Path

from .cue import cue_vet

REPO_ROOT = Path(__file__).resolve().parent.parent


def validate_profile(name: str) -> None:
    """Validate a profile against the schema. Raises CueError on failure."""
    profile_path = REPO_ROOT / "profiles" / f"{name}.cue"
    schema_path = REPO_ROOT / "profiles" / "schema.cue"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    cue_vet(profile_path, schema_path)


def validate_naming() -> None:
    """Validate naming.cue (self-validating; checks structure)."""
    naming_path = REPO_ROOT / "tools" / "naming.cue"
    cue_vet(naming_path)
