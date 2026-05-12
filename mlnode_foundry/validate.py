"""Profile validation via `cue vet` + cross-package referential checks."""

from __future__ import annotations

from pathlib import Path

from .cue import cue_export, cue_vet

REPO_ROOT = Path(__file__).resolve().parent.parent


class ModelRegistryError(ValueError):
    """Profile references (model, model_revision) tuple not in tools/model-registry.cue."""


def load_model_registry() -> list[dict]:
    """Load tools/model-registry.cue and return its models list."""
    data = cue_export(REPO_ROOT / "tools" / "model-registry.cue")
    return data.get("models", [])


def validate_model_revision(profile_name: str, profile: dict) -> None:
    """Cross-check profile (model, model_revision) against the registry."""
    axes = profile["identity"]["axes"]
    family = axes["model"]
    revision = axes["model_revision"]
    registry = load_model_registry()
    matches = [
        m for m in registry if m["family"] == family and m["revision"] == revision
    ]
    if not matches:
        valid_pairs = sorted({(m["family"], m["revision"]) for m in registry})
        raise ModelRegistryError(
            f"Profile '{profile_name}' references (model={family}, "
            f"model_revision={revision}) which is not in tools/model-registry.cue. "
            f"Add it to the registry or pick a known pair. "
            f"Known pairs: {valid_pairs}"
        )
    entry = matches[0]
    if entry.get("status") == "deprecated":
        eol = entry.get("eol_date", "unknown")
        print(
            f"WARNING: profile '{profile_name}' uses deprecated model "
            f"{family}:{revision} (EOL {eol}). Consider migrating."
        )


def validate_profile(name: str) -> None:
    """Validate a profile against the schema AND the model registry.

    Raises CueError on schema failure, ModelRegistryError on unknown model pair.
    """
    profile_path = REPO_ROOT / "profiles" / f"{name}.cue"
    schema_path = REPO_ROOT / "profiles" / "schema.cue"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    cue_vet(profile_path, schema_path)

    # Cross-package check that Cue cannot do at vet time.
    profile_data = cue_export(profile_path, schema_path)
    key = name.replace("-", "_")
    if key in profile_data:
        validate_model_revision(name, profile_data[key])


def validate_naming() -> None:
    """Validate naming.cue (self-validating; checks structure)."""
    naming_path = REPO_ROOT / "tools" / "naming.cue"
    cue_vet(naming_path)


def validate_model_registry() -> None:
    """Validate tools/model-registry.cue (self-validating; checks #ModelEntry shape)."""
    cue_vet(REPO_ROOT / "tools" / "model-registry.cue")
