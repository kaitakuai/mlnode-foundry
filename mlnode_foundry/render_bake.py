"""Render profile → docker buildx build args.

Profile loading + build-arg generation. Naming/tag rendering lives in
`render_name_tag.py` (separated for testability and to keep concerns narrow).
"""

from __future__ import annotations

from pathlib import Path

from .cue import cue_export

REPO_ROOT = Path(__file__).resolve().parent.parent

# Phase 2 placeholder: Stage 2 not yet built. Stage 3 inherits from upstream
# binary directly. Real BASE_IMAGE resolution from tools/stage2.lock.cue
# lands in Phase 3 (PR #2).
SPIKE_BASE_IMAGE = "ghcr.io/product-science/mlnode:3.0.13-alpha5"


def load_profile(name: str) -> dict:
    """Load profile by name; return resolved profile struct with schema applied.

    Convention: profile filename `b300-kimi-int4.cue` exports a top-level field
    named `b300_kimi_int4` (hyphens replaced by underscores). This avoids
    package-level unification conflicts between sibling profiles in same dir.
    """
    profile_path = REPO_ROOT / "profiles" / f"{name}.cue"
    schema_path = REPO_ROOT / "profiles" / "schema.cue"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    data = cue_export(profile_path, schema_path)
    key = name.replace("-", "_")
    if key not in data:
        raise ValueError(
            f"Profile file did not export top-level field '{key}' "
            f"(expected from filename {profile_path.name})"
        )
    return data[key]


def render_build_args(profile: dict, package: str, tag: str) -> dict[str, str]:
    """Build args to pass to docker buildx (--build-arg KEY=VALUE)."""
    axes = profile["identity"]["axes"]
    args: dict[str, str] = {
        "BASE_IMAGE":   SPIKE_BASE_IMAGE,
        "GPU":          axes["gpu"],
        "MODEL":        axes["model"],
        "QUANT":        axes.get("quant", ""),
        "PACKAGE_NAME": package,
        "TAG":          tag,
    }
    # ENV from profile → docker --build-arg ENV_<KEY>=VALUE
    # Dockerfile maps ENV_<KEY> ARG → ENV <KEY> for known keys.
    # Phase 3 will replace this hardcoded mapping with dynamic ENV injection.
    for k, v in profile.get("env", {}).items():
        args[f"ENV_{k}"] = str(v)
    return args
