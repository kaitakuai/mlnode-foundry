"""Render profile → docker buildx build args.

Resolves BASE_IMAGE per profile.mode:
- `kaitakuai-base`: from tools/stage3.lock.cue (Stage 3 published image)
- `upstream-overlay` / `upstream-test`: from profile.base (explicit image+digest);
  the two differ only in the word the tag carries, not in how the base resolves

MLNODE_FOUNDRY_SPIKE_BASE=1 replaces the resolved base with SPIKE_BASE_IMAGE
— an unconditional override for local dev before the first Stage 3 publish
(applies to every mode, including upstream-overlay).
"""

from __future__ import annotations

import os

from .cue import cue_export
from .paths import REPO_ROOT

# Phase 1 spike base — unconditional override when MLNODE_FOUNDRY_SPIKE_BASE=1 in env.
SPIKE_BASE_IMAGE = "ghcr.io/product-science/mlnode:3.0.13-alpha5"


def load_profile(name: str) -> dict:
    """Load profile by name; return resolved profile struct with schema applied."""
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


def load_stage3_lock() -> dict:
    """Load tools/stage3.lock.cue."""
    return cue_export(REPO_ROOT / "tools" / "stage3.lock.cue")


def resolve_base_image(profile: dict) -> str:
    """Resolve BASE_IMAGE for a profile based on its mode."""
    if os.environ.get("MLNODE_FOUNDRY_SPIKE_BASE") == "1":
        return SPIKE_BASE_IMAGE

    mode = profile["mode"]
    if mode == "kaitakuai-base":
        lock = load_stage3_lock()
        s3 = lock["stage3"]
        # Use tag reference; CI verifies digest separately.
        # Phase 4 will switch to digest pinning once stage 3 publishes consistently.
        return f"{s3['package']}:{s3['tag']}"
    elif mode in ("upstream-overlay", "upstream-test"):
        base = profile.get("base", {})
        digest = base.get("digest")
        if digest:
            return f"{base['image']}@{digest}"
        return f"{base['image']}:{base.get('upstream_version', 'latest')}"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def render_build_args(profile: dict, package: str, tag: str, profile_hash: str) -> dict[str, str]:
    """Build args to pass to docker buildx (--build-arg KEY=VALUE).

    Note: profile.env values are baked into the *rendered* Dockerfile by
    render_dockerfile.py (no longer passed as build-args). This function
    only emits identity + base image args.
    """
    axes = profile["identity"]["axes"]
    return {
        "BASE_IMAGE":     resolve_base_image(profile),
        "GPU":            axes["gpu"],
        "MODEL":          axes["model"],
        "MODEL_REVISION": axes["model_revision"],
        "QUANT":          axes.get("quant", ""),
        "PACKAGE_NAME":   package,
        "TAG":            tag,
        "PROFILE_HASH":   profile_hash,
        "RUNNER_PATCH":   profile.get("runner_patch", ""),
    }
