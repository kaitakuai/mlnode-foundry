"""Content hash for profile inputs — drives skip-if-unchanged in CI.

The hash covers everything that influences the resulting image:
  - profile .cue file content
  - schema.cue (changes invalidate validation logic)
  - all bases referenced via `bases.X` (resolved by `cue export`)
  - naming.cue (changes invalidate tag computation)
  - stage3/Dockerfile (template changes)
  - tools/stage2.lock.cue (Stage 2 pin)
  - hw-patches files referenced (Phase 3 — placeholder for now)
  - runner-patch file referenced (Phase 3 — placeholder)

Phase 2 implementation hashes the static set above. Phase 3 will add
hw-patches / runner-patches resolution from the profile.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .render_bake import REPO_ROOT


def _hash_file(path: Path) -> bytes:
    """SHA-256 of file content (returns raw digest bytes for chaining)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def compute_profile_hash(name: str) -> str:
    """Compute SHA-256 hex of all inputs that affect this profile's build.

    Includes both Stage 3 inputs (profile, schema, naming, Dockerfile.tmpl,
    bases/) AND the Stage 2 lineage (Dockerfile.patch-and-build, patches/*,
    stage2/scripts/*). Without the Stage 2 inputs, a vLLM patch added in
    Stage 2 would silently fail to invalidate Stage 3 skip-if-unchanged,
    because Stage 3 pulls Stage 2 by TAG (not digest) so the BASE_IMAGE
    build-arg stays the same while the underlying image content changes.
    """
    inputs = [
        REPO_ROOT / "profiles" / f"{name}.cue",
        REPO_ROOT / "profiles" / "schema.cue",
        REPO_ROOT / "tools" / "naming.cue",
        REPO_ROOT / "tools" / "stage2.lock.cue",
        REPO_ROOT / "tools" / "model-registry.cue",
        REPO_ROOT / "stage3" / "Dockerfile.tmpl",
        REPO_ROOT / "stage2" / "Dockerfile.patch-and-build",
    ]
    bases_dir = REPO_ROOT / "profiles" / "bases"
    if bases_dir.is_dir():
        inputs.extend(sorted(bases_dir.glob("*.cue")))

    # Source-level patches applied in Stage 2 Stage A (alpine/git container).
    patches_dir = REPO_ROOT / "patches"
    if patches_dir.is_dir():
        inputs.extend(sorted(patches_dir.glob("*.patch")))

    # In-image vLLM patchers run in Stage 2 Stage B (after Stage 1 inheritance).
    stage2_scripts = REPO_ROOT / "stage2" / "scripts"
    if stage2_scripts.is_dir():
        inputs.extend(sorted(stage2_scripts.glob("*.py")))

    aggregator = hashlib.sha256()
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"hash input missing: {p}")
        # Include path in hash so file additions/removals are visible.
        aggregator.update(str(p.relative_to(REPO_ROOT)).encode())
        aggregator.update(b"\0")
        aggregator.update(_hash_file(p))
        aggregator.update(b"\0")
    return aggregator.hexdigest()
