"""Render the dashboard-compatible registry-view JSON for a published profile.

This is the aggregation layer the legacy mlnode/registry/*.json provided —
combining profile spec, model registry, image metadata, and provenance into
the single shape the dashboard renders.

Inputs that MUST exist at render time:
  - profiles/<name>.cue                  (intent)
  - tools/model-registry.cue             (human-readable model metadata)
  - tools/stage3.lock.cue                (cuda + base image version)
  - tools/naming.cue                     (package + tag composition)

Inputs that CAN be passed by the workflow after push (else null):
  - digest                — image digest (sha256:...)
  - cosign_identity       — workflow identity that signed the image
  - size                  — humanized image size from buildx imagetools
  - uploaded_by           — last git author of the profile file
  - nonces / weight       — Tier 3 benchmark results (filled by poc-benchmark agent later)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .cue import cue_export
from .paths import REPO_ROOT
from .render_bake import load_profile
from .render_name_tag import load_naming, render_package_and_tag
from .validate import load_model_registry


def _git_last_author(profile_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an", "--", str(profile_path)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        name = result.stdout.strip()
        return name or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _flags(profile: dict) -> list[str]:
    """Flatten profile.env + profile.runtime_defaults into `key=value` strings."""
    out: list[str] = []
    for k, v in sorted(profile.get("env", {}).items()):
        out.append(f"{k}={v}")
    for k, v in sorted(profile.get("runtime_defaults", {}).items()):
        out.append(f"{k}={v}")
    return out


def _flag_descriptions(tuning_notes: list[dict]) -> dict[str, str]:
    """Every tuning note → entry in flag_descriptions (rendered as tooltip)."""
    return {n["knob"]: n["reason"] for n in tuning_notes}


def _flag_warnings(tuning_notes: list[dict]) -> dict[str, str]:
    """Tuning notes with severity=warning → entry in flag_warnings (triangle icon)."""
    return {
        n["knob"]: n["reason"]
        for n in tuning_notes
        if n.get("severity") == "warning"
    }


def _report_url(tuning_notes: list[dict]) -> str | None:
    """First tuning_notes.source pointing at an experiments runbook, if any."""
    for n in tuning_notes:
        src = n.get("source", "")
        if "experiments" in src or src.startswith("http"):
            return src
    return None


def render_registry_view(
    profile_name: str,
    *,
    digest: str | None = None,
    cosign_identity: str | None = None,
    size: str | None = None,
    nonces: int | None = None,
    weight: int | None = None,
) -> dict[str, Any]:
    """Assemble the full dashboard-shaped JSON for `profile_name`."""
    profile = load_profile(profile_name)
    naming = load_naming()
    pkg, tag = render_package_and_tag(profile, naming)

    axes = profile["identity"]["axes"]
    version = profile["identity"]["version"]
    tuning_notes = profile.get("tuning_notes", [])

    # A leaf may advertise a different checkpoint than the one its axes name:
    # b300-deepseek serves the NVFP4 conversion by network decision, while the
    # axes (and governance) name the official release. Stated in the profile so
    # a rebuild re-renders it instead of clobbering a hand-edited JSON.
    advertised = profile.get("advertised_model", {})

    registry = load_model_registry()
    entry = next(
        (
            m for m in registry
            if m["family"] == axes["model"] and m["revision"] == axes["model_revision"]
        ),
        None,
    )
    if entry is None:
        raise ValueError(
            f"Profile {profile_name} references unknown (model={axes['model']}, "
            f"revision={axes['model_revision']}); add it to tools/model-registry.cue"
        )

    stage3_lock = cue_export(REPO_ROOT / "tools" / "stage3.lock.cue")
    cuda = stage3_lock.get("stage2", {}).get("cuda")

    mode = profile["mode"]
    is_overlay = mode == "upstream-overlay"
    line = "mlnode-overlay" if is_overlay else "mlnode"

    full_tag = f"{pkg}:{tag}"
    # SLSA L3 attestation lives INSIDE the OCI manifest (BuildKit
    # `provenance: mode=max` + `sbom: true` in build-stage4.yml). Verifiable
    # via `cosign verify-attestation` or `docker buildx imagetools inspect`.
    # We do NOT use `actions/attest-build-provenance`, so the GitHub
    # /attestations page is not populated for our repo. Schema keeps the
    # field for future use (e.g., if Gonka governance ever requires a
    # publishable attestation URL alongside the in-manifest provenance).
    slsa_attestation_url = None

    return {
        "$schema": "./schema.json",
        "schema_version": 1,
        "line": line,
        "gpu": axes["gpu"],
        "model_family": axes["model"],
        "model_revision": axes["model_revision"],
        "quant": axes.get("quant"),
        "name": pkg,
        "tag": tag,
        "k_rev": version["rev"],
        "vllm_base_version": version.get("vllm") if not is_overlay else None,
        "mlnode_version": version.get("mlnode") if not is_overlay else None,
        "upstream_overlay_version": version.get("upstream") if is_overlay else None,
        "model": advertised.get("model", entry["hf_repo"]),
        "model_short": advertised.get("model_short", entry["display_name"]),
        "model_params_b": entry.get("params_b"),
        "model_context_max": entry.get("context_max"),
        "model_license": entry.get("license"),
        "cuda": cuda,
        "size": size,
        "runtime": f"docker run -d --gpus all -p 8080:8080 {full_tag}",
        "flags": _flags(profile),
        "flag_warnings": _flag_warnings(tuning_notes),
        "flag_descriptions": _flag_descriptions(tuning_notes),
        "digest": digest,
        "cosign_identity": cosign_identity,
        "slsa_attestation_url": slsa_attestation_url,
        "report_url": _report_url(tuning_notes),
        "uploaded_by": _git_last_author(REPO_ROOT / "profiles" / f"{profile_name}.cue"),
        "nonces": nonces,
        "weight": weight,
    }
