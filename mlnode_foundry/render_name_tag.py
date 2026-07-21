"""Render profile + naming policy → (package_name, tag) string.

This is org-wide naming policy made executable. The single source of truth
for "what should this profile's GHCR coordinates be" lives in
`tools/naming.cue`; this module reads it and applies it.
"""

from __future__ import annotations

from .cue import cue_export
from .paths import REPO_ROOT


def load_naming() -> dict:
    """Load `tools/naming.cue` policy as a dict."""
    return cue_export(REPO_ROOT / "tools" / "naming.cue")


def render_package(profile: dict, naming: dict) -> str:
    """Compute GHCR package name from profile.identity.axes + naming.package."""
    axes = profile["identity"]["axes"]
    pkg_prefix = naming["package"]["prefix"]
    pkg_axes = naming["package"]["axes"]
    parts = [axes[a] for a in pkg_axes]
    return f"{pkg_prefix}-{'-'.join(parts)}"


def render_tag(profile: dict, naming: dict) -> str:
    """Compute image tag from profile.identity + naming.tag.

    Axes that are absent in profile.identity.axes OR set to their default
    are omitted from the tag (keeps tags short).
    """
    axes = profile["identity"]["axes"]
    version = profile["identity"]["version"]
    mode = profile["mode"]
    axes_registry = naming["axes"]

    tag_axes_order = naming["tag"]["axes_order"]
    tag_axes_parts = []
    for axis_name in tag_axes_order:
        if axis_name not in axes:
            continue
        axis_def = axes_registry[axis_name]
        # Omit if value equals registry default
        default = axis_def.get("default")
        value = axes[axis_name]
        if default is not None and value == default:
            continue
        prefix = axis_def["prefix"]
        tag_axes_parts.append(f"-{prefix}.{value}")
    tag_axes = "".join(tag_axes_parts)

    template = naming["tag"]["modes"][mode]
    return template.format(
        mlnode=version.get("mlnode", ""),
        vllm=version.get("vllm", ""),
        upstream=version.get("upstream", ""),
        tag_axes=tag_axes,
        rev=version["rev"],
    )


def render_package_and_tag(profile: dict, naming: dict) -> tuple[str, str]:
    """Convenience wrapper returning (package_name, tag)."""
    return render_package(profile, naming), render_tag(profile, naming)
