"""Render profile + naming policy → docker buildx build args + image tag."""

from __future__ import annotations

from pathlib import Path

from .cue import cue_export

REPO_ROOT = Path(__file__).resolve().parent.parent

# Spike: Stage 2 not yet implemented; use upstream binary as base directly.
SPIKE_BASE_IMAGE = "ghcr.io/product-science/mlnode:3.0.13-alpha5"


def load_profile(name: str) -> dict:
    """Load profile by name; return resolved `profile` field with schema applied."""
    profile_path = REPO_ROOT / "profiles" / f"{name}.cue"
    schema_path = REPO_ROOT / "profiles" / "schema.cue"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    data = cue_export(profile_path, schema_path)
    if "profile" not in data:
        raise ValueError(f"Profile file did not export 'profile' field: {profile_path}")
    return data["profile"]


def load_naming() -> dict:
    """Load naming.cue policy."""
    return cue_export(REPO_ROOT / "tools" / "naming.cue")


def render_package_and_tag(profile: dict, naming: dict) -> tuple[str, str]:
    """Compute (package_name, tag) from profile + naming policy."""
    axes = profile["identity"]["axes"]
    version = profile["identity"]["version"]
    mode = profile["mode"]

    # Package name: prefix + axes from naming.package.axes
    pkg_axes = naming["package"]["axes"]
    pkg_prefix = naming["package"]["prefix"]
    name_parts = [axes[a] for a in pkg_axes]
    package = f"{pkg_prefix}-{'-'.join(name_parts)}"

    # Tag axes (axes that appear in the tag string)
    tag_axes_order = naming["tag"]["axes_order"]
    tag_axes_parts = []
    for axis_name in tag_axes_order:
        if axis_name in axes:
            prefix = naming["axes"][axis_name]["prefix"]
            value = axes[axis_name]
            tag_axes_parts.append(f"-{prefix}.{value}")
    tag_axes = "".join(tag_axes_parts)

    # Tag template by mode
    template = naming["tag"]["modes"][mode]
    tag = template.format(
        mlnode=version.get("mlnode", ""),
        vllm=version.get("vllm", ""),
        upstream=version.get("upstream", ""),
        tag_axes=tag_axes,
        rev=version["rev"],
    )

    return package, tag


def render_build_args(profile: dict, package: str, tag: str) -> dict[str, str]:
    """Build args to pass to docker buildx (--build-arg KEY=VALUE)."""
    args: dict[str, str] = {
        "BASE_IMAGE": SPIKE_BASE_IMAGE,
        "GPU": profile["identity"]["axes"]["gpu"],
        "MODEL": profile["identity"]["axes"]["model"],
        "QUANT": profile["identity"]["axes"].get("quant", ""),
        "PACKAGE_NAME": package,
        "TAG": tag,
    }
    # ENV vars from profile — passed as ENV_<KEY> build-args, mapped to ENV in Dockerfile
    for k, v in profile.get("env", {}).items():
        args[f"ENV_{k}"] = str(v)
    return args
