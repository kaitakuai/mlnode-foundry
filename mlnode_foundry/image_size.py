"""Compressed image size lookup for a published OCI image.

The previous workflow used `docker buildx imagetools inspect --raw | jq '.size'`
which read manifest-blob sizes (few KB), not the actual layer-byte total —
yielding 'size: "0 GB"' after rounding. This module does it properly:

  1. Fetch the OCI image index for the reference.
  2. Pick the linux/amd64 manifest (skip attestation manifests).
  3. Fetch that platform manifest.
  4. Sum config.size + Σ layers[].size.

Output is humanized (e.g. "15 GB") for the dashboard registry-view JSON.
"""

from __future__ import annotations

import json
import subprocess


class ImageSizeError(RuntimeError):
    """Raised when buildx inspect fails or the manifest is malformed."""


def _buildx_inspect_raw(ref: str) -> dict:
    """Run `docker buildx imagetools inspect --raw <ref>` and return parsed JSON."""
    try:
        result = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", "--raw", ref],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ImageSizeError(
            f"buildx inspect failed for {ref}: {exc.stderr.strip()}"
        ) from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ImageSizeError(f"buildx inspect for {ref} returned non-JSON") from exc


def _is_platform_manifest(entry: dict, arch: str = "amd64", os_: str = "linux") -> bool:
    plat = entry.get("platform") or {}
    if plat.get("architecture") != arch or plat.get("os") != os_:
        return False
    # Skip attestation manifests even if platform matches.
    annotations = entry.get("annotations") or {}
    if annotations.get("vnd.docker.reference.type") == "attestation-manifest":
        return False
    return True


def _sum_manifest_bytes(manifest: dict) -> int:
    """Sum config.size + every layer's size in a single-platform OCI manifest."""
    total = (manifest.get("config") or {}).get("size", 0)
    for layer in manifest.get("layers") or []:
        total += layer.get("size", 0)
    return total


def fetch_image_compressed_size(ref: str) -> int:
    """Return total compressed image size in bytes for the linux/amd64 platform.

    Works for both image-index (multi-platform) and direct manifest references.
    Returns 0 if no linux/amd64 manifest exists.
    """
    top = _buildx_inspect_raw(ref)

    # Direct manifest (rare) — no index wrap.
    if top.get("layers") is not None:
        return _sum_manifest_bytes(top)

    # OCI image index — find platform manifest digest.
    manifests = top.get("manifests") or []
    platform_digest = None
    for entry in manifests:
        if _is_platform_manifest(entry):
            platform_digest = entry["digest"]
            break

    if platform_digest is None:
        return 0

    # Strip both `:tag` and `@digest` so the platform lookup uses `name@digest` only.
    # Only strip the tag if `:` follows the final `/` — preserves any `registry:port/...` form.
    no_digest = ref.split("@", 1)[0]
    if "/" in no_digest:
        prefix, last = no_digest.rsplit("/", 1)
        last = last.split(":", 1)[0]
        base_ref = f"{prefix}/{last}"
    else:
        base_ref = no_digest.split(":", 1)[0]
    platform_ref = f"{base_ref}@{platform_digest}"
    platform_manifest = _buildx_inspect_raw(platform_ref)
    return _sum_manifest_bytes(platform_manifest)


def humanize_bytes(n: int) -> str:
    """Convert bytes to a short humanized string ('15 GB' / '512 MB')."""
    if n <= 0:
        return "0 B"
    for unit, factor in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if n >= factor:
            value = n / factor
            # Round GB to integer for clean dashboard look ("15 GB" not "15.3 GB").
            if unit == "GB":
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
    return f"{n} B"
