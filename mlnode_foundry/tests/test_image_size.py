"""Tests for compressed image size lookup."""

from __future__ import annotations

import json
import subprocess

import pytest

from mlnode_foundry import image_size
from mlnode_foundry.image_size import (
    ImageSizeError,
    _is_platform_manifest,
    _sum_manifest_bytes,
    fetch_image_compressed_size,
    humanize_bytes,
)


def test_humanize_zero() -> None:
    assert humanize_bytes(0) == "0 B"
    assert humanize_bytes(-1) == "0 B"


def test_humanize_units() -> None:
    assert humanize_bytes(512) == "512 B"
    assert humanize_bytes(2048) == "2.0 KB"
    assert humanize_bytes(2 * 1024 * 1024) == "2.0 MB"
    assert humanize_bytes(15 * 1024**3) == "15 GB"   # GB rounded to int for dashboard
    assert humanize_bytes(int(15.7 * 1024**3)) == "16 GB"


def test_is_platform_manifest_amd64() -> None:
    assert _is_platform_manifest({"platform": {"architecture": "amd64", "os": "linux"}})
    assert not _is_platform_manifest({"platform": {"architecture": "arm64", "os": "linux"}})
    # Attestation manifest with matching platform → still skipped.
    assert not _is_platform_manifest(
        {
            "platform": {"architecture": "amd64", "os": "linux"},
            "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
        }
    )


def test_sum_manifest_bytes() -> None:
    manifest = {
        "config": {"size": 1000},
        "layers": [{"size": 100}, {"size": 200}, {"size": 50}],
    }
    assert _sum_manifest_bytes(manifest) == 1350


def test_sum_manifest_bytes_missing_fields() -> None:
    assert _sum_manifest_bytes({}) == 0
    assert _sum_manifest_bytes({"layers": []}) == 0
    assert _sum_manifest_bytes({"config": {}, "layers": [{}]}) == 0


def _fake_run(index: dict, platform_manifest: dict | None = None):
    """Return a subprocess.run replacement that yields canned JSON in sequence."""
    payloads = [json.dumps(index)]
    if platform_manifest is not None:
        payloads.append(json.dumps(platform_manifest))
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        out = payloads[len(calls) - 1] if len(calls) <= len(payloads) else "{}"
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    return runner, calls


def test_fetch_size_multi_platform_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two-step lookup: index → platform manifest → sum config + layers."""
    index = {
        "manifests": [
            {
                "digest": "sha256:platformdigest",
                "platform": {"architecture": "amd64", "os": "linux"},
            },
            {
                "digest": "sha256:attestdigest",
                "platform": {"architecture": "amd64", "os": "linux"},
                "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
            },
        ]
    }
    platform_manifest = {
        "config": {"size": 10_000},
        "layers": [{"size": 1_000_000_000}, {"size": 500_000_000}],
    }
    runner, calls = _fake_run(index, platform_manifest)
    monkeypatch.setattr(subprocess, "run", runner)

    size = fetch_image_compressed_size("ghcr.io/kaitakuai/mlnode-b300-kimi-k26:test")
    assert size == 10_000 + 1_000_000_000 + 500_000_000
    # Verified the second call used the platform digest from the index.
    assert calls[1][-1] == "ghcr.io/kaitakuai/mlnode-b300-kimi-k26@sha256:platformdigest"


def test_fetch_size_direct_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the reference is a single-platform manifest, sum directly."""
    manifest = {
        "config": {"size": 500},
        "layers": [{"size": 2000}],
    }
    runner, _ = _fake_run(manifest)
    monkeypatch.setattr(subprocess, "run", runner)
    assert fetch_image_compressed_size("ghcr.io/example:tag") == 2500


def test_fetch_size_no_amd64_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index without a linux/amd64 manifest → 0."""
    index = {
        "manifests": [
            {"digest": "sha256:x", "platform": {"architecture": "arm64", "os": "linux"}}
        ]
    }
    runner, _ = _fake_run(index)
    monkeypatch.setattr(subprocess, "run", runner)
    assert fetch_image_compressed_size("ghcr.io/example:tag") == 0


def test_fetch_size_buildx_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="not found")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ImageSizeError):
        fetch_image_compressed_size("ghcr.io/example:tag")


def test_module_attr_exports() -> None:
    """Public API entrypoints are stable across refactors."""
    assert callable(image_size.fetch_image_compressed_size)
    assert callable(image_size.humanize_bytes)
