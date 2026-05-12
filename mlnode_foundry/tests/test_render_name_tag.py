"""Unit tests for naming/tag rendering — runs `cue export` once per test."""

from __future__ import annotations

import pytest

from mlnode_foundry.render_bake import load_profile
from mlnode_foundry.render_name_tag import (
    load_naming,
    render_package,
    render_package_and_tag,
    render_tag,
)


@pytest.fixture(scope="module")
def naming() -> dict:
    return load_naming()


def test_naming_loads(naming: dict) -> None:
    assert "axes" in naming
    assert "package" in naming
    assert "tag" in naming
    assert "gpu" in naming["axes"]
    assert naming["axes"]["gpu"]["name_axis_only"] is True


def test_b300_kimi_int4_tag(naming: dict) -> None:
    profile = load_profile("b300-kimi-k2-6-int4")
    pkg, tag = render_package_and_tag(profile, naming)
    assert pkg == "ghcr.io/kaitakuai/mlnode-b300-kimi-k2-6"
    assert tag == "0.2.13-vllm0.20.0-q.int4-k1"


def test_h100_qwen_tag_omits_default_axes(naming: dict) -> None:
    """h100-qwen has no quant set; tag should omit q.* segment."""
    profile = load_profile("h100-qwen3-235b-a22b")
    pkg, tag = render_package_and_tag(profile, naming)
    assert pkg == "ghcr.io/kaitakuai/mlnode-h100-qwen3-235b-a22b"
    assert tag == "0.2.13-vllm0.20.0-k1"
    assert "q." not in tag
    assert "f." not in tag


def test_render_package_uses_naming_package_axes(naming: dict) -> None:
    """Package name is constructed from naming.package.axes in order."""
    profile = load_profile("b300-kimi-k2-6-int4")
    pkg = render_package(profile, naming)
    # naming.package.axes is ["gpu", "model", "model_revision"]
    assert pkg.endswith("-b300-kimi-k2-6")
    assert pkg.startswith("ghcr.io/kaitakuai/mlnode")


def test_render_tag_includes_only_set_axes(naming: dict) -> None:
    """Tag only includes axes that are explicitly set in profile.identity.axes."""
    profile = load_profile("b300-kimi-k2-6-int4")
    tag = render_tag(profile, naming)
    assert "q.int4" in tag  # quant set → in tag
    assert "f." not in tag  # framework not set → not in tag
    assert "m." not in tag  # build_mode not set → not in tag
