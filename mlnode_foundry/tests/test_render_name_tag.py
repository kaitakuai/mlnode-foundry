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


def test_b200_kimi_tag(naming: dict) -> None:
    profile = load_profile("b200-kimi-k2-6")
    pkg, tag = render_package_and_tag(profile, naming)
    assert pkg == "ghcr.io/kaitakuai/mlnode-b200-kimi-k2-6"
    # rev=2 in profile.identity.version → -k2 suffix.
    # quant axis intentionally omitted from this profile, so the tag has no
    # `-q.int4` segment (Kimi-K2.6 ships only as INT4 today — see profile).
    assert tag == "0.2.13-vllm0.20.0-k2"
    assert "q." not in tag


def test_h100_minimax_tag_omits_default_axes(naming: dict) -> None:
    """h100-minimax has no quant set; tag should omit q.* segment."""
    profile = load_profile("h100-minimax-m2-7")
    pkg, tag = render_package_and_tag(profile, naming)
    assert pkg == "ghcr.io/kaitakuai/mlnode-h100-minimax-m2-7"
    assert tag == "0.2.13-vllm0.20.0-k1"
    assert "q." not in tag
    assert "f." not in tag


def test_render_package_uses_naming_package_axes(naming: dict) -> None:
    """Package name is constructed from naming.package.axes in order."""
    profile = load_profile("b200-kimi-k2-6")
    pkg = render_package(profile, naming)
    # naming.package.axes is ["gpu", "model", "model_revision"]
    assert pkg.endswith("-b200-kimi-k2-6")
    assert pkg.startswith("ghcr.io/kaitakuai/mlnode")


def test_render_tag_includes_only_set_axes(naming: dict) -> None:
    """Tag only includes axes that are explicitly set in profile.identity.axes."""
    profile = load_profile("b200-kimi-k2-6")
    tag = render_tag(profile, naming)
    # No identity axes set beyond what the package name carries (gpu/model/
    # model_revision are name-axes; quant intentionally omitted).
    assert "q." not in tag  # quant not set → not in tag
    assert "f." not in tag  # framework not set → not in tag
    assert "m." not in tag  # build_mode not set → not in tag
