"""Tests for content-hash stability and sensitivity."""

from __future__ import annotations

from mlnode_foundry.build_hash import compute_profile_hash


def test_hash_is_deterministic() -> None:
    h1 = compute_profile_hash("b200-kimi-k2-6-int4")
    h2 = compute_profile_hash("b200-kimi-k2-6-int4")
    assert h1 == h2


def test_hash_is_64_hex_chars() -> None:
    h = compute_profile_hash("b200-kimi-k2-6-int4")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_differs_between_profiles() -> None:
    h1 = compute_profile_hash("b200-kimi-k2-6-int4")
    h2 = compute_profile_hash("h100-minimax-m2-7")
    assert h1 != h2
