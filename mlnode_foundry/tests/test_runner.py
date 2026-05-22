"""Tests for runner inventory and selection."""

from __future__ import annotations

import pytest

from mlnode_foundry.runner import list_runners, select_runner


def test_list_runners_returns_registered() -> None:
    runners = list_runners()
    assert "vast-b300-1x" in runners
    assert "cherry-b300-8x" in runners
    assert runners["vast-b300-1x"]["kind"] == "vast.ai"
    assert runners["cherry-b300-8x"]["kind"] == "ssh"


def test_select_runner_no_affinity_in_phase2() -> None:
    """Phase 2 profiles don't declare validation_targets; selection should fail."""
    with pytest.raises(ValueError, match="declares no runner affinity"):
        select_runner("b200-kimi-k2-6-int4", "smoke")
