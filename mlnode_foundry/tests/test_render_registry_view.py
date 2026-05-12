"""Tests for the registry-view aggregator (dashboard-compatible JSON)."""

from __future__ import annotations

import pytest

from mlnode_foundry.render_registry_view import (
    _flag_descriptions,
    _flag_warnings,
    _flags,
    _report_url,
    render_registry_view,
)


def test_flags_flattens_env_and_runtime_defaults() -> None:
    profile = {
        "env": {"VLLM_USE_V1": "1"},
        "runtime_defaults": {"tensor_parallel_size": 4},
    }
    out = _flags(profile)
    assert "VLLM_USE_V1=1" in out
    assert "tensor_parallel_size=4" in out


def test_flag_warnings_only_severity_warning() -> None:
    notes = [
        {"knob": "VLLM_FOO=1", "reason": "perf gain", "severity": "info"},
        {"knob": "max_model_len=120000", "reason": "below native context", "severity": "warning"},
    ]
    warnings = _flag_warnings(notes)
    assert "VLLM_FOO=1" not in warnings
    assert "max_model_len=120000" in warnings
    assert warnings["max_model_len=120000"] == "below native context"


def test_flag_descriptions_includes_all_notes_regardless_of_severity() -> None:
    notes = [
        {"knob": "A=1", "reason": "r1", "severity": "info"},
        {"knob": "B=2", "reason": "r2", "severity": "warning"},
    ]
    descriptions = _flag_descriptions(notes)
    assert descriptions == {"A=1": "r1", "B=2": "r2"}


def test_report_url_prefers_first_experiments_link() -> None:
    notes = [
        {"knob": "X=1", "source": "internal-decision-log#42", "reason": "r"},
        {
            "knob": "Y=2",
            "source": "https://github.com/kaitakuai/experiments/2026-05/foo",
            "reason": "r",
        },
    ]
    assert _report_url(notes) == "https://github.com/kaitakuai/experiments/2026-05/foo"


def test_report_url_none_when_no_url_notes() -> None:
    assert _report_url([]) is None
    assert _report_url([{"knob": "K=v", "source": "n/a", "reason": "r"}]) is None


def test_render_registry_view_b300_kimi_int4() -> None:
    """End-to-end render: profile + model-registry + stage2.lock → dashboard JSON."""
    view = render_registry_view(
        "b300-kimi-int4",
        digest="sha256:" + "a" * 64,
        cosign_identity="https://github.com/kaitakuai/mlnode-foundry/.github/workflows/build-stage3.yml@refs/heads/main",
        size="42 GB",
    )
    assert view["line"] == "mlnode"
    assert view["gpu"] == "b300"
    assert view["model_family"] == "kimi"
    assert view["model_revision"] == "k26"
    assert view["quant"] == "int4"
    assert view["model"] == "moonshotai/Kimi-K2.6"
    assert view["model_short"] == "Moonshot Kimi-K2.6"
    assert view["cuda"] == "13.0"
    assert view["size"] == "42 GB"
    assert "tensor_parallel_size=4" in view["flags"]
    assert "VLLM_USE_FLASHINFER_MOE_INT4=1" in view["flags"]
    # b300-kimi-int4 has two tuning_notes (both info) → all in descriptions, none in warnings
    assert view["flag_warnings"] == {}
    assert "VLLM_USE_FLASHINFER_MOE_INT4=1" in view["flag_descriptions"]
    assert view["report_url"] is not None
    assert view["report_url"].startswith("https://github.com/kaitakuai/experiments")
    assert view["digest"].startswith("sha256:")
    assert view["nonces"] is None  # filled later by benchmark agent
    assert view["weight"] is None


def test_render_registry_view_unknown_profile_raises() -> None:
    with pytest.raises(FileNotFoundError):
        render_registry_view("nonexistent-profile")
