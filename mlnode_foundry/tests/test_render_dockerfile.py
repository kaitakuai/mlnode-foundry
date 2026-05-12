"""Tests for stage3 Dockerfile template renderer."""

from __future__ import annotations

from pathlib import Path

from mlnode_foundry.render_bake import load_profile
from mlnode_foundry.render_dockerfile import (
    _render_env_block,
    _render_tuning_label,
    render_dockerfile,
)


def test_env_block_sorted_and_chained() -> None:
    """Env block emits sorted keys joined with backslash-newline continuation."""
    block = _render_env_block({"B_VAR": "2", "A_VAR": "1"})
    assert block.startswith("ENV ")
    assert "A_VAR=1" in block
    assert "B_VAR=2" in block
    # Sorted: A before B
    assert block.index("A_VAR") < block.index("B_VAR")
    assert " \\\n    " in block


def test_env_block_empty() -> None:
    """Empty env → harmless comment, no ENV directive."""
    block = _render_env_block({})
    assert "ENV " not in block
    assert "empty" in block


def test_tuning_label_empty() -> None:
    """No tuning notes → count=0 label keeps LABEL chain valid."""
    label = _render_tuning_label(None)
    assert "tuning_notes_count" in label
    assert '"0"' in label


def test_tuning_label_serializes_compact_json() -> None:
    notes = [
        {
            "knob": "VLLM_FOO=1",
            "source": "exp-id",
            "reason": "bench-driven",
            "added_at": "2026-05-12",
        }
    ]
    label = _render_tuning_label(notes)
    assert "gonka.kaitaku.tuning_notes" in label
    # JSON is escaped for Dockerfile LABEL value
    assert '\\"knob\\"' in label
    assert "VLLM_FOO=1" in label


def test_render_dockerfile_b300_kimi(tmp_path: Path) -> None:
    """Real profile renders without leftover {{...}} placeholders."""
    out = tmp_path / "Dockerfile.rendered"
    profile = load_profile("b300-kimi-int4")
    path = render_dockerfile(profile, out)
    text = path.read_text()
    assert "{{ENV_BLOCK}}" not in text
    assert "{{TUNING_LABEL}}" not in text
    # b300-kimi-int4 sets VLLM_USE_FLASHINFER_MOE_INT4=1 via KIMI_INT4 base
    assert "VLLM_USE_FLASHINFER_MOE_INT4=1" in text
    # b300-kimi-int4 has tuning_notes → label present, not the count-zero fallback
    assert "tuning_notes_count" not in text
    assert "gonka.kaitaku.tuning_notes=" in text


def test_render_dockerfile_profile_without_tuning_notes(tmp_path: Path) -> None:
    """Profile without tuning_notes falls back to count=0 label."""
    out = tmp_path / "Dockerfile.rendered"
    profile = load_profile("h100-qwen")
    path = render_dockerfile(profile, out)
    text = path.read_text()
    assert "tuning_notes_count" in text
