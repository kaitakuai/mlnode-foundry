"""Tests for stage3 Dockerfile template renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlnode_foundry.render_bake import load_profile
from mlnode_foundry.render_dockerfile import (
    _render_env_block,
    _render_hw_patches_block,
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
    profile = load_profile("b300-kimi-k2-6-int4")
    path = render_dockerfile(profile, out)
    text = path.read_text()
    assert "{{ENV_BLOCK}}" not in text
    assert "{{TUNING_LABEL}}" not in text
    # b300-kimi-k2-6-int4 sets VLLM_USE_FLASHINFER_MOE_INT4=1 via KIMI_INT4 base
    assert "VLLM_USE_FLASHINFER_MOE_INT4=1" in text
    # b300-kimi-k2-6-int4 has tuning_notes → label present, not the count-zero fallback
    assert "tuning_notes_count" not in text
    assert "gonka.kaitaku.tuning_notes=" in text


def test_render_dockerfile_profile_without_tuning_notes(tmp_path: Path) -> None:
    """Profile without tuning_notes falls back to count=0 label."""
    out = tmp_path / "Dockerfile.rendered"
    profile = load_profile("h100-qwen3-235b-a22b")
    path = render_dockerfile(profile, out)
    text = path.read_text()
    assert "tuning_notes_count" in text


def test_hw_patches_block_empty_list() -> None:
    """Empty hw_patches → comment-only placeholder, no fragment content."""
    block = _render_hw_patches_block([])
    assert "(profile.hw_patches empty" in block
    assert "RUN " not in block


def test_hw_patches_block_none() -> None:
    """Missing hw_patches → same comment-only placeholder."""
    block = _render_hw_patches_block(None)
    assert "(profile.hw_patches empty" in block


def test_hw_patches_block_single_patch_includes_fragment() -> None:
    """Declared patch → its `RUN ...` body appears inline."""
    block = _render_hw_patches_block(["poc-householder-compile"])
    # Separator header names the patch and the source file
    assert "# --- hw-patch: poc-householder-compile" in block
    assert "tools/hw-patches/poc-householder-compile.dockerfile" in block
    # Actual content from the fragment file is inlined
    # (the python heredoc that wraps apply_householder)
    assert "apply_householder" in block
    assert "torch.compile" in block
    assert "RUN python3" in block


def test_hw_patches_block_preserves_declared_order() -> None:
    """Patches inline in the order the profile declares them, not alphabetic."""
    block = _render_hw_patches_block(
        ["poc-householder-compile", "nvidia-headers-symlinks"]
    )
    i_householder = block.index("hw-patch: poc-householder-compile")
    i_headers = block.index("hw-patch: nvidia-headers-symlinks")
    assert i_householder < i_headers


def test_hw_patches_block_unknown_patch_raises() -> None:
    """Profile referencing a nonexistent patch fails loud — caught at render."""
    with pytest.raises(FileNotFoundError) as exc:
        _render_hw_patches_block(["definitely-not-a-real-patch"])
    assert "definitely-not-a-real-patch" in str(exc.value)
    assert "tools/hw-patches" in str(exc.value)


def test_render_dockerfile_inlines_hw_patches(tmp_path: Path) -> None:
    """End-to-end: h100-minimax profile has poc-householder-compile applied inline."""
    out = tmp_path / "Dockerfile.rendered"
    profile = load_profile("h100-minimax-m2-7")
    assert "poc-householder-compile" in profile["hw_patches"]
    text = render_dockerfile(profile, out).read_text()
    # No leftover placeholder
    assert "{{HW_PATCHES_BLOCK}}" not in text
    # Patch fragment is in the rendered file
    assert "hw-patch: poc-householder-compile" in text
    assert "@torch.compile" in text
    # /tmp/hw-patches COPY is gone — fragments are inlined, no need for the dir
    assert "/tmp/hw-patches" not in text


def test_render_dockerfile_empty_hw_patches_for_h100_qwen3(tmp_path: Path) -> None:
    """H100 base contributes 0 patches; h100-qwen3 opts in to poc-householder-compile only."""
    # h100-qwen3-235b-a22b uses list.Concat([H100.hw_patches, ["poc-householder-compile"]])
    # = [] + ["poc-householder-compile"] = single patch.
    profile = load_profile("h100-qwen3-235b-a22b")
    out = tmp_path / "Dockerfile.rendered"
    text = render_dockerfile(profile, out).read_text()
    assert "{{HW_PATCHES_BLOCK}}" not in text
    assert "hw-patch: poc-householder-compile" in text
