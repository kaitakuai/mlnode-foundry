"""Tests for the registry-view CLI's preserve-hand-edited-metrics helper.

Plan B benchmark workflow: operator hand-edits `registry-view/<file>.json`
to set `nonces` (and optionally `weight`) after testing the image on real
hardware. Subsequent Stage 4 rebuilds MUST NOT clobber that number back
to `null` — there's no profile change to trigger a re-edit, and asking
the operator to re-set the value after every rebuild defeats the whole
"build once, validate once, publish forever" workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlnode_foundry.cli import _preserve_hand_edited_metrics


def _write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content))


def test_no_existing_file_returns_view_unchanged(tmp_path: Path) -> None:
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, tmp_path / "does-not-exist.json")
    assert out["nonces"] is None
    assert out["weight"] is None


def test_preserves_hand_edited_nonces(tmp_path: Path) -> None:
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": 4736, "weight": None})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] == 4736
    assert out["weight"] is None


def test_preserves_both_when_both_set(tmp_path: Path) -> None:
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": 2624, "weight": 18560})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] == 2624
    assert out["weight"] == 18560


def test_does_not_overwrite_explicit_caller_value(tmp_path: Path) -> None:
    """If CLI ever passes nonces explicitly (future use), it beats preservation."""
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": 4736, "weight": None})
    # Simulate: caller set nonces=1000 explicitly (current CLI doesn't, but the
    # invariant is "explicit > preserved").
    view = {"nonces": 1000, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] == 1000


def test_skips_negative_values(tmp_path: Path) -> None:
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": -5, "weight": None})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] is None


def test_skips_non_numeric_values(tmp_path: Path) -> None:
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": "lots", "weight": [4736]})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] is None
    assert out["weight"] is None


def test_skips_boolean_disguised_as_int(tmp_path: Path) -> None:
    """Python: True == 1, False == 0. Explicit reject to avoid pseudo-nonces."""
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": True, "weight": False})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] is None
    assert out["weight"] is None


def test_corrupt_json_silently_falls_back(tmp_path: Path) -> None:
    """Corrupt existing file should not fail the rebuild — the operator can re-edit."""
    existing = tmp_path / "view.json"
    existing.write_text("{not valid json")
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] is None
    assert out["weight"] is None


def test_existing_file_with_zero_value_is_preserved(tmp_path: Path) -> None:
    """Zero is a legitimate measurement ("we tried; it didn't earn"). Preserve it."""
    existing = tmp_path / "view.json"
    _write(existing, {"nonces": 0, "weight": 0})
    view = {"nonces": None, "weight": None}
    out = _preserve_hand_edited_metrics(view, existing)
    assert out["nonces"] == 0
    assert out["weight"] == 0
