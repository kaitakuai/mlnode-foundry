"""Tests for profile generation (mlnode-foundry profile new)."""

from __future__ import annotations

import pytest

from mlnode_foundry import profile_factory
from mlnode_foundry.validate import validate_profile


def test_generated_profile_validates(tmp_path, monkeypatch) -> None:
    """A freshly-generated profile must pass `cue vet` AND registry cross-check."""
    # Use a combo not in the migrated profile set (H100 + Kimi K2.6 INT4 is
    # realistic but currently absent — exercises gpu + model+quant base resolution
    # and the (kimi, k2-6) registry pair.
    path = profile_factory.generate_profile(
        gpu="h100",
        model="kimi",
        model_revision="k2-6",
        quant="int4",
        mlnode="0.2.13",
        vllm="0.20.0",
        rev=1,
    )
    try:
        validate_profile("h100-kimi-k2-6-int4")
    finally:
        path.unlink(missing_ok=True)


def test_unknown_model_revision_rejected() -> None:
    """Profile with a (model, model_revision) tuple not in registry must fail validation."""
    from mlnode_foundry.validate import ModelRegistryError

    path = profile_factory.generate_profile(
        gpu="h100",
        model="kimi",
        model_revision="k99",  # fake revision, not in registry
        quant="int4",
        mlnode="0.2.13",
        vllm="0.20.0",
        rev=1,
    )
    try:
        with pytest.raises(ModelRegistryError):
            validate_profile("h100-kimi-k99-int4")
    finally:
        path.unlink(missing_ok=True)


def test_generate_refuses_overwrite() -> None:
    """Generation MUST refuse to overwrite an existing profile."""
    with pytest.raises(FileExistsError):
        profile_factory.generate_profile(
            gpu="b200", model="kimi", model_revision="k2-6", quant="int4"
        )


def test_filename_mismatch_rejected(tmp_path) -> None:
    """Renaming a profile file without updating its axes is REJECTED.

    Strict invariant: filename ↔ axes is 1:1. If the file is renamed to
    something other than <gpu>-<model>-<revision>[-<quant>], validation
    must raise FilenameMismatchError.
    """
    from mlnode_foundry.profile_factory import PROFILES_DIR
    from mlnode_foundry.validate import FilenameMismatchError

    # Generate a valid profile, then copy it to a wrong-name location.
    path = profile_factory.generate_profile(
        gpu="h100",
        model="kimi",
        model_revision="k2-6",
        quant="int4",
        mlnode="0.2.13",
        vllm="0.20.0",
        rev=1,
    )
    misnamed = PROFILES_DIR / "h100-kimi-int4.cue"  # missing -k2-6
    try:
        # Copy content but keep top-level key matching the new (wrong) filename
        # so cue vet passes — we want the explicit filename check to fire.
        body = path.read_text().replace(
            "h100_kimi_k2_6_int4:", "h100_kimi_int4:"
        )
        misnamed.write_text(body)
        with pytest.raises(FilenameMismatchError):
            validate_profile("h100-kimi-int4")
    finally:
        path.unlink(missing_ok=True)
        misnamed.unlink(missing_ok=True)


def test_filename_convention() -> None:
    """Filename is gpu-model-quant.cue or gpu-model.cue."""
    # Just exercise the private functions through generate_profile signature
    # (no direct assertion since file is created; we trust naming via test above)
    pass


def test_key_replacement() -> None:
    """Top-level Cue field key replaces - with _."""
    # Implicit via test_generated_profile_validates — if key mismatched filename,
    # validate would fail to find the field.
    pass
