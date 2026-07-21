"""Render stage4/Dockerfile.tmpl → concrete stage4/Dockerfile.rendered per profile.

Substitutes three placeholders:

  {{HW_PATCHES_BLOCK}} — concatenated Dockerfile fragments from
                         tools/hw-patches/<name>.dockerfile for each name in
                         profile.hw_patches, in declared order. Each fragment
                         is idempotent and self-contained (see
                         tools/hw-patches/README.md).

  {{ENV_BLOCK}}        — `ENV K1=V1 \\\n    K2=V2 \\\n    ...` from profile.env.

  {{TUNING_LABEL}}     — `gonka.kaitaku.tuning_notes="<compact-json>"` from
                         profile.tuning_notes (empty tuning_notes still
                         emits a harmless count=0 label so the chained
                         LABEL continuation stays valid).

This removes the fixed-list ENV ARG/ENV block that lived in the old static
Dockerfile, so any profile can introduce arbitrary ENV vars without editing
the template. Per-profile hw-patches selection (opt-in via the .cue
profile's `hw_patches: [...]` list) is honoured here — fragments declared
by a profile end up inlined into that profile's Stage 4 image; fragments
not declared are absent entirely (no /tmp/hw-patches dir, no leftover
scripts).
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import REPO_ROOT

TEMPLATE_PATH = REPO_ROOT / "stage4" / "Dockerfile.tmpl"
DEFAULT_OUTPUT = REPO_ROOT / "stage4" / "Dockerfile.rendered"
HW_PATCHES_DIR = REPO_ROOT / "tools" / "hw-patches"


def _render_env_block(env: dict[str, str]) -> str:
    if not env:
        return "# (profile.env empty)"
    pairs = sorted(env.items())
    lines = [f"{k}={v}" for k, v in pairs]
    body = " \\\n    ".join(lines)
    return f"ENV {body}"


def _render_tuning_label(tuning_notes: list[dict] | None) -> str:
    # Render to a single LABEL key=value line that fits into the chained LABEL block.
    # Empty tuning_notes still emits a harmless count=0 label so the chained
    # LABEL continuation stays valid.
    if not tuning_notes:
        return 'gonka.kaitaku.tuning_notes_count="0"'
    payload = json.dumps(tuning_notes, separators=(",", ":"), sort_keys=True)
    # Escape backslashes and quotes for safe embedding in a Dockerfile LABEL value.
    payload = payload.replace("\\", "\\\\").replace('"', '\\"')
    return f'gonka.kaitaku.tuning_notes="{payload}"'


def _render_hw_patches_block(hw_patches: list[str] | None) -> str:
    """Inline hw-patch Dockerfile fragments in declared order.

    Each name in `hw_patches` must correspond to an existing
    `tools/hw-patches/<name>.dockerfile`. Missing files raise
    FileNotFoundError at render time — this is a profile authoring bug
    (typo in the patch name) that should fail loud, not silently produce
    an image missing a patch.

    Fragments are joined with a separator comment that names which patch
    contributed each block, so the rendered Dockerfile remains readable.

    An empty/missing `hw_patches` renders to a single placeholder comment,
    keeping the build deterministic.
    """
    if not hw_patches:
        return "# (profile.hw_patches empty — no hardware-specific fragments to inline)"
    chunks: list[str] = []
    for name in hw_patches:
        fragment = HW_PATCHES_DIR / f"{name}.dockerfile"
        if not fragment.exists():
            raise FileNotFoundError(
                f"hw-patch fragment not found: {fragment} "
                f"(referenced from profile.hw_patches as {name!r}). "
                f"Either add the file under tools/hw-patches/ or correct the typo."
            )
        chunks.append(f"# --- hw-patch: {name} (from tools/hw-patches/{name}.dockerfile) ---")
        chunks.append(fragment.read_text().rstrip("\n"))
    return "\n".join(chunks)


def _replace_unique(template: str, placeholder: str, value: str) -> str:
    """str.replace but assert placeholder appears EXACTLY once.

    If a placeholder is mentioned in a comment alongside its real location,
    str.replace silently substitutes BOTH — content meant for line 50 also
    appears in a comment on line 10, and the Dockerfile breaks in ways that
    are hard to diagnose (e.g., `FROM` vanishes if a multi-line replacement
    lands inside a comment block). Fail loud here.
    """
    count = template.count(placeholder)
    if count != 1:
        raise ValueError(
            f"Stage 4 template expects exactly one occurrence of "
            f"{placeholder!r}, found {count}. Check stage4/Dockerfile.tmpl "
            f"for stale mentions in comments / doc headers."
        )
    return template.replace(placeholder, value)


def render_dockerfile(profile: dict, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Render the Stage 4 Dockerfile template for `profile` to `output_path`."""
    template = TEMPLATE_PATH.read_text()
    env = profile.get("env", {})
    tuning_notes = profile.get("tuning_notes")
    hw_patches = profile.get("hw_patches")
    rendered = template
    rendered = _replace_unique(
        rendered, "{{HW_PATCHES_BLOCK}}", _render_hw_patches_block(hw_patches)
    )
    rendered = _replace_unique(rendered, "{{ENV_BLOCK}}", _render_env_block(env))
    rendered = _replace_unique(
        rendered, "{{TUNING_LABEL}}", _render_tuning_label(tuning_notes)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return output_path
