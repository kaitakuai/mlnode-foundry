"""Render stage3/Dockerfile.tmpl → concrete stage3/Dockerfile.rendered per profile.

Substitutes:
  {{ENV_BLOCK}}    — `ENV K1=V1 \\\n    K2=V2 \\\n    ...` from profile.env
  {{TUNING_LABEL}} — `gonka.kaitaku.tuning_notes="<compact-json>"` from profile.tuning_notes
                     (omitted if profile has no tuning_notes; renders as empty placeholder)

This removes the fixed-list ENV ARG/ENV block that lived in the old static
Dockerfile, so any profile can introduce arbitrary ENV vars without editing
the template.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_PATH = REPO_ROOT / "stage3" / "Dockerfile.tmpl"
DEFAULT_OUTPUT = REPO_ROOT / "stage3" / "Dockerfile.rendered"


def _render_env_block(env: dict[str, str]) -> str:
    if not env:
        return "# (profile.env empty)"
    pairs = sorted(env.items())
    lines = [f"{k}={v}" for k, v in pairs]
    body = " \\\n    ".join(lines)
    return f"ENV {body}"


def _render_tuning_label(tuning_notes: list[dict] | None) -> str:
    # Render to a single LABEL key=value line that fits into the chained LABEL block.
    # Empty profile → empty (just trailing backslash continuation) — drop the line entirely
    # so the LABEL chain stays valid.
    if not tuning_notes:
        # Keep one harmless label so the line continuation in the template stays valid.
        return 'gonka.kaitaku.tuning_notes_count="0"'
    payload = json.dumps(tuning_notes, separators=(",", ":"), sort_keys=True)
    # Escape backslashes and quotes for safe embedding in a Dockerfile LABEL value.
    payload = payload.replace("\\", "\\\\").replace('"', '\\"')
    return f'gonka.kaitaku.tuning_notes="{payload}"'


def render_dockerfile(profile: dict, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Render the Stage 3 Dockerfile template for `profile` to `output_path`."""
    template = TEMPLATE_PATH.read_text()
    env = profile.get("env", {})
    tuning_notes = profile.get("tuning_notes")
    rendered = (
        template
        .replace("{{ENV_BLOCK}}", _render_env_block(env))
        .replace("{{TUNING_LABEL}}", _render_tuning_label(tuning_notes))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return output_path
