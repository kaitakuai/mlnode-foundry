"""Self-contained profile snapshot — bases inlined, resolved JSON.

Useful for:
  - audit: see exactly what `cue export` resolved a profile to
  - sharing: send one JSON instead of the whole repo
  - debugging tag rendering: what axes are actually set?
"""

from __future__ import annotations

from .render_bake import load_profile


def expand_profile(name: str) -> dict:
    """Return fully-resolved profile dict (bases unified, defaults applied)."""
    return load_profile(name)
