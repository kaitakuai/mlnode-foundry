"""Runner inventory access — list and select runners from tools/runners.cue.

Affinity-based selection: profile declares `validation_targets.{smoke,benchmark}`
as a list of runner names; this module picks the first registered runner from
that list. Budget filtering (current month spend, instance availability) is
Phase 3 work — see ADR-0009.
"""

from __future__ import annotations

from .cue import cue_export
from .render_bake import REPO_ROOT, load_profile


def list_runners() -> dict[str, dict]:
    """Return all registered runners from tools/runners.cue."""
    data = cue_export(REPO_ROOT / "tools" / "runners.cue")
    return data.get("runners", {})


def select_runner(profile_name: str, tier: str) -> tuple[str, dict]:
    """Select a runner for a profile + tier via affinity matching.

    Args:
        profile_name: profile filename without .cue
        tier: "smoke" or "benchmark"

    Returns:
        (runner_name, runner_dict)

    Raises:
        ValueError if no matching runner is registered.
    """
    profile = load_profile(profile_name)
    targets = profile.get("validation_targets", {})
    affinity = targets.get(tier, [])
    if not affinity:
        raise ValueError(
            f"Profile '{profile_name}' declares no runner affinity for tier '{tier}'. "
            f"Add `validation_targets.{tier}: [\"<runner-name>\", ...]` to the profile."
        )
    runners = list_runners()
    for name in affinity:
        if name in runners:
            return name, runners[name]
    raise ValueError(
        f"None of profile '{profile_name}' tier '{tier}' affinity runners {affinity} "
        f"are registered in tools/runners.cue (registered: {sorted(runners)})"
    )
