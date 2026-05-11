"""Cold-start tolerance — env-driven patches for MLNode watcher + vLLM runner.

Lets mlnode survive multi-instance vLLM cold-starts on high-end GPU hosts
(observed 2026-04-29 on a fresh 8×B300 host with 4× TP=2 instances and
`--compilation-config '{"mode":1}'`, where the combined DeepGEMM warmup
+ FlashInfer TRTLLM autotune + per-shape mode=1 compile took ~22 min,
exceeding the runner's WAIT_FOR_SERVER_TIMEOUT=1200 (20 min)).

Pattern: turns hardcoded constants into env-readable defaults and replaces
the watcher's main loop with a session-aware "first-healthy grace"
implementation. Source-side diff stays minimal at the constant level —
the actual values live in `compose.yml` env or the image's ENV block.
Operators can tune per-deployment without an image rebuild; reverting
to vanilla behavior is just unsetting the env (or setting GRACE=0).

Env vars introduced (all optional, defaults preserve upstream behavior):

    VLLM_RUNNER_TIMEOUT             default 1200   (sec)
        How long runner.start() waits for vLLM to open its port before
        raising. Bump for slow cold starts (8×B300 + mode=1 needs ≥3000).

    WATCHER_MAX_UNHEALTHY_COUNT     default 3
        Number of consecutive unhealthy reports before the watcher
        calls os._exit(1). Operators rarely need to change this.

    WATCHER_GRACE_FIRST_HEALTHY     default "0" (off)
        When "1": until a manager has reported is_healthy()=True at
        least once IN AN ACTIVE SESSION (state != STOPPED), the watcher
        does NOT count unhealthy reports against the kill threshold —
        just logs them. Once any first-active-healthy is seen, normal
        MAX_UNHEALTHY_COUNT-based behavior kicks back in for that
        manager — so post-startup crashes still trigger the
        operator-expected fast restart. The grace window is RE-ARMED
        on each STOPPED→active session transition (a stop+start cycle
        gets a fresh grace window).

The patches:

  runner.py — one-line:
    -  WAIT_FOR_SERVER_TIMEOUT = 1200
    +  WAIT_FOR_SERVER_TIMEOUT = int(os.environ.get("VLLM_RUNNER_TIMEOUT", "1200"))

  watcher.py — two replacements:
    1. Constant declaration becomes env-readable (MAX_UNHEALTHY_COUNT) +
       new GRACE_FIRST_HEALTHY env flag is added.
    2. `watch_managers` body is replaced with a session-aware first-healthy
       implementation. The body uses `manager.get_state().name != "STOPPED"`
       (a public IManager API) to distinguish the trivial STOPPED-state
       healthy shortcut (`is_healthy()` returns True when `not _is_active`)
       from "really running and operational". `ever_healthy` is only
       set True when the manager is healthy AND in an active session,
       and is reset whenever the manager returns to STOPPED (so a
       subsequent `up/async` gets a fresh grace window).

Idempotent: re-running on already-patched files is a no-op. Fails loud
if any anchor missing — that is the signal upstream refactored the
layout and the patch needs re-verification before regenerating.

Usage inside a Dockerfile RUN step (see _shared/cold-start-tolerance.dockerfile):

    COPY tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py /tmp/cst.py
    RUN python3 /tmp/cst.py && rm /tmp/cst.py
    ENV WATCHER_GRACE_FIRST_HEALTHY=1
    ENV VLLM_RUNNER_TIMEOUT=3600
"""

from __future__ import annotations

import re
import sys

RUNNER_PY = "/app/packages/api/src/api/inference/vllm/runner.py"
WATCHER_PY = "/app/packages/api/src/api/watcher.py"

# --------- runner.py ----------

RUNNER_OLD = "WAIT_FOR_SERVER_TIMEOUT = 1200"
RUNNER_NEW = 'WAIT_FOR_SERVER_TIMEOUT = int(os.environ.get("VLLM_RUNNER_TIMEOUT", "1200"))'

# --------- watcher.py ---------

# Replacement 1: constant declaration — env-readable + new GRACE flag.
WATCHER_CONST_OLD = "MAX_UNHEALTHY_COUNT = 3"
WATCHER_CONST_NEW = (
    'MAX_UNHEALTHY_COUNT = int(os.environ.get("WATCHER_MAX_UNHEALTHY_COUNT", "3"))\n'
    "# When \"1\", grants a 'first-healthy grace' window per session: until a\n"
    "# manager has reported is_healthy()=True at least once IN AN ACTIVE\n"
    "# session (state != STOPPED), unhealthy reports do NOT count toward\n"
    "# MAX_UNHEALTHY_COUNT — only logged. Once any first-active-healthy\n"
    "# is seen, normal kill-threshold behavior kicks back in. Re-armed on\n"
    "# each STOPPED→active session transition. Lets long cold-starts\n"
    "# (e.g. 8×B300 4× TP=2 with mode=1, ~22 min) complete without\n"
    "# mlnode terminating itself, while preserving fast crash-restart\n"
    "# semantics for post-startup failures.\n"
    'GRACE_FIRST_HEALTHY = os.environ.get("WATCHER_GRACE_FIRST_HEALTHY", "0") == "1"'
)

# Replacement 2: full watch_managers function — session-aware first-healthy grace.
# Match the function from `async def watch_managers(` to its last meaningful line,
# tolerating arbitrary trailing whitespace on blank lines (the upstream source uses
# trailing spaces on indented blank lines, which exact-string match would miss).
WATCHER_FN_PATTERN = re.compile(
    r"^async def watch_managers\(.*?unhealthy_counts\[manager\] = 0\s*$",
    re.MULTILINE | re.DOTALL,
)

WATCHER_FN_NEW = '''async def watch_managers(
    app: FastAPI,
    managers: List[IManager],
    interval: int = 2
):
    unhealthy_counts = {manager: 0 for manager in managers}
    # Session-aware first-healthy grace tracking. `ever_healthy` flips
    # to True only when a manager is healthy AND in an active session
    # (state != STOPPED), and resets when the manager returns to
    # STOPPED. This way the trivial STOPPED-shortcut healthy reported
    # by IManager.is_healthy() (returns True when `not _is_active`)
    # does not consume the grace window.
    ever_healthy = {manager: False for manager in managers}
    prev_in_session = {manager: False for manager in managers}

    while True:
        await asyncio.sleep(interval)
        for manager in managers:
            in_session = manager.get_state().name != "STOPPED"

            # Session-end transition (active → STOPPED): re-arm the grace
            # window so the next up/async gets its own cold-start grace.
            if prev_in_session[manager] and not in_session and ever_healthy[manager]:
                logger.info(f"Manager {manager.__class__.__name__} returned to STOPPED — resetting cold-start grace for next session")
                ever_healthy[manager] = False
            prev_in_session[manager] = in_session

            if not manager.is_healthy():
                if GRACE_FIRST_HEALTHY and not ever_healthy[manager]:
                    logger.info(f"Manager {manager.__class__.__name__} not yet healthy (cold-start grace; kill threshold inactive until first healthy in active session)")
                    continue
                unhealthy_counts[manager] += 1
                logger.error(f"Manager {manager.__class__.__name__} is unhealthy (count: {unhealthy_counts[manager]}/{MAX_UNHEALTHY_COUNT})")

                if unhealthy_counts[manager] >= MAX_UNHEALTHY_COUNT:
                    logger.critical(f"Manager {manager.__class__.__name__} has been unhealthy {MAX_UNHEALTHY_COUNT} times in a row. Shutting down the application.")
                    # Use the proper stop() interface for all managers
                    manager.stop()
                    os._exit(1)
            else:
                # Only count first-healthy when manager is actually in an
                # active session — skip the trivial STOPPED-shortcut.
                if in_session and not ever_healthy[manager]:
                    ever_healthy[manager] = True
                    logger.info(f"Manager {manager.__class__.__name__} reached healthy state in active session — MAX_UNHEALTHY_COUNT={MAX_UNHEALTHY_COUNT} kill threshold now active")
                if unhealthy_counts[manager] > 0:
                    logger.info(f"Manager {manager.__class__.__name__} is healthy again, resetting unhealthy count")
                    unhealthy_counts[manager] = 0'''


def _ensure_import_os(src: str, file_label: str) -> str:
    """Make sure `import os` is present at the top of the file."""
    if "import os" in src:
        return src
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
        elif insert_at > 0 and not line.strip():
            break
    lines.insert(insert_at, "import os\n")
    print(f"  {file_label}: added missing 'import os'")
    return "".join(lines)


def _replace_once(src: str, old: str, new: str, label: str) -> str:
    if new in src:
        print(f"  {label}: already patched — skipping")
        return src
    if old not in src:
        sys.stderr.write(
            f"ERROR: cold-start-tolerance: anchor {label!r} not found. "
            "Upstream may have refactored — re-verify the patch.\n"
        )
        raise SystemExit(1)
    print(f"  {label}: applying")
    return src.replace(old, new, 1)


def _replace_regex_once(
    src: str, pattern: re.Pattern[str], new: str, label: str
) -> str:
    """Regex-based replacement, idempotent on already-patched files."""
    if new in src:
        print(f"  {label}: already patched — skipping")
        return src
    if not pattern.search(src):
        sys.stderr.write(
            f"ERROR: cold-start-tolerance: anchor {label!r} not found. "
            "Upstream may have refactored — re-verify the patch.\n"
        )
        raise SystemExit(1)
    print(f"  {label}: applying")
    return pattern.sub(new, src, count=1)


def _patch_runner() -> None:
    print(f"== patching {RUNNER_PY}")
    with open(RUNNER_PY) as f:
        src = f.read()
    src = _ensure_import_os(src, "runner.py")
    src = _replace_once(src, RUNNER_OLD, RUNNER_NEW, "runner.WAIT_FOR_SERVER_TIMEOUT")
    with open(RUNNER_PY, "w") as f:
        f.write(src)


def _patch_watcher() -> None:
    print(f"== patching {WATCHER_PY}")
    with open(WATCHER_PY) as f:
        src = f.read()
    src = _ensure_import_os(src, "watcher.py")
    src = _replace_once(
        src, WATCHER_CONST_OLD, WATCHER_CONST_NEW,
        "watcher.MAX_UNHEALTHY_COUNT + GRACE flag",
    )
    src = _replace_regex_once(
        src, WATCHER_FN_PATTERN, WATCHER_FN_NEW,
        "watcher.watch_managers (session-aware grace)",
    )
    with open(WATCHER_PY, "w") as f:
        f.write(src)


def main() -> int:
    _patch_runner()
    _patch_watcher()
    print("== cold-start-tolerance: all patches applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
