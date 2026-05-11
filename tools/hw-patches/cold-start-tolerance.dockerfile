# Cold-start tolerance for MLNode watcher + vLLM runner.
# Applies env-driven patches that let mlnode survive long multi-instance
# vLLM cold-starts on high-end GPU hosts (e.g. 8×B300 4× TP=2 with
# compilation_mode=1 ≈ 22 min, exceeding the runner's default 20-min
# WAIT_FOR_SERVER_TIMEOUT and tripping watcher's MAX_UNHEALTHY_COUNT=3
# kill threshold via the sticky `manager._exception` set on timeout).
#
# Two layered defenses:
#
# 1. `runner.WAIT_FOR_SERVER_TIMEOUT` becomes env-readable. Default
#    stays at upstream's 1200, but the b300 image overrides via ENV
#    `VLLM_RUNNER_TIMEOUT=3600` so runner.start() doesn't raise on
#    long cold starts (no `_exception` → watcher stays happy).
#
# 2. `watcher.watch_managers` gains a session-aware first-healthy
#    grace window when ENV `WATCHER_GRACE_FIRST_HEALTHY=1`. The grace
#    is re-armed on every STOPPED→active session transition. Until a
#    manager reaches healthy state IN AN ACTIVE SESSION (state !=
#    STOPPED), unhealthy reports do NOT count toward
#    MAX_UNHEALTHY_COUNT — only logged. Once any first-active-healthy
#    is seen, normal kill-threshold behavior kicks back in for that
#    manager — so post-startup crashes still trigger the
#    operator-expected fast restart.
#
# See: tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py

COPY tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py /tmp/cold-start-tolerance.py
RUN python3 /tmp/cold-start-tolerance.py && rm /tmp/cold-start-tolerance.py

ENV VLLM_RUNNER_TIMEOUT=3600
ENV WATCHER_GRACE_FIRST_HEALTHY=1
