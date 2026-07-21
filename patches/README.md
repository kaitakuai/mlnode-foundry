# `patches/` — upstream source patches for Stage 3

Patches in this directory are applied by `stage3/Dockerfile.patch-and-build` to the upstream `gonka-ai/gonka` source tree (at the commit pinned in `tools/stage3.lock.cue::upstream.commit`) before invoking upstream's `mlnode/packages/api/Dockerfile`.

## Format

Standard `git format-patch -1 <sha>` output, applied via `git apply --3way`.

## Current patches

| File | Files touched | What it does |
|------|---------------|--------------|
| `0001-content-type-middleware.patch` | `mlnode/packages/api/src/api/app.py`, `proxy.py` | Adds `ContentTypeInjector` ASGI middleware injecting `Content-Type: application/json` for header-less POSTs from Go-http-client. Needed for vLLM 0.20 (Pydantic v2 strict mode). |
| `0002-api-watcher-grace.patch` | `mlnode/packages/api` (watcher) | Session-aware watcher grace window replacing the 3-strike auto-shutdown (slow cold starts of big-TP MoE models). |
| `0003-mlnode-heartbeat-liveness.patch` | `runner.py`, `proxy.py` (+unit tests) | Scheduler-heartbeat liveness (`vllm:iteration_tokens_total_count` instead of `/health`) + damped proxy probe (3-strike, 10s timeout). Extracted from upstream PR gonka-ai/gonka#1421 (open; drop this patch and bump the pin once merged). `MLNODE_HANG_GRACE_SEC` default 120, `0` disables. |

## Patch source / provenance

Patches are extracted from upstream commits and applied on top of a clean upstream checkout. They are NOT a fork — they're independently-versioned changes layered on `tools/stage3.lock.cue::upstream.commit`.

`0001-content-type-middleware.patch` was extracted from commit `827d5ffe401f0482c46090fbf79ec693b385a5b0` in `gonka-ai/gonka`. Verified end-to-end on production RTX PRO 6000 SE running vLLM 0.20.0 PoC v2.

## Adding new patches

```bash
# Extract from any commit on any branch/fork
git -C /path/to/gonka-ai-clone format-patch -1 <sha> --stdout > patches/000N-<short-name>.patch

# Test apply
git -C /path/to/gonka-ai-clone apply --3way --check patches/000N-<short-name>.patch

# Add to stage3.lock.cue patches list (in order applied)
```

Patches must be **commutative** — order shouldn't matter beyond standard `git apply --3way` 3-way-merge resolution.
