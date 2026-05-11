# `patches/` — upstream source patches for Stage 2

Patches in this directory are applied by `stage2/Dockerfile.patch-and-build` to the upstream `gonka-ai/gonka` source tree (at the commit pinned in `tools/stage2.lock.cue::upstream.commit`) before invoking upstream's `mlnode/packages/api/Dockerfile`.

## Format

Standard `git format-patch -1 <sha>` output, applied via `git apply --3way`.

## Current patches

| File | Status | What it does |
|------|--------|--------------|
| `0001-content-type-middleware.patch.todo` | **Placeholder** | Adds `ContentTypeInjector` ASGI middleware injecting `Content-Type: application/json` for header-less POSTs from Go-http-client. Needed for vLLM 0.20 (Pydantic v2 strict mode). |

## Status: placeholder

The Content-Type middleware patch from legacy commit `827d5ffe401f0482c46090fbf79ec693b385a5b0` is currently **a metadata placeholder**, not a usable patch file. The legacy submodule was a sparse checkout without parent history, so `git format-patch` against it produces an unusable 4.6 M-line "add everything" diff.

**Resolution in Phase 3 (PR #2):**

1. Do a full clone of `gonka-ai/gonka` with history
2. `git format-patch -1 827d5ffe` produces a clean patch
3. Replace `0001-content-type-middleware.patch.todo` with the real `.patch` file
4. Update `tools/stage2.lock.cue::patches` list

Until Phase 3 lands real Stage 2, Stage 3 profiles use upstream `product-science/mlnode:3.0.13-alpha5` directly as `BASE_IMAGE` (bypassing Stage 2 entirely), so this patch is not yet load-bearing.

## Files touched by the patch (for reference)

- `mlnode/packages/api/src/api/app.py` — middleware class definition + registration
- `mlnode/packages/api/src/api/proxy.py` — middleware usage
- And related test/Dockerfile changes per commit message
