# ADR-0009 — Validation tiers and benchmark integration

**Status:** Accepted
**Date:** 2026-05-10

## Context

Image validation has wildly different costs:

- Schema check: free (cue vet)
- Build smoke: free (CI runner CPU only)
- Real-GPU smoke: $0.50-3 per run (vast.ai ephemeral B300)
- Full benchmark: $5-50 per run (1000+ nonces, multi-GPU)

Running real-GPU validation on every PR would burn budget in days. Running zero validation lets regressions reach prod.

## Decision

**Four-tier validation**, opt-in at higher tiers:

| Tier | What | Where | Cost | Trigger |
|------|------|-------|------|---------|
| 0 — Static | cue vet, ruff, mypy, pytest | GHA ubuntu-latest | $0 | every PR/push |
| 1 — Build-only smoke | image builds, entrypoint exists, CPU-only `import vllm` | GHA ubuntu-latest | $0 | every PR with profile change |
| 2 — Real-GPU smoke | image runs, `/api/v1/inference/up`, 1-10 nonces | vast.ai or ssh-host | $0.50-3 | PR label `/validate-gpu`, first publication |
| 3 — Full benchmark | 1000+ nonces, logprobs, cross-validation | vast.ai or ssh via `poc-benchmark` agent | $5-50 | PR label `/benchmark`, cron, major bumps |

Lifecycle states: `draft` → (Tier 2) `validated` → (Tier 3) `benchmarked` → `deprecated`.

Image is published in any state. Dashboard shows lifecycle badge. Node operators filter by `status >= validated` for production use.

## Bidirectional flow with experiments

After Tier 3 the `poc-benchmark` agent (separate process, see `.claude/agents/poc-benchmark.md`) writes results to `experiments/<YYYY-MM>/<package-tag>/{README.md, nonces.json, metrics.json}` and opens an auto-PR in `mlnode-foundry` updating `state/<package-tag>.json` (status, validation.benchmark, metrics).

The auto-PR does NOT trigger a Stage 4 rebuild — `state/` files are excluded from `profile_hash` input.

## Consequences

- **CI cycle stays fast** — Tier 0+1 only on every push (~5-10 min)
- **Budget controllable** — Tier 2/3 require explicit label, can be rate-limited
- **Profile lifecycle visible** — operators see `draft` vs `benchmarked` and choose accordingly
- **No coupling of build to benchmark** — eventual-consistency between image publish and metrics

## Alternatives considered

- **Always run Tier 2 on every PR**: rejected — burns budget on draft profiles
- **Manual validation outside CI**: rejected — easy to forget; no audit trail
- **Single combined Tier**: rejected — sweeping all checks into one $5-50 invocation makes "is this image valid?" expensive
