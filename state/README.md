# `state/` — observed state files

This directory holds **machine-written** JSON files recording what actually happened to each published image. Schema is in [`schema.cue`](./schema.cue).

## Authorship policy

| Who writes | What | When |
|------------|------|------|
| `build-stage4.yml` workflow | `state/mlnode-<gpu>-<model>-<tag>.json` initial entry + `status: draft` | After successful Stage 4 build + push |
| `build-stage3.yml` workflow | `state/_stage3-vllm-env.json` (introspected vLLM env vars) | After Stage 3 build |
| `poc-benchmark` agent | Updates `validation.benchmark`, `metrics`, `status: benchmarked` | After Tier 3 benchmark run; opens auto-PR to this repo |
| CI workflows | `status: validated` (Tier 2 pass) | After GPU smoke test |

## What humans MUST NOT do

❌ Edit `state/*.json` by hand.

State is **observation**, not intent. Editing here desynchronizes from reality. If you need to override metrics or skip a tier, change the spec in `profiles/<x>.cue` or the workflow input, not the observation.

Enforced via `CODEOWNERS` — manual edits to `state/*.json` require infra-team review and explanation in PR.

## Validation

```bash
cue vet state/<file>.json state/schema.cue
```

CI runs this on every PR that touches `state/` (rare — only auto-PRs from agents).

## Schema (`schema.cue`)

Cue schema with:

- **Required**: `profile`, `profile_hash`, `status`, `image.{package, tag, digest, built_at, cosign_identity}`
- **Optional**: `validation`, `metrics`
- **Constraints**: `profile_hash` is 64-char hex; `digest` is `sha256:...`; `cosign_identity` is `https://github.com/...`

Profile authors don't touch this file — schema rarely changes; only ADR-driven changes require infra review.
