# ADR-0006 — Rendered Dockerfile policy

**Status:** Accepted
**Date:** 2026-05-10

## Context

Legacy generates per-(GPU, model) Dockerfiles from fragments + a Python generator, commits the rendered output to git. Two problems:

1. **Drift** — generator output checked into git diverges from source (kimi-variants in legacy were "hand-maintained, not generator output" per the old docstring)
2. **Combinatorial sprawl** — 8 GPUs × 3 models = 24 nearly-identical Dockerfile commits to scan in PR review

We need a way for ops/audit to inspect "what Dockerfile produced this image" without inviting drift.

## Decision

**Rendered Dockerfiles are NEVER committed.** They are:

1. **Rendered on demand** by `mlnode-foundry print-dockerfile <profile>` (writes to stdout)
2. **PR-bot diff** — `validate-profiles.yml` GHA renders Dockerfile diff for changed profiles, posts as PR comment
3. **Attached as attestation** — BuildKit `--attest type=dockerfile` records the actual Dockerfile content in the image's in-toto predicates, accessible via `cosign download attestation`

Source of truth is `stage4/Dockerfile` (parametric template) + profile inputs. The "rendered" form exists only as a build action and an attestation predicate.

## Consequences

- **No drift possible** — there's no second copy to drift from
- **PR review preserved** — diff-bot comment shows effective change
- **Audit trail preserved** — anyone can `cosign download attestation` to see what Dockerfile produced any given digest
- **Combinatorial sprawl eliminated** — git history shows profile diff + 1 Dockerfile template, not N Dockerfiles

## Alternatives considered

- **Commit rendered Dockerfiles** (legacy): drift risk, sprawl
- **Snapshot only on release** (committed `released/<tag>/Dockerfile`): partial mitigation; rejected because `--attest type=dockerfile` already provides cryptographically signed snapshot at zero cost
