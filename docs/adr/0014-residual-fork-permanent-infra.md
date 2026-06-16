# ADR-0014 — Residual vLLM fork as permanent infrastructure (amends ADR-0013)

**Status:** Accepted (amends ADR-0013 §Layer-3)
**Date:** 2026-06-16
**Owners:** @baychak
**Related:** [ADR-0013](0013-poc-integration-architecture.md) (PoC integration architecture). Branches `kaitakuai/vllm@poc-sampler-residual-v0.23`, `kaitakuai/gonka-poc@main`.

## Context

- ADR-0013 framed Layer 3 (upstream sampler-stack / `SamplingParams` extension hooks to `vllm-project/vllm`) as the eventual exit strategy for the thin fork — once those primitives land, the 6 sampler-stack commits in `kaitakuai/vllm@poc-sampler-residual-vX.YY` migrate into the `gonka-poc` plugin and the fork goes away.
- That pathway is OFF the table: Kaitaku does not have the bandwidth or acceptance channel to drive upstream PRs through the `vllm-project` review process.
- Evaluating three porting strategies post-0.20→0.23 port (full-fork monkey-patch, plugin+thin-fork, full-fork rebase), the plugin+thin-fork approach is the best of the three even WITHOUT a Layer 3 exit. The thin fork is small (6 commits, ~150 lines), localized to sampler hot-spots; per-minor rebase is mechanical hours, not heroic days.
- The risk is that without an exit strategy, the fork accumulates technical debt. Mitigation = invest in maintenance pipeline so debt does not silently grow.

## Decision

Treat `kaitakuai/vllm@poc-sampler-residual-vX.YY` as **permanent infrastructure** rather than a temporary bridge. Invest in tooling that makes the per-minor rebase deterministic.

Specifically:

1. **Per-residual-branch contract tests** (`tests/contract/test_sampler_surface.py`) pinning `SamplingParams` fields, `Sampler.__call__` signature, `TopKTopPSampler.forward_*` kwargs, `InputBatch.logprobs_modes` attribute, structured-output graceful-degradation hook.
2. **CI workflow** `contract-tests-residual.yml` on the residual branch with two jobs:
   - `in-fork`: install the residual fork + run tests (verifies our patches actually applied).
   - `upstream-drift`: install latest stable upstream `vllm` + run the same tests with `continue-on-error`; failure ALERTS us before manual rebase.

   Daily cron schedule on the workflow so we catch upstream patch-release drift.
3. **REBASE.md** at the root of the residual branch documenting the mechanical rebase procedure (cherry-pick order, version-bump location, foundry overlay update).
4. **README header** marking the branch as permanent infra + linking REBASE.md + ADR-0014.

## Options considered

### A. Status quo: ADR-0013 as written, Layer 3 as deferred exit
- Pro: leaves the door open to upstreaming if circumstances change.
- Con: misleads the team about expected lifetime of the fork. Tooling that assumes 'temporary' (e.g., minimal documentation, no contract tests on the fork itself) becomes liability.

### B. Monkey-patch sampler inside the plugin instead of maintaining a thin fork
- Pro: 0 forks of vllm; 1 artifact (plugin only); operator install is `pip install vllm==X && pip install gonka-poc`.
- Con: ADR-0013 explicitly rejected Option A monkey-patching as fragile. `SamplingParams` is a frozen dataclass — extending fields requires class re-creation which breaks `isinstance` and serialization identity. The risk surface is wider than a 6-commit fork.
- Rejected: the failure mode of monkey-patching is silent breakage on every minor. Thin fork's failure mode is loud failure in CI before deploy.

### C. Full-fork rebase (current ADR-0013 Option A under another name)
- Pro: one artifact, familiar workflow.
- Con: 46-file diff per minor; same hardware-only-discovery failure mode that ADR-0013 already rejected.

### D. Thin fork as permanent infrastructure (chosen)
- Pro: PoC math (the majority) stays in plugin via stable public extension APIs; sampler residual is isolated and small; per-minor cost bounded; CI catches drift.
- Con: forever maintenance ownership of the residual branch; need explicit rebase procedure to prevent bus-factor; longer-running version skew between residual and upstream than originally anticipated.
- Risks: if sampler-stack semantics change radically in a future minor, the 6 commits may not cherry-pick cleanly and require semantic re-derivation. Mitigation = contract tests catch the surface change ahead of the rebase.

## Consequences

- Positive: explicit and honest documentation of fork lifetime; CI gate on drift; rebase procedure is doc-driven not tribal-knowledge-driven.
- Negative: residual fork is now an explicit team artifact with maintenance ownership; if the team disbands, the fork goes stale. Bus factor needs to be acknowledged.
- Operational: each vLLM minor release triggers (a) automated CI alert when upstream surface drifts, (b) explicit rebase task per REBASE.md, (c) version-tag bump in residual fork + foundry stage1 lock.

## Acceptance criteria

1. The residual branch carries `tests/contract/test_sampler_surface.py` asserting at least 6 pins on the sampler / `SamplingParams` / `InputBatch` / structured-output surface.
2. The residual branch carries a CI workflow that runs those tests on push, PR, `workflow_dispatch`, AND daily cron — with a `continue-on-error` upstream-drift job that alerts before manual rebase.
3. The residual branch carries REBASE.md with the 10-step rebase procedure + cherry-pick order.
4. ADR-0013 §Rollout-plan step 4 (Layer 3 upstreaming) is marked as **DEFERRED-INDEFINITELY** in cross-reference text, not as a planned milestone.
5. `mlnode-foundry/docs/decision-log.md` has a 2026-06-16 entry pointing at this ADR.

## Links

- ADR-0013 (this ADR amends): [docs/adr/0013-poc-integration-architecture.md](0013-poc-integration-architecture.md)
- Residual branch: https://github.com/kaitakuai/vllm/tree/poc-sampler-residual-v0.23
- Plugin: https://github.com/kaitakuai/gonka-poc
