# Decision Log

This file records decisions made for this repository.
Each entry should be short, factual, and link to supporting artifacts (issues/PRs/ADRs).

## Format

- YYYY-MM-DD — <Decision> (Owner: @<DRI>) — <link(s)>

## Entries

- 2026-06-16 — ADR-0013 amended in-place with DEFERRED-INDEFINITELY status on §Layer-3 + §Rollout-step-4, per ADR-0014 acceptance criterion #4 (Owner: @baychak) — [ADR-0013](adr/0013-poc-integration-architecture.md)
- 2026-06-16 — ADR-0014 amends 0013: residual fork as permanent infra (Owner: @baychak) — [ADR-0014](adr/0014-residual-fork-permanent-infra.md)
- 2026-06-12 — PoC integration architecture: move from fork+monkey-patch to out-of-tree plugin (`gonka-poc` via `worker_extension_cls`/`collective_rpc` + composed entrypoint) with per-version compat shim and contract tests; upstream enabling primitives in parallel; baked deploy defaults move from the vLLM fork into foundry profiles (Owner: @baychak) — [ADR-0013](adr/0013-poc-integration-architecture.md)
- 2026-06-12 — Initialized decision log; prior decisions are recorded as ADR-0001…0012 (Owner: @baychak)
