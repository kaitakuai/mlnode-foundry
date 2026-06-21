# Decision Log

This file records decisions made for this repository.
Each entry should be short, factual, and link to supporting artifacts (issues/PRs/ADRs).

## Format

- YYYY-MM-DD — <Decision> (Owner: @<DRI>) — <link(s)>

## Entries

- 2026-06-21 — Renumbered the build pipeline to S0..S5 after the Stage-1 vllm-poc split (ADR-0013/0014): old monolithic Stage 1 splits into Stage 1 (vllm-sampler-residual) + Stage 2 (vllm-poc), shifting mlnode-base S2→S3, per-profile S3→S4, reserved minimization S4→S5. Pure rename — image VALUE pins unchanged; plugin migration is a separate change. Moved `stage2/`→`stage3/`, `stage3/`→`stage4/`, `build-stage2.yml`→`build-stage3.yml`, `build-stage3.yml`→`build-stage4.yml`, `tools/stage2.lock.{cue,schema.cue}`→`stage3.lock.*`, `state/_stage2-vllm-env.json`→`_stage3-vllm-env.json`; OCI `gonka.kaitaku.stage` labels 2→3 / 3→4; ADR-0001 amended to "Five-stage build pipeline" and ADR-0010 to "Stage 5 placeholder" (filenames kept). Full-matrix rebuild is the intentional one-time cost (build_hash hashes path strings) (Owner: @baychak) — [ADR-0001](adr/0001-four-stage-pipeline.md), [ADR-0010](adr/0010-image-minimization-stage4.md)
- 2026-06-16 — ADR-0013 amended in-place with DEFERRED-INDEFINITELY status on §Layer-3 + §Rollout-step-4, per ADR-0014 acceptance criterion #4 (Owner: @baychak) — [ADR-0013](adr/0013-poc-integration-architecture.md)
- 2026-06-16 — ADR-0014 amends 0013: residual fork as permanent infra (Owner: @baychak) — [ADR-0014](adr/0014-residual-fork-permanent-infra.md)
- 2026-06-12 — PoC integration architecture: move from fork+monkey-patch to out-of-tree plugin (`gonka-poc` via `worker_extension_cls`/`collective_rpc` + composed entrypoint) with per-version compat shim and contract tests; upstream enabling primitives in parallel; baked deploy defaults move from the vLLM fork into foundry profiles (Owner: @baychak) — [ADR-0013](adr/0013-poc-integration-architecture.md)
- 2026-06-12 — Initialized decision log; prior decisions are recorded as ADR-0001…0012 (Owner: @baychak)
