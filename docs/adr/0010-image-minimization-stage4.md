# ADR-0010 — Image minimization (Stage 4 placeholder)

**Status:** Reserved
**Date:** 2026-05-10

## Context

Real Stage 3 images are 45 GB. ~50% is reusable runtime (Python venv, vLLM kernels, CUDA libs); the rest is build-time noise (apt cache, OpenSSL build artifacts, FlashInfer JIT cache for irrelevant SMs, etc.).

For most node operators 45 GB is acceptable (one-time pull, persistent on disk). For some scenarios (k8s ephemeral pods, edge deployment, fast cold-start) we may want minimization to ~15-20 GB.

This is **not implemented** in the initial system; reserved for future Stage 4.

## Decision

The architecture **reserves** an axis `transform: full | slim` with `status: reserved` in `tools/naming.cue`. Profiles **may** declare an optional `minimization:` block in spec; today it is ignored by the build pipeline.

When Stage 4 is implemented:

1. New `stage4/Dockerfile.minimize` (or `apko`-based recipe) consumes a Stage 3 image and produces a slimmed variant
2. Profile `minimization.strategy` selects the recipe (`multistage-copy`, `apko-distroless`, etc.)
3. Slimmed image gets `transform: slim` axis → tag suffix `-t.slim`
4. Same supply-chain attestations apply

Until implementation:

- Profiles that include `minimization:` block are validated for shape but not acted on
- `transform` axis present in `naming.cue` registry but profiles don't set it (defaults to `full`)

## Consequences

- **Forward-compatible**: profiles authored today won't break when Stage 4 lands
- **No premature implementation** — slimming strategy decision deferred until real demand materializes
- **Honest documentation** — researchers see `transform` reserved, not "available"

## Implementation triggers

- A specific node operator scenario requires < 20 GB images
- Stage 4 effort estimated separately; not in current epic
