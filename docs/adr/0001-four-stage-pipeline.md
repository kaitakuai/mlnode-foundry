# ADR-0001 — Four-stage build pipeline

**Status:** Accepted
**Date:** 2026-05-10

## Context

Legacy `kaitakuai/mlnode` builds two image lines (`mlnode-full`, `mlnode-overlay`) in parallel pipelines. Each per-(GPU, model) Dockerfile re-runs `apt install` + OpenSSL + Poetry venv from scratch — ~7-10 min of identical work × 8 GPU variants = wasted cycles every build.

## Decision

Adopt a **4-stage pipeline** with explicit intermediate images stored in GHCR:

```
Stage 0: vllm/vllm-openai (upstream, external)
Stage 1: kaitakuai/vllm:<vllm-ver>-poc-k<rev>           (PoC v2 patches)
Stage 2: kaitakuai/mlnode-base:<mlnode>-vllm<vllm>-k<rev> (mlnode source + venv)
Stage 3: kaitakuai/mlnode-<gpu>-<model>:<tag>           (hw-patches + tuning)
```

Stage 2 is GPU-agnostic, built once per (mlnode, vllm) bump. Stage 3 is per-(GPU, model), inherits from Stage 2 by digest.

The two legacy lines (`full`, `overlay`) collapse into Stage 3 mode discriminator: `kaitakuai-base` (default) vs `upstream-overlay` (for fast-path overlays on `product-science/mlnode`).

## Consequences

- **8× cache reuse** for Stage 2 across GPU variants
- **Bumping vLLM** rebuilds Stage 2 once, not 8 times
- **Profile drift** between similar GPU variants eliminated — they share Stage 2 layer hash
- `full`/`overlay` distinction becomes a profile field, not a separate pipeline

## Alternatives considered

- **Single-stage parametric Dockerfile** (current legacy): rejected for the duplication problem above
- **Flat Bazel target graph**: tracked separately (see ADR-0008); adds Bazel ramp-up cost not justified at current scale
