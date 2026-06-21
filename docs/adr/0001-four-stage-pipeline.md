# ADR-0001 — Five-stage build pipeline

**Status:** Accepted (amended 2026-06-21 — Stage 1 split into residual + vllm-poc; pipeline renumbered S0..S4 + reserved S5, see ADR-0013/0014)
**Date:** 2026-05-10 (amended 2026-06-21)

## Context

Legacy `kaitakuai/mlnode` builds two image lines (`mlnode-full`, `mlnode-overlay`) in parallel pipelines. Each per-(GPU, model) Dockerfile re-runs `apt install` + OpenSSL + Poetry venv from scratch — ~7-10 min of identical work × 8 GPU variants = wasted cycles every build.

The original decision (2026-05-10) defined a four-stage pipeline whose Stage 1 was a single monolithic `kaitakuai/vllm` image carrying the vLLM fork **plus** the Gonka PoC patches. Per ADR-0013 (PoC integration architecture) and ADR-0014 (residual fork as permanent infrastructure), that monolith is split into a thin sampler-residual fork and a `pip install gonka-poc` overlay. This amendment (2026-06-21) records the resulting **five-stage** pipeline and the +1 renumbering of every downstream foundry layer.

## Decision

Adopt a **5-stage pipeline** (plus a reserved Stage 5 for image minimization) with explicit intermediate images stored in GHCR:

```
Stage 0: vllm/vllm-openai (upstream, external)
Stage 1: kaitakuai/vllm-sampler-residual:<vllm-ver>-k<rev>   (thin sampler-stack fork, ADR-0014)
Stage 2: kaitakuai/vllm-poc:<vllm-ver>-k<rev>                (Stage 1 + pip install gonka-poc, ADR-0013)
Stage 3: kaitakuai/mlnode-base:<mlnode>-vllm<vllm>-k<rev>    (mlnode source + venv)
Stage 4: kaitakuai/mlnode-<gpu>-<model>:<tag>               (hw-patches + tuning)
```

Stages 1 and 2 live in the `kaitakuai/vllm` repository. Stage 3 is GPU-agnostic, built once per (mlnode, vllm) bump. Stage 4 is per-(GPU, model), inherits from Stage 3 by digest.

The two legacy lines (`full`, `overlay`) collapse into a Stage 4 mode discriminator: `kaitakuai-base` (default) vs `upstream-overlay` (for fast-path overlays on `product-science/mlnode`).

## Consequences

- **8× cache reuse** for Stage 3 (mlnode-base) across GPU variants
- **Bumping vLLM** rebuilds Stage 1+2 (residual fork + gonka-poc) and Stage 3 once, not 8 times
- **Profile drift** between similar GPU variants eliminated — they share the Stage 3 layer hash
- **PoC decoupling**: vLLM patch releases that don't touch private APIs require no plugin rebuild (Stage 2 = official base + residual + `pip install gonka-poc`)
- `full`/`overlay` distinction becomes a profile field, not a separate pipeline

## Alternatives considered

- **Single-stage parametric Dockerfile** (current legacy): rejected for the duplication problem above
- **Keep the monolithic Stage 1** (vLLM + PoC in one image): rejected per ADR-0013 — couples every vLLM patch release to a full plugin rebuild
- **Flat Bazel target graph**: tracked separately (see ADR-0008); adds Bazel ramp-up cost not justified at current scale
