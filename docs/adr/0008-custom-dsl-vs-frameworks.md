# ADR-0008 — Why custom thin DSL over Bazel/apko/Dagger

**Status:** Accepted
**Date:** 2026-05-10

## Context

When designing the build system, we evaluated established frameworks for image production:

- **Bazel + rules_oci/rules_img** — industrial-grade, hermetic, content-addressable cache, lazy CUDA pull
- **Chainguard apko + melange** — declarative no-RUN images, distroless-friendly
- **Dagger SDK** — code-as-CICD, programmable pipelines
- **Custom**: Cue spec + Python CLI + docker buildx

## Decision

**Custom thin DSL** (Cue + Python + buildx) chosen because:

1. **2-dev team** with active Claude Code Opus 4.7 — Bazel ramp-up (2-4 weeks compressed, 3-6 months without AI) doesn't fit our bandwidth
2. **Infrequent builds (≤ 10/month) of 45 GB images** — Bazel's content-addressable wins are wasted; bottleneck is push bandwidth (see ADR-0007)
3. **vLLM + CUDA stack is Debian-based** — `apko + melange` (APK-based) doesn't fit; would require forking vLLM build chain
4. **No requirement for runtime programmability** — Dagger's Turing-complete pipelines unnecessary; declarative profiles are simpler to reason about

The choice is **right-sized for actual constraints**, not optimal in a vacuum.

## Triggers for revisiting

| Trigger | Reconsider |
|---------|-----------|
| > 150 profiles AND active tuning development | Bazel `rules_img` — lazy CUDA pull becomes critical |
| Compliance L4 (full hermeticity, offline-buildable) | Bazel + air-gapped builder, or Nix `dockerTools` |
| Team grows to 5+ devs with build-infra-engineer role | Any framework — onboarding cost amortizes |
| Stage 5 minimization becomes default (slim images) | apko as `scratch-repack` strategy |
| Pipeline logic genuinely programmable (loops, conditional steps) | Dagger SDK |

Currently all triggers below the horizon.

## Consequences

- **Cue ramp-up**: 1-2 weeks for team (compressed by AI). Acceptable investment.
- **Custom `profile_hash`** instead of Bazel native cache (see ADR-0007)
- **No gold-standard hermeticity** — docker buildx is "loose"; acceptable for our threat model (cosign signature provides verification, not bit-perfect reproducibility)

## Alternatives considered (in detail)

See `docs/architecture-bazel.md` and `docs/architecture-dagger.md` from initial design phase (deleted post-decision; preserved in git history).
