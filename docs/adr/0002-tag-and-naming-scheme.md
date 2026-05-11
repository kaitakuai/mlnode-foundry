# ADR-0002 — Tag and naming scheme

**Status:** Accepted
**Date:** 2026-05-10

## Context

Legacy tag scheme `0.2.12-vllm0.20.0-b300-k5-kimi-1` mixes 4-5 axes (gpu, model, quant variant, mlnode/vllm versions, kaitaku rev) without separators. Adding a new axis (e.g., `framework`) requires regex updates in dashboard, `experiments` pin migrations, and risks accidental collisions.

Profile authors must NOT be free to choose names — names should be **derived** from a single source of truth (the profile + global naming policy). This eliminates "typo in tag" bugs and keeps name consistency org-wide.

## Decision

**GHCR package name** = `<prefix>-<axis-1>-<axis-2>` from `naming.yaml::package_axes` (default `[gpu, model]`):

- `ghcr.io/kaitakuai/mlnode-b300-kimi`
- `ghcr.io/kaitakuai/mlnode-h100-qwen`

**Tag** = `<base-version>(<-prefix.value>)*-k<rev>` with prefix-separated identity axes:

- `kaitakuai-base` mode: `<mlnode>-vllm<vllm>(-q.<value>)?(-f.<value>)?(-t.<value>)?-k<rev>`
- `upstream-overlay` mode: `<upstream>-overlay(-q.<value>)?...-k<rev>`

Examples:

- `0.2.13-vllm0.20.0-q.int4-k1`
- `0.2.13-vllm0.20.0-q.int4-f.sglang-k1`
- `3.0.13-alpha5-overlay-k1`

Axes registered in `tools/naming.cue` with explicit `prefix` (e.g., `quant: prefix: "q"`).

The CLI computes name + tag from profile; profile author cannot override.

## Consequences

- **Dashboard regex parses by prefix-разделитель**, not by position — adding new axis doesn't break old tags
- **Single source of truth** for naming convention (`tools/naming.cue`), changeable in one place
- **Strict invariant**: `mlnode-foundry tag <profile>` is the only way to know what name an image will get
- Profile names with embedded version (`b200-kimi-int4-cutlass`) become illegal — must be axes only

## Alternatives considered

- **Free-form tags** (legacy): too error-prone
- **Hash-only tags** (Nix-style): unreadable for humans
- **Position-based composite** (`b300-kimi-int4-vllm0.20.0-mlnode0.2.13-k1`): hard to extend; chose prefix-separated instead
