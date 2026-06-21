# ADR-0007 — Build optimization and caching policy

**Status:** Accepted
**Date:** 2026-05-10

## Context

Real images are 45 GB (validated against local `docker image ls`). Builds happen infrequently (≤ 10/month) on GitHub Actions free tier. Cache strategies that optimize layer-level reuse have limited ROI on this cadence/size.

The actual bottleneck is **push bandwidth** — 45 GB to GHCR per Stage 4 publish (~10-15 min). Build time itself is dominated by `docker pull` of Stage 1-3 base layers (already cached on GHCR CDN).

## Decision

Use **GHCR registry-backed cache** as primary mechanism:

```yaml
- uses: docker/build-push-action@v6
  with:
    cache-from: type=registry,ref=ghcr.io/kaitakuai/mlnode-<gpu>-<model>:buildcache
    cache-to: type=registry,ref=ghcr.io/kaitakuai/mlnode-<gpu>-<model>:buildcache,mode=max
```

This:

1. **Survives across workflow runs** (unlike GHA cache which evicts at 10 GB)
2. **Free** (GHCR storage is free for public repos)
3. **mode=max** exports cache for all intermediate stages, not just final

**Smart skip-if-unchanged** via `profile_hash`:

`mlnode-foundry hash <profile>` computes SHA-256 over (profile.cue + naming.cue + hw_patches/ files referenced + runner_patch + Stage 3 digest from the base-stage lock). On `build-stage4.yml` execution, compare against `gonka.kaitaku.profile_hash` label on the current GHCR tag. Match → skip rebuild, retag.

**No remote cache services** (no BuildBuddy, no Depot). Justified by ADR-0008 + the bandwidth-bottleneck observation above.

## Retention

- Stage 1 / Stage 2 / Stage 3 tags (residual fork, vllm-poc, mlnode-base): keep ≥ 3 months for traceability
- Stage 4 tags: **keep indefinitely** (public artifact pinned by node operators, experiments)
- `:buildcache` tags: keep last 5, GC weekly

## Consequences

- **Zero infrastructure cost** beyond GHCR (free for public)
- **Cold rebuild ~ 25 min**, warm-cache delta-rebuild ~ 1-2 min
- **Push bandwidth** (45 GB) remains bottleneck — not addressable by build-engine choice
- Custom `profile_hash` does NOT match Bazel's content-addressable rigor; acceptable trade-off (rebuilds we falsely skip = caught by smoke tests)
