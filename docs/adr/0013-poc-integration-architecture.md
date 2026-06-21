# ADR-0013 — PoC integration architecture: fork+monkey-patch → plugin+shim+upstream

**Status:** Accepted (Layer 3 — upstreaming track — requires external buy-in: vLLM working group / gonka-ai / vllm-project)
**Date:** 2026-06-12
**Owners:** @baychak
**Related:** [gonka-ai/gonka#1135](https://github.com/gonka-ai/gonka/issues/1135), port branches `kaitakuai/vllm` `mb/feat/port-pocv2-vllm-*`, porting methodology study (workspace `.work/1135/dpoc-porting-methodology.md`)

## Context

vLLM releases frequently; every release forces a re-port of PoC. Empirical data from the 0.15.1 → 0.19 → 0.20 port history:

- Port surface is **46 files, +5633/−62** vs the vanilla base — but the PoC math itself is stable: **11/12 files of `vllm/poc/` are blob-identical between the 0.19 and 0.20 ports**. Version sensitivity is concentrated in one file (`poc_model_runner.py`) plus integration touchpoints.
- The expensive, breakage-prone parts of each port are **not** the `AsyncLLM.poc_request` monkey-patch (one cheap attach point) but:
  1. **Private-internals coupling**: `CommonAttentionMetadata` changed import path and kwargs between versions; a silently lost kwarg (`seq_lens_cpu_upper_bound`) crashed MLA backends (Kimi-K2.6 TP=4/EP=4) and was only caught on hardware (`f79d6b1b5`).
  2. **Sampler-stack invasions** (dual logprobs / enforced tokens) spread across 5 files of private code.
  3. **Non-PoC baggage in the fork**: baked deployment defaults (attention backend, gpu_mem, dtype). The `b90121a27 → f16047bd6 → 6c1075749` saga showed defaults embedded in config dataclasses break on every release.
  4. **No drift contract**: API drift is discovered by crashes, not by tests at version-bump time.
- Precedent that upstreaming works: `skip_compiled` became a vanilla vLLM mechanism in 0.20 — the fork's compilation-layer patches **disappeared** from the port.

Constraint: PoC requires in-process access to the model (inputs_embeds forward, per-layer hooks, worker execution) — a separate-process sidecar cannot work.

## Decision

Adopt a three-layer architecture; do not block the in-flight decode-PoC port on it, but structure new code plugin-ready.

**Layer 1 — out-of-tree plugin package `gonka-poc`** on official vLLM extension points:
- Worker side: `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension` — the official mechanism (used by RLHF frameworks) to add worker methods callable via the **public** `collective_rpc` API. Replaces the monkey-patch for `execute_poc_forward`.
- Server side: thin entrypoint `gonka-vllm-serve` that composes vLLM's stock `build_app` + PoC router + gating middleware (503 + abort in-flight). Composition instead of patching chat/completions routers.
- Torch-level pieces (layer Householder hooks via `register_forward_hook`, vanilla `skip_compiled`) work from the plugin as-is.
- Image becomes: **official `vllm/vllm-openai:<ver>` + `pip install gonka-poc==X`**. The Stage 2 (vllm-poc) overlay build trivializes; vLLM patch releases that don't touch private APIs require no plugin rebuild.

**Layer 2 — compat shim + contract tests**:
- All touches of private internals (attention metadata builders, KV-scratch, worker model access) isolated in `gonka_poc/_compat/v0_XX.py` behind an explicit interface. Porting to a new vLLM = writing **one new compat file**.
- Contract tests assert the private surface (e.g. `CommonAttentionMetadata` field list, builder signatures) on plain import — drift is caught in CI at version bump, not on H100 in production.

**Layer 3 — upstream enabling primitives** to vllm-project (via the vLLM working group; shared pain across teams): stable hook for synthetic-embeds forward on workers; enforced-token replay / dual logprobs (verified-inference demand exists industry-wide). Each merged primitive permanently deletes a fork patch (proof: `skip_compiled`).
**Status (2026-06-16):** DEFERRED-INDEFINITELY — see ADR-0014. The upstream-PR pathway to vllm-project is not viable at present; the residual fork is treated as permanent infrastructure with a maintenance pipeline.

**Hygiene:** baked deployment defaults move out of the fork into mlnode-foundry profiles (env/CLI — Stage 4 concern, not engine code). The fork remains only as a staging area for patches not yet plugin-ized/upstreamed, with auto-rebase CI onto upstream tags and the bit-test verification ladder.

## Options considered

### A. Status quo: fork + monkey-patch + overlay image
- Pros: full internals access; works today; no upstream coordination.
- Cons: 46-file re-port per version; silent private-API drift; fork entangles PoC with deploy policy; rebuild on every release.
- Risks: hardware-only crash classes (MLA kwarg) recur.

### B. Sidecar process (PoC server talks to vLLM over API)
- Rejected: PoC needs in-process inputs_embeds forward + layer hooks + worker execution; impossible over HTTP.

### C. Upstream-first (push everything into vllm-project before restructuring)
- Rejected as sole strategy: too slow, external acceptance uncertain; kept as Layer 3 running in parallel.

### D. Plugin + compat shim + upstream track (chosen)
- Pros: per-version cost collapses to one compat file; module ships via pip, image from official base; drift caught by contract tests; upstream track shrinks the shim over time.
- Cons: extension points (`worker_extension_cls`, `collective_rpc`, entrypoint composition) are more stable than internals but not frozen; sampler-stack (enforced tokens) is the hardest to plugin-ize and leaves the fork last.
- Risks: attention-metadata building stays version-sensitive (shim cannot eliminate it, only contain it) until Layer 3 lands a stable hook.

## Consequences

- Positive: vLLM version bumps stop being projects; PoC math versioned/released independently of vLLM; official images consumable directly; clearer security/supply-chain story (pip package + digest pinning vs full-image overlay).
- Negative: initial extraction effort; two artifacts to release (plugin + image profile) instead of one; enforced-tokens remains fork-bound until upstreamed.
- Operational: the vLLM base stages of the foundry pipeline (Stage 1 residual + Stage 2 vllm-poc, ADR-0001) eventually reduce to "official base + pip install"; contract tests added to CI gates.

## Rollout plan

1. **Now**: continue decode-PoC port (`mb/feat/port-dpoc-vllm-0-20-0`) on the current skeleton; route all new internals touches through the same narrow interface as prefill (plugin-ready structure).
2. After decode-PoC parity: extract `gonka-poc` package (math + worker extension + entrypoint), `_compat/v0_20.py`, contract tests; ship one release where fork-image and plugin-image are byte-equivalent on the golden ladder.
3. Switch the foundry vLLM base stages (Stage 1 residual + Stage 2 vllm-poc) to official base + residual + pip install; keep fork build as fallback for one release cycle.
4. Layer 3: bring primitive proposals to the vLLM working group (2026-06-15 call agenda fits: "перенос на новые версии"). DEFERRED-INDEFINITELY per ADR-0014.

Backout: plugin and fork builds coexist; revert = pin the vLLM base stage (Stage 2 vllm-poc) back to the monolith fork image.

## Acceptance criteria

- Port to next vLLM version = 1 new compat file + green contract tests + green golden ladder (no other diffs).
- `vllm/vllm-openai:<ver>` + `pip install gonka-poc` passes the full verification ladder (bit-tests, live PoC, golden artifacts, hardware matrix).
- vLLM patch release without private-API changes requires zero plugin changes.

## Links

- Porting methodology + decode-PoC donor map: workspace `.work/1135/dpoc-porting-methodology.md`
- ADR-0001 (four-stage pipeline), ADR-0004 (supply-chain attestations)
- Precedent: `skip_compiled` vanilla in vLLM 0.20 (forward_context / compilation decorators)
