# ADR-0003 — Profile DSL and axis types

**Status:** Accepted
**Date:** 2026-05-10

## Context

A profile declares "what image to build for `(gpu, model[, quant])`". The DSL needs to:

1. Be type-safe (catch typos and incompatible combinations at validation time)
2. Support discrimination by `mode` (`kaitakuai-base` vs `upstream-overlay` have different required fields)
3. Support composition (a B300 profile shares many fields with another B300 profile)
4. Distinguish "different image" from "same image, different runtime arg"

## Decision

**Profile DSL is Cue.** Schema in `profiles/schema.cue`:

```cue
#Profile: #BaseProfile | #OverlayProfile
```

Sum type discriminated by `mode`. Cue compiler enforces required fields per branch.

**Axes have two types:**

- **identity** — value change ⇒ different image (different GHCR coord or digest). Examples: `gpu`, `model`, `quant`, `framework`, `transform`.
- **runtime** — value change ⇒ same image, different `docker run` flag. Examples: `tensor_parallel_size`, `gpu_memory_utilization`, vLLM CLI flags.

Identity axes go in `identity.axes`; runtime config goes in `runtime_defaults` (suggested at startup; nod-operator can override).

A previously considered "tuning" type (same image, different tag) was **rejected** as artificial — if axes don't change content, no reason to publish a second tag.

## Consequences

- **Native discriminated union** via Cue `|` — invalid combinations rejected at `cue vet`
- **Composition** via `&` unification — bases (e.g., `_base/b300.cue`) merged with leaf profile, conflicts raise compiler error
- **Boundary enforced**: changing `runtime_defaults` doesn't trigger image rebuild (it's not in `profile_hash`)

## Alternatives considered

- **YAML + JSON Schema**: rejected — runtime-only validation, no sum types, no unification
- **Pydantic in Python**: rejected — Turing-complete profiles allow arbitrary code in spec
- **Starlark / Bazel macros**: rejected (see ADR-0008) — Bazel ramp-up not justified at current scale
