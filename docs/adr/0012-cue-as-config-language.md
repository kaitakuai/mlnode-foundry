# ADR-0012 — Cue as config language

**Status:** Accepted
**Date:** 2026-05-10

## Context

Initially considered YAML + JSON Schema for profile DSL (per ADR-0003). Three pain points showed up immediately:

1. **YAML Norway problem** (`no` → false) and indentation pitfalls
2. **No native sum types** — `mode: kaitakuai-base | upstream-overlay` requires `oneOf`/`if`/`then` boilerplate in JSON Schema; runtime-only discrimination
3. **Schema and data in different files/languages** — schema in JSON Schema (also JSON), data in YAML; two parsing chains, validator fragility

## Decision

Adopt [Cue](https://cuelang.org/) as the single language for:

- **Profile spec** (`profiles/*.cue`)
- **Profile schemas** (`profiles/schema.cue` — defines `#Profile`)
- **Policy** (`tools/naming.cue`, `tools/runners.cue`)
- **State schemas** (`state/schema.cue` — validates JSON state files)

Cue gives us:

- **Native sum types** — `#Profile: #BaseProfile | #OverlayProfile`, compiler discriminates by mode field
- **Unification** (`&`) — commutative composition, conflicts raise loud compiler errors (vs silent overwrite in deep-merge)
- **Constraints in types** — `int & >=1`, `=~ "^sha256:[a-f0-9]{64}$"` inline
- **Schema = data in same language** — no separate JSON Schema file
- **Validates JSON natively** — `cue vet state.json schema.cue` works without code changes

JSON remains the format for machine-written state files (see ADR-0011); Cue schema validates them.

## Consequences

- **Type-safety at validation time** — schema mismatches caught before build, not at runtime
- **One language to learn** for spec / schema / policy (Python remains for orchestration; ramp-up reasonable with active Claude Code Opus 4.7)
- **Cue runtime dependency** — adds ~15 MB binary; pinned via `mise.toml`
- **Smaller community than YAML/Pydantic** — fewer Stack Overflow answers; AI compensates

## Fall-back trigger

If team finds Cue ramp-up exceeds 2 weeks (2x our budget), fall back to TOML + Pydantic — at the cost of losing native sum types (re-opens "mode-discriminated-union" as a runtime smell) and unification (reverts to dict deep-merge with conflict detection in Python).

Not expected to fire; documented for accountability.

## Alternatives considered

- **YAML + JSON Schema**: see Context
- **TOML + Pydantic**: viable fallback (see above)
- **Dhall, Pkl, Jsonnet**: each has trade-offs; Cue chosen for sum types + unification + JSON-validation combination
