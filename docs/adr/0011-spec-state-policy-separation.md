# ADR-0011 — Spec / state / policy separation

**Status:** Accepted
**Date:** 2026-05-10

## Context

Three different kinds of information live in the system:

1. **Spec / intent** — what humans want built (profiles)
2. **State / observation** — what actually happened (build results, validation outcomes, benchmarks)
3. **Policy** — how axes serialize, what tags look like, which runners exist (org-wide rules)

Mixing these in one file (e.g., a profile that has both intent and observed metrics) leads to:

- Polluted `git blame` (automated metric updates clutter human commits)
- Cyclic edits (agent writes metric → CI re-evaluates → re-runs → updates metric)
- Conflated authority (humans and machines both edit the same fields)

## Decision

**Three distinct file conventions** with non-overlapping authorship:

| Kind | Location | Format | Author |
|------|----------|--------|--------|
| Spec | `profiles/*.cue`, `profiles/bases/*.cue` | Cue | humans |
| Schema | `profiles/schema.cue`, `state/schema.cue` | Cue | humans (rare changes) |
| Policy | `tools/naming.cue`, `tools/runners.cue`, `tools/stage3.lock.cue` | Cue | humans (rare changes) |
| State | `state/<package-tag>.json` | JSON | machines (CI, agent) |

Cue is the language for **human-authored intent** (composition, types, constraints, sum types). JSON is for **machine-written observations** (no need for type system at write-time; validated by Cue schema at read-time via `cue vet`).

CI workflows write `state/*.json` directly; humans don't edit these. Schema enforcement via `cue vet state/<x>.json state/schema.cue` in `validate.yml`.

## Consequences

- **Clean `git blame` on profiles** — only human changes
- **No write-loop**: agent writes state.json → does NOT trigger Stage 4 rebuild (state files excluded from `profile_hash`)
- **Reset state without touching spec** — if benchmark fails or budget burns, `rm state/<x>.json` and re-run; profile untouched
- **Auditability**: dashboard can `cosign verify` + read state.json + compare to spec.cue, all three from different sources

## Alternatives considered

- **Single profile file with status block** (initial draft): rejected — exactly the write-loop problem above
- **State in separate orphan branch**: rejected — adds operational complexity for marginal cleanliness gain
- **State in external store (DB)**: rejected — adds another infra dependency
