# `registry-view/` — dashboard-facing image registry

One JSON per published Stage 3 image, written by CI after `cosign sign`. The
shape replaces the legacy `kaitakuai/mlnode/registry/*.json` aggregator —
single file the dashboard reads to render the "image card" view.

## What it aggregates

| Source | Fields |
|--------|--------|
| `profiles/<x>.cue` (env + runtime_defaults) | `flags[]` |
| `profiles/<x>.cue` (tuning_notes) | `flag_descriptions{}`, `flag_warnings{}`, `report_url` |
| `tools/model-registry.cue` | `model`, `model_short`, `model_params_b`, `model_context_max`, `model_license` |
| `tools/stage2.lock.cue` | `cuda` |
| `tools/naming.cue` | `name`, `tag` |
| Stage 3 build outputs (CI-supplied) | `digest`, `size`, `cosign_identity`, `slsa_attestation_url` |
| `git log` | `uploaded_by` |
| poc-benchmark agent (Tier 3, separate PRs) | `nonces`, `weight` (null until measured) |

## Severity vs description

Every `profile.tuning_notes[]` entry lands in `flag_descriptions` (neutral
tooltip). Entries with `severity: "warning"` ALSO appear in `flag_warnings`,
so the dashboard renders the triangle icon.

## Workflow

```
profile.cue change → push to main
  → build-stage3 matrix
     → docker buildx build + push
     → cosign sign
     → mlnode-foundry registry-view <profile> --digest --cosign-identity --size
     → commit registry-view/<package>-<tag>.json to main
```

The file is regenerated on every Stage 3 publish. Do NOT edit by hand —
edit `profiles/<x>.cue`, `tools/model-registry.cue`, or `tools/stage2.lock.cue`
and let CI re-emit.
