# Add a hardware-validation report to an image

> Once an image has been smoke-tested on real GPU hardware and behaves
> correctly (PoC nonces flowing, no engine crashes, no chain divergence),
> link the evidence to its registry entry so the public dashboard shows
> the **verified** chip with a clickable link to the report instead of
> the **in progress** chip.

## Pre-conditions

- Image has been **published** to GHCR by the Stage 3 CI
  (`registry-view/<package-tag>.json` exists in this repo).
- Image has been **run on real hardware** and the operator has evidence
  of correct behaviour. Evidence MAY be a multi-batch nonce sweep with
  measured throughput, MAY be a single-batch smoke confirming nonces are
  generated, MAY be a screenshot of healthy operator dashboard for ≥1 hour.
  Anything below "image starts and produces non-zero nonces" is NOT enough.

## Procedure

### 1. Write the report in `kaitakuai/experiments`

Path convention: `kaitakuai/experiments/<YYYY-MM>/<short-name>/README.md`.

The short-name SHOULD encode the image axes for searchability. Examples:

- `kimi_k26_int4_4xb200_q-int4-k2` (model_family_revision_quant_hwcount×gpu_tag)
- `minimax_m27_fp8_4xh100`
- `qwen3-235b_fp8_2xb200`

Required content (concise — full operator log lives in chat, not here):

- Image full ref (`ghcr.io/kaitakuai/mlnode-<gpu>-<family>-<rev>:<tag>` + digest)
- Stage 1 / Stage 2 / Stage 3 digests so the build chain is reproducible
- Hardware (provider, GPU count + model, host CPU/RAM if relevant)
- What was tested (PoC nonce generation, /chat/completions correctness, multi-batch sweep, …)
- Result (numeric where applicable: nonces/min, batch size, p50/p99 latency)
- Known caveats from this run (OOM at batch=N, slow cold-start, etc.)
- Links to artifacts in subdirectories if relevant (`artifacts/nonces_1000.json`, etc.)

Push to `experiments/main` directly (or PR + merge). The URL you will
reference next is **stable** as long as the file is on `main`:
`https://github.com/kaitakuai/experiments/blob/main/<YYYY-MM>/<short-name>/README.md`.

### 2. Add a `validation-report` tuning_note **first** in the profile

In `kaitakuai/mlnode-foundry/profiles/<gpu>-<family>-<rev>[-<quant>].cue`,
prepend (yes, **first**) entry to `tuning_notes:`:

```cue
tuning_notes: [
    {
        knob:     "validation-report"
        source:   "https://github.com/kaitakuai/experiments/blob/main/<YYYY-MM>/<short-name>/README.md"
        reason:   "Hardware validation — <gpu count>× <gpu>, <who>, <when>, <key finding>."
        added_at: "<YYYY-MM-DD>"
    },
    // ... existing notes
]
```

Why **first**: `mlnode_foundry/render_registry_view.py::_report_url` walks
`tuning_notes` and returns the first `source` that starts with `http` or
contains `experiments`. The first match wins.

### 3. (Optional, immediate effect) Patch the published `registry-view/` JSON

The `registry-view/<package>-<tag>.json` for this profile is auto-generated
by Stage 3 CI. If you do not patch it manually, the new `report_url` only
lands when the next Stage 3 rebuild for this profile happens — possibly
days or weeks away.

For immediate dashboard pickup, also edit the JSON in this same PR:

```diff
-  "report_url": null,
+  "report_url": "https://github.com/kaitakuai/experiments/blob/main/<YYYY-MM>/<short-name>/README.md",
```

This is safe: the next Stage 3 build will regenerate the same value
deterministically from the profile change in step 2.

### 4. Open the PR + merge

Squash + delete branch. Single PR carries both edits (profile + JSON).
Commit message convention:

```
chore(<gpu>-<family>): link validation report after hardware smoke
```

### 5. Verify the chip flipped (≤ 5 min after merge)

The dashboard auto-sync timer (`dashboard-sync-registry.timer` on
`88.216.70.137`) runs every 5 minutes and pulls
`kaitakuai/mlnode-foundry/registry-view/*.json`. After the next tick:

- API: `curl -s https://registry.kaitaku.ai/api/images | jq '.images[] | select(.tag == "<tag>") | .report_url'` → returns the report URL, not `null`.
- UI: open https://registry.kaitaku.ai/, locate the image card, the
  amber **in progress** chip becomes a blue **verified** link pointing at
  the report.

If the chip didn't flip after 5 min, check `journalctl -u dashboard-sync-registry --since "10 minutes ago"` on the host for sync errors (expired token, FK violation, etc).

## Reverting a verification

If a "validated" image is later found to misbehave under load, the report
should NOT silently disappear — it documents what was tested at the time.
Instead:

- Open a follow-up note in the same experiments report (`## Update YYYY-MM-DD — regression observed under <conditions>`).
- Bump the profile `rev` to a new revision (k3, k4, …) carrying the fix.
- Build + publish; the new image starts in the **in progress** state again
  until its own validation report is linked.

The old report stays in place as a historical record of the (since-revoked)
validation; the affected image keeps its `report_url` for traceability.
