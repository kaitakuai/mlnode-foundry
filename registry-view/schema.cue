// Package registry_view — formal shape of the dashboard-facing image entry.
//
// One registry-view/<package-basename>-<tag>.json file per published Stage 3
// image, written by CI (mlnode_foundry.render_registry_view). Validated via
// `cue vet registry-view/<x>.json registry-view/schema.cue` so any
// drift between the Python renderer and dashboard contract fails in CI.
//
// Field-level comments document the SEMANTIC contract for any human or AI
// agent that writes / edits these values (e.g., poc-benchmark agent that
// later fills `nonces` after Tier 3 measurement on real hardware).

package registry_view

#RegistryView: {
	// Self-link to this schema; standard JSON Schema convention, ignored by
	// validators but useful for IDE autocompletion when humans inspect the file.
	"$schema": *"./schema.json" | string

	// Schema version bumped on breaking shape changes. Dashboard / consumers
	// MUST check this before parsing; refuse to render unknown versions.
	schema_version: int & >=1

	// Image line — high-level grouping for dashboard filtering.
	//   - "mlnode":         Stage 3 image built on our kaitakuai/mlnode-base.
	//   - "mlnode-overlay": Stage 3 built directly on product-science/mlnode upstream binary.
	// "vllm" line is reserved for future Stage 1 publishes.
	line: "mlnode" | "mlnode-overlay" | "vllm"

	// Identity axes (mirrors profile.identity.axes). Each value MUST satisfy
	// the same regex as in profiles/schema.cue::#Identity to stay consistent.
	gpu:            =~"^[a-z][a-z0-9]+$"
	model_family:   =~"^[a-z][a-z0-9]+$"
	model_revision: =~"^[a-z0-9][a-z0-9-]*$"
	quant:          null | "int4" | "fp8" | "nvfp4" | "awq4bit" | "mxfp4"

	// Full GHCR package path (no tag). MUST match the package name policy
	// from tools/naming.cue. Old `mlnode-full`/`mlnode-overlay` legacy names
	// are NOT valid here — those live in archived registry from kaitakuai/mlnode.
	name: =~"^ghcr\\.io/kaitakuai/mlnode-[a-z0-9-]+$"

	// Image tag (no leading colon). Computed deterministically by render_name_tag
	// from profile.identity.version + axes. MUST NOT be 'latest' or any mutable
	// alias (immutable-tag policy).
	tag: string & !="latest"

	// Kaitaku revision counter, monotonically increasing per content change at
	// the same (axes, mlnode_version, vllm_version) tuple. Matches profile.identity.version.rev.
	k_rev: int & >=1

	// Mode-specific upstream versions. Exactly one of these MUST be set non-null:
	//   - kaitakuai-base   mode → vllm_base_version + mlnode_version
	//   - upstream-overlay mode → upstream_overlay_version only
	vllm_base_version:        null | string
	mlnode_version:           null | string
	upstream_overlay_version: null | string

	// Model metadata (from tools/model-registry.cue). `model` is the HuggingFace
	// repo path that operators actually pull; `model_short` is the dashboard label.
	model:             string
	model_short:       string
	model_params_b:    null | (number & >0)  // parameter count in billions
	model_context_max: null | (int & >0)     // native context window
	model_license:     null | string         // SPDX-style identifier when available

	// CUDA toolkit baked into Stage 1 base. Surfaced for compatibility hints.
	// Null if unknown (legacy entries before tools/stage2.lock.cue had a cuda field).
	cuda: null | =~"^[0-9]+\\.[0-9]+$"

	// Humanized compressed image size for linux/amd64 platform (e.g. "15 GB").
	// Filled by mlnode-foundry image-size <ref>. Null if buildx inspect failed —
	// dashboard should render '—' in that case rather than '0 GB'.
	size: null | =~"^[0-9]+(\\.[0-9]+)? (B|KB|MB|GB)$"

	// Suggested local `docker run` line. Convention only — operators may override.
	runtime: =~"^docker run "

	// Effective flags = profile.env + profile.runtime_defaults, flattened to
	// "key=value" strings (sorted, deterministic). Each entry should appear
	// either in flag_descriptions or flag_warnings (or both) when context exists.
	flags: [...string]

	// flag → reason for any tuning_note with severity=warning. Dashboard renders
	// these with a triangle/caution icon. Keep messages short and operator-focused.
	flag_warnings: [string]: string

	// flag → reason for any tuning_note. Dashboard renders as a neutral tooltip.
	// Every entry in flag_warnings MUST also exist here (warning is a superset).
	flag_descriptions: [string]: string

	// Image digest (sha256:...) — content-addressable identifier.
	digest: =~"^sha256:[a-f0-9]{64}$"

	// Cosign keyless signing identity (the GHA workflow URL that signed the image).
	// Verifiable by anyone via `cosign verify --certificate-identity-regexp`.
	cosign_identity: =~"^https://github\\.com/kaitakuai/mlnode-foundry/.*"

	// SLSA L3 attestation URL — points to the GitHub attestation index for this digest.
	slsa_attestation_url: =~"^https://github\\.com/kaitakuai/mlnode-foundry/attestations"

	// Link to the experiments runbook that justifies the tuning choices, if any.
	// Picked from the first tuning_notes[].source that points at an experiments URL.
	// Null when the profile carries no tuning_notes with a URL source.
	report_url: null | =~"^https?://"

	// Git author of the last commit touching profiles/<x>.cue. Surfaced so operators
	// know who to ask about tuning choices. Auto-derived from `git log -1`.
	uploaded_by: null | string

	// Tier 3 benchmark results, filled by the poc-benchmark agent in a separate PR
	// AFTER the image is validated on real hardware. MUST remain null until then —
	// dashboard distinguishes "draft" (null) from "benchmarked" (number).
	nonces: null | (int & >=0)
	weight: null | (number & >=0)
}
