// Package state — schema for observed state files written by CI / poc-benchmark agent.
//
// state/<package-tag>.json files are machine-written JSON, validated against
// this schema via `cue vet state/<x>.json state/schema.cue` (Cue natively
// validates JSON files against Cue schemas).
//
// See ADR-0011 (spec/state/policy separation).
//
// Field-level comments below document the SEMANTIC contract for any human
// or AI agent that writes / edits state files (including the poc-benchmark
// agent that fills .metrics after Tier 3 measurement).
package state

#ValidationResult: {
	// Outcome of one validation tier.
	//   - "pass":    completed successfully.
	//   - "fail":    completed with failure; check the linked report for details.
	//   - "pending": started but not yet concluded (rare; mostly used to mark
	//                long-running Tier 3 work in progress between PRs).
	result: "pass" | "fail" | "pending"

	// ISO 8601 timestamp of when the validation concluded (or started, for "pending").
	at?: string

	// Additional context (commit SHA, vast.ai instance id, ...) — schema is open.
	...
}

#State: {
	// Path (relative to repo root) of the profile that produced this image.
	// MUST start with `profiles/` and end with `.cue`. Cross-reference for any
	// consumer that wants to inspect the source spec.
	profile: string & =~"^profiles/.+\\.cue$"

	// Content-addressable profile hash (matches gonka.kaitaku.profile_hash OCI label).
	// Used by skip-if-unchanged in CI: same hash → existing image is reused.
	profile_hash: =~"^[a-f0-9]{64}$"

	// Lifecycle stage (see ADR-0009):
	//   - "draft":      build smoke passed, no hardware validation yet.
	//                   DO NOT route production traffic to this tag.
	//   - "validated":  manual GPU smoke on real hardware passed.
	//   - "benchmarked": Tier 3 PoC nonces/min measured; .metrics populated.
	//   - "deprecated": superseded; new deployments should pick a newer tag.
	status: "draft" | "validated" | "benchmarked" | "deprecated"

	// Image identity in registry — the coordinates of the published artifact.
	image: {
		// Full GHCR package path (no tag).
		package: =~"^ghcr\\.io/"
		// Tag the CI assigned. NOT 'latest' (immutable tag policy).
		tag: string
		// Immutable sha256 digest. Cosign and SLSA attestations reference this.
		digest: =~"^sha256:[a-f0-9]{64}$"
		// ISO 8601 build timestamp.
		built_at: string
		// Cosign keyless signing identity (GHA workflow URL).
		cosign_identity: =~"^https://github\\.com/"
	}

	// Per-tier validation results. Tier names are free-form so new tiers can
	// be added without schema bumps, but conventional keys are:
	//   - "build_smoke":  Stage 3 RUN steps completed without error.
	//   - "tier1_local":  local single-prompt smoke on developer hardware.
	//   - "tier2_remote": remote single-instance soak (1-hour) on target GPU.
	//   - "tier3_pocv2":  full PoC v2 nonce generation benchmark.
	validation: [string]: #ValidationResult

	// Metrics from Tier 3 benchmark. Filled by the poc-benchmark agent in
	// a separate PR after measurement on real hardware. The agent MUST NOT
	// touch other fields — keep concerns separated.
	metrics?: {
		// PoC v2 throughput in nonces per minute (integer, > 0).
		// Used by tokenomics model to compute expected token weight.
		expected_nonces_per_min?: int & >0

		// ISO 8601 timestamp when the measurement was taken.
		measured_at?: string

		// Approximate USD cost per nonces/min slot at the GPU rental rate
		// observed during measurement. Informational for economic comparisons.
		cost_estimate_usd?: number & >=0

		// Path (within kaitakuai/experiments repo) to the full benchmark report.
		// Format: 2026-MM/<experiment-name>/README.md
		benchmark_report?: string
	}
}
