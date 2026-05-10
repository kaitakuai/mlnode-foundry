// Package state — schema for observed state files written by CI / poc-benchmark agent.
//
// state/<package-tag>.json files are machine-written JSON, validated against
// this schema via `cue vet state/<x>.json state/schema.cue` (Cue natively
// validates JSON files against Cue schemas).
//
// See ADR-0011 (spec/state/policy separation).
package state

#ValidationResult: {
	result: "pass" | "fail" | "pending"
	at?:    string  // ISO 8601 date
	...               // additional fields allowed (commit, instance, etc.)
}

#State: {
	// Reference to the profile that produced this image
	profile:      string & =~"^profiles/.+\\.cue$"
	profile_hash: =~"^[a-f0-9]{64}$"

	// Lifecycle (see ADR-0009)
	status: "draft" | "validated" | "benchmarked" | "deprecated"

	// Image identity in registry
	image: {
		package:         =~"^ghcr\\.io/"
		tag:             string
		digest:          =~"^sha256:[a-f0-9]{64}$"
		built_at:        string  // ISO 8601 timestamp
		cosign_identity: =~"^https://github\\.com/"
	}

	// Per-tier validation results
	validation: [string]: #ValidationResult

	// Metrics from Tier 3 benchmark (filled by poc-benchmark agent)
	metrics?: {
		expected_nonces_per_min?: int & >0
		measured_at?:             string
		cost_estimate_usd?:       number & >=0
		benchmark_report?:        string  // path within experiments repo
	}
}
