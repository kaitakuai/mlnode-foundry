// Package profiles defines the schema for build profiles.
//
// A profile is a *pure spec* (intent written by humans) — it declares what
// image to build for a (gpu, model[, quant]) target. Observed state (build
// results, validation, metrics) lives in `state/<x>.json`, not here.
//
// Profile is a discriminated union by `mode`:
//   - kaitakuai-base   → built FROM kaitakuai/mlnode-base (our Stage 2)
//   - upstream-overlay → built FROM product-science/mlnode binary directly
//
// Cue's sum type catches mismatched fields at validation time.
package profiles

#Identity: {
	axes: {
		gpu:            =~"^[a-z][a-z0-9]+$"
		model:          =~"^[a-z][a-z0-9]+$"
		// Required since model_revision PR. Disambiguates Kimi-K2.5 vs K2.6,
		// Qwen3-235B vs Qwen3-72B, etc. Cross-checked against tools/model-registry.cue
		// by the Python validator (Cue cannot import across packages at vet time).
		model_revision: =~"^[a-z][a-z0-9-]*$"
		quant?:         "int4" | "fp8" | "nvfp4" | "awq4bit" | "mxfp4"
		framework?:     "vllm" | "sglang" | "trtllm"
		transform?:     "full" | "slim"
	}
	version: {
		rev: int & >=1
		...
	}
}

// Provenance entry for one tuning knob discovered post-benchmark.
// Profiles SHOULD attach a TuningNote whenever they set a non-default env or
// runtime_defaults value that came from empirical tuning (not from a base).
// The CI bakes the array as compact JSON into the gonka.kaitaku.tuning_notes
// OCI label so it travels with the image.
#TuningNote: {
	knob:     string                 // e.g., "VLLM_ATTENTION_BACKEND=FLASHINFER"
	source:   string                 // experiment id, PR link, or decision-log entry
	reason:   string                 // one-line why
	added_at: =~"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
}

// Common shape declared once; mode-specific narrowing applied below.
// `mode` and optional `base` declared here so closed-struct semantics
// allow #BaseProfile / #OverlayProfile to narrow them via &.
#CommonProfile: {
	mode:          "kaitakuai-base" | "upstream-overlay"
	identity:      #Identity
	hw_patches:    [...string]
	runner_patch:  string
	env: [string]: string
	runtime_defaults: {...}
	description:   string
	notes?:        string
	tuning_notes?: [...#TuningNote]
	base?: {
		image:            =~"^ghcr\\.io/"
		digest:           =~"^sha256:[a-f0-9]{64}$"
		upstream_version: string
	}
}

// Profile mode A: Stage 3 builds on top of our `mlnode-base` (Stage 2).
// Requires mlnode + vllm versions in identity.version.
#BaseProfile: #CommonProfile & {
	mode: "kaitakuai-base"
	identity: version: {
		mlnode: =~"^[0-9]+\\.[0-9]+\\.[0-9]+"
		vllm:   =~"^[0-9]+\\.[0-9]+\\.[0-9]+"
		rev:    int & >=1
	}
}

// Profile mode B: Stage 3 builds on top of an upstream binary image
// (product-science/mlnode). Requires explicit base.image + base.digest.
#OverlayProfile: #CommonProfile & {
	mode: "upstream-overlay"
	identity: version: {
		upstream: string
		rev:      int & >=1
	}
	base: {
		image:            =~"^ghcr\\.io/"
		digest:           =~"^sha256:[a-f0-9]{64}$"
		upstream_version: string
	}
}

// Discriminated union — Cue compiler discriminates by `mode`.
#Profile: #BaseProfile | #OverlayProfile
