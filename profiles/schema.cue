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
		gpu:        =~"^[a-z][a-z0-9]+$"
		model:      =~"^[a-z][a-z0-9]+$"
		quant?:     "int4" | "fp8" | "nvfp4" | "awq4bit" | "mxfp4"
		framework?: "vllm" | "sglang" | "trtllm"
		transform?: "full" | "slim"
	}
	version: {
		rev: int & >=1
		...
	}
}

// Common shape declared once; mode-specific narrowing applied below.
// `mode` and optional `base` declared here so closed-struct semantics
// allow #BaseProfile / #OverlayProfile to narrow them via &.
#CommonProfile: {
	mode:         "kaitakuai-base" | "upstream-overlay"
	identity:     #Identity
	hw_patches:   [...string]
	runner_patch: string
	env: [string]: string
	runtime_defaults: {...}
	description: string
	notes?:      string
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
