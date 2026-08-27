// Package profiles defines the schema for build profiles.
//
// A profile is a *pure spec* (intent written by humans) — it declares what
// image to build for a (gpu, model[, quant]) target. Observed state (build
// results, validation, metrics) lives in `state/<x>.json`, not here.
//
// Profile is a discriminated union by `mode` — it selects WHERE the Stage 4
// build gets its base, i.e. how much of the stack we build ourselves:
//
//   - kaitakuai-base   → FROM kaitakuai/mlnode-base, our own Stage 3, pinned
//                        in tools/stage3.lock.cue. Use while iterating on the
//                        mlnode source or on local patches/: we control every
//                        layer and can rebuild in minutes.
//   - upstream-overlay → FROM a published binary image pinned by digest in
//                        `base`. Two flavours in practice, distinguished only
//                        by what the digest points at:
//                          * gonka-ai/mlnode (the release image cortima ships,
//                            their Stage-3 equivalent) — preferred once a
//                            release exists, because it removes a whole build
//                            stage and makes drifting from their mlnode
//                            impossible;
//                          * an older product-science/mlnode binary — the
//                            original reason this mode exists.
//
// Rule of thumb: test-and-iterate with kaitakuai-base, ship on top of the
// published image with upstream-overlay. Whichever is chosen, Stage 4 still
// applies our hw-patches, runner patch and ENV, so the two modes differ in
// base provenance only — not in what we add.
//
// Cue's sum type catches mismatched fields at validation time.
//
// Field-level comments below are the contract for any human or AI agent
// authoring or editing a profile: they explain WHAT each field carries,
// what values are legal, and WHY constraints exist.
package profiles

#Identity: {
	axes: {
		// GPU class identifier — lowercase, ascii alphanumeric, starts with a letter.
		// Examples: h100, h200, a100, b200, b300, rtx6000.
		// Used in package name (mlnode-<gpu>-...) and OCI label gonka.kaitaku.gpu.
		gpu: =~"^[a-z][a-z0-9]+$"

		// Model family — top-level grouping by vendor/architecture (kimi, qwen, minimax, ...).
		// Cross-checked together with model_revision against tools/model-registry.cue
		// by mlnode_foundry.validate.validate_model_revision.
		model: =~"^[a-z][a-z0-9]+$"

		// Specific revision within the model family. Disambiguates Kimi-K2.5 vs K2.6,
		// Qwen3-235B vs Qwen3-72B, etc. Lowercase + digits + hyphens; NO dots
		// (would break GHCR package names which require [a-z0-9_-]).
		// May start with a digit (Qwen3 → "235b-a22b") since the family name
		// (e.g., "qwen3") carries the architecture-generation context.
		// Required since the model_revision PR. Cross-checked against the registry.
		model_revision: =~"^[a-z0-9][a-z0-9-]*$"

		// Quantization scheme. Omit when the deployment uses the model's native
		// precision (typically FP8 for production); set when an explicit quant
		// is loaded (int4 for Kimi MoE, nvfp4 experiments, etc.).
		quant?: "int4" | "fp8" | "nvfp4" | "awq4bit" | "mxfp4"

		// Inference engine. Defaults to "vllm" (omitted from tag when default).
		// Add a value only when porting to sglang/trtllm; profile_hash will
		// produce a distinct image automatically.
		framework?: "vllm" | "sglang" | "trtllm"

		// Stage 5 post-build transform (reserved for ADR-0010 minimization work).
		// "full" = untouched image; "slim" = minimized via Stage 5 transform.
		// Today only "full" is implemented.
		transform?: "full" | "slim"
	}
	version: {
		// Kaitaku revision counter for the SAME (axes, upstream-versions) tuple.
		// Bumped monotonically on every content change (e.g., new tuning_notes,
		// patched hw_patches). Old k-revs stay pullable as historical record.
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
	// Exact "key=value" string identifying the knob. MUST match a string in
	// flags[] of the rendered registry-view; renderer maps via this field.
	knob: string

	// Where the decision is justified. Prefer in this order:
	//   1. URL to experiments runbook (https://github.com/kaitakuai/experiments/...)
	//   2. URL to a PR / decision-log entry
	//   3. Free-form description (last resort; provides no audit trail)
	source: string

	// One-line operator-facing rationale. Appears in dashboard tooltip
	// (flag_descriptions) AND, when severity=warning, in flag_warnings.
	// Keep it short and concrete — numbers and trade-offs, not platitudes.
	reason: string

	// Dashboard rendering signal:
	//   - "info":    neutral tooltip; default for performance optimizations.
	//   - "warning": triangle/caution icon; use when the knob represents a
	//                constraint operators MUST notice (max_model_len cut
	//                below model native context; max_num_batched_tokens
	//                below baseline that caps effective batch; quality
	//                regression in some configs; etc.).
	severity: *"info" | "warning"

	// Date (UTC) the note was added. ISO YYYY-MM-DD. Useful for spotting
	// stale tuning that may need re-verification on newer base images.
	added_at: =~"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
}

// Common shape declared once; mode-specific narrowing applied below.
// `mode` and optional `base` declared here so closed-struct semantics
// allow #BaseProfile / #OverlayProfile to narrow them via &.
#CommonProfile: {
	// Discriminator for the Cue sum type. See #BaseProfile / #OverlayProfile.
	mode: "kaitakuai-base" | "upstream-overlay" | "upstream-test"

	// See #Identity.
	identity: #Identity

	// Ordered list of tools/hw-patches/<x>.dockerfile basenames to apply on
	// top of the base image. Each entry MUST correspond to an existing file
	// in tools/hw-patches/. Patches are applied in declared order; idempotency
	// is the patch author's responsibility (re-apply must not error).
	hw_patches: [...string]

	// Basename of tools/runner-patches/<x>.py to inject into the built image
	// at /tmp/runner-patch.py and execute once. Empty string = no runner patch.
	// Use for runtime-flag mutations that hw_patches dockerfile fragments
	// cannot express (e.g., hardcoding vLLM TP/MOE flags inside mlnode runner.py).
	runner_patch: string

	// Environment variables set on the final image (ENV K=V in Dockerfile).
	// Rendered into Dockerfile.tmpl as one ENV directive per key, sorted.
	// Profile.env composes via Cue & with base profiles (B300, KIMI_INT4, ...)
	// — leaves override base values when bases use *default | type syntax.
	env: [string]: string

	// Default runtime knobs the dashboard surfaces (tensor_parallel_size,
	// gpu_memory_utilization, max_num_batched_tokens, max_model_len, ...).
	// Aggregated with `env` into flags[] in registry-view.
	runtime_defaults: {...}

	// Checkpoint the registry card advertises, when it differs from the one
	// `identity.axes` names. Only case today: b300-deepseek serves the NVFP4
	// conversion by network decision while the axes (and the chain) name the
	// official release. Absent = the model registry entry is used.
	advertised_model?: {
		model:       =~"^[^/]+/[^/]+$"
		model_short: string
	}

	// Operator-facing one-line summary. Shown in dashboard image card.
	description: string

	// Optional multi-line notes — design rationale, gotchas, links to PoC reports.
	// Shown in dashboard expanded view.
	notes?: string

	// Provenance for non-base tuning knobs. See #TuningNote.
	// Strongly recommended for profiles that override base env / runtime_defaults
	// or use hw_patches outside what the GPU base provides.
	tuning_notes?: [...#TuningNote]

	// Required ONLY for mode="upstream-overlay" — see #OverlayProfile narrowing.
	base?: {
		image:            =~"^ghcr\\.io/"
		digest:           =~"^sha256:[a-f0-9]{64}$"
		upstream_version: string
	}
}

// Profile mode A: Stage 4 builds on top of our `mlnode-base` (Stage 3).
// Requires mlnode + vllm versions in identity.version.
#BaseProfile: #CommonProfile & {
	mode: "kaitakuai-base"
	identity: version: {
		// Gonka mlnode source version (matches upstream commit in tools/stage3.lock.cue).
		mlnode: =~"^[0-9]+\\.[0-9]+\\.[0-9]+"
		// vLLM major.minor.patch baked into Stage 2 (kaitakuai/vllm-poc).
		vllm: =~"^[0-9]+\\.[0-9]+\\.[0-9]+"
		// Kaitaku k-rev counter (see #Identity.version.rev).
		rev: int & >=1
	}
}

// Profile mode B: Stage 4 builds on top of a PUBLISHED binary image, pinned
// by digest — either cortima's release mlnode (gonka-ai/mlnode) or a legacy
// product-science/mlnode. Requires explicit base.image + base.digest so a
// rebuild is reproducible even if the publisher re-pushes the tag.
#OverlayProfile: #CommonProfile & {
	mode: "upstream-overlay" | "upstream-test"
	identity: version: {
		// Upstream product-science/mlnode version tag.
		upstream: string
		rev:      int & >=1
	}
	base: {
		// Full published image package path (no tag). Typically
		// ghcr.io/gonka-ai/mlnode (release) or ghcr.io/kaitakuai/mlnode-base
		// when pinning one of our own Stage-3 builds explicitly.
		image: =~"^ghcr\\.io/"
		// Immutable upstream digest. Pinning here, not the tag, ensures
		// our overlay rebuild is reproducible even if upstream re-pushes.
		digest: =~"^sha256:[a-f0-9]{64}$"
		// Upstream version string for OCI label + dashboard.
		upstream_version: string
	}
}

// Discriminated union — Cue compiler discriminates by `mode`.
#Profile: #BaseProfile | #OverlayProfile
