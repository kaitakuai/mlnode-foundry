// Package tools — model registry.
//
// Single source of truth for "what (model, model_revision) tuples are valid".
// The Python validator cross-checks each profile's identity.axes.model +
// identity.axes.model_revision against this list before building.
//
// Adding a new model variant:
//   1. Add an entry to `models` below with HF coordinates + license.
//   2. Reference it from a profile via identity.axes.model + model_revision.
//   3. `mlnode-foundry validate <profile>` will succeed.
//
// Deprecating a model:
//   - Set `status: "deprecated"` and `eol_date: "YYYY-MM-DD"`.
//   - Profiles already in production keep building; new profiles get a warning.
//
// Cue cannot enforce cross-package referential integrity at `cue vet` time
// (profiles package vs tools package). Enforcement lives in
// mlnode_foundry/validate.py:validate_model_revision().

package tools

#ModelEntry: {
	// Model family — matches profile.identity.axes.model (kimi, qwen, deepseek, ...).
	// Lowercase, alphanumeric, starts with a letter. NO hyphens (use revision for sub-variants).
	family: =~"^[a-z][a-z0-9]+$"

	// Specific revision within the family. Identifies a unique HuggingFace
	// checkpoint. Examples: "k26" (Kimi-K2.6), "v3-235b" (Qwen3 235B variant).
	// Lowercase + digits + hyphens. NO dots — package names forbid them.
	revision: =~"^[a-z][a-z0-9-]*$"

	// Human-readable name shown in the dashboard. Operators look at this,
	// not the family/revision codes. Keep it accurate to the model spec.
	display_name: string

	// HuggingFace repository ID (org/name). Operators paste this into the
	// vLLM --model flag; it MUST be pullable as-is.
	hf_repo: string

	// Optional pinned git SHA / branch / tag at the HF repo. Use when a
	// specific weights revision is required (uncommon — most prod runs
	// pull the default branch HEAD).
	hf_revision?: string

	// Parameter count in billions (e.g., 1060.0 for Kimi-K2.6, 235.0 for Qwen3-235B).
	// Used for capacity planning and dashboard summary.
	params_b: number & >0

	// Native context window the model supports. Profile.runtime_defaults
	// MAY cut max_model_len below this; if so, a tuning_note with
	// severity=warning is expected.
	context_max: int & >0

	// SPDX-style license identifier (MIT, Apache-2.0, ...). Surfaces in
	// dashboard for compliance review.
	license: string

	// Lifecycle:
	//   - "active":     supported, recommended for new profiles.
	//   - "deprecated": existing profiles keep building; validator warns
	//                   when a new profile references it.
	status: *"active" | "deprecated"

	// End-of-life date for deprecated entries. ISO YYYY-MM-DD. Operators
	// should plan migration off the model before this date.
	eol_date?: =~"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"

	// Free-form notes — architecture, quantization details, known quirks.
	// Shown verbatim in dashboard expanded view.
	notes?: string
}

models: [...#ModelEntry]
models: [
	{
		family:       "kimi"
		revision:     "k26"
		display_name: "Moonshot Kimi-K2.6"
		hf_repo:      "moonshotai/Kimi-K2.6"
		params_b:     1060.0
		context_max:  262144
		license:      "MIT"
		notes:        "DeepseekV3-style MoE, 384 routed experts × top_k=8, vision tower. INT4 (compressed-tensors W4A16, group_size=32)."
	},
	{
		family:       "qwen"
		revision:     "v3-235b"
		display_name: "Qwen3-235B-A22B-Instruct FP8"
		hf_repo:      "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
		params_b:     235.0
		context_max:  262144
		license:      "Apache-2.0"
		notes:        "Qwen3 235B with 22B active params (MoE). FP8 quantized native checkpoint."
	},
]
