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
	family:       =~"^[a-z][a-z0-9]+$"
	revision:     =~"^[a-z][a-z0-9-]*$"
	display_name: string
	hf_repo:      string
	hf_revision?: string  // optional pinned git SHA / branch / tag
	params_b:     number  // billions of parameters
	context_max:  int     // native context window
	license:      string
	status:       *"active" | "deprecated"
	eol_date?:    =~"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
	notes?:       string
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
