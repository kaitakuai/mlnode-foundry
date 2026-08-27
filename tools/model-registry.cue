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
	// checkpoint. Examples: "k2-6" (Kimi-K2.6), "235b-a22b" (Qwen3 235B
	// MoE variant; major version is in family="qwen3"), "m2-7" (MiniMax-M2.7).
	// Lowercase + digits + hyphens. NO dots — package names forbid them.
	// May start with a digit (revision goes into image name *after* the
	// family which provides context, e.g. mlnode-h100-qwen3-235b-a22b).
	revision: =~"^[a-z0-9][a-z0-9-]*$"

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
		revision:     "k2-6"
		display_name: "Kimi K2.6"
		hf_repo:      "moonshotai/Kimi-K2.6"
		params_b:     1060.0
		context_max:  262144
		license:      "MIT"
		notes:        "DeepseekV3-style MoE, 384 routed experts × top_k=8, vision tower. INT4 (compressed-tensors W4A16, group_size=32)."
	},
	{
		family:       "qwen3"
		revision:     "235b-a22b"
		display_name: "Qwen3 235B"
		hf_repo:      "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
		params_b:     235.0
		context_max:  262144
		license:      "Apache-2.0"
		notes:        "Qwen3 235B-parameter MoE with 22B active params per token. FP8 native checkpoint released 2026-07 (2507 in HF repo)."
	},
	{
		family:       "minimax"
		revision:     "m2-7"
		display_name: "MiniMax M2.7"
		hf_repo:      "MiniMaxAI/MiniMax-M2.7"
		// Pinned by the upstream chain governance model at v0.2.13 upgrade
		// (inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel).
		// Operators that drift from this commit will fail chain validation.
		hf_revision:  "d494266a4affc0d2995ba1fa35c8481cbd84294b"
		params_b:     230.0
		// Chain governance `--max-model-len 180000`. Larger contexts are truncated
		// (operators previously ran 131072 in PoC experiments; production cap = 180000).
		context_max:  180000
		license:      "Apache-2.0"
		status:       "active"
		notes:        "Activates on chain at epoch 271 (v0.2.13 upgrade). vLLM args mandated by chain: --enable-auto-tool-choice --kv-cache-dtype fp8 --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think. PoC seq_len=1024, ThroughputPerNonce=5000, VRam=320 GB total."
	},
	{
		family:       "glm"
		revision:     "5-3-flash"
		display_name: "GLM 5.3 Flash"
		hf_repo:      "zai-org/GLM-5.3-Flash"
		// PROVISIONAL pin = main HEAD (2026-08-27). No chain governance model for
		// this checkpoint yet, so this SHA has to be reconciled with the future
		// governance hf_revision before any production use — same rule as
		// deepseek/v4-flash below. Pinned explicitly so a silent upstream re-push
		// cannot change what our image resolves to.
		hf_revision: "04c4e9e95c5da8862dced7e5056455116f83a7e0"
		// Estimated from the checkpoint: 328 GiB of FP8 e4m3 weights is ~328B
		// parameters at one byte each. Replace with the published figure when
		// a governance registration states one.
		params_b:    328.0
		context_max: 1048576
		license:     "MIT"
		// The registry has no "experimental" state; this entry is active in the
		// sense that profiles may reference it, not that it is production-ready.
		status: "active"
		notes:  "Glm5NextForConditionalGeneration, model_type glm5_next — 45 layers, 288 experts × top-8, index_topk=2048, FP8 e4m3 dynamic (62 shards, 328 GiB), multimodal (image-text-to-text). num_nextn_predict_layers=1, i.e. MTP speculation is on by default: a replay MUST carry kaitakuai/vllm#21 or validation diverges from the executor (measured 82/100 length mismatches on Hy3 without it). NOT in any upstream vLLM release — support lives in the open PR vllm-project/vllm#53906, which touches .cu and cmake, so the kernels come pre-built from vllm/vllm-openai:glm53-flash. Chain has no model record yet. See PORTING.md on kaitakuai/vllm@poc-residual-glm53."
	},
	{
		family:       "glm"
		revision:     "5-2"
		display_name: "GLM 5.2"
		hf_repo:      "zai-org/GLM-5.2-FP8"
		hf_revision:  "31cba24fb749908a485082bdeed6eb1ac6cffc2f"
		params_b:     753.0
		// Native context; the b200 profile caps max_model_len at 350000 (warning).
		context_max:  1048576
		license:      "MIT"
		status:       "active"
		notes:        "GlmMoeDsaForCausalLM (verbatim DeepseekV2 subclass) — DeepSeek-style MLA + DSA sparse attention (index_topk=2048), 753B total / ~40B active, 256 experts × top-8, 78 layers, FP8 block-wise e4m3 [128,128]. DeepGEMM split on vLLM 0.23: MoE experts run on DeepGEMM (throughput lever) but the block-FP8 LINEAR kernel must be routed to Cutlass via VLLM_DISABLED_KERNELS — GLM's fused_qkv_a_proj N=2624 (N%128==64 partial block) + E8M0 requant crashes the DeepGEMM linear kernel (cudaErrorInvalidValue) at memory profiling. MoE workspace lock handled by gonka-poc df73e1c. See b200-glm-5-2 profile + experiments/2026-06/glm-5.2-fp8-8xb200."
	},
	{
		family:       "deepseek"
		revision:     "v4-flash"
		display_name: "DeepSeek V4 Flash"
		hf_repo:      "deepseek-ai/DeepSeek-V4-Flash"
		// PROVISIONAL pin = current main HEAD (2026-07-20). V4-Flash has no
		// chain governance model yet (gonka-ai/gonka#1408 is still discussion).
		// This SHA MUST be reconciled with — and match — the future governance
		// hf_revision before any production/chain use, or operators fail chain
		// validation (see minimax hf_revision precedent). Kept explicit so a
		// silent upstream re-push does not change what our image resolves to.
		hf_revision:  "60d8d70770c6776ff598c94bb586a859a38244f1"
		params_b:     291.0
		context_max:  1048576
		license:      "MIT"
		status:       "active"
		notes:        "DeepseekV4ForCausalLM, model_type deepseek_v4 — ~291B total / 149 GiB FP8 weights, 43 layers, 256 experts × top-6, MXFP4 experts (group-32 ue8m0) + FP8-block linears with NATIVE ue8m0 scales (unlike GLM, no float32-requant DeepGEMM crash), 3 hash-MoE layers (route via tid2eid[input_ids] — PoC needs deterministic pseudo input_ids, gonka-poc#14), sparse MLA index_topk=512, SWA compressor block=8. Requires --kv-cache-dtype fp8 (FlashMLA fp8_ds_mla layout; assert otherwise). Attention backend deterministic FlashMLA-DSV4 (do NOT pin — FlashInfer-DSV4 has placeholder FP8 scales, consensus-unsafe). vLLM auto-forces CompilationMode.NONE + breakable-cudagraph. sm_80 (A100) and sm_120 (RTX PRO 6000) statically excluded — both sparse backends need sm major in [9,10]. FIRST-CLASS support in vLLM 0.25.1 (0.23 works via gonka-poc#14). See .work/deepseek-v4-flash/ + project_deepseek_v4_flash_plan."
	},
	{
		family:       "deepseek"
		revision:     "v4-flash-0731"
		display_name: "DeepSeek V4 Flash 0731"
		hf_repo:      "deepseek-ai/DeepSeek-V4-Flash-0731"
		// Pinned to the revision the gonka-ai release configs use
		// (deploy/join/node-config-deepseekv4flash0731-*.json, gonka#1536).
		// NOTE: DeepSeek re-pushed the checkpoint on 2026-08-01 03:07, ~15h
		// after release. 9e165c30 (2026-07-31 12:02) was HEAD when our first
		// campaign was pinned and is now stale -- artifacts produced against
		// it will not match the fleet. Consensus makes this a hard pin, not a
		// preference.
		hf_revision:  "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
		params_b:     291.0
		context_max:  1048576
		license:      "MIT"
		status:       "active"
		notes:        "Weight refresh of v4-flash: same DeepseekV4ForCausalLM architecture, 43 layers, MXFP4+FP8 mix; adds dspark_* speculator config (V2-runner-only, optional). PoC throughput identical to v4-flash on every tested topology (1xB300, 2xB200, 2xH200, 4xH100 — experiments 2026-08). Same operational constraints as v4-flash: --kv-cache-dtype fp8 mandatory, backend NOT pinned, sm in [9,10]. Release config adds --tokenizer-mode deepseek_v4, tool-call and reasoning parsers (gonka#1536)."
	},
]
