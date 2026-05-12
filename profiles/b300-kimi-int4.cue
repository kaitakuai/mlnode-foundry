// Profile: B300 Blackwell Ultra + Kimi-K2.6 INT4
//
// Composition: #BaseProfile constraint + B300 base + KIMI_INT4 base + leaf overrides.
// Cue unification (&) merges all four; conflicts on any field raise compiler errors.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

// Top-level field name matches filename with `-` → `_`.
// This avoids package-level unification conflicts between profiles.
// Python CLI reads file, derives key from filename.
b300_kimi_int4: #BaseProfile & bases.B300 & bases.KIMI_INT4 & {
	identity: {
		axes: {
			gpu:   "b300"
			model: "kimi"
			model_revision: "k2-6"
			quant: "int4"
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.20.0"
			rev:    1
		}
	}
	mode:         "kaitakuai-base"
	runner_patch: "b300-kimi"
	// hw_patches inherited from B300; no leaf-specific patches
	env: {
		// VLLM_USE_V1, RUNNER_TIMEOUT, WATCHER_GRACE — from B300
		// VLLM_USE_FLASHINFER_MOE_INT4, MOE_BACKEND, POC_BATCH_SIZE — from KIMI_INT4
		// Leaf-specific: none in spike
	}
	runtime_defaults: {
		// compilation_mode, cudagraph_mode — from KIMI_INT4
		tensor_parallel_size:   4
		gpu_memory_utilization: 0.85
		max_num_batched_tokens: 131072
	}
	description: "B300 Blackwell Ultra + Kimi-K2.6 INT4 (4×B300 SXM)"
	notes: """
		Why VLLM_USE_FLASHINFER_MOE_INT4=1: +138% throughput vs Marlin on Blackwell sm_100.
		Phase 2 skeleton — real hw_patches application wires in Phase 3.
		"""
	// Provenance for every non-base tuning knob, baked into the
	// gonka.kaitaku.tuning_notes OCI label.
	tuning_notes: [
		{
			knob:     "VLLM_USE_FLASHINFER_MOE_INT4=1"
			source:   "https://github.com/kaitakuai/experiments/2026-05/kimi_k26_b300_eager_flashinfer"
			reason:   "+138% PoC throughput vs Marlin on Blackwell sm_100 (5120 nonces/min @ 4×B300)"
			added_at: "2026-05-02"
		},
		{
			knob:     "max_num_batched_tokens=131072"
			source:   "https://github.com/kaitakuai/experiments/2026-05/kimi_k26_4xb200_b200-k5-kimi-1"
			reason:   "Vision-encoder profile_run OOMs above 131072 on Kimi-K2.6 (verified on B200; same envelope used for B300)"
			added_at: "2026-05-02"
		},
	]
}
