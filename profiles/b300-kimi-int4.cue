// Profile: B300 Blackwell Ultra + Kimi-K2.6 INT4
//
// Discovery spike profile. Inline (no _base/ composition yet) — Phase 2 will
// extract shared B300 / Kimi-INT4 fields into profiles/_base/*.cue.
//
// For the spike, BASE_IMAGE is upstream `product-science/mlnode:3.0.13-alpha5`
// directly (skip our Stage 2 mlnode-base). Real Stage 2 lands in Phase 3.
package profiles

profile: #BaseProfile & {
	identity: {
		axes: {
			gpu:   "b300"
			model: "kimi"
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
	hw_patches: []  // none for spike — Phase 3 wires real patches
	env: {
		VLLM_USE_FLASHINFER_MOE_INT4: "1"
		VLLM_FLASHINFER_MOE_BACKEND:  "latency"
		VLLM_USE_V1:                  "1"
		POC_BATCH_SIZE_DEFAULT:       "64"
	}
	runtime_defaults: {
		tensor_parallel_size:    4
		gpu_memory_utilization:  0.85
		max_num_batched_tokens:  131072
	}
	description: "B300 Blackwell Ultra + Kimi-K2.6 INT4 (4×B300 SXM)"
	notes: """
		Why VLLM_USE_FLASHINFER_MOE_INT4=1: +138% throughput vs Marlin on Blackwell sm_100.
		Discovery-spike profile — real hw_patches and Stage 2 wired in Phase 3.
		"""
}
