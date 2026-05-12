// Profile: B200 Blackwell + Kimi-K2.6 INT4
//
// Blackwell FP4/INT4 path: FlashInfer MoE INT4 backend (+138% throughput
// vs Marlin on sm_100). Kimi-K2.6 has slow cold-start with INT4 — add
// cold-start-tolerance hw-patch on top of B200 base.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b200_kimi_k2_6_int4: #BaseProfile & bases.B200 & bases.KIMI_INT4 & {
	identity: {
		axes: {
			gpu:   "b200"
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
	mode: "kaitakuai-base"
	// Extend B200 base with cold-start tolerance for slow Kimi INT4 init.
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
	]
	runner_patch: ""
	env: {
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		tensor_parallel_size:   4
		gpu_memory_utilization: 0.85
		max_num_batched_tokens: 131072
	}
	description: "B200 Blackwell SXM + Kimi-K2.6 INT4 (TP=4)"
	notes: """
		Same hw-patches as B300 (Blackwell family compile fixes + slow-init grace).
		FlashInfer MoE INT4 backend inherited from KIMI_INT4 base.
		"""
}
