// Profile: A100 Ampere + Qwen3-235B-A22B FP8
//
// Ampere (sm_80) is fully supported by upstream CUDA/Triton — no patches.
// FP8 quant runs through Marlin kernels (no native FP8 tensor cores on A100).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

a100_qwen: #BaseProfile & bases.A100 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "a100"
			model: "qwen"
			model_revision: "v3-235b"
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.20.0"
			rev:    1
		}
	}
	mode:         "kaitakuai-base"
	runner_patch: ""
	env: {}
	runtime_defaults: {
		gpu_memory_utilization: 0.90
	}
	description: "A100 Ampere SXM + Qwen3-235B-A22B FP8 (TP=4)"
	notes: """
		Ampere FP8 via Marlin (no native FP8 tensor cores). TRITON MoE backend.
		Throughput materially lower than Hopper — kept for fallback availability.
		"""
}
