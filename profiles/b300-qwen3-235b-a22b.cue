// Profile: B300 Blackwell Ultra + Qwen3-235B-A22B FP8
//
// Single-B300 Qwen baseline: TP=1 forced by b300 runner-patch
// (the network node broadcasts H100-tuned vLLM flags that do not fit
// B300's 275 GiB/GPU topology at TP=1, so the runner patch hardcodes
// gpu_memory_utilization, max_model_len, logprobs_mode).
// Blackwell FP8 path enabled via VLLM_USE_FLASHINFER_MOE_FP8=1 (overrides QWEN base).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b300_qwen3_235b_a22b: #BaseProfile & bases.B300 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "b300"
			model: "qwen3"
			model_revision: "235b-a22b"
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.20.0"
			rev:    1
		}
	}
	mode:         "kaitakuai-base"
	runner_patch: "b300"
	env: {
		// Overrides QWEN base default of "0" — Blackwell FP8 MoE backend.
		VLLM_USE_FLASHINFER_MOE_FP8: "1"
		VLLM_FLASHINFER_MOE_BACKEND: "latency"
	}
	runtime_defaults: {
		tensor_parallel_size:   1
		gpu_memory_utilization: 0.85
	}
	description: "B300 Blackwell Ultra (1×B300) + Qwen3-235B-A22B FP8"
	notes: """
		Single-GPU profile (TP=1). The b300 runner-patch hardcodes runner.py
		flags overriding the network node broadcast; see tools/runner-patches/b300.py.
		Memory: 832 nonces/min baseline on vLLM 0.19; revalidate on vLLM 0.20 base.
		"""
}
