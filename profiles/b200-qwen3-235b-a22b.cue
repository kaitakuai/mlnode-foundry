// Profile: B200 Blackwell + Qwen3-235B-A22B FP8
//
// B200 (sm_100) needs the standard Blackwell compile-time fixes
// (Triton ptxas, FlashInfer JIT, libcuda compat, nvidia headers).
// For Qwen FP8 we keep TRITON MoE backend (Blackwell FlashInfer FP8 path
// has perf regressions on early driver builds — revisit per benchmark).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b200_qwen3_235b_a22b: #BaseProfile & bases.B200 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "b200"
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
	runner_patch: ""
	env: {}
	runtime_defaults: {
		gpu_memory_utilization: 0.85
	}
	description: "B200 Blackwell SXM + Qwen3-235B-A22B FP8 (TP=4)"
	notes: """
		Blackwell compile-time patches applied via B200 base.
		gpu_memory_utilization=0.85 (Blackwell HBM3e is faster — lower headroom acceptable).
		"""
}
