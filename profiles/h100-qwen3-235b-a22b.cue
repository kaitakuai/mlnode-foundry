// Profile: H100 Hopper + Qwen3-235B-A22B FP8
//
// Minimal baseline — H100 needs no SM-specific patches; Qwen FP8 uses
// TRITON MoE backend per ADR.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

// Top-level field name matches filename with `-` → `_`.
h100_qwen3_235b_a22b: #BaseProfile & bases.H100 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "h100"
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
	hw_patches:  list.Concat([bases.H100.hw_patches, ["poc-householder-compile"]])
	runner_patch: "h100"
	env: {}
	runtime_defaults: {
		gpu_memory_utilization: 0.90
	}
	description: "H100 Hopper SXM + Qwen3-235B-A22B FP8 (TP=4)"
	notes: """
		Baseline H100 profile. No SM-specific patches needed (mature CUDA support).
		Qwen FP8 with TRITON MoE backend (VLLM_USE_FLASHINFER_MOE_FP8=0 from QWEN base).
		"""
}
