// Profile: H200 Hopper + Qwen3-235B-A22B FP8
//
// H200 inherits H100's mature CUDA support — no SM-specific patches.
// Larger HBM (141 GB/GPU vs 80 GB on H100) enables higher max_model_len
// without dropping into CPU offload.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

h200_qwen: #BaseProfile & bases.H200 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "h200"
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
		gpu_memory_utilization: 0.90
	}
	description: "H200 Hopper SXM + Qwen3-235B-A22B FP8 (TP=4, 141 GB HBM/GPU)"
	notes: """
		H200 = H100 with 1.76× HBM; no SM-specific patches needed.
		TRITON MoE backend (VLLM_USE_FLASHINFER_MOE_FP8=0 from QWEN base).
		"""
}
