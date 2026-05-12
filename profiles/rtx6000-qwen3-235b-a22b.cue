// Profile: RTX PRO 6000 Blackwell SE + Qwen3-235B-A22B FP8
//
// Consumer Blackwell (sm_120) on 4× cards. FlashInfer JIT must be cleared
// so the host SM is detected (shipped wheel includes sm_120 but baked cache
// targets a different chip). nvidia-* dev headers symlinked for JIT compile.
// Memory: GPUHub snapshot vllm0.20.0-sm120-fi-attn-v1 — 576 nonces/min @ batch=32.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

rtx6000_qwen3_235b_a22b: #BaseProfile & bases.RTX6000 & bases.QWEN & {
	identity: {
		axes: {
			gpu:   "rtx6000"
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
	env: {
		POC_BATCH_SIZE_DEFAULT: "32"
	}
	runtime_defaults: {
		tensor_parallel_size:   4
		gpu_memory_utilization: 0.90
	}
	description: "RTX PRO 6000 Blackwell SE (4× consumer) + Qwen3-235B-A22B FP8"
	notes: """
		Consumer Blackwell — libcuda-compat patch not applicable (different driver path).
		FlashInfer JIT cleared so first-launch compile detects host sm_120.
		POC_BATCH_SIZE_DEFAULT=32 matches GPUHub fleet snapshot tuning.
		"""
}
