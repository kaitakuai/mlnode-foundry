// Profile: B200 Blackwell + MiniMax-M2.7 FP8
//
// Composition: #BaseProfile & B200 (sm_100 patches) & MINIMAX_M2_7 (chain args).
// Activates on chain at epoch 271 (v0.2.13 upgrade).
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b200_minimax_m2_7: #BaseProfile & bases.B200 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "minimax"
			model_revision: "m2-7"
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.20.0"
			rev:    1
		}
	}
	mode:         "kaitakuai-base"
	hw_patches:  list.Concat([bases.B200.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 2 × B200 is the minimum that fits the 320 GB chain VRam requirement
		// (2 × 180 GB HBM = 360 GB total).
		tensor_parallel_size: 2
		// FLASHINFER_TRTLLM is auto-selected on Blackwell sm_100; no force needed.
		// Attention backend auto.
	}
	description: "B200 Blackwell SXM (2×) + MiniMax-M2.7 FP8"
	notes: """
		Throughput proven on 2×B200 (Vast.ai inst 41067, vLLM 0.20 PoC v2): 2624 nonces/min @ batch=32.
		MoE backend FLASHINFER_TRTLLM auto-selected on sm_100.
		"""
	tuning_notes: [
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only up to 131072. First production deployment under load is the proof point. Set high enough that cross-node PoC validation matches; lower will diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=2"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-2xb200"
			reason:   "2×B200 = 360 GB HBM, fits chain VRam=320 GB requirement with headroom for FP8 KV cache. Best throughput at batch=32 (2624 nonces/min); batch=64 OOMs the PoC engine."
			added_at: "2026-05-25"
		},
	]
}
