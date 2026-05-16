// Profile: H100 Hopper + MiniMax-M2.7 FP8
//
// 4×H100 SXM is the proven configuration. Same Hopper-FP8 caveat as H200:
// TRITON MoE + FLASHINFER attention forced.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

h100_minimax_m2_7: #BaseProfile & bases.H100 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "h100"
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
	hw_patches:  list.Concat([bases.H100.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 4 × H100 = 320 GB HBM, exactly matches chain VRam=320 GB.
		// gpu_memory_utilization=0.92 + FP8 KV cache leaves enough headroom.
		tensor_parallel_size: 4
		moe_backend:          "triton"
		attention_backend:    "FLASHINFER"
	}
	description: "H100 Hopper SXM (4×) + MiniMax-M2.7 FP8 (TP=4, TRITON MoE)"
	notes: """
		Throughput proven on 4×H100 SXM5 (shadecloud orion, vLLM 0.20 PoC v2):
		2368 nonces/min @ batch=32 — matches 2×B200 within margin, Hopper SXM5
		is genuinely competitive on this model.
		"""
	tuning_notes: [
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain mandate; experiments validated only 131072. First production load is the validation; long-context divergence risk if lower."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "moe_backend=triton"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-4xh100"
			reason:   "FlashInfer FP8 MoE paths underperform on Hopper sm_90 in vLLM 0.20; TRITON is the chosen production backend."
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=4"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-4xh100"
			reason:   "4×H100 = 320 GB HBM exactly matches chain VRam=320 GB requirement. Best at batch=32 (2368 nonces/min); batch=64 OOMs the PoC engine."
			added_at: "2026-05-25"
		},
	]
}
