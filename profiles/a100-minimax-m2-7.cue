// Profile: A100 Ampere + MiniMax-M2.7 FP8
//
// 4×A100 SXM4 80GB is the proven configuration. Ampere sm_80 has no native
// FP8 hardware — MARLIN is the only MoE backend that works for FP8 weights;
// FlashInfer FP8 / DeepGEMM both reject the device.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

a100_minimax_m2_7: #BaseProfile & bases.A100 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "a100"
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
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 4 × A100 80GB = 320 GB HBM, exactly matches chain VRam=320 GB.
		tensor_parallel_size: 4
		moe_backend:          "marlin"
		// Optimal batch is 16 on A100 (not 32) — beyond batch=16 there's no
		// throughput gain and batch=64 OOMs.
		max_num_seqs:          64
	}
	description: "A100 Ampere SXM4 80GB (4×) + MiniMax-M2.7 FP8 (TP=4, MARLIN MoE)"
	notes: """
		Throughput proven on 4×A100 SXM4 80GB (Vast.ai, vLLM 0.20 PoC v2):
		896 nonces/min @ batch=16. Significantly below Hopper/Blackwell — Ampere
		has no native FP8 hardware, MARLIN is a software emulation fallback.
		Kept as a deployment option for operators with A100 fleets.
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
			knob:     "moe_backend=marlin"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-4xa100"
			reason:   "Ampere sm_80 has no native FP8 — FlashInfer FP8 and DeepGEMM both reject the device. MARLIN software emulation is the only working FP8 MoE backend; ~3× slower than Blackwell native FP8."
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=4"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-4xa100"
			reason:   "4×A100 80GB = 320 GB HBM exactly matches chain VRam=320 GB. Best at batch=16 (896 nonces/min); higher batch saturates without throughput gain, batch=64 OOMs."
			added_at: "2026-05-25"
		},
	]
}
