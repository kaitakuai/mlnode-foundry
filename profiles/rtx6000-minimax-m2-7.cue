// Profile: RTX PRO 6000 Blackwell SE + MiniMax-M2.7 FP8
//
// 4× RTX PRO 6000 SE (consumer Blackwell sm_120) is the proven configuration
// — TP=4. TRITON is the ONLY working FP8 MoE backend on sm_120 with MiniMax's
// 128×128 block-wise quant layout; FlashInfer FP8 paths reject the layout,
// DeepGEMM rejects the device.
//
// Caveat: experiments are from 2026-04 on vLLM 0.19. Re-validate on vLLM
// 0.20 base before declaring this profile production-ready at the current
// Stage 1 pin.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

rtx6000_minimax_m2_7: #BaseProfile & bases.RTX6000 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "rtx6000"
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
	hw_patches:  list.Concat([bases.RTX6000.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 4 × RTX PRO 6000 SE = 4 × 96 GB = 384 GB HBM; fits chain VRam=320 GB
		// with reasonable headroom.
		tensor_parallel_size: 4
		moe_backend:          "triton"
		// Optimal batch is 8 on RTX 6000 (NOT 32) — higher batches OOM the
		// PoC engine on this card.
		max_num_seqs:         32
	}
	description: "RTX PRO 6000 Blackwell SE (4× consumer) + MiniMax-M2.7 FP8 (TP=4, TRITON MoE)"
	notes: """
		Throughput proven on 4×RTX PRO 6000 SE (vLLM 0.19, 2026-04 experiment):
		848 nonces/min @ batch=8 (TP=4); engine OOMs at batch≥16.
		Re-validate at vLLM 0.20 before fleet rollout — Stage 1 base changed.
		"""
	tuning_notes: [
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain mandate; experiments validated only up to 131072. First production load is the validation; long-context divergence risk if lower."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "moe_backend=triton"
			source:   "https://github.com/kaitakuai/experiments/2026-04/minimax-m27-fp8-4xrtxpro6000"
			reason:   "TRITON is the only working FP8 MoE backend on sm_120 with MiniMax's 128×128 block-wise quant — FlashInfer FP8 paths reject the layout (verified FI 0.6.6 + 0.6.7.post3); DeepGEMM rejects the device entirely (no sm_120 support). Caps throughput ~30-50% below vendor-tuned kernels would on the same silicon."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "max_num_seqs=32"
			source:   "https://github.com/kaitakuai/experiments/2026-04/minimax-m27-fp8-4xrtxpro6000"
			reason:   "Optimal effective batch on this card is 8 (848 nonces/min); engine OOMs at batch≥16. Lower max_num_seqs vs the 128 used on Hopper/Blackwell SXM."
			severity: "warning"
			added_at: "2026-05-25"
		},
	]
}
