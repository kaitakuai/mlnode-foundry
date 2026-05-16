// Profile: H200 Hopper + MiniMax-M2.7 FP8
//
// 2×H200 SXM is the proven configuration. Hopper sm_90 needs MoE backend
// forced to TRITON and attention backend forced to FLASHINFER (the FP8 path
// via FlashInfer MoE has perf regressions on Hopper in vLLM 0.20).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

h200_minimax_m2_7: #BaseProfile & bases.H200 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "h200"
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
		// 2 × H200 = 282 GB HBM. Just under chain VRam=320 GB on paper, but
		// FP8 KV cache + tight gpu_memory_utilization=0.92 makes it fit in
		// practice (proven in experiments).
		tensor_parallel_size: 2
		moe_backend:          "triton"
		attention_backend:    "FLASHINFER"
	}
	description: "H200 Hopper SXM (2×) + MiniMax-M2.7 FP8 (TP=2, TRITON MoE)"
	notes: """
		Throughput proven on 2×H200 (Vast.ai): 1728 nonces/min @ batch=32.
		Hopper FP8 paths via FlashInfer MoE underperform — TRITON MoE forced.
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
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-2xh200"
			reason:   "FlashInfer FP8 MoE paths underperform on Hopper sm_90 in vLLM 0.20; TRITON is the chosen production backend."
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=2"
			source:   "https://github.com/kaitakuai/experiments/2026-05/minimax-m27-fp8-2xh200"
			reason:   "2×H200 = 282 GB HBM, under chain VRam=320 GB nominal — fits in practice via FP8 KV cache + gpu_mem_util=0.92. Best at batch=32 (1728 nonces/min); batch=64 OOMs."
			added_at: "2026-05-25"
		},
	]
}
