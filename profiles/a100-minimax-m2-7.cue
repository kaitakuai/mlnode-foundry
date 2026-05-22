// Profile: A100 Ampere SXM4 80GB + MiniMax-M2.7 FP8
//
// Ampere has no native FP8 hardware — MoE backend MARLIN forced (software-
// emulated W4A8/FP8 path; the only one that works on sm_80 for this model).
// Slowest GPU in the supported set (~3× B200, ~2× H200) but still valid for
// Gonka chain participation; useful for spot/lease economics on cheap A100.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

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
	hw_patches:   list.Concat([bases.A100.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {
		// Belt-and-suspenders with moe_backend=marlin: even though we force
		// MARLIN, some vLLM paths probe VLLM_USE_FLASHINFER_MOE_FP8 separately
		// and raise NotImplementedError on Ampere. Pin to 0 at env level so
		// the backend selector never even attempts the FlashInfer FP8 path.
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
	}
	runtime_defaults: {
		// 4 × A100 80GB = 320 GB HBM, exactly matches chain VRam=320 GB.
		// gpu_memory_utilization=0.92 + FP8 KV cache leaves enough headroom.
		tensor_parallel_size: 4
		// MARLIN is the ONLY MoE backend that works on Ampere for FP8 weights.
		// Without forcing it, vLLM tries FLASHINFER_CUTLASS Fp8 MoE and fails:
		//   NotImplementedError: VLLM_USE_FLASHINFER_MOE_FP8=1, but no FlashInfer
		//   FP8 MoE backend supports the configuration
		// (A100 sm_80 lacks native FP8 tensor cores — MARLIN is the software
		// path; compute-bound, hence the saturation at batch≥16 in the sweep.)
		moe_backend: "marlin"
		// attention_backend left to vLLM auto-select. Report didn't pin one;
		// FLASHINFER on Ampere is slower than the default FLASH_ATTN/XFORMERS
		// in vLLM 0.20.0 for FP8 KV cache, so we let auto pick.
	}
	description: "A100 Ampere SXM4 80GB (4×) + MiniMax-M2.7 FP8 (TP=4, MARLIN MoE)"
	notes: """
		Throughput proven on 4×A100 SXM4 80GB (Vast.ai Massachusetts, vLLM 0.20 PoC v2):
		896 nonces/min @ batch=16 — slowest GPU in the supported set
		(~3× slower than 2×B200's 2624, ~2× slower than 2×H200's 1728).
		8-GPU normalized (2 × 4-GPU instances): 1 792 nonces/min.
		+4% vs published 2026-04 baseline (864 → 896).
		MARLIN saturates at batch=16; batch=32 produces same nonces (compute-bound,
		not memory-bound). batch=64 hangs the PoC engine.
		"""
	tuning_notes: [
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xa100/README.md"
			reason:   "Hardware validation — 4×A100 SXM4 80GB (Vast.ai Massachusetts, 2026-05). PoC throughput 896 nonces/min @ batch=16 (MARLIN saturation; batch=32 same). +4% vs published 2026-04. Cross-hardware: A100 is the slowest GPU in the supported set."
			added_at: "2026-05-22"
		},
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only 131072 (on 4×A100). First production load is the validation; long-context divergence risk if lower. Set high enough that cross-node PoC validation matches; lower would diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-22"
		},
		{
			knob:     "VLLM_USE_FLASHINFER_MOE_FP8=0"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xa100/README.md"
			reason:   "Env-level disable for FlashInfer FP8 MoE. Pairs with moe_backend=marlin: even with MARLIN forced, some vLLM code paths probe this env var independently and raise NotImplementedError on Ampere sm_80. Pinning to 0 stops the FlashInfer path from being attempted at all."
			added_at: "2026-05-22"
		},
		{
			knob:     "moe_backend=marlin"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xa100/README.md"
			reason:   "Forced because A100 sm_80 has NO native FP8 tensor cores — MARLIN is the only MoE backend that works for FP8 weights on Ampere. Without it vLLM tries FLASHINFER_CUTLASS Fp8 MoE and raises NotImplementedError. Compute-bound (software-emulated FP8), which is why throughput saturates at batch=16 and the GPU is ~3× slower than B200."
			added_at: "2026-05-22"
		},
		{
			knob:     "tensor_parallel_size=4"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xa100/README.md"
			reason:   "4×A100 80GB = 320 GB HBM exactly matches chain VRam=320 GB requirement. Phase 3 batch sweep: 2→604, 8→832, 16→896 (best), 32→896 (saturated), 64→hang."
			added_at: "2026-05-22"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xa100/README.md"
			reason:   "Phase 3 sweep observed batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Same failure mode as b200-minimax-m2-7 and h100-minimax-m2-7."
			severity: "warning"
			added_at: "2026-05-22"
		},
	]
}
