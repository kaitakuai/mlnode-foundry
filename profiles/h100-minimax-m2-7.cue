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
		2368 nonces/min @ batch=32 — matches 2×B200 (2624) within margin;
		Hopper SXM5 is genuinely competitive on this model.
		8-GPU normalized (2 × 4-GPU instances): 4 736 nonces/min.
		+42% vs published 2026-04 baseline (1664 → 2368), attributed to
		PR#36 (apply_householder torch.compile) + TRITON MoE forced
		(vs FLASHINFER_CUTLASS auto) + FLASHINFER attention forced
		(vs FLASH_ATTN auto) + shadecloud H100 hardware (NV18 NVLink mesh).
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
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xh100/README.md"
			reason:   "FlashInfer FP8 MoE paths (FLASHINFER_CUTLASS) underperform on Hopper sm_90 in vLLM 0.20 — auto-select picks them and silently regresses. TRITON forced explicitly so the 2368 nonces/min measurement is reproducible across vLLM versions."
			added_at: "2026-05-25"
		},
		{
			knob:     "attention_backend=FLASHINFER"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xh100/README.md"
			reason:   "vLLM auto-default on Hopper is FLASH_ATTN; FLASHINFER measured faster on the 2026-05 sweep. Pinning explicitly prevents the auto-selector heuristic from silently regressing this profile across vLLM releases."
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=4"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xh100/README.md"
			reason:   "4×H100 = 320 GB HBM exactly matches chain VRam=320 GB requirement. Phase 3 batch sweep: 2→888, 8→2048, 16→2240, 32→2368 (best), 64→hang. shadecloud orion, 2026-05."
			added_at: "2026-05-25"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-4xh100/README.md"
			reason:   "Phase 3 sweep observed batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Same failure mode as b200-minimax-m2-7 and the flashinfer_moe_int4_blackwell.md memory entry."
			severity: "warning"
			added_at: "2026-05-25"
		},
	]
}
