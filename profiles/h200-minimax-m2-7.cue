// Profile: H200 Hopper SXM 141GB + MiniMax-M2.7 FP8
//
// 2×H200 = 282 GB HBM, fits chain VRam=320 GB requirement with TP=2 (one
// instance per pair, two pairs per 4×H200 host). Same Hopper-FP8 caveat as
// H100: TRITON MoE + FLASHINFER attention must be forced — auto-select picks
// the slower FLASHINFER_CUTLASS / FLASH_ATTN paths on sm_90.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

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
	hw_patches:   list.Concat([bases.H200.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 2 × H200 141GB = 282 GB HBM — fits chain VRam=320 GB requirement
		// at TP=2 (single 2-GPU instance) with FP8 KV cache headroom.
		// On an 8×H200 host run 4 × 2-GPU instances for 8-GPU normalised
		// 6 912 nonces/min (4 × 1728).
		tensor_parallel_size: 2
		moe_backend:          "triton"
		attention_backend:    "FLASHINFER"
	}
	description: "H200 Hopper SXM 141GB (2×) + MiniMax-M2.7 FP8 (TP=2, TRITON MoE)"
	notes: """
		Throughput proven on 2×H200 SXM (Vast.ai, vLLM 0.20 PoC v2):
		1728 nonces/min @ batch=32. 0% delta vs 2026-04 baseline (exact
		match — Hopper-FP8 path is stable across the bump).
		8-GPU normalized (4 × 2-GPU instances): 6 912 nonces/min.
		Cross-hardware @ batch=32: 2×B200 (2624) > 4×H100 (2304) > 2×H200
		(1728) > 4×A100 (896).
		"""
	tuning_notes: [
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xh200/README.md"
			reason:   "Hardware validation — 2×H200 SXM (Vast.ai), 2026-05. PoC throughput 1728 nonces/min @ batch=32. 0% delta vs 2026-04 published baseline (exact match)."
			added_at: "2026-05-22"
		},
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only 131072 (on 2×H200). First production load is the validation; long-context divergence risk if lower. Set high enough that cross-node PoC validation matches; lower would diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-22"
		},
		{
			knob:     "moe_backend=triton"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xh200/README.md"
			reason:   "FLASHINFER_CUTLASS Fp8 MoE — vLLM's auto-default on Hopper sm_90 — underperforms TRITON for this model. Forced explicitly so the 1728 nonces/min measurement is reproducible across vLLM versions."
			added_at: "2026-05-22"
		},
		{
			knob:     "attention_backend=FLASHINFER"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xh200/README.md"
			reason:   "vLLM auto-default on Hopper is FLASH_ATTN (also valid but slightly slower). FLASHINFER pinned explicitly so the auto-selector heuristic doesn't silently regress this profile across vLLM releases."
			added_at: "2026-05-22"
		},
		{
			knob:     "tensor_parallel_size=2"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xh200/README.md"
			reason:   "2×H200 = 282 GB HBM, fits chain VRam=320 GB requirement at TP=2 with FP8 KV cache headroom. Phase 3 batch sweep: 2→1072, 8→1568, 16→1696, 32→1728 (best), 64→hang."
			added_at: "2026-05-22"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xh200/README.md"
			reason:   "Phase 3 sweep observed batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Same failure mode as the other MiniMax profiles."
			severity: "warning"
			added_at: "2026-05-22"
		},
	]
}
