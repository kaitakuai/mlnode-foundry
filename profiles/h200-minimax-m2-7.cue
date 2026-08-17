// Profile: H200 Hopper SXM 141GB + MiniMax-M2.7 FP8
//
// 2×H200 = 282 GB HBM, fits chain VRam=320 GB requirement with TP=2 (one
// instance per pair, two pairs per 4×H200 host). Same Hopper-FP8 caveat as
// H100: TRITON MoE + FLASHINFER attention must be forced — auto-select picks
// the slower FLASHINFER_CUTLASS / FLASH_ATTN paths on sm_90.
// 0.25.1 MIGRATION NOTE (2026-08-06): base flipped to the release-line
// mlnode-base 0.2.14-vllm0.25.1-k5 via tools/stage3.lock.cue; serving flags
// are INHERITED from the vLLM 0.20 fat-fork validation campaigns
// (experiments 2026-05/minimax-m27-*) and are NOT yet revalidated on the
// 0.25.1 plugin stack. Backend selection is the sensitive part (TRTLLM
// auto-select on Blackwell, forced triton on Hopper, marlin on A100 —
// consensus-relevant, see the Marlin/DeepGEMM cross-hw precedent). Treat
// these images as CANDIDATES until a hardware pass equivalent to the
// deepseek 2026-08 campaigns is run.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

h200_minimax_m2_7: #OverlayProfile & bases.H200 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "h200"
			model:          "minimax"
			model_revision: "m2-7"
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// rev=7 — max-model-len demoted to a standalone default so the
			// node-config 180000 (deploy f8e469fc) is no longer overridden.
			upstream: "3.0.16"
			rev:      7
		}
	}
	mode: "upstream-overlay"
	base: {
		// Cortima's PUBLISHED release image — their Stage-3 equivalent
		// (mlnode/packages/api/Dockerfile over ghcr.io/gonka-ai/vllm:v0.25.1-poc-v2).
		// Taking it as our base drops a whole build stage and makes drifting
		// from their mlnode source impossible; everything we still add lives
		// in hw_patches + runner_patch below. See schema.cue on the switch.
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:1b9b7ce55feecab837f1d7ce974fc5f377ae0a04a4fb403eeeb50130e7728ee1"
		upstream_version: "3.0.16"
	}
	hw_patches: list.Concat([bases.H200.hw_patches, bases.GONKA_BASE_PATCHES])
	runner_patch: "h200-minimax-m2-7-plugin"
	env: {
		// Plugin entrypoint + worker-extension RPC channel (0.25.1 line).
		MLNODE_VLLM_MODULE:                "gonka_poc.entrypoint.api_router"
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
	}
	env: {
		// Stage 2/3 base bakes in VLLM_USE_FLASHINFER_MOE_FP8=1, which on
		// Hopper sm_90 routes MoE through FLASHINFER_CUTLASS — measurably
		// slower than TRITON for MiniMax-M2.7. CLI `--moe-backend triton`
		// overrides this in normal operation, but a flat `docker run` without
		// the override would silently regress. Pin to 0 so the env layer
		// agrees with moe_backend=triton from runtime_defaults out of the box.
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
	}
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
			knob:     "VLLM_USE_FLASHINFER_MOE_FP8=0"
			source:   "image inspection — Stage 2/3 base bakes in VLLM_USE_FLASHINFER_MOE_FP8=1"
			reason:   "Env-level disable for FlashInfer FP8 MoE on Hopper. Pairs with moe_backend=triton: CLI args take priority in vLLM, but a flat `docker run` without `--moe-backend triton` would silently regress to FLASHINFER_CUTLASS (slower). Pinning to 0 makes the env layer agree with the profile's TRITON choice out of the box."
			added_at: "2026-05-26"
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
