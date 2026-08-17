// Profile: H100 Hopper + MiniMax-M2.7 FP8
//
// 4×H100 SXM is the proven configuration. Same Hopper-FP8 caveat as H200:
// TRITON MoE + FLASHINFER attention forced.
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

h100_minimax_m2_7: #OverlayProfile & bases.H100 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "h100"
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
	hw_patches: list.Concat([bases.H100.hw_patches, bases.GONKA_BASE_PATCHES])
	runner_patch: "h100-minimax-m2-7-plugin"
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
			knob:     "VLLM_USE_FLASHINFER_MOE_FP8=0"
			source:   "image inspection — Stage 2/3 base bakes in VLLM_USE_FLASHINFER_MOE_FP8=1"
			reason:   "Env-level disable for FlashInfer FP8 MoE on Hopper. Pairs with moe_backend=triton: CLI args take priority in vLLM, but a flat `docker run` without `--moe-backend triton` would silently regress to FLASHINFER_CUTLASS (slower). Pinning to 0 makes the env layer agree with the profile's TRITON choice out of the box."
			added_at: "2026-05-26"
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
