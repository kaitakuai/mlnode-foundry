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
		// FLASHINFER_TRTLLM is auto-selected on Blackwell sm_100 — no force needed
		// for MoE backend. Attention backend is set explicitly to FLASHINFER so we
		// don't depend on vLLM's auto-selection heuristic changing across releases
		// (the 2026-05 2×B200 measurement at 2624 nonces/min was made with this
		// backend).
		attention_backend: "FLASHINFER"
	}
	description: "B200 Blackwell SXM (2×) + MiniMax-M2.7 FP8"
	notes: """
		Throughput proven on 2×B200 (Vast.ai inst 41067, vLLM 0.20 PoC v2): 2624 nonces/min @ batch=32.
		8-GPU normalized (4 × 2-GPU instances): 10 496 nonces/min.
		+11% vs published 2026-04 baseline (2367 → 2624).
		MoE backend FLASHINFER_TRTLLM auto-selected on sm_100; attention backend FLASHINFER (explicit).
		"""
	tuning_notes: [
		// FIRST entry with an experiments URL — picked up by
		// render_registry_view._report_url for the dashboard "verified" chip.
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_2xb200_ohio_pr36/README.md"
			reason:   "Hardware validation — 2×B200 SXM (Vast.ai Ohio inst 37296947), Pasha 2026-05-21. Validates the foundry image with PR#36 (apply_householder torch.compile) confirmed baked in. PoC throughput 2624 nonces/min @ batch=32 — bit-for-bit identical to the published 2×B200 reference (mean L2 = 0.0000, 0/1000 mismatch under strict vLLM self-validation 0.02). PR#36 numerically neutral on B200 (contrast: on 4×H100 same patch shifts L2≈0.23). Inference (chain §3.2.3): TTFT 2.12 s, TPOT 24.9 ms/tok, 803 output tok/s, RPS 2.68, 0/60 failures."
			added_at: "2026-05-22"
		},
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only up to 131072. First production deployment under load is the proof point. Set high enough that cross-node PoC validation matches; lower will diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "tensor_parallel_size=2"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xb200/README.md"
			reason:   "2×B200 = 360 GB HBM, fits chain VRam=320 GB requirement with headroom for FP8 KV cache. Phase 3 batch sweep: 2→1364, 8→2256, 16→2560, 32→2624 (best), 64→hang. Vast.ai inst 41067, 2026-05."
			added_at: "2026-05-25"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xb200/README.md"
			reason:   "Phase 3 sweep observed batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Same failure as flashinfer_moe_int4_blackwell.md memory entry."
			severity: "warning"
			added_at: "2026-05-25"
		},
		{
			knob:     "post-sweep memory accumulation"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xb200/README.md"
			reason:   "After Phase 3 batch sweep the PyTorch caching allocator accumulates ~170 GB of unreleased buffers; subsequent 1000-nonce collection requires a vLLM restart to clear cache. Not a fatal issue under normal chain workload (no multi-batch sweep), but operators running benchmark + production back-to-back should restart between."
			added_at: "2026-05-25"
		},
		{
			knob:     "attention_backend=FLASHINFER (explicit)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax-m27-fp8-2xb200/README.md"
			reason:   "The 2624 nonces/min measurement was made with FLASHINFER attention. vLLM's auto-selector picks the same on Blackwell sm_100 today, but pinning explicitly defends against the heuristic changing across vLLM versions and silently regressing throughput on this profile."
			added_at: "2026-05-25"
		},
	]
}
