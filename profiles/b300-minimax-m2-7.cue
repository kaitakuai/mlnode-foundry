// Profile: B300 Blackwell Ultra + MiniMax-M2.7 FP8 (TP=1, single-GPU)
//
// Composition: #BaseProfile & B300 & MINIMAX_M2_7 with TP=1.
// Validated 2026-05-23 on a 1×B300 SXM6 (275 GiB HBM) bare server: the
// 230 GB model fits at 1.23× concurrency at 180000-token max_model_len.
// 1×B300 (275 GiB) is below the v0.2.13 chain VRam=320 GB nominal
// requirement, but the requirement is conservative and the B300 nonces
// PASS the MiniMax chain gate vs the 2×B200 baseline (mean L2 0.266,
// 0.9% mismatch — well within 0.75/0.10).
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b300_minimax_m2_7: #BaseProfile & bases.B300 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "b300"
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
	hw_patches:   list.Concat([bases.B300.hw_patches, ["poc-householder-compile"]])
	runner_patch: ""
	env: {}
	runtime_defaults: {
		// 1 × B300 SXM6 = 275 GiB HBM. MiniMax-M2.7 FP8 (230 GB) fits with
		// 1.23× concurrency at max_model_len=180000 (KV pool 221616 tokens,
		// GPU ~91% full). Highest per-GPU PoC throughput of any tested
		// hardware (1792 nonces/min/GPU, vs B200 1312, H100 576, A100 224).
		tensor_parallel_size: 1
		// FLASHINFER_TRTLLM is auto-selected on Blackwell (sm_100 reported
		// by CUDA on B300) — no MoE backend force needed (image starts
		// out-of-the-box, no env override required). Attention backend
		// pinned to FLASHINFER explicitly so the auto-selector heuristic
		// doesn't silently regress this profile across vLLM releases.
		attention_backend: "FLASHINFER"
	}
	description: "B300 Blackwell Ultra SXM6 (1×) + MiniMax-M2.7 FP8 (TP=1, FLASHINFER_TRTLLM MoE auto)"
	notes: """
		Throughput proven on 1×B300 SXM6 (bare server, vLLM 0.20 PoC v2):
		1792 nonces/min @ batch=32 — highest per-GPU value of any tested
		hardware (B200 1312/GPU, H100 576/GPU, H200 864/GPU, A100 224/GPU).
		8-GPU normalized (8 × 1-GPU instances): 14 336 nonces/min.
		Inference: TTFT 1.28 s (fastest), TPOT 26.6 ms/tok, 746 output tok/s,
		RPS 2.51, 0/60 failures — a single B300 ≈ a 2×B200 instance (803 tok/s).
		Nonces cross-validate with the 2×B200 baseline: mean L2 0.266,
		0.9% > thr=0.75 — PASS p=1.0 under the MiniMax chain gate (0.75/0.10).
		(L2 0.27 vs B200 reflects different TP topology / reduction order,
		not bit-identical like B200↔B200 TP=2, but well within the gate.)
		"""
	tuning_notes: [
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Hardware validation — 1×B300 SXM6 (bare server, 275 GiB HBM, 2026-05-23). PoC throughput 1792 nonces/min @ batch=32 — highest per-GPU of any hardware. Inference TTFT 1.28 s (fastest), 746 output tok/s. Nonces cross-validate with 2×B200 baseline: mean L2 0.266, PASS under MiniMax chain gate (0.75/0.10). Image starts out-of-the-box (no env override needed — unlike A100; Blackwell auto-selects FLASHINFER_TRTLLM correctly)."
			added_at: "2026-05-23"
		},
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated KV-pool fit (221616 tokens, 1.23× concurrency at 180000) on B300, but generation quality at full 180k context not specifically swept. Set high enough that cross-node PoC validation matches; lower would diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-23"
		},
		{
			knob:     "tensor_parallel_size=1"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Validated single-B300 (275 GiB HBM) configuration — MiniMax-M2.7 FP8 (230 GB) fits with 1.23× concurrency at max_model_len=180000. Phase 3 batch sweep on B300: 8→1392, 16→1728, 32→1792 (best), 64→hang. The chain VRam=320 GB nominal requirement is conservative; 1×B300 (275 GB) cross-validates with the B200 fleet under the MiniMax chain gate."
			added_at: "2026-05-23"
		},
		{
			knob:     "attention_backend=FLASHINFER"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "vLLM auto-selector picks FLASHINFER on Blackwell; B300 image starts out-of-the-box with no env override. Pinning explicitly so the auto-selector heuristic doesn't silently regress this profile across vLLM versions."
			added_at: "2026-05-23"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Phase 3 sweep on B300 confirms batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). Same failure mode as B200/H200/H100/A100 MiniMax profiles. max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override."
			severity: "warning"
			added_at: "2026-05-23"
		},
	]
}
