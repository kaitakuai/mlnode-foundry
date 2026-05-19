// Profile: B300 Blackwell Ultra (1×) + MiniMax-M2.7 FP8 — experimental
//
// Composition: #BaseProfile & B300 (sm_103a patches) & MINIMAX_M2_7 (chain args).
// Sister profile to b200-minimax-m2-7, differing in:
//   - TP=1 (single GPU) vs TP=2 (B200 needs 2× to clear chain VRam=320 GB)
//   - attention_backend left to vLLM auto-select (no B300 measurement yet)
//
// EXPERIMENTAL/TEST IMAGE. No hardware-validated throughput numbers yet —
// this is the starting point for B300 throughput tuning on MiniMax-M2.7.
// Once we have a 1×B300 (and possibly multi-B300) tuning sweep, a
// production rev=2 (or a new multi-GPU sister profile) will be carved out
// with measured knobs the way b200-minimax-m2-7 carries 2624 nonces/min.
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
		// 1×B300 (275 GiB HBM) is the experimental starting point. Sister
		// profile b200-minimax-m2-7 uses TP=2 because chain VRam=320 GB
		// does not fit on a single B200 (192 GB); on B300 a single GPU
		// has enough memory for MiniMax-M2.7 FP8 weights + 180k-token
		// FP8 KV cache, but throughput vs multi-B300 is unmeasured.
		// Production rev will pin TP after measurement.
		tensor_parallel_size: 1
		// No attention_backend set: vLLM auto-select on sm_103a until
		// a B300 measurement tells us which to pin (b200-minimax pins
		// FLASHINFER based on 2026-05 2×B200 evidence).
	}
	description: "B300 Blackwell Ultra (1×) + MiniMax-M2.7 FP8 (experimental, untuned)"
	notes: """
		EXPERIMENTAL / test image — no hardware-validated throughput numbers yet.
		Sister of b200-minimax-m2-7 with TP=1 instead of TP=2.
		Will be improved iteratively on B300 hardware; production rev=2 (or a
		new multi-GPU sister profile) will be carved out once a B300 tuning
		sweep produces measured optima.
		"""
	tuning_notes: [
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only up to 131072 (on B200). First B300 deployment will measure whether 180000 fits at TP=1 on a 275 GiB GPU plus FP8 KV cache; if not, a multi-GPU sister profile will be needed."
			severity: "warning"
			added_at: "2026-05-19"
		},
		{
			knob:     "tensor_parallel_size=1"
			source:   "operator request 2026-05-19"
			reason:   "Experimental starting point: 1×B300 = 275 GiB, large enough for MiniMax-M2.7 FP8 (~230 GB weights at fp8 + 180k-token KV cache). Whether 1×B300 throughput beats 2×B200 (2624 nonces/min, see b200-minimax-m2-7) is the question this image answers. Production rev will pin TP after measurement."
			severity: "warning"
			added_at: "2026-05-19"
		},
	]
}
