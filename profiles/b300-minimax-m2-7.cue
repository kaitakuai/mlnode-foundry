// Profile: B300 Blackwell Ultra (sm_103a) + MiniMax-M2.7 FP8 (experimental TP=1)
//
// Composition: #BaseProfile & B300 & MINIMAX_M2_7 with TP=1.
// 1×B300 = 288 GB HBM3e — sits BELOW the chain VRam=320 GB requirement, so
// this profile is for single-B300 benchmarking only, not production chain
// participation. For chain duty on Blackwell Ultra hosts, run two instances
// (TP=2 per instance) or stick to the B200 profile.
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
		// 1 × B300 = 288 GB HBM3e — below chain VRam=320 GB requirement.
		// Experimental single-GPU configuration for benchmarking the
		// Blackwell Ultra MoE path against the 2×B200 baseline; not for
		// production chain participation.
		tensor_parallel_size: 1
		// FLASHINFER_TRTLLM is auto-selected on Blackwell sm_103a (same family
		// as sm_100) — no MoE backend force needed. Attention backend pinned
		// to FLASHINFER explicitly so the auto-selector heuristic doesn't
		// silently regress this profile across vLLM releases.
		attention_backend: "FLASHINFER"
	}
	description: "B300 Blackwell Ultra SXM (1×) + MiniMax-M2.7 FP8 — experimental TP=1 (below chain VRam=320 GB)"
	notes: """
		Experimental single-B300 (TP=1) profile for benchmarking the Blackwell
		Ultra MoE path. 1×B300 = 288 GB HBM3e is BELOW the chain VRam=320 GB
		requirement, so this image is not eligible for production chain duty
		under v0.2.13 governance. Validated throughput / nonces TBD — first
		hardware run is the benchmark.
		"""
	tuning_notes: [
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated only 131072 (on B200/H200/H100/A100). No B300 validation yet — first production load is the proof point. Set high enough that cross-node PoC validation matches; lower would diverge on long-context prompts."
			severity: "warning"
			added_at: "2026-05-23"
		},
		{
			knob:     "tensor_parallel_size=1"
			source:   "operator iteration 2026-05-23 — experimental single-B300 benchmark"
			reason:   "Single-B300 configuration for benchmarking the Blackwell Ultra MoE path against the 2×B200 baseline. Not chain-eligible on its own (1×B300 = 288 GB HBM3e, chain governance asks 320 GB); operators wanting production duty should run two instances at TP=2, or use b200-minimax-m2-7."
			added_at: "2026-05-23"
		},
		{
			knob:     "attention_backend=FLASHINFER"
			source:   "carry-over from b200-minimax-m2-7 (sm_100 sibling of sm_103a)"
			reason:   "vLLM auto-selector picks FLASHINFER on Blackwell sm_100; sm_103a is the same family and the heuristic applies. Pinning explicitly so the auto-selector heuristic doesn't silently regress this profile across vLLM versions."
			added_at: "2026-05-23"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "carry-over from b200/h200/h100/a100 MiniMax profiles"
			reason:   "Phase 3 sweeps across all other MiniMax profiles observed batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Assumed same failure mode on B300 until proven otherwise."
			severity: "warning"
			added_at: "2026-05-23"
		},
	]
}
