// Profile base: MiniMax-M2.7 common tunings.
//
// Two sources of truth converge here:
//
// 1. Chain governance (v0.2.13 `minimaxGovernanceModel`) MANDATES these vLLM
//    args. A node that omits them will produce outputs that fail cross-node
//    PoC validation:
//
//      --enable-auto-tool-choice
//      --max-model-len 180000
//      --kv-cache-dtype fp8
//      --tool-call-parser minimax_m2
//      --reasoning-parser minimax_m2_append_think
//
//    (See inference-chain/app/upgrades/v0_2_13/upgrades.go in gonka-ai/gonka.)
//
// 2. PoC v2 experiments (kaitakuai/experiments/2026-05/minimax-m27-fp8-*) found
//    a stable common set across B200 / H200 / H100 / A100 of:
//
//      gpu_memory_utilization=0.92, max_num_seqs=128,
//      logprobs_mode=processed_logprobs, trust_remote_code=true,
//      max_model_len=131072 (NOT 180000 — see warning below).
//
// MoE backend / TP differ PER GPU — those land in the leaf profile, not here.
//
// **Known gap**: experiments validated `--max-model-len 131072`, but chain
//   governance demands `180000`. We declare 180000 (chain wins) and surface a
//   warning tuning_note in each leaf. The first MiniMax production deployment
//   under real load is what proves 180000 stable end-to-end.
package bases

MINIMAX_M2_7: {
	env: {
		// MiniMax-M2.7 cold-start with FP8 KV + 180000 max-model-len is slow
		// on production TPs. Grace window suppresses the 3-strike kill-threshold
		// UNTIL the manager first reports healthy=True in an active session;
		// after that, normal post-startup crash detection is restored.
		// See patches/0002-api-watcher-grace.patch.
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		// From chain governance v0.2.13 (mandatory — outputs must match).
		max_model_len:           180000
		kv_cache_dtype:          "fp8"
		enable_auto_tool_choice: true
		tool_call_parser:        "minimax_m2"
		reasoning_parser:        "minimax_m2_append_think"

		// From experiments (common across all FP8 PoC v2 measurements).
		gpu_memory_utilization: *0.92 | number
		max_num_seqs:           *128 | int
		logprobs_mode:          "processed_logprobs"
		trust_remote_code:      true
	}
}
