// Profile base: Kimi-K2.6 INT4 quantization common tunings.
// Imported by all kimi+int4 profiles.
package bases

KIMI_INT4: {
	env: {
		POC_BATCH_SIZE_DEFAULT: "64"
		// Kimi-K2.6 INT4 weights load (~1 TB on TP=4) + MLA compile takes
		// 10-22 min on production topologies. Grace window suppresses the
		// 3-strike kill-threshold UNTIL the manager first reports healthy=True
		// in an active session. Same env on every Kimi-K2.6 profile regardless
		// of GPU (B300 already sets it via B300 base too — Cue unifies same
		// value without conflict).
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		// Defaults established by 2026-04 / 2026-05 production experiments
		// (cudagraph FULL blocks batch≥128 on Kimi MLA → eager mode).
		// Defaultable so leaves can override when later A/B finds a different
		// optimum for their (GPU, batch) tuple — e.g., b200-kimi rev=2
		// flips to mode=3 + FULL_AND_PIECEWISE for an experimental tune.
		compilation_mode: *0 | int
		cudagraph_mode:   *"NONE" | string
	}
}

// FlashInfer's INT4 MoE kernels, which exist only for Blackwell (trtllm-gen).
// Unified in by the b200/b300 Kimi leaves; Hopper leaves leave the MoE backend
// to vLLM's auto-selection, matching gonka's own H200 Kimi profile.
KIMI_INT4_FLASHINFER_MOE: {
	env: {
		VLLM_USE_FLASHINFER_MOE_INT4: "1"
		VLLM_FLASHINFER_MOE_BACKEND:  "latency"
	}
}
