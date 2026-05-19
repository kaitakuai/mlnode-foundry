// Profile base: Kimi-K2.6 INT4 quantization common tunings.
// Imported by all kimi+int4 profiles.
package bases

KIMI_INT4: {
	env: {
		VLLM_USE_FLASHINFER_MOE_INT4: "1"
		VLLM_FLASHINFER_MOE_BACKEND:  "latency"
		POC_BATCH_SIZE_DEFAULT:       "64"
		// Kimi-K2.6 INT4 weights load (~1 TB on TP=4) + MLA compile takes
		// 10-22 min on production topologies. Grace window suppresses the
		// 3-strike kill-threshold UNTIL the manager first reports healthy=True
		// in an active session. Same env on every Kimi-K2.6 profile regardless
		// of GPU (B300 already sets it via B300 base too — Cue unifies same
		// value without conflict).
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		compilation_mode: 0
		cudagraph_mode:   "NONE"
	}
}
