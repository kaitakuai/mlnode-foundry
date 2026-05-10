// Profile base: Kimi-K2.6 INT4 quantization common tunings.
// Imported by all kimi+int4 profiles.
package bases

KIMI_INT4: {
	env: {
		VLLM_USE_FLASHINFER_MOE_INT4: "1"
		VLLM_FLASHINFER_MOE_BACKEND:  "latency"
		POC_BATCH_SIZE_DEFAULT:       "64"
	}
	runtime_defaults: {
		compilation_mode: 0
		cudagraph_mode:   "NONE"
	}
}
