// Profile base: Qwen3-235B-A22B FP8 common tunings.
// Imported by all qwen profiles.
package bases

QWEN: {
	env: {
		VLLM_USE_FLASHINFER_MOE_FP8: "0"  // TRITON FP8 MoE backend preferred on H100/H200
	}
	runtime_defaults: {
		tensor_parallel_size: 4
	}
}
