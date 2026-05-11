// Profile base: Qwen3-235B-A22B FP8 common tunings.
// Imported by all qwen profiles.
//
// Defaults use Cue's `*` marker so leaves can override via & without conflict.
package bases

QWEN: {
	env: {
		// TRITON FP8 MoE backend preferred on H100/H200/A100.
		// Blackwell profiles (b300-qwen, b200-qwen) override to "1".
		VLLM_USE_FLASHINFER_MOE_FP8: *"0" | string
	}
	runtime_defaults: {
		// 4× SXM topology is the common deployment; single-GPU profiles override to 1.
		tensor_parallel_size: *4 | int
	}
}
