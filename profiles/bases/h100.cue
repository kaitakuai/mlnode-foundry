// Profile base: H100 Hopper common tunings.
// H100 needs minimal hw-patches (mature CUDA support, no SM-specific fixes).
package bases

H100: {
	// Defaultable so leaf profiles (e.g. h100-qwen3) can extend via list.Concat
	// when a model base wants to add patches like poc-householder-compile.
	hw_patches: *[] | [...string]
	env: {
		VLLM_USE_V1: "1"
	}
}
