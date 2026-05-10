// Profile base: H100 Hopper common tunings.
// H100 needs minimal hw-patches (mature CUDA support, no SM-specific fixes).
package bases

H100: {
	hw_patches: []
	env: {
		VLLM_USE_V1: "1"
	}
}
