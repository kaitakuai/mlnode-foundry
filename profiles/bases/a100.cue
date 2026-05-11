// Profile base: A100 Ampere common tunings.
// Mature CUDA support (sm_80) — no SM-specific patches needed.
package bases

A100: {
	hw_patches: []
	env: {
		VLLM_USE_V1: "1"
	}
}
