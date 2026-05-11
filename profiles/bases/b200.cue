// Profile base: B200 Blackwell common tunings (sm_100).
// Same compile-time fixes as B300 minus cold-start-tolerance
// (B200 with Qwen FP8 has fast cold start; add per-leaf if using
// Kimi-INT4 or other slow-init models).
package bases

B200: {
	// Default list; leaves may override (e.g., b200-kimi-int4 adds cold-start-tolerance).
	hw_patches: *[
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
	] | [...string]
	env: {
		VLLM_USE_V1: "1"
	}
}
