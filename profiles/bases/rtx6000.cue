// Profile base: RTX PRO 6000 Blackwell SE common tunings (sm_120).
// Consumer Blackwell needs FlashInfer JIT compile-on-first-launch
// (shipped wheel covers sm_120 but the JIT cache must be cleared so
// the actual host SM is detected) and the nvidia-* dev headers symlink
// for inline kernel compile. libcuda-compat fix is not applicable
// (consumer driver path differs).
package bases

RTX6000: {
	hw_patches: [
		"flashinfer-jit-uninstall",
		"nvidia-headers-symlinks",
	]
	env: {
		VLLM_USE_V1: "1"
	}
}
