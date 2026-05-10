// Profile base: B300 Blackwell Ultra (sm_103a) common tunings.
// Imported by all b300-* profiles and unified with leaf overrides.
package bases

B300: {
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
	]
	env: {
		VLLM_USE_V1:                  "1"
		VLLM_RUNNER_TIMEOUT:          "3600"
		WATCHER_GRACE_FIRST_HEALTHY:  "1"
	}
}
