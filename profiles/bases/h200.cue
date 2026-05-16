// Profile base: H200 Hopper common tunings.
// Same SM family as H100 (sm_90a) — no SM-specific patches needed.
// Distinguished from H100 mainly by larger HBM (141 GB vs 80 GB).
package bases

H200: {
	hw_patches: *[] | [...string]
	env: {
		VLLM_USE_V1: "1"
	}
}
