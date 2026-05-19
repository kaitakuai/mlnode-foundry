// Package tools — immutable upstream pin for Stage 2 (`mlnode-base`).
//
// CI reads this file, fetches the pinned upstream commit, applies
// `patches/*.patch`, runs upstream's mlnode/packages/api/Dockerfile with
// BASE_IMAGE overridden to our pinned Stage 1 digest, and publishes the
// resulting image to `ghcr.io/kaitakuai/mlnode-base:<tag>`.
//
// Bumped by:
//   - Renovate bot when a new vLLM PoC image (Stage 1) is published
//   - Human in PR when gonka-ai/gonka mlnode source needs bumping
//   - Human in PR to apply a new patch in patches/
//
// Stage 1 digest format: sha256:<64-hex> (immutable; tag is mutable metadata).
package tools

upstream: {
	repo: "gonka-ai/gonka"
	// f3b38936 = parent of 827d5ffe. The 827d5ffe commit IS the Content-Type middleware
	// fix; pinning here to its parent so patches/0001-content-type-middleware.patch
	// applies cleanly (vs trying to apply a patch on top of an already-patched commit).
	commit:         "f3b3893687d8a13a078955fad07879ea2b0ce2d0"
	mlnode_version: "0.2.13"
}

stage1: {
	image:  "ghcr.io/kaitakuai/vllm"
	tag:    "0.20.0-pocv2"  // existing kaitakuai/vllm tag (will retag to 0.20.0-poc-k1 in a separate kaitakuai/vllm PR)
	digest: "sha256:2025cd0dfd682bd66327959493e47ddcc45ec3c9dd9660e93086c9056e3fb819"
	cuda:   "13.0"          // CUDA toolkit shipped inside Stage 1 (vLLM PoC base); used by dashboard renderer
}

stage2: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	// k2 = k1 + CommonAttentionMetadata.seq_lens_cpu_upper_bound restore in
	// poc_model_runner.py (kwarg lost in Stage 1 0.20.0-pocv2 rebase; without
	// it MLA backends hit `assert seq_lens_cpu is not None` and every PoC
	// step on Kimi-K2.6 + B200 (CUTLASS_MLA / FLASHINFER_MLA) crashes).
	tag: "0.2.13-vllm0.20.0-k2"
}

patches: [
	"patches/0001-content-type-middleware.patch",
]
