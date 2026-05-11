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
	repo:           "gonka-ai/gonka"
	commit:         "827d5ffe401f0482c46090fbf79ec693b385a5b0"
	mlnode_version: "0.2.13"
}

stage1: {
	image:  "ghcr.io/kaitakuai/vllm"
	tag:    "0.20.0-pocv2"  // existing kaitakuai/vllm tag (will retag to 0.20.0-poc-k1 in a separate kaitakuai/vllm PR)
	digest: "sha256:2025cd0dfd682bd66327959493e47ddcc45ec3c9dd9660e93086c9056e3fb819"
}

stage2: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	tag:     "0.2.13-vllm0.20.0-k1"
}

patches: [
	"patches/0001-content-type-middleware.patch",
]
