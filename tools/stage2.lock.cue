// Package stage2_lock — immutable upstream pin for Stage 2 (`mlnode-base`).
//
// This file declares what to build Stage 2 FROM. CI reads it, fetches the
// pinned upstream commit, applies `patches/*.patch`, runs upstream Dockerfile,
// publishes the resulting image to `ghcr.io/kaitakuai/mlnode-base:<tag>`.
//
// Bumped by:
//   - Renovate bot when a new vLLM PoC image (Stage 1) is published
//   - Renovate bot when gonka-ai/gonka mlnode source bumps mlnode-ver
//   - Human in PR to apply a new patch in patches/
//
// **PLACEHOLDER** — real upstream commit + Stage 1 digest land in PR #2 (Phase 3).
// Until then, Stage 3 builds use `product-science/mlnode:3.0.13-alpha5` directly
// as BASE_IMAGE, skipping Stage 2 entirely.
package tools

upstream: {
	repo:           "gonka-ai/gonka"
	commit:         "PLACEHOLDER_REPLACE_IN_PHASE_3"  // 40-char SHA expected; will be validated by render-bake
	mlnode_version: "0.2.13"
}

stage1: {
	image:  "ghcr.io/kaitakuai/vllm"
	tag:    "0.20.0-poc-k1"
	digest: "sha256:PLACEHOLDER_REPLACE_IN_PHASE_3"  // 64-char hex expected
}

stage2: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	tag:     "0.2.13-vllm0.20.0-k1"
}

patches: [
	"patches/0001-content-type-middleware.patch",
]
