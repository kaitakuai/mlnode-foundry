// Package tools — immutable upstream pin for Stage 3 (`mlnode-base`).
//
// CI reads this file, fetches the pinned upstream commit, applies
// `patches/*.patch`, runs upstream's mlnode/packages/api/Dockerfile with
// BASE_IMAGE overridden to our pinned Stage 2 (vllm-poc) digest, and publishes
// the resulting image to `ghcr.io/kaitakuai/mlnode-base:<tag>`.
//
// Bumped by:
//   - Renovate bot when a new vLLM PoC image (Stage 2) is published
//   - Human in PR when gonka-ai/gonka mlnode source needs bumping
//   - Human in PR to apply a new patch in patches/
//
// Stage 2 (vllm-poc) digest format: sha256:<64-hex> (immutable; tag is mutable metadata).
package tools

upstream: {
	repo: "gonka-ai/gonka"
	// f3b38936 = parent of 827d5ffe. The 827d5ffe commit IS the Content-Type middleware
	// fix; pinning here to its parent so patches/0001-content-type-middleware.patch
	// applies cleanly (vs trying to apply a patch on top of an already-patched commit).
	commit:         "f3b3893687d8a13a078955fad07879ea2b0ce2d0"
	mlnode_version: "0.2.13"
}

stage2: {
	// Repointed to the vllm-poc PLUGIN base (residual vLLM + gonka-poc package:
	// worker extension + composed entrypoint; see ADR-0013) as part of the
	// b300-minimax fork→plugin migration. The legacy `ghcr.io/kaitakuai/vllm`
	// monolith remains a valid Stage 2 lineage for the 5 non-migrated profiles
	// (a100/b200/h100/h200-minimax + b200-kimi) — the schema accepts both.
	image: "ghcr.io/kaitakuai/vllm-poc"
	// 0.23.0 is the vllm-poc tag (vLLM 0.23 residual base + gonka-poc); build
	// chain pins by digest below. Digest is a PLACEHOLDER (sixty-four zeros) so
	// `cue vet` passes the ^sha256:[a-f0-9]{64}$ regex; it is NOT a real image.
	// TODO(user): swap placeholder for real vllm-poc digest after build+push
	// (the S2 vllm-poc 0.23.0 image must be built and pushed first).
	tag:    "0.23.0"
	digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	cuda:   "13.0" // CUDA toolkit shipped inside Stage 2 (vllm-poc plugin base); used by dashboard renderer
}

stage3: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	tag:     "0.2.13-vllm0.23.0-k1"
}

patches: [
	"patches/0001-content-type-middleware.patch",
]
