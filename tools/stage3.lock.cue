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
	image: "ghcr.io/kaitakuai/vllm"
	// 0.20.0-pocv2 is the mutable tag; build chain pins by digest below.
	// Image rebuilt by kaitakuai/vllm build-stage1 CI on 2026-05-19 from
	// branch mb/feat/port-pocv2-vllm-0.20 HEAD = ccbe7cd8d (merge of PRs #9 + #10):
	//   - kaitakuai/vllm#9: restore seq_lens_cpu_upper_bound in
	//     _create_v1_attn_metadata; without it MLA backends crash on the
	//     first PoC step with `assert seq_lens_cpu is not None` (reported
	//     by Паша on b200-kimi-k2-6 0.2.13-q.int4-k2 image).
	//   - kaitakuai/vllm#10: actions/workflows/build-stage1.yml — makes
	//     future Stage 1 rebuilds reproducible CI artifacts (cosign, SLSA,
	//     SBOM) instead of manual `docker push` from a laptop.
	tag:    "0.20.0-pocv2"
	digest: "sha256:7955b84635f3138ec61bd612d682ad73588305113ed7f3291f34b786f4bb14df"
	cuda:   "13.0" // CUDA toolkit shipped inside Stage 2 (vLLM PoC base); used by dashboard renderer
}

stage3: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	tag:     "0.2.13-vllm0.20.0-k1"
}

patches: [
	"patches/0001-content-type-middleware.patch",
]
