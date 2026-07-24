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
	// Pinned to our fork branch kaitakuai/gonka `feat/mlnode-metrics-exporter-v0.2.14`
	// (26f7db5 = mlnode 0.2.14 + node-side metrics exporter #14, rebased onto
	// gonka-ai/gonka v0.2.14 ee730031). Sourced from the fork because the exporter is
	// not yet merged upstream; recorded in the kaitakuai/gonka fork README per rules.
	// patches/0001 and 0003 were regenerated to apply directly onto this commit — the
	// exporter shifted context around their anchors in runner.py/proxy.py/app.py.
	repo:           "kaitakuai/gonka"
	commit:         "26f7db559711ab5368633a46c70a192472e2fa12"
	mlnode_version: "0.2.14"
}

stage2: {
	// Repointed to the vllm-poc PLUGIN base (residual vLLM + gonka-poc package:
	// worker extension + composed entrypoint; see ADR-0013) as part of the
	// b300-minimax fork→plugin migration. The legacy `ghcr.io/kaitakuai/vllm`
	// monolith remains a valid Stage 2 lineage for the 5 non-migrated profiles
	// (a100/b200/h100/h200-minimax + b200-kimi) — the schema accepts both.
	image: "ghcr.io/kaitakuai/vllm-poc"
	// 0.25.1 is the vllm-poc tag (vLLM 0.25.1 residual S1 + gonka-poc plugin
	// @e24861a03). Build chain pins by the digest below — the verified working S2
	// (residual + out-of-tree plugin, DeepSeek-V4 capable). Was 0.23.0 @835aa90…
	// before the 0.23 -> 0.25.1 base migration.
	tag:    "0.25.1"
	digest: "sha256:72423e85385aad9eb0e4de38d82dd4b554baf72f679d8f8cd27adb8f966c3976"
	// CUDA 13.0: the residual S1 bases on vLLM's recommended default image
	// (vllm/vllm-openai:v0.25.1 → CUDA 13.0.2), not the pinned cu129 (12.9).
	// This matches the 5 fat-fork 0.20 profiles (also CUDA 13.0), so the single
	// SHARED cuda field is accurate for every profile — no per-profile cuda
	// split needed.
	cuda: "13.0"
}

stage3: {
	package: "ghcr.io/kaitakuai/mlnode-base"
	tag:     "0.2.14-vllm0.25.1-k1"
}

patches: [
	"patches/0001-content-type-middleware.patch",
	"patches/0002-api-watcher-grace.patch",
	"patches/0003-mlnode-heartbeat-liveness.patch",
]
