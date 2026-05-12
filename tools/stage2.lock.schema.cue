// Package tools — schema for the Stage 2 immutable pin (stage2.lock.cue).
//
// Validate with: `cue vet tools/stage2.lock.cue tools/stage2.lock.schema.cue`
// Run as part of the validate-profiles workflow.
//
// The pattern: stage2.lock.cue holds CONCRETE values; this file holds the
// TYPE definitions and per-field semantic guidance. A human or AI editing
// the lock file reads field-level comments to know what each value MUST
// satisfy and WHY the constraint exists.

package tools

#Stage2Lock: {
	// Pinned upstream gonka-ai/gonka source.
	// Bumped by Renovate on new Stage 1 images and by humans on patch updates.
	upstream: #Upstream

	// Stage 1 base image — the immutable digest the Stage 2 build pulls from.
	// Provenance for the chain: Stage 2 → Stage 1 → upstream vLLM.
	stage1: #Stage1

	// Stage 2 publish target. CI uses these coordinates verbatim.
	stage2: #Stage2

	// Source patches applied to the upstream tree before building.
	// Order matters; conflicts are detected by `git apply --3way`.
	// Each entry MUST be a path relative to repo root.
	patches: [...=~"^patches/[0-9]{4}-.+\\.patch$"]
}

#Upstream: {
	// GitHub org/repo of upstream source (always 'gonka-ai/gonka' today).
	repo: =~"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"

	// 40-char SHA of the upstream commit Stage 2 builds from.
	// MUST be the parent of the commit that introduces our patches/
	// so the patches apply cleanly (otherwise git apply detects already-applied).
	commit: =~"^[a-f0-9]{40}$"

	// Human-readable mlnode version string — used in OCI labels and tags.
	// Format: semver (X.Y.Z[-suffix]); does NOT need to match upstream tag.
	mlnode_version: =~"^[0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9.]+)?$"
}

#Stage1: {
	// Full GHCR package path (no tag) of the kaitakuai/vllm PoC base.
	image: =~"^ghcr\\.io/kaitakuai/vllm$"

	// Human-readable tag of the Stage 1 image. NOT used for builds (digest is);
	// kept for traceability and dashboard. Must not be 'latest' (mutable tag policy).
	tag: string & !="latest"

	// Immutable content-addressable Stage 1 digest. THIS is what build-stage2
	// passes as --build-arg BASE_IMAGE. Pinning to the digest, not the tag, is
	// the only way to guarantee Stage 2 reproducibility.
	digest: =~"^sha256:[a-f0-9]{64}$"

	// CUDA toolkit version baked into Stage 1. Surfaced in registry-view's
	// `cuda` field so the dashboard can render compatibility info.
	cuda: =~"^[0-9]+\\.[0-9]+$"
}

#Stage2: {
	// Full GHCR package path (no tag) where Stage 2 publishes (mlnode-base).
	package: =~"^ghcr\\.io/kaitakuai/mlnode-base$"

	// Tag for the Stage 2 publish. Convention: <mlnode_version>-vllm<vllm_version>-k<rev>.
	// Bumped on every patch change (k-rev counter) so old Stage 2 tags stay pullable.
	tag: =~"^[0-9]+\\.[0-9]+\\.[0-9]+-vllm[0-9]+\\.[0-9]+\\.[0-9]+-k[0-9]+$"
}

// Top-level keys MUST conform to #Stage2Lock.
upstream: #Upstream
stage1:   #Stage1
stage2:   #Stage2
patches: [...string]
