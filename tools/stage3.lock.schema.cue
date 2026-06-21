// Package tools — schema for the Stage 3 immutable pin (stage3.lock.cue).
//
// Validate with: `cue vet tools/stage3.lock.cue tools/stage3.lock.schema.cue`
// Run as part of the validate-profiles workflow.
//
// The pattern: stage3.lock.cue holds CONCRETE values; this file holds the
// TYPE definitions and per-field semantic guidance. A human or AI editing
// the lock file reads field-level comments to know what each value MUST
// satisfy and WHY the constraint exists.

package tools

#Stage3Lock: {
	// Pinned upstream gonka-ai/gonka source.
	// Bumped by Renovate on new Stage 2 (vllm-poc) images and by humans on patch updates.
	upstream: #Upstream

	// Stage 2 (vllm-poc) base image — the immutable digest the Stage 3 build pulls from.
	// Provenance for the chain: Stage 3 → Stage 2 (vllm-poc) → upstream vLLM.
	stage2: #Stage2

	// Stage 3 publish target. CI uses these coordinates verbatim.
	stage3: #Stage3

	// Source patches applied to the upstream tree before building.
	// Order matters; conflicts are detected by `git apply --3way`.
	// Each entry MUST be a path relative to repo root.
	patches: [...=~"^patches/[0-9]{4}-.+\\.patch$"]
}

#Upstream: {
	// GitHub org/repo of upstream source (always 'gonka-ai/gonka' today).
	repo: =~"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"

	// 40-char SHA of the upstream commit Stage 3 builds from.
	// MUST be the parent of the commit that introduces our patches/
	// so the patches apply cleanly (otherwise git apply detects already-applied).
	commit: =~"^[a-f0-9]{40}$"

	// Human-readable mlnode version string — used in OCI labels and tags.
	// Format: semver (X.Y.Z[-suffix]); does NOT need to match upstream tag.
	mlnode_version: =~"^[0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9.]+)?$"
}

#Stage2: {
	// Full GHCR package path (no tag) of the Stage 2 base. Either the
	// vllm-poc plugin base (residual + gonka-poc) OR the legacy vllm monolith.
	// `vllm`     → legacy fat-fork monolith (PoC math baked into the vLLM tree).
	// `vllm-poc` → plugin base: official-style residual vLLM + the gonka-poc
	//              package (worker extension + composed entrypoint) layered on top
	//              (see ADR-0013). Both are valid Stage 2 publish targets during
	//              the fork→plugin migration; profiles pick which lineage they
	//              build on via the digest pinned in stage2.digest below.
	image: =~"^ghcr\\.io/kaitakuai/vllm(-poc)?$"

	// Human-readable tag of the Stage 2 base image. NOT used for builds (digest is);
	// kept for traceability and dashboard. Must not be 'latest' (mutable tag policy).
	// vllm-poc plugin base (residual + gonka-poc) OR legacy vllm monolith.
	tag: string & !="latest"

	// Immutable content-addressable Stage 2 digest. THIS is what build-stage3
	// passes as --build-arg BASE_IMAGE. Pinning to the digest, not the tag, is
	// the only way to guarantee Stage 3 reproducibility. Points at either the
	// vllm-poc plugin base (residual + gonka-poc) OR the legacy vllm monolith.
	digest: =~"^sha256:[a-f0-9]{64}$"

	// CUDA toolkit version baked into Stage 2 (vllm-poc plugin base, residual +
	// gonka-poc, OR legacy vllm monolith). Surfaced in registry-view's `cuda`
	// field so the dashboard can render compatibility info.
	cuda: =~"^[0-9]+\\.[0-9]+$"
}

#Stage3: {
	// Full GHCR package path (no tag) where Stage 3 publishes (mlnode-base).
	package: =~"^ghcr\\.io/kaitakuai/mlnode-base$"

	// Tag for the Stage 3 publish. Convention: <mlnode_version>-vllm<vllm_version>-k<rev>.
	// Bumped on every patch change (k-rev counter) so old Stage 3 tags stay pullable.
	tag: =~"^[0-9]+\\.[0-9]+\\.[0-9]+-vllm[0-9]+\\.[0-9]+\\.[0-9]+-k[0-9]+$"
}

// Top-level keys MUST conform to #Stage3Lock.
upstream: #Upstream
stage2:   #Stage2
stage3:   #Stage3
patches: [...string]
