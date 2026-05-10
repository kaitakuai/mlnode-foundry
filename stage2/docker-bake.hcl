// Stage 2 bake config — placeholder. Real CI invocation lands in PR #2.
//
// `mlnode-foundry build-stage2` (Phase 3) will populate this dynamically
// from tools/stage2.lock.cue.

variable "BASE_IMAGE" {
  default = ""  // overridden by CI from stage2.lock.cue::stage1.digest
}

variable "UPSTREAM_REPO" {
  default = "gonka-ai/gonka"
}

variable "UPSTREAM_COMMIT" {
  default = ""  // overridden by CI from stage2.lock.cue::upstream.commit
}

variable "MLNODE_VERSION" {
  default = "0.2.13"
}

target "default" {
  context    = "."
  dockerfile = "stage2/Dockerfile.patch-and-build"
  args = {
    BASE_IMAGE      = BASE_IMAGE
    UPSTREAM_REPO   = UPSTREAM_REPO
    UPSTREAM_COMMIT = UPSTREAM_COMMIT
    MLNODE_VERSION  = MLNODE_VERSION
  }
  tags = [
    "ghcr.io/kaitakuai/mlnode-base:${MLNODE_VERSION}-vllm0.20.0-k1",
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}
