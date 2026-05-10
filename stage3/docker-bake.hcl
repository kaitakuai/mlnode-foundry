// Stage 3 bake config — placeholder, populated dynamically by Python CLI.
//
// `mlnode-foundry build <profile>` (per ADR-0008) computes build args from
// the profile + naming.cue and invokes `docker buildx build` directly.
// This HCL file exists so `docker buildx bake` works for manual debugging
// (e.g., `BASE_IMAGE=... GPU=... docker buildx bake -f stage3/docker-bake.hcl`).

variable "BASE_IMAGE" {
  default = ""  // resolved by CLI from tools/stage2.lock.cue OR profile overlay
}

variable "GPU" {
  default = ""
}

variable "MODEL" {
  default = ""
}

variable "QUANT" {
  default = ""
}

variable "PACKAGE_NAME" {
  default = ""
}

variable "TAG" {
  default = ""
}

target "default" {
  context    = "."
  dockerfile = "stage3/Dockerfile"
  args = {
    BASE_IMAGE   = BASE_IMAGE
    GPU          = GPU
    MODEL        = MODEL
    QUANT        = QUANT
    PACKAGE_NAME = PACKAGE_NAME
    TAG          = TAG
  }
  tags = [
    "${PACKAGE_NAME}:${TAG}",
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}
