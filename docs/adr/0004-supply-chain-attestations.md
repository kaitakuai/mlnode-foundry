# ADR-0004 — Supply-chain attestations

**Status:** Accepted
**Date:** 2026-05-10

## Context

Node operators pull images from GHCR and need to verify they are produced by our CI from a known commit, with a known dependency tree, and not tampered with. Standard 2026 supply chain practices: SLSA L3 provenance, SBOM, cryptographic signing.

## Decision

Every published Stage 2 and Stage 3 image carries:

1. **SLSA L3 provenance attestation** via BuildKit `--attest type=provenance,mode=max` — records build environment, source commit, and command line
2. **SBOM (SPDX format)** via BuildKit `--sbom=true` — full pip + apt manifest
3. **Rendered Dockerfile attestation** via BuildKit `--attest type=dockerfile` (in-toto predicate) — exact content used to build this digest
4. **Cosign signature** via [Sigstore Public Good Instance](https://www.sigstore.dev/) — keyless flow using GitHub OIDC token from the `build-stage*.yml` workflow

All four artifacts are stored as OCI Referrers next to the image in GHCR, downloadable via `cosign download attestation` or `crane manifest`.

## Verify-on-pull (dashboard sync)

Dashboard `sync_registry.py` runs `cosign verify` with identity-regex pinned to `https://github.com/kaitakuai/mlnode-foundry/.github/workflows/.*` before recording a new image. Mismatched signatures are rejected.

## Consequences

- **No private signing keys** to manage — keyless OIDC eliminates that class of secret
- **Free** (Sigstore Public Good)
- **Verifiable by anyone** — cosign verify against our public OIDC issuer regex
- **Standard format** — SPDX SBOM, in-toto attestations, OCI Referrers — interoperable with any tooling

## Alternatives considered

- **PGP-signed releases**: rejected — key management overhead
- **Internal-only verification**: rejected — supply chain transparency is the point
