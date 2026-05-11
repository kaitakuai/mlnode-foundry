# Security policy

## Scope

This policy covers:

- Published container images under `ghcr.io/kaitakuai/mlnode-full` and `ghcr.io/kaitakuai/mlnode-overlay` (every Kaitaku-revision tag).
- The build pipeline that produces them — [`.github/workflows/build-full.yml`](.github/workflows/build-full.yml), [`scripts/build-full.sh`](scripts/build-full.sh), [`scripts/build-overlay.sh`](scripts/build-overlay.sh), the generator in [`tools/`](tools/), and the Dockerfiles/`docker-bake.hcl` under [`full/`](full/) and [`overlay/`](overlay/).
- The trusted-source pinning process — [`.github/trusted-sources.yaml`](.github/trusted-sources.yaml) and the policy in [`.github/ALLOWLIST.md`](.github/ALLOWLIST.md).

Out of scope: vulnerabilities in the upstream projects themselves (`gonka-ai/gonka`, `product-science/mlnode`, `vllm-project/vllm`). Those belong upstream; we will coordinate disclosure if a Kaitaku image ships a known-vulnerable pin.

## Why this matters

Gonka operators who run these images stake collateral and face slashing risk on an incompatible or compromised release. A supply-chain vulnerability in a Kaitaku image is therefore high-blast-radius: it can zero stakes across every operator that pulls the affected tag. We take reports accordingly.

## Reporting a vulnerability

**Do not file a public GitHub issue.** Report privately:

- Preferred: GitHub's [private vulnerability reporting](https://github.com/kaitakuai/mlnode/security/advisories/new) on this repository.
- Fallback: email `vakula.kolia@gmail.com` with subject `[kaitakuai/mlnode security]`.

Include:

- The affected image tag (or script/workflow path) and digest if applicable.
- Steps to reproduce, or a proof-of-concept if you have one.
- Your assessment of impact and who is at risk.
- Whether you are willing to be credited publicly.

We aim to acknowledge within 72 hours and to agree on a fix timeline within one week of acknowledgement. We do not offer a paid bounty.

## Coordinated disclosure

- We prefer coordinated disclosure: we will not publish details until a fix is available to operators, unless exploitation is already observed in the wild.
- If the root cause lives upstream (for example in `product-science/mlnode` or a pinned pip package), we will loop in the upstream maintainer and coordinate a joint advisory. Credit in the advisory goes to the reporter unless they ask otherwise.

## What constitutes a security issue

Examples of in-scope findings:

- A mechanism by which an unexpected commit SHA / OCI digest lands in [`.github/trusted-sources.yaml`](.github/trusted-sources.yaml) without CODEOWNER review.
- A way to make the build produce an image whose layers do not match the pinned upstream.
- A way to bypass the drift check for [`full/Dockerfile`](full/Dockerfile) so a hand-edited Dockerfile ships.
- A privilege-escalation or credential-exfiltration primitive reachable from pulling one of the published images with default environment.
- A workflow misconfiguration granting write access to GHCR that we did not intend.

Out-of-scope examples (please do not report these privately — open a normal GitHub issue or PR):

- Documentation typos, broken links, stale examples.
- Minor CI timeouts or flakiness.
- Operator-side configuration questions.
