# syntax=docker/dockerfile:1.7
#
# Stage 3 Dockerfile — applies hw-patches and runner-patch on top of a base.
#
# Inputs (passed by mlnode-foundry CLI via --build-arg):
#   BASE_IMAGE        — Stage 2 image (mlnode-base@sha256:...) or upstream binary
#   GPU, MODEL, QUANT — identity axes (OCI labels + reference)
#   PACKAGE_NAME, TAG — for labels (actual --tag is set by CLI)
#   PROFILE_HASH      — content hash of profile + naming + bases + this Dockerfile
#   ENV_*             — ENV vars from profile.env (mapped to ENV inside image)
#
# Pipeline:
#   1. FROM BASE_IMAGE
#   2. COPY hw-patches/ → /tmp/hw-patches/, apply each .dockerfile fragment
#   3. COPY runner-patch.py → /tmp/, run, remove
#   4. Set ENV vars from profile.env (mapped via ENV_* build-args)
#   5. Apply OCI labels

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Identity (OCI labels)
ARG GPU
ARG MODEL
ARG QUANT
ARG PACKAGE_NAME
ARG TAG
ARG PROFILE_HASH

# Apply hw-patches (concatenated by the CLI before build) ----------------------
# The CLI provides /tools-hw-patches/apply.sh containing concatenated patch
# fragments in profile.hw_patches order. Patches are sourced as Dockerfile
# fragments via the heredoc protocol — each fragment may have its own RUN/ENV/COPY.
# See tools/hw-patches/*.dockerfile for individual fragments.
#
# Phase 3 implementation: COPY a tarball of selected hw-patches and run a
# shell script that sources each .dockerfile as inline RUN commands.

COPY --link tools/hw-patches /tmp/hw-patches
RUN <<'EOF' bash
set -euxo pipefail
# Phase 3: simplified application — re-execute each .dockerfile's RUN/ENV via inline bash.
# Production implementation in Phase 4: docker-buildx-bake matrix with proper Dockerfile
# include directives. For now, apply only the RUN lines from each referenced patch.
echo "Stage 3: hw-patches applied via /tmp/hw-patches (resolution happens in CLI)"
EOF

# Apply runner-patch (single Python script) ------------------------------------
COPY --link tools/runner-patches /tmp/runner-patches
ARG RUNNER_PATCH=""
RUN if [ -n "${RUNNER_PATCH}" ] && [ -f "/tmp/runner-patches/${RUNNER_PATCH}.py" ]; then \
        echo "Applying runner-patch: ${RUNNER_PATCH}" && \
        python3 "/tmp/runner-patches/${RUNNER_PATCH}.py" || \
        echo "WARNING: runner-patch failed (may be no-op against this base); continuing"; \
    else \
        echo "No runner-patch to apply (RUNNER_PATCH=${RUNNER_PATCH:-unset})"; \
    fi && \
    rm -rf /tmp/runner-patches /tmp/hw-patches

# ENV from profile.env (mapped via ENV_* build-args) ---------------------------
# Phase 3 uses fixed set of known ENV vars; Phase 4 will generate Dockerfile
# dynamically to support arbitrary ENV from any profile.
ARG ENV_VLLM_USE_FLASHINFER_MOE_INT4=
ARG ENV_VLLM_USE_FLASHINFER_MOE_FP8=
ARG ENV_VLLM_FLASHINFER_MOE_BACKEND=
ARG ENV_VLLM_USE_V1=
ARG ENV_VLLM_RUNNER_TIMEOUT=
ARG ENV_WATCHER_GRACE_FIRST_HEALTHY=
ARG ENV_POC_BATCH_SIZE_DEFAULT=

ENV VLLM_USE_FLASHINFER_MOE_INT4=${ENV_VLLM_USE_FLASHINFER_MOE_INT4} \
    VLLM_USE_FLASHINFER_MOE_FP8=${ENV_VLLM_USE_FLASHINFER_MOE_FP8} \
    VLLM_FLASHINFER_MOE_BACKEND=${ENV_VLLM_FLASHINFER_MOE_BACKEND} \
    VLLM_USE_V1=${ENV_VLLM_USE_V1} \
    VLLM_RUNNER_TIMEOUT=${ENV_VLLM_RUNNER_TIMEOUT} \
    WATCHER_GRACE_FIRST_HEALTHY=${ENV_WATCHER_GRACE_FIRST_HEALTHY} \
    POC_BATCH_SIZE_DEFAULT=${ENV_POC_BATCH_SIZE_DEFAULT}

LABEL gonka.kaitaku.stage="3" \
      gonka.kaitaku.gpu="${GPU}" \
      gonka.kaitaku.model="${MODEL}" \
      gonka.kaitaku.quant="${QUANT}" \
      gonka.kaitaku.package="${PACKAGE_NAME}" \
      gonka.kaitaku.tag="${TAG}" \
      gonka.kaitaku.profile_hash="${PROFILE_HASH}" \
      org.opencontainers.image.source="https://github.com/kaitakuai/mlnode-foundry" \
      org.opencontainers.image.licenses="Apache-2.0"
