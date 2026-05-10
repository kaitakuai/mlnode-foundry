# syntax=docker/dockerfile:1.7
#
# Stage 3 Dockerfile — Discovery spike skeleton.
#
# Inputs are passed as docker buildx --build-arg:
#   BASE_IMAGE        — Stage 2 image (or upstream image for spike)
#   GPU, MODEL, QUANT — identity axes (used for OCI labels)
#   PACKAGE_NAME, TAG — computed name + tag (used for labels only; actual
#                       --tag is passed separately by the CLI)
#   ENV_<KEY>         — ENV vars from profile.env, mapped to ENV inside image
#
# This skeleton:
#  - Inherits from BASE_IMAGE
#  - Sets a fixed set of known ENV vars from ENV_* build-args
#  - Adds OCI labels with kaitaku metadata
#
# Phase 3 will replace this with real Stage 3 (hw_patches application,
# runner_patch invocation, dynamic ENV from profile, etc.).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Identity (OCI labels)
ARG GPU
ARG MODEL
ARG QUANT
ARG PACKAGE_NAME
ARG TAG

# ENV from profile.env — known set for spike scope
ARG ENV_VLLM_USE_FLASHINFER_MOE_INT4=
ARG ENV_VLLM_FLASHINFER_MOE_BACKEND=
ARG ENV_VLLM_USE_V1=
ARG ENV_POC_BATCH_SIZE_DEFAULT=

ENV VLLM_USE_FLASHINFER_MOE_INT4=${ENV_VLLM_USE_FLASHINFER_MOE_INT4} \
    VLLM_FLASHINFER_MOE_BACKEND=${ENV_VLLM_FLASHINFER_MOE_BACKEND} \
    VLLM_USE_V1=${ENV_VLLM_USE_V1} \
    POC_BATCH_SIZE_DEFAULT=${ENV_POC_BATCH_SIZE_DEFAULT}

LABEL gonka.kaitaku.gpu="${GPU}" \
      gonka.kaitaku.model="${MODEL}" \
      gonka.kaitaku.quant="${QUANT}" \
      gonka.kaitaku.package="${PACKAGE_NAME}" \
      gonka.kaitaku.tag="${TAG}" \
      org.opencontainers.image.source="https://github.com/kaitakuai/mlnode-foundry" \
      org.opencontainers.image.licenses="Apache-2.0"
