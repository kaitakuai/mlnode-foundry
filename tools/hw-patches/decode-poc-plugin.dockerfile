# Decode-PoC plugin swap. Replaces the gonka-poc package baked into the base
# image (prefill scheme, v0.1.x line) with the decode-PoC branch, pinned by
# commit SHA. The decode branch DROPS the prefill scheme entirely — an image
# with this fragment cannot serve the current network's prefill PoC and is a
# CANDIDATE for the coordinated consensus switch only (see profile notes).
#
# Pin: kaitakuai/gonka-vllm-plugins @ feat/decode-poc-v2, HEAD c0a19daf
# (28 commits over v0.1.3; hardware-validated in the 2026-08-15/16 campaign:
# self-validation 0 at TP=1, 0.20-golden match under the consensus rule,
# cross-node tau=0.05 -> 0). Upstream WIP PR: gonka-ai/gonka-vllm-plugins#6.
#
# --no-deps: the base already carries vllm + runtime deps; the plugin must not
# drag anything in. --force-reinstall: the base has gonka-poc pre-installed and
# pip would otherwise consider the requirement satisfied. Tarball over
# git+https so the build needs no git binary.
COPY tools/runner-patches/decode-poc-plugin-check.py /tmp/decode-poc-plugin-check.py
RUN set -e; \
    PLUGIN_SHA="c0a19daf1342d8b5b6d886f18e21ca84d321b0db"; \
    pip install --no-cache-dir --no-deps --force-reinstall \
        "https://github.com/kaitakuai/gonka-vllm-plugins/archive/${PLUGIN_SHA}.tar.gz" && \
    python3 /tmp/decode-poc-plugin-check.py && \
    rm /tmp/decode-poc-plugin-check.py
