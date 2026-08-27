# Pin FlashInfer to the 0.6.18 nightly of 2026-08-19, at Crash_Bash_FL's
# request for the GLM-5.3-Flash bring-up.
#
# There is no stable 0.6.18: PyPI stops at 0.6.17, and so does the
# flashinfer.ai index. 0.6.18 exists only as release candidates (v0.6.18rc2 …
# rc9) and nightlies, so the pin has to be a nightly wheel URL rather than a
# version specifier. The base image ships 0.6.17, which is also what vLLM
# itself pins in requirements/cuda.txt and FLASHINFER_VERSION.
#
# All three distributions move together — flashinfer-python, flashinfer-cubin
# and flashinfer-jit-cache share a version, and jit-cache is additionally built
# per CUDA minor. The base is CUDA 13.0.1, hence +cu130.
#
# ORDER MATTERS: on profiles that also carry flashinfer-jit-uninstall, this
# fragment must come BEFORE it. All three distributions are version-checked
# against each other at import, so they move together; the uninstall then drops
# the jit-cache again, which is what a non-sm_120 GPU wants — JIT-compiling for
# the real arch beats shipping kernels built for another one. Reversed, this
# fragment would reinstall the cache the uninstall had just removed.
ARG FLASHINFER_NIGHTLY=0.6.18.dev20260819
ARG FLASHINFER_RELEASE=nightly-v0.6.18-20260819
ARG FLASHINFER_BASE_URL=https://github.com/flashinfer-ai/flashinfer/releases/download

RUN pip install --no-cache-dir --force-reinstall --no-deps \
      "${FLASHINFER_BASE_URL}/${FLASHINFER_RELEASE}/flashinfer_python-${FLASHINFER_NIGHTLY}-py3-none-any.whl" \
      "${FLASHINFER_BASE_URL}/${FLASHINFER_RELEASE}/flashinfer_cubin-${FLASHINFER_NIGHTLY}-py3-none-any.whl" \
      "${FLASHINFER_BASE_URL}/${FLASHINFER_RELEASE}/flashinfer_jit_cache-${FLASHINFER_NIGHTLY}+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" \
    && python3 -c "import flashinfer, sys; v = flashinfer.__version__; sys.exit(0 if v.startswith('0.6.18') else f'expected 0.6.18*, got {v}')"
