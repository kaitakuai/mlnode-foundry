# Remove precompiled FlashInfer JIT cache. The shipped pip package targets
# sm_120; on any other Blackwell-class SM (sm_103a, etc.) those kernels miss
# entirely and FlashInfer falls back to a slow path. Uninstalling forces
# JIT-compile-on-first-launch for the actual GPU (~10 min first run, then
# cached at /root/.cache/flashinfer/<ver>/<arch>/).
#
# The `rm -rf` afterwards is mandatory: pip leaves the now-empty
# `flashinfer_jit_cache/` directory in site-packages, which `importlib.util
# .find_spec("flashinfer_jit_cache")` still resolves as a *namespace package*.
# In FlashInfer 0.6.8.post1 (vLLM 0.20.0 base image) `flashinfer/jit/env.py`
# unconditionally reads `flashinfer_jit_cache.__version__` when find_spec
# returns non-None — and a namespace package has no `__version__` attribute,
# so vLLM fails to import flashinfer at startup with
# `AttributeError: module 'flashinfer_jit_cache' has no attribute '__version__'`.
# Deleting the empty dir makes find_spec return None, and FlashInfer falls
# back to its default in-package AOT path. (FlashInfer 0.6.6 / vLLM 0.19 base
# did not query `__version__` at import time, so the rm was not needed there;
# the regression is specific to the 0.6.8.post1 base.)
RUN pip uninstall flashinfer-jit-cache -y && \
    rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache
