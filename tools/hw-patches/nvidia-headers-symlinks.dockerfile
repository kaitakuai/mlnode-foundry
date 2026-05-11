# NVIDIA dev headers live inside the `nvidia-*` pip packages; FlashInfer JIT
# (and a few other JIT-compilers) look for them at the standard
# /usr/local/cuda/include path. Symlink every header from every nvidia-*
# package into that path.
#
# Some subpackages ship no *.h (e.g. nvtx stubs) — `if [ -f ... ]` so the
# missing-file branch returns 0 cleanly under `set -e`. The naive
# `[ -f X ] && ln -sf` chain trips set -e on empty globs.
RUN set -eux; \
    for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/include/; do \
      for f in "$d"*.h; do \
        if [ -f "$f" ]; then \
          ln -sf "$f" "/usr/local/cuda/include/$(basename "$f")"; \
        fi; \
      done; \
    done
