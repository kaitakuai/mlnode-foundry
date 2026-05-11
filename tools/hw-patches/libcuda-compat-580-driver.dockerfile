# libcuda compat symlink fix. NVIDIA driver 580.126 (current Cherry/Hetzner
# default for B300/Blackwell hosts) ships libcuda.so.1 at
# /usr/lib/x86_64-linux-gnu instead of the legacy /usr/local/cuda/compat path.
# The bundled CUDA compat directory keeps a stub libcuda.so that the runtime
# loader picks up FIRST, hiding the real driver and breaking GPU detection.
# Replace the stub with a symlink to the real driver so dynamic linking
# resolves correctly.
#
# Path uses the `/usr/local/cuda` symlink so the same fragment applies to
# kaitakuai/vllm:v0.19.0-pocv2-* (CUDA 12.9) and kaitakuai/vllm:0.20.0-pocv2
# (CUDA 13.0).
RUN rm -f /usr/local/cuda-*/compat/libcuda.so* && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
           /usr/local/cuda/compat/libcuda.so
