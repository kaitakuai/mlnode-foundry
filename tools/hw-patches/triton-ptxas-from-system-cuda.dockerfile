# Triton bundled `ptxas` lacks newer SM targets (sm_103a, sm_120, etc.) —
# overwrite with the system CUDA `ptxas`, which knows the full set.
# Required for any Blackwell+ variant (B300 sm_103a today, RTX PRO 6000 SM_120
# tomorrow) where Triton fails to emit native-code for the actual GPU.
#
# Path uses the `/usr/local/cuda` symlink (resolves to `cuda-12.9` on
# kaitakuai/vllm:v0.19.0-pocv2-* and to `cuda-13.0` on
# kaitakuai/vllm:0.20.0-pocv2). Same fragment works against both bases.
RUN cp /usr/local/cuda/bin/ptxas \
      /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas

ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
