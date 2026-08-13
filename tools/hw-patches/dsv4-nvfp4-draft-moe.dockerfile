# DSpark draft experts off the NVFP4 path — S4 form of kaitakuai/vllm#20, for
# images that overlay gonka's published mlnode. Only the b300 DeepSeek leaf
# references this: NVFP4 is the technically-primary variant on B300 alone.
# Fails the build if the quant config no longer matches; no-op once the fix
# reaches the base image.
COPY tools/runner-patches/dsv4-nvfp4-draft-moe.py /tmp/dsv4-nvfp4-draft-moe.py
RUN python3 /tmp/dsv4-nvfp4-draft-moe.py && rm /tmp/dsv4-nvfp4-draft-moe.py
