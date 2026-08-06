# DeepSeek-V4-Flash-0731 needs the V2 model runner reachable for DSpark
# speculation (V2-runner-only; config/vllm.py force-selects V2 when
# --speculative-config method=dspark is set). The S2 base image
# (docker/Dockerfile.gonka-poc) bakes VLLM_USE_V2_MODEL_RUNNER=0 and the GPU
# bases set VLLM_USE_V1=1, both of which pin V1 and make DSpark impossible.
# Replay validation on V2 is correct since vllm#92 + kaitakuai/vllm#18, so
# clearing the pins is safe: vLLM then auto-selects V1 for plain serving and
# V2 only where DSpark asks for it. Empty ENV overrides the inherited value.
ENV VLLM_USE_V2_MODEL_RUNNER=""
ENV VLLM_USE_V1=""
