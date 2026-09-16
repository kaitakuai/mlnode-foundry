# Stage-4 form of gonka-ai/vllm#111: drop the replay max_tokens pin from
# kaitakuai/vllm#21. The pin rewrote SamplingParams after validation, so a
# client request with min_tokens above the recorded length killed the engine's
# input thread while /health stayed green (two GLM-5.3-Flash hangs, 2026-09-16).
# The replay block goes back to what release/v0.25.1 ships. No-op once the base
# residual is past #111.
# See tools/runner-patches/replay-max-tokens-pin-revert.py for the script body.
COPY tools/runner-patches/replay-max-tokens-pin-revert.py /tmp/replay-max-tokens-pin-revert.py
RUN python3 /tmp/replay-max-tokens-pin-revert.py && rm /tmp/replay-max-tokens-pin-revert.py
