# Scheduler KeyError guard — S4 form of kaitakuai/vllm#19, for images that
# overlay gonka's published mlnode instead of building from our residual tree.
# Fails the build if vLLM's scheduler no longer matches; no-op once the fix
# reaches the base image.
COPY tools/runner-patches/sched-req-index-guard.py /tmp/sched-req-index-guard.py
RUN python3 /tmp/sched-req-index-guard.py && rm /tmp/sched-req-index-guard.py
