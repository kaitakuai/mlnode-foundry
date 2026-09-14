# Force MLNode's PoC batch size to the image's POC_BATCH_SIZE_DEFAULT.
#
# The batch size belongs to the hardware the image was tuned for, so the image
# outranks the caller. MLNode proxies PoC requests to the vLLM backends in two
# places; both are rewritten to overwrite batch_size when the variable is set,
# and left alone when it is not.
#
# Without this the variable is dead on the production path: gonka_poc honours it
# only when the request omits batch_size, dapi omits it on purpose, and MLNode
# fills its own hardcoded 32 in. Batch 32 is fatal for GLM-5.3-Flash on Hopper
# (FlashInfer sparse-MLA illegal memory access on H200, OOM on 8xH100).
#
# See tools/runner-patches/poc-batch-size-force.py for the script body.
COPY tools/runner-patches/poc-batch-size-force.py /tmp/poc-batch-size-force.py
RUN python3 /tmp/poc-batch-size-force.py && rm /tmp/poc-batch-size-force.py
