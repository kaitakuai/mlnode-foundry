# Make MLNode's PoC batch-size default read POC_BATCH_SIZE_DEFAULT.
#
# gonka_poc falls back to that variable only when the request carries no
# batch_size. dapi omits it on purpose, but MLNode's own request models declare
# `batch_size: int = 32`, so the proxied payload always carries 32 and the env
# knob never applies on a real node. Batch 32 is fatal for GLM-5.3-Flash on
# Hopper (FlashInfer sparse-MLA illegal memory access on H200, OOM on 8xH100).
#
# Pairs with the profile's POC_BATCH_SIZE_DEFAULT; a no-op for the value when
# the variable is unset, since the rewritten default still falls back to 32.
# See tools/runner-patches/poc-batch-size-from-env.py for the script body.
COPY tools/runner-patches/poc-batch-size-from-env.py /tmp/poc-batch-size-from-env.py
RUN python3 /tmp/poc-batch-size-from-env.py && rm /tmp/poc-batch-size-from-env.py
