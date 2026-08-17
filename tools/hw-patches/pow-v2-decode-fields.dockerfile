# Decode-PoC pass-through in MLNode's pow_v2 routes — Stage-4 mirror of
# kaitakuai/gonka@feat/decode-poc-mlnode 86ff63d (decode params + k_points_steps
# + decode stat-test defaults + backend-owned batch_size). Anchored, additive,
# fails the build loudly if the base's pow_v2_routes.py drifted; a no-op once
# the base image ships the mlnode fix itself.
COPY tools/runner-patches/pow-v2-decode-fields.py /tmp/pow-v2-decode-fields.py
RUN python3 /tmp/pow-v2-decode-fields.py && rm /tmp/pow-v2-decode-fields.py
