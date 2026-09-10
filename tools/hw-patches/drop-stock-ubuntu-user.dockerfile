# Drop the stock `ubuntu` account that Ubuntu 24.04 base images ship with
# UID/GID 1000. mlnode's packages/api/entrypoint.sh creates `appuser` with
# HOST_UID (default 1000) under `set -e`, so on a 24.04 base the container
# dies before `exec "$@"` with `useradd: UID 1000 is not unique` and three
# lines of log. First seen on ghcr.io/gonka-ai/mlnode:3.0.17 (vLLM 0.28 base,
# 24.04); 3.0.16 was 22.04 and had UID 1000 free. Found by Crash_Bash_FL on
# 4xB200 and 4xH200, 2026-09-10.
#
# Idempotent and a no-op on 22.04 bases (no such user). The upstream fix is
# the same line in mlnode/packages/api/Dockerfile.
RUN if id ubuntu >/dev/null 2>&1; then \
        userdel -r ubuntu 2>/dev/null || userdel ubuntu; \
        echo "drop-stock-ubuntu-user: removed stock 'ubuntu' (UID 1000) so entrypoint.sh can create appuser"; \
    else \
        echo "drop-stock-ubuntu-user: no stock 'ubuntu' user; no-op"; \
    fi
