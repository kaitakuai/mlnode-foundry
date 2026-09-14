"""Make MLNode's PoC batch-size default read POC_BATCH_SIZE_DEFAULT.

`gonka_poc` takes the PoC batch size from the request and falls back to the
POC_BATCH_SIZE_DEFAULT environment variable. MLNode never reads that variable:
`pow_v2_routes.py` declares `batch_size: int = 32` on both request models, so
FastAPI fills 32 in, `body.model_dump()` puts it in the payload, and the
plugin sees an explicit value. The chain relies on this default -- dapi omits
batch_size on purpose ("MLNode will use its default", poc_v2_requests.go) --
so on a real node the env knob is dead and every model runs at 32.

That is fatal on Hopper with GLM-5.3-Flash: batch 32 hits an illegal memory
access in the FlashInfer sparse-MLA kernel on H200 and goes OOM on 8xH100,
while batch 16 runs for hours. Same knob, same dead end, on the h100 DeepSeek
and Kimi profiles.

This rewrites both defaults to read the environment at import time, keeping 32
when the variable is unset. Idempotent; fails loudly if the anchors move.
"""
import os
import sys

_CANDIDATES = (
    "/app/packages/api/src/api/inference/pow_v2_routes.py",
    "/app/src/api/inference/pow_v2_routes.py",
)
FILE = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])

MARKER = "    batch_size: int = 32\n"
REPLACEMENT = '    batch_size: int = int(os.environ.get("POC_BATCH_SIZE_DEFAULT", "32"))\n'
IMPORT_ANCHOR = "import asyncio\n"
IMPORT_LINE = "import os\n"


def main() -> int:
    with open(FILE) as f:
        src = f.read()

    if "POC_BATCH_SIZE_DEFAULT" in src:
        print("poc-batch-size-from-env: already patched; no-op")
        return 0

    count = src.count(MARKER)
    if count != 2:
        sys.stderr.write(
            f"ERROR: poc-batch-size-from-env: expected 2 batch_size defaults in {FILE}, "
            f"found {count}. MLNode's PoC routes may have been refactored - re-verify.\n"
        )
        return 1
    src = src.replace(MARKER, REPLACEMENT)

    if IMPORT_LINE not in src:
        if IMPORT_ANCHOR not in src:
            sys.stderr.write("ERROR: poc-batch-size-from-env: import anchor not found.\n")
            return 1
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)

    with open(FILE, "w") as f:
        f.write(src)
    print(f"poc-batch-size-from-env: patched {count} batch_size defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
