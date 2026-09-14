"""Force MLNode's PoC batch size to POC_BATCH_SIZE_DEFAULT from the image.

The batch size is a property of the hardware the image was tuned for, so the
image outranks the caller. MLNode proxies PoC requests to the vLLM backends in
two places; both take the payload from `body.model_dump()`, which carries
whatever batch_size the request had, or MLNode's own hardcoded default of 32
when it had none. This rewrites both sites to overwrite the field when
POC_BATCH_SIZE_DEFAULT is set, and leaves them alone when it is not.

Why it is needed at all: gonka_poc falls back to POC_BATCH_SIZE_DEFAULT only
when the request omits batch_size, and MLNode never omits it. dapi does omit it
on purpose -- "MLNode will use its default", poc_v2_requests.go, on both the
generation and the validation structs -- so before this patch every node ran at
32 regardless of the image. That is fatal for GLM-5.3-Flash on Hopper: batch 32
hits an illegal memory access in the FlashInfer sparse-MLA kernel on H200 and
goes OOM on 8xH100, where batch 16 ran 66 minutes for 101k nonces.

Forcing rather than defaulting costs one thing: a sweep that posts batch_size
through MLNode no longer takes effect. Talk to the plugin directly, or unset
the variable, when sweeping.

Idempotent; fails loudly if the anchors move.
"""
import os
import sys

_CANDIDATES = (
    "/app/packages/api/src/api/inference/pow_v2_routes.py",
    "/app/src/api/inference/pow_v2_routes.py",
)
FILE = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])

IMPORT_ANCHOR = "import asyncio\n"
IMPORT_LINE = "import os\n"
HELPER_ANCHOR = 'router = APIRouter(prefix="/inference/pow", tags=["PoC v2"])\n'
HELPER = '''
_POC_BATCH_SIZE = os.environ.get("POC_BATCH_SIZE_DEFAULT")


def _poc_payload(body) -> dict:
    """Backend payload with the image's PoC batch size forced.

    POC_BATCH_SIZE_DEFAULT states what this hardware can run, so it outranks
    the caller. Unset leaves the request in charge.
    """
    payload = body.model_dump()
    if _POC_BATCH_SIZE:
        payload["batch_size"] = int(_POC_BATCH_SIZE)
    return payload
'''
SITES = (
    ("        payload = body.model_dump()\n", "        payload = _poc_payload(body)\n"),
    ('await call_backend(port, "POST", "/api/v1/pow/generate", body.model_dump())',
     'await call_backend(port, "POST", "/api/v1/pow/generate", _poc_payload(body))'),
)


def main() -> int:
    with open(FILE) as f:
        src = f.read()

    if "_poc_payload" in src:
        print("poc-batch-size-force: already patched; no-op")
        return 0

    for old, _ in SITES:
        if src.count(old) != 1:
            sys.stderr.write(
                f"ERROR: poc-batch-size-force: expected exactly one occurrence of\n"
                f"  {old.strip()}\nin {FILE}, found {src.count(old)}. "
                "MLNode's PoC routes may have been refactored - re-verify.\n"
            )
            return 1
    if HELPER_ANCHOR not in src or IMPORT_ANCHOR not in src:
        sys.stderr.write("ERROR: poc-batch-size-force: import or router anchor not found.\n")
        return 1

    if IMPORT_LINE not in src:
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER, 1)
    for old, new in SITES:
        src = src.replace(old, new, 1)

    with open(FILE, "w") as f:
        f.write(src)
    print("poc-batch-size-force: both proxy payloads now take the image's batch size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
