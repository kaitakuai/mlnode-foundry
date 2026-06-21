#!/usr/bin/env python3
"""Idempotently patch the vLLM package installed in the Stage 2 (vllm-poc) base image.

ONE universal edit for any PoC v2 image:

  vllm/poc/poc_model_runner.py — kv_scratch dtype skip.
    For FP8 KV cache models (--kv-cache-dtype fp8 → uint8/float8_e4m3fn
    storage), the PoC fast-path reuses a KV cache slab as inputs_embeds
    scratch. Pre-patch code did NOT check the slab's dtype: the byte/fp8
    buffer became `inputs_embeds`, downstream per_token_group_quant kernel
    received a non-float tensor and crashed.

    Patch: when iterating kv_caches looking for a slab to reuse, SKIP slabs
    whose dtype is uint8/int8/float8_e4m3fn/float8_e5m2. If no eligible
    slab is found, the else-branch's gen_fn allocates a fresh bf16 tensor
    (correct dtype, slight perf hit relative to in-place reuse, but only
    for FP8-KV models — bf16/fp16 KV models hit the fast path unchanged).

Not patched here:
  - layernorm.py: kaitakuai/vllm PR #8 targeted a refactored-out
    `def rms_norm()` that no longer exists in 0.20.0-pocv2 (now methods on
    an Op class with multiple forward variants). The guard there is needed
    only for NVFP4/int8 *weight* quant, which we don't deploy today.
  - gpu_random.py @torch.compile(apply_householder): moved from Stage 3
    (universal) to Stage 4 (opt-in) via tools/hw-patches/poc-householder-compile
    so Kimi-K2.6-INT4 profiles can opt out — legacy production Kimi image
    explicitly omits this decorator (see kaitakuai/mlnode-full:
    0.2.12-vllm0.20.0-b300-k5-kimi-1), and PR #36's measured +10-12% gain
    was on Qwen3-235B-FP8, not Kimi-INT4. Until A/B'd on Kimi, leave it off.

Script is idempotent (detects already-patched files); the anchor is an
exact regex — if vLLM upstream moves it, the script exits 1 with a clear
pointer back to tools/stage3.lock.cue.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _resolve_vllm_root() -> Path:
    """Return the on-disk root of the installed vLLM package (e.g.,
    /usr/local/lib/python3.12/dist-packages/vllm/)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        sys.exit("ERROR: vllm package not importable; is this a Stage 2 (vllm-poc) base image?")
    root = Path(spec.origin).parent
    if not root.is_dir():
        sys.exit(f"ERROR: vllm import resolved to {root!r} which is not a directory")
    return root


_POC_RUNNER_MARKER = "if kv.dtype in (torch.uint8, torch.int8"


def _patch_poc_runner(path: Path) -> str:
    """In the `for kv in kv_caches:` scratch-search loop, skip uint8/int8/fp8 slabs.

    Anchors on the `for kv in kv_caches:` loop body which contains the
    `if kv.numel() >= needed_elems:` check — we insert the dtype skip
    immediately above that check.
    """
    src = path.read_text()
    if _POC_RUNNER_MARKER in src:
        return "skip (already patched)"
    anchor_re = re.compile(
        r"(for kv in kv_caches:\n)(?P<indent>\s+)(if kv\.numel\(\) >= needed_elems:)",
    )
    m = anchor_re.search(src)
    if m is None:
        sys.exit(f"ERROR: kv_caches scratch loop anchor not found in {path}; "
                 "vLLM upstream moved? Verify against tools/stage3.lock.cue.")
    indent = m.group("indent")
    insertion = (
        f"{indent}if kv.dtype in (torch.uint8, torch.int8, "
        f"torch.float8_e4m3fn, torch.float8_e5m2):\n"
        f"{indent}    continue  # FP8 KV cache; reusing as inputs_embeds "
        f"crashes per_token_group_quant\n"
    )
    new_src = anchor_re.sub(r"\1" + insertion + r"\g<indent>\3", src, count=1)
    path.write_text(new_src)
    return "patched"


def main() -> None:
    root = _resolve_vllm_root()
    target = root / "poc" / "poc_model_runner.py"
    if not target.exists():
        sys.exit(f"ERROR: expected vLLM file missing: {target}")
    print(f"→ {target}: {_patch_poc_runner(target)}")


if __name__ == "__main__":
    main()
