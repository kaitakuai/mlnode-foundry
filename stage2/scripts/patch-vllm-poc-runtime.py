#!/usr/bin/env python3
"""Idempotently patch the vLLM package installed in the Stage 1 base image.

Two edits derived from the 2026-05 MiniMax-M2.7 PoC v2 experiments:

  1. vllm/poc/poc_model_runner.py — kv_scratch dtype skip.
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

  2. vllm/poc/gpu_random.py — decorate apply_householder with
     @torch.compile(dynamic=False, fullgraph=True) — product-science/vllm
     PR #36. +10-12% PoC throughput (measured 2026-05). Safe no-op on
     first call; applies to every PoC v2 profile regardless of GPU.

Not patched here:
  - layernorm.py: kaitakuai/vllm PR #8 targeted a refactored-out
    `def rms_norm()` that no longer exists in 0.20.0-pocv2 (now methods on
    an Op class with multiple forward variants). The guard there is needed
    only for NVFP4/int8 *weight* quant, which we don't deploy today.

Script is idempotent (detects already-patched files); anchors are exact
regexes — if vLLM upstream moves them, the script exits 1 with a clear
pointer back to tools/stage2.lock.cue.
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
        sys.exit("ERROR: vllm package not importable; is this a Stage 1 base image?")
    root = Path(spec.origin).parent
    if not root.is_dir():
        sys.exit(f"ERROR: vllm import resolved to {root!r} which is not a directory")
    return root


_POC_RUNNER_MARKER = "if kv.dtype in (torch.uint8, torch.int8"
_POC_RUNNER_PATCH = """\
            if kv.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
                continue  # FP8 KV cache; reusing as inputs_embeds crashes per_token_group_quant
"""

_HOUSEHOLDER_DECORATOR = "@torch.compile(dynamic=False, fullgraph=True)\n"


def _patch_poc_runner(path: Path) -> str:
    """In the `for kv in kv_caches:` scratch-search loop, skip uint8/int8/fp8 slabs.

    Anchors on the `for kv in kv_caches:` loop body which contains the
    `if kv.numel() >= needed_elems:` check — we insert the dtype skip
    immediately above that check.
    """
    src = path.read_text()
    if _POC_RUNNER_MARKER in src:
        return "skip (already patched)"
    # Anchor: the first line in the loop body checking kv.numel(). Capture
    # its leading whitespace to keep indentation consistent.
    anchor_re = re.compile(
        r"(for kv in kv_caches:\n)(?P<indent>\s+)(if kv\.numel\(\) >= needed_elems:)",
    )
    m = anchor_re.search(src)
    if m is None:
        sys.exit(f"ERROR: kv_caches scratch loop anchor not found in {path}; "
                 "vLLM upstream moved? Verify against tools/stage2.lock.cue.")
    indent = m.group("indent")
    # Re-render the inserted block with the discovered indentation so the
    # patch works regardless of whether the loop is at 8 / 12 / 16-space level.
    insertion = (
        f"{indent}if kv.dtype in (torch.uint8, torch.int8, "
        f"torch.float8_e4m3fn, torch.float8_e5m2):\n"
        f"{indent}    continue  # FP8 KV cache; reusing as inputs_embeds "
        f"crashes per_token_group_quant\n"
    )
    new_src = anchor_re.sub(r"\1" + insertion + r"\g<indent>\3", src, count=1)
    path.write_text(new_src)
    return "patched"


def _patch_gpu_random(path: Path) -> str:
    """Add @torch.compile decorator above `def apply_householder(`."""
    src = path.read_text()
    # Idempotency: the decorator line immediately precedes the def, so check
    # that the line above the def already starts with @torch.compile.
    lines = src.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("def apply_householder("):
            if i > 0 and lines[i - 1].startswith("@torch.compile"):
                return "skip (already patched)"
            lines.insert(i, _HOUSEHOLDER_DECORATOR)
            path.write_text("".join(lines))
            return "patched"
    sys.exit(f"ERROR: `def apply_householder(` not found in {path}; "
             "vLLM upstream moved? Verify against tools/stage2.lock.cue.")


def main() -> None:
    root = _resolve_vllm_root()
    targets = [
        (root / "poc" / "poc_model_runner.py", _patch_poc_runner),
        (root / "poc" / "gpu_random.py", _patch_gpu_random),
    ]
    for path, fn in targets:
        if not path.exists():
            sys.exit(f"ERROR: expected vLLM file missing: {path}")
        status = fn(path)
        print(f"→ {path}: {status}")


if __name__ == "__main__":
    main()
