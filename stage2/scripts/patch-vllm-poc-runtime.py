#!/usr/bin/env python3
"""Idempotently patch the vLLM package installed in the Stage 1 base image.

Two small edits derived from the 2026-05 MiniMax-M2.7 PoC v2 experiments:

  1. vllm/poc/poc_model_runner.py — when model config dtype is uint8/int8,
     cast to bfloat16 before building input embeddings. Otherwise the PoC
     reuses uint8-storage KV cache as inputs_embeds and feeds a Byte tensor
     to per_token_group_quant — crash on FP8 KV cache models (MiniMax-M2.7).

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


_POC_RUNNER_MARKER = "if dtype in (torch.uint8, torch.int8):"
_POC_RUNNER_PATCH = """\
    if dtype in (torch.uint8, torch.int8):
        dtype = torch.bfloat16
"""

_HOUSEHOLDER_DECORATOR = "@torch.compile(dynamic=False, fullgraph=True)\n"


def _patch_poc_runner(path: Path) -> str:
    """Insert int8/uint8 dtype guard right after dtype is read from worker.model_config."""
    src = path.read_text()
    if _POC_RUNNER_MARKER in src:
        return "skip (already patched)"
    anchor_re = re.compile(r"(dtype = worker\.model_config\.dtype\n)")
    new_src, n = anchor_re.subn(r"\1" + _POC_RUNNER_PATCH, src, count=1)
    if n != 1:
        sys.exit(f"ERROR: model_config.dtype anchor not found in {path}; "
                 "vLLM upstream moved? Verify against tools/stage2.lock.cue.")
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
