#!/usr/bin/env python3
"""Idempotently patch the vLLM package installed in the Stage 1 base image.

Three small edits, all derived from kaitakuai/vllm PR #8 (kv_scratch dtype
guard) and product-science/vllm PR #36 (apply_householder torch.compile):

  1. vllm/model_executor/layers/layernorm.py — guard rms_norm() against
     uint8/int8 input dtypes by casting to bfloat16. NVFP4 / int8 quantized
     models pass integer tensors that ops.rms_norm cannot handle.

  2. vllm/poc/poc_model_runner.py — same guard in execute_poc_forward():
     when the model config dtype is uint8/int8, cast to bfloat16 before
     building input embeddings. Otherwise the PoC reuses uint8-storage KV
     cache as inputs_embeds and feeds a Byte tensor to per_token_group_quant.

  3. vllm/poc/gpu_random.py — decorate apply_householder with
     @torch.compile(dynamic=False, fullgraph=True). +10-12% PoC throughput
     (measured in 2026-05 PoC v2 experiments). Safe no-op on all models.

Run unconditionally; the script detects already-patched files and skips.
Required for any FP8-KV-cache, NVFP4, or int8-quantized model (MiniMax-M2.7,
Kimi-K2.6-INT4 with FP8 KV experiments, future int8 deployments). Harmless
for fp16 / bf16 / fp32 models.

Why baked here, not at Stage 3: same patches apply to every PoC v2 profile
regardless of GPU. Doing it once in Stage 2 (after Stage 1 base inherits
vLLM into the image) avoids duplicating five-line edits across every Stage 3
hw-patch list.
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


_LAYERNORM_MARKER = "if x.dtype in (torch.uint8, torch.int8):"
_LAYERNORM_PATCH = """\
    if x.dtype in (torch.uint8, torch.int8):
        x = x.to(torch.bfloat16)
"""

_POC_RUNNER_MARKER = "if dtype in (torch.uint8, torch.int8):"
_POC_RUNNER_PATCH = """\
    if dtype in (torch.uint8, torch.int8):
        dtype = torch.bfloat16
"""

_HOUSEHOLDER_DECORATOR = "@torch.compile(dynamic=False, fullgraph=True)\n"


def _patch_layernorm(path: Path) -> str:
    """Insert int8/uint8 dtype guard before the rms_norm() ops call.

    Anchors on the `if envs.VLLM_BATCH_INVARIANT:` line because the surrounding
    function body is short and stable across recent vLLM releases.
    """
    src = path.read_text()
    if _LAYERNORM_MARKER in src:
        return "skip (already patched)"
    anchor_re = re.compile(
        r"(if envs\.VLLM_BATCH_INVARIANT:\n\s+return rms_norm_batch_invariant\([^)]+\)\n)"
    )
    new_src, n = anchor_re.subn(r"\1" + _LAYERNORM_PATCH, src, count=1)
    if n != 1:
        sys.exit(f"ERROR: rms_norm anchor not found in {path}; vLLM upstream moved? "
                 "Verify against the version pinned in tools/stage2.lock.cue.")
    path.write_text(new_src)
    return "patched"


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
        (root / "model_executor" / "layers" / "layernorm.py", _patch_layernorm),
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
