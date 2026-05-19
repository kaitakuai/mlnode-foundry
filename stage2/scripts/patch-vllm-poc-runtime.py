#!/usr/bin/env python3
"""Idempotently patch the vLLM package installed in the Stage 1 base image.

TWO universal edits for any PoC v2 image, both targeting
`vllm/poc/poc_model_runner.py`:

  1. kv_scratch dtype skip.
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

  2. CommonAttentionMetadata.seq_lens_cpu_upper_bound restore.
    `_create_v1_attn_metadata` constructs a CommonAttentionMetadata for
    the PoC forward pass. MLA-style attention backends (CUTLASS_MLA,
    FLASHINFER_MLA) read `seq_lens_cpu_upper_bound` in their
    `metadata_builder.build(...)` path and `assert is not None`. In
    Stage 1 `kaitakuai/vllm:0.20.0-pocv2`, this kwarg was lost from the
    constructor call (likely during the gonka v0.20→v0.20.0-pocv2
    rebase). The old 0.2.12 mlnode-full image had it.

    Symptom on Kimi-K2.6 + B200 (TP=EP=4): all worker procs raise
    `AssertionError` at mla_attention.py:1843 on the first PoC step;
    `/v1/health` stays 200 but no nonces are produced.

    Patch: re-insert
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    immediately after `_seq_lens_cpu=seq_lens_cpu,` in the
    `CommonAttentionMetadata(...)` kwargs. Non-MLA backends ignore the
    field, so the fix is universal/safe across profiles.

Not patched here:
  - layernorm.py: kaitakuai/vllm PR #8 targeted a refactored-out
    `def rms_norm()` that no longer exists in 0.20.0-pocv2 (now methods on
    an Op class with multiple forward variants). The guard there is needed
    only for NVFP4/int8 *weight* quant, which we don't deploy today.
  - gpu_random.py @torch.compile(apply_householder): moved from Stage 2
    (universal) to Stage 3 (opt-in) via tools/hw-patches/poc-householder-compile
    so Kimi-K2.6-INT4 profiles can opt out — legacy production Kimi image
    explicitly omits this decorator (see kaitakuai/mlnode-full:
    0.2.12-vllm0.20.0-b300-k5-kimi-1), and PR #36's measured +10-12% gain
    was on Qwen3-235B-FP8, not Kimi-INT4. Until A/B'd on Kimi, leave it off.

Script is idempotent (detects already-patched files); the anchor is an
exact regex — if vLLM upstream moves it, the script exits 1 with a clear
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
_SEQ_LENS_UPPER_BOUND_MARKER = "seq_lens_cpu_upper_bound=seq_lens_cpu"


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
                 "vLLM upstream moved? Verify against tools/stage2.lock.cue.")
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


def _patch_seq_lens_upper_bound(path: Path) -> str:
    """Insert the missing `seq_lens_cpu_upper_bound=seq_lens_cpu,` kwarg.

    Anchors on `_seq_lens_cpu=seq_lens_cpu,` inside the
    `CommonAttentionMetadata(...)` call in `_create_v1_attn_metadata`, then
    inserts the upper-bound kwarg immediately after it at the same indent.
    """
    src = path.read_text()
    if _SEQ_LENS_UPPER_BOUND_MARKER in src:
        return "seq_lens_upper_bound: skip (already patched)"
    anchor_re = re.compile(
        r"(?P<indent>[ \t]+)_seq_lens_cpu=seq_lens_cpu,\n",
    )
    m = anchor_re.search(src)
    if m is None:
        sys.exit(
            f"ERROR: `_seq_lens_cpu=seq_lens_cpu,` anchor not found in {path}; "
            "vLLM upstream moved? Verify against tools/stage2.lock.cue."
        )
    indent = m.group("indent")
    insertion = f"{indent}seq_lens_cpu_upper_bound=seq_lens_cpu,\n"
    new_src = src[: m.end()] + insertion + src[m.end():]
    path.write_text(new_src)
    return "seq_lens_upper_bound: patched"


def main() -> None:
    root = _resolve_vllm_root()
    target = root / "poc" / "poc_model_runner.py"
    if not target.exists():
        sys.exit(f"ERROR: expected vLLM file missing: {target}")
    print(f"→ {target}: {_patch_poc_runner(target)}")
    print(f"→ {target}: {_patch_seq_lens_upper_bound(target)}")


if __name__ == "__main__":
    main()
