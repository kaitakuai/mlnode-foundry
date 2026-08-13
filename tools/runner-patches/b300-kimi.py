"""B300 Kimi-K2.6 INT4 hardcodes for MLNode runner.py.

Inserts a small block right after the constructor anchor
in `VLLMRunner.__init__`. Additive — no upstream code is removed; fails loud if
the marker line is missing (= upstream refactored, re-verify the patch).

Why: chain epoch_models broadcasts CUDA-graph-friendly defaults that don't fit
Kimi-K2.6 INT4 on B300 with vision tower + MLA. The Kimi-specific tuning that
hits the measured 5120 nonces/min @ batch=64 peak on 8×B300 (TP=4 × 2 instances)
needs:
    * TP=4 — the only way 1.06T-param Kimi INT4 (~530 GiB) splits onto B300
      cards with headroom for KV cache + 384 routed experts × top_k=8.
    * gpu_memory_utilization=0.85 — leaves headroom for vision-encoder
      profile_run on top of the 4×B300 SXM mesh; 0.95 OOMs at warmup.
    * max_num_batched_tokens=131072 — one prefill chunk for batch=128 ×
      seq_len=1024; lifts batch=128 above the default 65536 stall point.
    * compilation_config={"mode":0, "cudagraph_mode":"NONE"} — eager mode
      saves 5% throughput vs cudagraph FULL on this model AND unblocks
      batch=128 (cudagraph hangs at b=128 on Kimi MLA).
    * --logprobs-mode processed_logprobs — PoC v2 correctness.

NOT forced (intentionally):
    * --max-model-len — chain epoch_models drives this. Kimi-K2.6 native
      256K context fits on 4×B300 with ~3-4× concurrency; a forced
      120K cap (as in the Qwen b300-k5 image) would silently cut the
      usable context window in half.

Env vars set at the image level (NOT in this patcher; see Dockerfile):
    VLLM_USE_FLASHINFER_MOE_INT4=1
        FlashInfer mxint4 MoE kernel — Blackwell sm_103a only. +138% vs
        Marlin (W4A16 dequant→bf16 every matmul) on Kimi-K2.6, measured
        clean A/B on 8×B300 (4864 vs 2048 nonces/min @ batch=32).

Defaults (only set if absent — chain may override):
    --max-num-seqs=128                  (avoid sampler-warmup OOM)

Operator note — chain epoch_models for Kimi-K2.6 typically passes
`--attention-backend CUTLASS_MLA --enable-expert-parallel --enforce-eager`
which COMPLEMENTS the forced flags above. CUTLASS_MLA is required on
Blackwell (FLASHINFER doesn't implement MLA). EP=4 yields +5% over plain
TP=4 on this model.

Idempotent: re-running on an already-patched file is a no-op.

Usage inside a Dockerfile RUN step:
    COPY tools/fragments/hw-patches/runner-py-patches/b300-kimi.py /tmp/b300-kimi.py
    RUN python3 /tmp/b300-kimi.py && rm /tmp/b300-kimi.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8

INJECTION_LINES = [
    "",
    "# --- Kaitaku B300-Kimi hardcodes (tools/fragments/hw-patches/runner-py-patches/b300-kimi.py) ---",
    "_b300_kimi_defaults = {",
    "    '--max-num-seqs': '128',",
    "}",
    "for _flag, _value in _b300_kimi_defaults.items():",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.extend([_flag, _value])",
    "_b300_kimi_forced = {",
    "    '--tensor-parallel-size': '4',",
    "    '--gpu-memory-utilization': '0.85',",
    "    '--max-num-batched-tokens': '131072',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--compilation-config': '{\"mode\": 0, \"cudagraph_mode\": \"NONE\"}',",
    "}",
    "for _flag, _value in _b300_kimi_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# --- end Kaitaku B300-Kimi hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: Kaitaku B300-Kimi patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    if "Kaitaku B300-Kimi hardcodes" in src:
        print("runner.py already patched — skipping")
        return 0

    lines = src.splitlines(keepends=True)
    new_lines: list[str] = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and MARKER in line:
            new_lines.append(injection)
            inserted = True

    if not inserted:
        sys.stderr.write(
            "ERROR: Kaitaku B300-Kimi patch: marker present but insertion did not fire\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.writelines(new_lines)
    print("runner.py patched for B300-Kimi hardcodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
