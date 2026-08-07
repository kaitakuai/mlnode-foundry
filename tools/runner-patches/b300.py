"""B300-sizing hardcodes for MLNode runner.py.

Inserts a small block right after the constructor anchor
in `VLLMRunner.__init__`. Additive — no upstream code is removed; fails loud if
the marker line is missing (= upstream refactored, re-verify the patch).

Why: the network node broadcasts H100-tuned vLLM flags that do not fit B300's
275 GiB/GPU topology at TP=1, and PoC v2 needs a few B300-specific overrides
to hit the 1280 nonces/min peak measured on this silicon — see
`experiments/2026-04/qwen235b-fp8-1xb300-OVERVIEW.md` and the corresponding
`vllm020-stockcompile/` run for the validation data.

Forces (overwrite if present):
    --gpu-memory-utilization=0.95
    --max-model-len=120000          (under the 125 408-token KV pool ceiling
                                     observed when max-num-batched-tokens=65536
                                     and STOCK_TORCH_COMPILE is on)
    --max-num-batched-tokens=65536  (one prefill chunk for batch=64,
                                     seq_len=1024; default 8192 chunks PoC
                                     prefill and stalls batch>=64)
    --logprobs-mode=processed_logprobs  (PoC v2 correctness)
    --compilation-config={"mode": 1}  (CompilationMode.STOCK_TORCH_COMPILE —
                                     keeps `torch.compile` active for the
                                     non-PoC pipeline but skips vLLM's
                                     piecewise/CUDA-graph machinery that
                                     was capping PoC throughput at 1024
                                     nonces/min on B300. PoC's own
                                     `skip_compiled=True` keeps the PoC
                                     forward eager regardless.
                                     Mode 1 measured +11% over enforce-eager
                                     and +54% over the original TRITON
                                     baseline on 1×B300.)

Defaults (only set if absent):
    --max-num-seqs=128              (avoid sampler-warmup OOM)

Note on `--tensor-parallel-size=1` (forced):
    1× independent vLLM instance per GPU is the throughput-optimal topology
    for Qwen3-235B-A22B FP8 PoC v2 on B300 (8 × 1280 = 10240 nonces/min on
    8×B300, vs ~9200-9500 measured for TP=2 × 4 instances). The network
    node sometimes passes `--tensor-parallel-size 2` in additional_args
    based on its own model-topology config; we overwrite that with TP=1
    so the b300 image always converges on the measured-optimal layout.
    Operators wanting TP>1 should use a non-b300 image variant.

Operator note — `compilation_mode=1` recompiles per (batch, seq_len) shape on
first encounter; cold-cache start swallows the 30 s PoC measurement window
for small batch sizes and yields zero nonces. A pre-warm script that issues
one prefill at each expected (batch, seq_len) pair is required before serving
live PoC traffic. After the first warm-up, the compile cache lives in
/root/.cache/vllm/torch_compile_cache/ and subsequent restarts are fast.

Idempotent: re-running on an already-patched file is a no-op.

Usage inside a Dockerfile RUN step:
    COPY tools/fragments/hw-patches/runner-py-patches/b300.py /tmp/b300-patch.py
    RUN python3 /tmp/b300-patch.py && rm /tmp/b300-patch.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8  # VLLMRunner.__init__ method body indent

INJECTION_LINES = [
    "",
    "# --- Kaitaku B300 hardcodes (tools/fragments/hw-patches/runner-py-patches/b300.py) ---",
    "_b300_defaults = {",
    "    '--max-num-seqs': '128',",
    "}",
    "for _flag, _value in _b300_defaults.items():",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.extend([_flag, _value])",
    "_b300_forced = {",
    "    '--tensor-parallel-size': '1',",
    "    '--gpu-memory-utilization': '0.95',",
    "    '--max-model-len': '120000',",
    "    '--max-num-batched-tokens': '65536',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--compilation-config': '{\"mode\": 1}',",
    "}",
    "for _flag, _value in _b300_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# --- end Kaitaku B300 hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: Kaitaku B300 patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    if "Kaitaku B300 hardcodes" in src:
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
            "ERROR: Kaitaku B300 patch: marker present but insertion did not fire\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.writelines(new_lines)
    print("runner.py patched for B300 hardcodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
