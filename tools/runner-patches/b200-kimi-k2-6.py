"""B200 Kimi-K2.6 INT4 (rev=2 experimental tune) hardcodes for MLNode runner.py.

Forces a specific vLLM args set into VLLMRunner.__init__ so the image runs
the operator's iteration-2026-05-19 tune regardless of what chain
epoch_models broadcasts. Same pattern as b300-kimi.py but with the rev=2
config:

  --tensor-parallel-size 4
  --gpu-memory-utilization 0.93
  --max-num-batched-tokens 32768
  --max-model-len 120000
  --logprobs-mode processed_logprobs
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
  --attention-backend CUTLASS_MLA
  --max-num-seqs 32
  --tool-call-parser kimi_k2
  --reasoning-parser kimi_k2
  --mm-encoder-tp-mode data
  --trust-remote-code            (flag, no value)
  --enable-auto-tool-choice      (flag, no value)
  --enable-expert-parallel       (flag, no value)

Also explicitly REMOVES --enforce-eager if it's present, because compilation
mode=3 + cudagraph FULL_AND_PIECEWISE is incompatible with eager.

Idempotent: re-running on an already-patched file is a no-op.
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8

INJECTION_LINES = [
    "",
    "# --- Kaitaku B200-Kimi-K2.6 rev=2 hardcodes (tools/runner-patches/b200-kimi-k2-6-int4.py) ---",
    "_b200_kimi_k26_forced = [",
    "    ('--tensor-parallel-size', '4'),",
    "    ('--gpu-memory-utilization', '0.93'),",
    "    ('--max-num-batched-tokens', '32768'),",
    "    ('--max-model-len', '120000'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--compilation-config', '{\"mode\": 3, \"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'),",
    "    ('--attention-backend', 'CUTLASS_MLA'),",
    "    ('--max-num-seqs', '32'),",
    "    ('--tool-call-parser', 'kimi_k2'),",
    "    ('--reasoning-parser', 'kimi_k2'),",
    "    ('--mm-encoder-tp-mode', 'data'),",
    "]",
    "_b200_kimi_k26_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--enable-expert-parallel',",
    "]",
    "_b200_kimi_k26_remove = ['--enforce-eager']",
    "for _flag, _value in _b200_kimi_k26_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b200_kimi_k26_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "for _flag in _b200_kimi_k26_remove:",
    "    while _flag in self.additional_args:",
    "        self.additional_args.pop(self.additional_args.index(_flag))",
    "# --- end Kaitaku B200-Kimi-K2.6 rev=2 hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B200-Kimi-K26 patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    if "Kaitaku B200-Kimi-K2.6 rev=2 hardcodes" in src:
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
            "ERROR: B200-Kimi-K26 patch: marker present but insertion did not fire\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.writelines(new_lines)
    print("runner.py patched for B200-Kimi-K2.6 rev=2 hardcodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
