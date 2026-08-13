"""H200 Kimi-K2.6 INT4 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner (same shape as every *-plugin.py patch): a forced-args
block after the constructor anchor, and the launch-module swap.

Forced config — 8x H200/141 GiB. Every serving flag is copied from gonka's own
H200 Kimi profile (deploy/join/node-config-kimik26-H200.json); this is the only
Kimi leaf we have not tuned ourselves:

    --tensor-parallel-size 8
    --gpu-memory-utilization 0.90
    --max-model-len 240000
    --attention-backend FLASHMLA     (Hopper MLA; the Blackwell leaves pin
                                      CUTLASS_MLA, which is sm_100+ only)
    --mm-encoder-tp-mode data
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
    (flags) --trust-remote-code --enable-auto-tool-choice --enable-expert-parallel

Added on top, as on every PoC leaf:

    --logprobs-mode processed_logprobs
    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension

NOT forced, unlike the b200/b300 Kimi leaves:
    - NO --compilation-config: gonka's H200 profile leaves compilation to vLLM,
      and the eager pin on Blackwell exists for a batch=128 cudagraph hang that
      has never been reproduced on Hopper. Consequently --enforce-eager is left
      alone too — with no --compilation-config there is nothing for it to
      conflict with. The PoC forward runs eager regardless, via gonka-poc
      skip_compiled.
    - NO batch sizing: max-num-seqs / max-num-batched-tokens stay at vLLM's
      defaults until an H200 Kimi benchmark says otherwise.
"""

import os
import sys

# Release-line Dockerfile installs mlnode under /app/packages/api/src; the
# pre-0.25.1 fat-fork used /app/src. Resolve at runtime so one patch works
# on both, and FAIL if neither exists (a silent skip ships an unconfigured
# image -- see the stage-4 fail-loud guard).
_CANDIDATES = (
    "/app/packages/api/src/api/inference/vllm/runner.py",
    "/app/src/api/inference/vllm/runner.py",
)
FILE = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8

MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'

INJECTION_LINES = [
    "",
    "# --- Kaitaku H200-Kimi-K2.6 plugin hardcodes (tools/runner-patches/h200-kimi-k2-6-plugin.py) ---",  # noqa: E501
    "_h200_kimi_forced = [",
    "    ('--tensor-parallel-size', '8'),",
    "    ('--gpu-memory-utilization', '0.90'),",
    "    ('--max-model-len', '240000'),",
    "    ('--attention-backend', 'FLASHMLA'),",
    "    ('--mm-encoder-tp-mode', 'data'),",
    "    ('--tool-call-parser', 'kimi_k2'),",
    "    ('--reasoning-parser', 'kimi_k2'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_h200_kimi_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--enable-expert-parallel',",
    "]",
    "for _flag, _value in _h200_kimi_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _h200_kimi_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# --- end Kaitaku H200-Kimi-K2.6 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join((INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES)

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            "ERROR: h200-kimi-k2-6 patch: forced-args marker not found. "
            "Upstream runner.py may have been refactored - re-verify the patch.\n"
        )
        return 1
    if "_h200_kimi_forced" in src:
        sys.stderr.write("patch already applied; skipping\n")
        return 0

    idx = src.index(MARKER) + len(MARKER)
    src = src[:idx] + "\n" + injection + src[idx:]

    if MODULE_MARKER in src:
        src = src.replace(MODULE_MARKER, MODULE_REPLACEMENT)
    elif "MLNODE_VLLM_MODULE" not in src:
        sys.stderr.write(
            "ERROR: h200-kimi-k2-6 patch: launch-module marker not found and no "
            "MLNODE_VLLM_MODULE support present.\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
