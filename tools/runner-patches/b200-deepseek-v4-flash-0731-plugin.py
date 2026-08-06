"""B200 DeepSeek-V4-Flash-0731 PLUGIN-base hardcodes for MLNode runner.py.

Production profile for the vLLM 0.25.1 release line. Two edits to VLLMRunner
(same shape as every *-plugin.py patch): a forced-args block after
`self.additional_args = additional_args or []`, and the launch-module swap.

Forced config — 2x B200/180 GiB: weights split across the pair, 400k KV fits at gmu 0.90 (experiments 2026-08/deepseek-v4-flash-0731-2xb200):

    --tensor-parallel-size 2
    --gpu-memory-utilization 0.90
    --max-model-len 400000          (release parameter, confirmed by gonka-ai
                                     2026-08-05; native context is 1,048,576)
    --max-num-batched-tokens 32768
    --kv-cache-dtype fp8            (MANDATORY: FlashMLA fp8_ds_mla assert)
    --logprobs-mode processed_logprobs
    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
    --tokenizer-mode deepseek_v4    (release config, gonka#1536)
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    (flags) --trust-remote-code --enable-auto-tool-choice

NOT forced — deliberately, for consensus safety:
    - NO --attention-backend: default is deterministic FlashMLA-DSV4; pinning
      FLASHINFER_MLA_SPARSE_DSV4 engages placeholder FP8 scales and blows
      cross-hardware L2.
    - NO --speculative-config: DSpark is an operator opt-in (costs nothing in
      PoC, up to 2.98x on long single-stream decode — see the 2026-08
      experiments); replay validation under speculation works since vllm#92+#18
      but is slower, so the default stays off.
"""

import sys

FILE = "/app/src/api/inference/vllm/runner.py"
MARKER = "self.additional_args = additional_args or []"
INDENT = " " * 8

MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = (
    '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'
)

INJECTION_LINES = [
    "",
    "# --- Kaitaku B200-DeepSeek-V4-Flash-0731 plugin hardcodes (tools/runner-patches/b200-deepseek-v4-flash-0731-plugin.py) ---",
    "_dsv4_0731_forced = [",
    "    ('--tensor-parallel-size', '2'),",
    "    ('--gpu-memory-utilization', '0.90'),",
    "    ('--max-model-len', '400000'),",
    "    ('--max-num-batched-tokens', '32768'),",
    "    ('--kv-cache-dtype', 'fp8'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "    ('--tokenizer-mode', 'deepseek_v4'),",
    "    ('--tool-call-parser', 'deepseek_v4'),",
    "    ('--reasoning-parser', 'deepseek_v4'),",
    "]",
    "_dsv4_0731_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "]",
    "for _flag, _value in _dsv4_0731_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _dsv4_0731_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# NOTE: attention-backend / speculative-config intentionally NOT forced --",
    "# see the module docstring.",
    "# --- end Kaitaku B200-DeepSeek-V4-Flash-0731 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            "ERROR: b200-deepseek-v4-flash-0731 patch: forced-args marker not found. "
            "Upstream runner.py may have been refactored - re-verify the patch.\n"
        )
        return 1
    if "_dsv4_0731_forced" in src:
        sys.stderr.write("patch already applied; skipping\n")
        return 0

    idx = src.index(MARKER) + len(MARKER)
    src = src[:idx] + "\n" + injection + src[idx:]

    if MODULE_MARKER in src:
        src = src.replace(MODULE_MARKER, MODULE_REPLACEMENT)
    elif "MLNODE_VLLM_MODULE" not in src:
        sys.stderr.write(
            "ERROR: b200-deepseek-v4-flash-0731 patch: launch-module marker not "
            "found and no MLNODE_VLLM_MODULE support present.\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
