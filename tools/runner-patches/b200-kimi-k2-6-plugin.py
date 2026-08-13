"""B200 Kimi-K2.6 INT4 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b300-kimi-k2-6-plugin.py;
B200 differs in the memory envelope + uses CUDA graphs):

1. Insert a forced-args block after the constructor anchor
   in __init__. Forces the validated B200 Kimi tune (experiment
   kimi_k26_int4_4xb200_q-int4-k2) PLUS the plugin worker wiring:

     --tensor-parallel-size 4
     --gpu-memory-utilization 0.93
     --max-num-batched-tokens 32768   (caps PoC batch at 32 = 32×1024; batch 64/128
                                       OOM at B200's 178 GiB/GPU under cudagraph FULL)
     --max-num-seqs 32
     --max-model-len 120000           (capped below Kimi's native 262144 — cudagraph
                                       FULL captures KV shapes at 256k that OOM on
                                       B200; B300 leaves it uncapped on 275 GiB/GPU)
     --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
                                       (CUDA graphs — at B200's batch=32 they work
                                       fine, UNLIKE B300 batch=128 where cudagraph
                                       hangs and b300-kimi forces eager. The
                                       experiment showed CUDA graphs do NOT change
                                       PoC throughput, so mode=3 keeps faster
                                       inference at no PoC cost. PoC forward is eager
                                       on its own via gonka-poc skip_compiled.)
     --attention-backend CUTLASS_MLA
     --tool-call-parser kimi_k2
     --reasoning-parser kimi_k2
     --mm-encoder-tp-mode data
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-
                                       side PoC via collective_rpc)
     (flags) --trust-remote-code, --enable-auto-tool-choice, --enable-expert-parallel

   REMOVES --enforce-eager: it conflicts with the forced --compilation-config; the
   compiled path is set via mode=3. PoC-forward eager (bit-compat) is handled inside
   gonka-poc (poc/poc_model_runner.py skip_compiled=True).

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.

Surgical / fail-loud: errors if either anchor is missing. Idempotent.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b200-kimi-k2-6-plugin.py /tmp/b200-kimi-k2-6-plugin.py
    RUN python3 /tmp/b200-kimi-k2-6-plugin.py && rm /tmp/b200-kimi-k2-6-plugin.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8

# Edit 2: launch-module swap (runner.py already imports `os`).
MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = (
    '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'
)

INJECTION_LINES = [
    "",
    "# --- Kaitaku B200-Kimi-K2.6 plugin hardcodes (tools/runner-patches/b200-kimi-k2-6-plugin.py) ---",
    "_b200_kimi_plugin_forced = [",
    "    ('--tensor-parallel-size', '4'),",
    "    ('--gpu-memory-utilization', '0.93'),",
    "    ('--max-num-batched-tokens', '32768'),",
    "    ('--max-num-seqs', '32'),",
    "    ('--max-model-len', '120000'),",
    "    ('--compilation-config', '{\"mode\": 3, \"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'),",
    "    ('--attention-backend', 'CUTLASS_MLA'),",
    "    ('--tool-call-parser', 'kimi_k2'),",
    "    ('--reasoning-parser', 'kimi_k2'),",
    "    ('--mm-encoder-tp-mode', 'data'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b200_kimi_plugin_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--enable-expert-parallel',",
    "]",
    "# --enforce-eager conflicts with --compilation-config; the compiled path is mode=3.",
    "_b200_kimi_plugin_remove = ['--enforce-eager']",
    "for _flag, _value in _b200_kimi_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b200_kimi_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "for _flag in _b200_kimi_plugin_remove:",
    "    while _flag in self.additional_args:",
    "        self.additional_args.pop(self.additional_args.index(_flag))",
    "# --- end Kaitaku B200-Kimi-K2.6 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B200-Kimi-K2.6 plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # mlnode 0.25.1 reads MLNODE_VLLM_MODULE natively, in a different textual
    # form -- any support for the variable counts as swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: B200-Kimi-K2.6 plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku B200-Kimi-K2.6 plugin hardcodes" in src
    if already_injected and already_swapped:
        print("runner.py already patched — skipping")
        return 0

    out = src

    if not already_injected:
        lines = out.splitlines(keepends=True)
        new_lines: list[str] = []
        inserted = False
        for line in lines:
            new_lines.append(line)
            if not inserted and MARKER in line:
                new_lines.append(injection)
                inserted = True
        if not inserted:
            sys.stderr.write(
                "ERROR: B200-Kimi-K2.6 plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: B200-Kimi-K2.6 plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B200-Kimi-K2.6 plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
