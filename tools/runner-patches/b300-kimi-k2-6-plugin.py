"""B300 Kimi-K2.6 INT4 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b300-minimax-plugin.py):

1. Insert a forced-args block after the constructor anchor
   in __init__. Forces the validated B300 Kimi tune (from tools/runner-patches/
   b300-kimi.py — measured 5120 nonces/min @ batch=64 on 8×B300) PLUS the plugin
   worker-side wiring, so the image runs the right config regardless of what the
   chain epoch_models broadcasts:

     --tensor-parallel-size 4          (1.06T-param Kimi INT4 ~530 GiB → TP=4)
     --gpu-memory-utilization 0.85     (headroom for the vision encoder)
     --max-num-batched-tokens 131072   (one prefill chunk for batch=128)
     --max-num-seqs 128
     --compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'  (EAGER — cudagraph
                                        hangs at batch=128 on Kimi MLA; eager also
                                        saves ~5% vs cudagraph FULL on this model)
     --attention-backend CUTLASS_MLA   (required on TP=4 for this model)
     --tool-call-parser kimi_k2
     --reasoning-parser kimi_k2
     --mm-encoder-tp-mode data
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-side
                                        PoC via collective_rpc — chain never sends this)
     (flags) --trust-remote-code, --enable-auto-tool-choice, --enable-expert-parallel

   NOT forced: --max-model-len. Kimi's native 256K context fits on 4×B300 (~3-4×
   concurrency); leave it to the chain rather than capping (unlike B200, which had
   to cap at 120000 to fit cudagraph capture on 178 GiB).

   REMOVES --enforce-eager: we set eager explicitly via --compilation-config
   (mode=0), and vLLM rejects --enforce-eager alongside --compilation-config.
   PoC-forward eager (bit-compat) is handled inside gonka-poc's poc_model_runner
   (skip_compiled=True), independent of this engine-level setting.

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.
   Required — stock api_server does NOT mount the PoC routes (see
   b300-minimax-plugin.py / api_router.build_gonka_app for the full rationale).

Surgical / fail-loud: errors if either anchor is missing (upstream refactor →
re-verify). Idempotent: re-running on an already-patched file is a no-op.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b300-kimi-k2-6-plugin.py /tmp/b300-kimi-k2-6-plugin.py
    RUN python3 /tmp/b300-kimi-k2-6-plugin.py && rm /tmp/b300-kimi-k2-6-plugin.py
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
    "# --- Kaitaku B300-Kimi-K2.6 plugin hardcodes (tools/runner-patches/b300-kimi-k2-6-plugin.py) ---",
    "_b300_kimi_plugin_forced = [",
    "    ('--tensor-parallel-size', '4'),",
    "    ('--gpu-memory-utilization', '0.85'),",
    "    ('--max-num-batched-tokens', '131072'),",
    "    ('--max-num-seqs', '128'),",
    "    ('--compilation-config', '{\"mode\": 0, \"cudagraph_mode\": \"NONE\"}'),",
    "    ('--attention-backend', 'CUTLASS_MLA'),",
    "    ('--tool-call-parser', 'kimi_k2'),",
    "    ('--reasoning-parser', 'kimi_k2'),",
    "    ('--mm-encoder-tp-mode', 'data'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b300_kimi_plugin_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--enable-expert-parallel',",
    "]",
    "# --enforce-eager conflicts with --compilation-config; eager is set via mode=0.",
    "_b300_kimi_plugin_remove = ['--enforce-eager']",
    "for _flag, _value in _b300_kimi_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b300_kimi_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "for _flag in _b300_kimi_plugin_remove:",
    "    while _flag in self.additional_args:",
    "        self.additional_args.pop(self.additional_args.index(_flag))",
    "# --- end Kaitaku B300-Kimi-K2.6 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B300-Kimi-K26 plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # mlnode 0.25.1 reads MLNODE_VLLM_MODULE natively, in a different textual
    # form -- any support for the variable counts as swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: B300-Kimi-K26 plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku B300-Kimi-K2.6 plugin hardcodes" in src
    if already_injected and already_swapped:
        print("runner.py already patched — skipping")
        return 0

    out = src

    # Edit 1: inject the forced-args block after the additional_args marker.
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
                "ERROR: B300-Kimi-K26 plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    # Edit 2: swap the launched module to honour MLNODE_VLLM_MODULE.
    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: B300-Kimi-K26 plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B300-Kimi-K2.6 plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
