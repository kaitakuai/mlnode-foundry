"""B200 GLM-5.2 FP8 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b300-kimi-k2-6-plugin.py):

1. Insert a forced-args block after the constructor anchor
   in __init__. Forces the recommended GLM-5.2 config (operator-supplied) PLUS
   the plugin worker-side wiring + PoC-eager, so the image runs the right config
   regardless of what the chain epoch_models broadcasts:

     --tensor-parallel-size 8
     --gpu-memory-utilization 0.85
     --max-model-len 400000           (< native 1048576 — fits 8×B200 with margin)
     --max-num-batched-tokens 16384
     --max-num-seqs 16
     --kv-cache-dtype fp8_e4m3        (accepted natively on sm_100)
     --tool-call-parser glm47
     --reasoning-parser glm45
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-
                                       side PoC via collective_rpc)
     (flags) --trust-remote-code, --enable-auto-tool-choice

   NOT forced: --attention-backend. GLM-5.2 is a DSA model (NOT MLA), so do NOT
   pin CUTLASS_MLA (that is a Kimi/DeepSeek-MLA backend). MoE-DeepGEMM is enabled
   while the block-FP8 LINEAR kernel is routed to Cutlass — both via env in the
   GLM_5_2 base (VLLM_MOE_USE_DEEP_GEMM=1 + VLLM_DISABLED_KERNELS=...), not here.

   NOT forced: compilation / --enforce-eager. Inference runs COMPILED by default
   (vLLM CompilationMode.VLLM_COMPILE + CUDA graphs) for decode throughput; the
   PoC forward runs eager on its own via gonka-poc poc_model_runner
   (set_forward_context skip_compiled=True). So one image gives eager PoC +
   compiled inference. Forcing global eager (--compilation-config mode:0 /
   --enforce-eager) would drop CUDA graphs for inference (~5x decode loss on GLM:
   157 vs 817 tok/s). For the EXCEPTIONAL case where inference must also be eager,
   the operator passes --enforce-eager at launch; this patch neither forces nor
   strips it, so the override is honored. (Same policy as b300-minimax-plugin.py;
   b300-kimi-k2-6-plugin.py is the documented exception — cudagraph hangs on Kimi
   MLA at batch=128, so it does force eager.)

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.

Surgical / fail-loud: errors if either anchor is missing. Idempotent.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b200-glm-5-2-plugin.py /tmp/b200-glm-5-2-plugin.py
    RUN python3 /tmp/b200-glm-5-2-plugin.py && rm /tmp/b200-glm-5-2-plugin.py
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
    "# --- Kaitaku B200-GLM-5.2 plugin hardcodes (tools/runner-patches/b200-glm-5-2-plugin.py) ---",
    "_b200_glm_plugin_forced = [",
    "    ('--tensor-parallel-size', '8'),",
    "    ('--gpu-memory-utilization', '0.85'),",
    "    ('--max-model-len', '400000'),",
    "    ('--max-num-batched-tokens', '16384'),",
    "    ('--max-num-seqs', '16'),",
    "    ('--kv-cache-dtype', 'fp8_e4m3'),",
    "    ('--tool-call-parser', 'glm47'),",
    "    ('--reasoning-parser', 'glm45'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b200_glm_plugin_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "]",
    "for _flag, _value in _b200_glm_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b200_glm_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# NOTE: compilation/eager is intentionally NOT forced here. Inference runs",
    "# COMPILED by default (vLLM CompilationMode.VLLM_COMPILE + CUDA graphs) for",
    "# decode throughput; the PoC forward runs eager on its own via gonka_poc",
    "# poc_model_runner (set_forward_context skip_compiled=True), so we get",
    "# eager PoC + compiled inference in one image. Forcing global eager",
    "# (--compilation-config mode:0 / --enforce-eager) would drop CUDA graphs",
    "# for inference (~5x decode throughput loss on GLM). For the EXCEPTIONAL",
    "# case where inference must also be eager, the operator passes",
    "# --enforce-eager (or --compilation-config '{\"mode\":0,...}') at launch;",
    "# we neither force nor strip it so that override is honored.",
    "# --- end Kaitaku B200-GLM-5.2 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B200-GLM-5.2 plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    already_swapped = MODULE_REPLACEMENT in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: B200-GLM-5.2 plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku B200-GLM-5.2 plugin hardcodes" in src
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
                "ERROR: B200-GLM-5.2 plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: B200-GLM-5.2 plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B200-GLM-5.2 plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
