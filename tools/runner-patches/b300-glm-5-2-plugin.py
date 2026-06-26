"""B300 GLM-5.2 FP8 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b200-glm-5-2-plugin.py;
B300 differs only in the per-engine tune):

1. Insert a forced-args block after `self.additional_args = additional_args or []`
   in __init__. Forces the B300 GLM-5.2 config (from experiments/2026-06/
   glm-5.2-deepgemm-4xb300) PLUS the plugin worker wiring:

     --tensor-parallel-size 4         (753B FP8 ~704 GB fits on 4×B300 / 275 GiB;
                                       an 8-GPU box auto-runs 2 TP=4 engines →
                                       1792 nonces/min/box vs 1078 for 1×TP=8)
     --gpu-memory-utilization 0.92    (leaves PoC headroom even at full context)
     --max-model-len 400000           (operator-forced; B300 fits the full
                                       1048576 at TP=4, so 400000 is comfortable)
     --max-num-batched-tokens 16384   (DeepGEMM survives memory profiling only at
                                       small mnbt; batch>16 → cudaErrorInvalidValue)
     --max-num-seqs 64
     --kv-cache-dtype fp8_e4m3        (B300 sm_100 accepts e4m3; no fp8_ds_mla
                                       needed, unlike H200)
     --tool-call-parser glm47
     --reasoning-parser glm45
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-
                                       side PoC via collective_rpc)
     (flags) --trust-remote-code, --enable-auto-tool-choice

   NOT forced: --attention-backend. GLM-5.2 is DSA (NOT MLA) — do NOT pin
   CUTLASS_MLA. The DeepGEMM split (MoE-on + linear→Cutlass) is set via env in
   the b300-glm-5-2 profile leaf (VLLM_USE_DEEP_GEMM=1 + VLLM_MOE_USE_DEEP_GEMM=1
   + VLLM_DISABLED_KERNELS=...), not here.

   NOT forced: compilation / --enforce-eager. Inference runs COMPILED by default
   (vLLM VLLM_COMPILE + CUDA graphs); the PoC forward runs eager on its own via
   gonka-poc poc_model_runner (skip_compiled=True). For the EXCEPTIONAL eager-
   inference (pure-mining) case the operator passes --enforce-eager; this patch
   neither forces nor strips it. NOTE: the deepgemm-4xb300 experiment's 896/engine
   (1792/box) number was measured under a FORCED
   --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'; the
   shipped image uses the vLLM compiled default (cudagraph mode may differ) — to
   reproduce the experiment exactly the operator adds that flag.

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.

Surgical / fail-loud: errors if either anchor is missing. Idempotent.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b300-glm-5-2-plugin.py /tmp/b300-glm-5-2-plugin.py
    RUN python3 /tmp/b300-glm-5-2-plugin.py && rm /tmp/b300-glm-5-2-plugin.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.additional_args = additional_args or []"
INDENT = " " * 8

# Edit 2: launch-module swap (runner.py already imports `os`).
MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = (
    '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'
)

INJECTION_LINES = [
    "",
    "# --- Kaitaku B300-GLM-5.2 plugin hardcodes (tools/runner-patches/b300-glm-5-2-plugin.py) ---",
    "_b300_glm_plugin_forced = [",
    "    ('--tensor-parallel-size', '4'),",
    "    ('--gpu-memory-utilization', '0.92'),",
    "    ('--max-model-len', '400000'),",
    "    ('--max-num-batched-tokens', '16384'),",
    "    ('--max-num-seqs', '64'),",
    "    ('--kv-cache-dtype', 'fp8_e4m3'),",
    "    ('--tool-call-parser', 'glm47'),",
    "    ('--reasoning-parser', 'glm45'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b300_glm_plugin_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "]",
    "for _flag, _value in _b300_glm_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b300_glm_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# NOTE: compilation/eager is intentionally NOT forced here. Inference runs",
    "# COMPILED by default (vLLM CompilationMode.VLLM_COMPILE + CUDA graphs); the",
    "# PoC forward runs eager on its own via gonka_poc poc_model_runner",
    "# (set_forward_context skip_compiled=True). For the EXCEPTIONAL eager-",
    "# inference (pure-mining) case the operator passes --enforce-eager; we",
    "# neither force nor strip it so that override is honored.",
    "# --- end Kaitaku B300-GLM-5.2 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B300-GLM-5.2 plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    already_swapped = MODULE_REPLACEMENT in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: B300-GLM-5.2 plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku B300-GLM-5.2 plugin hardcodes" in src
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
                "ERROR: B300-GLM-5.2 plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: B300-GLM-5.2 plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B300-GLM-5.2 plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
