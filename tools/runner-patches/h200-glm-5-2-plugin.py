"""H200 GLM-5.2 FP8 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b200-glm-5-2-plugin.py;
H200 = Hopper sm_90, so the backend/KV story differs materially):

1. Insert a forced-args block after the constructor anchor
   in __init__. Forces the H200 GLM-5.2 config (from experiments/2026-06/
   glm-5.2-fp8-8xh200) PLUS the plugin worker wiring:

     --tensor-parallel-size 8
     --gpu-memory-utilization 0.90    (sparse_decode_fwd needs ~2 GB/card scratch;
                                       gmu 0.97 OOMs with 500 errors)
     --max-model-len 400000           (operator-forced; comfortable with fp8 KV)
     --max-num-batched-tokens 16384   (Pasha-verified on the built H200 image; the
                                       compressed fp8_ds_mla KV makes the small mnbt
                                       fit, and it matches B200/B300)
     --max-num-seqs 128
     --kv-cache-dtype fp8             (LOAD-BEARING H200: on sm_90 GLM-DSA the backend
                                       is FLASHMLA_SPARSE, where 'fp8' is the supported
                                       alias for fp8_ds_mla (656-byte compressed MLA
                                       KV) — plain fp8_e4m3 is REJECTED ('No valid
                                       attention backend'). Pasha-verified post-test
                                       2026-06-26. NONCE: fp8 changes attention math vs
                                       bf16 — re-confirm nonce L2 on fp8 if mining.)
     --tool-call-parser glm45         (H200 difference: glm45, NOT glm47 like B200/B300)
     --reasoning-parser glm45
     --moe-backend triton             (DeepGEMM is off on sm_90; force triton
                                       explicitly so the FP8 MoE auto-selector
                                       cannot regress to FLASHINFER_CUTLASS — pairs
                                       with VLLM_USE_FLASHINFER_MOE_FP8=0 env)
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-
                                       side PoC via collective_rpc)
     (flags) --trust-remote-code, --enable-auto-tool-choice, --enable-expert-parallel
                                       (EP is UNIQUE to H200 per the experiment)

   NOT forced: --attention-backend. GLM-5.2 is DSA (GlmMoeDsaForCausalLM) —
   FLASHMLA_SPARSE auto-selects on H200; do NOT pin CUTLASS_MLA. DeepGEMM is off
   (sm_90, triton MoE) — set via env in the h200-glm-5-2 profile leaf
   (VLLM_USE_DEEP_GEMM=0 / VLLM_MOE_USE_DEEP_GEMM=0 / VLLM_USE_FLASHINFER_MOE_FP8=0).

   NOT forced: compilation / --enforce-eager. Inference runs COMPILED by default;
   the PoC forward runs eager on its own via gonka-poc skip_compiled. For the
   EXCEPTIONAL eager-inference (pure-mining) case the operator passes
   --enforce-eager; this patch neither forces nor strips it. NOTE: the H200 L2-PASS
   vs B200 (mean 0.188, 0.7% mismatch) was collected on a FULL-EAGER engine; the
   shipped image mines on a COMPILED engine with skip_compiled PoC — re-confirm
   H200 nonce L2 on the compiled engine before trusting cross-hw validity.

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.

Surgical / fail-loud: errors if either anchor is missing. Idempotent.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/h200-glm-5-2-plugin.py /tmp/h200-glm-5-2-plugin.py
    RUN python3 /tmp/h200-glm-5-2-plugin.py && rm /tmp/h200-glm-5-2-plugin.py
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
    "# --- Kaitaku H200-GLM-5.2 plugin hardcodes (tools/runner-patches/h200-glm-5-2-plugin.py) ---",
    "_h200_glm_plugin_forced = [",
    "    ('--tensor-parallel-size', '8'),",
    "    ('--gpu-memory-utilization', '0.90'),",
    "    ('--max-model-len', '400000'),",
    "    ('--max-num-batched-tokens', '16384'),",
    "    ('--max-num-seqs', '128'),",
    "    ('--kv-cache-dtype', 'fp8'),",
    "    ('--tool-call-parser', 'glm45'),",
    "    ('--reasoning-parser', 'glm45'),",
    "    ('--moe-backend', 'triton'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_h200_glm_plugin_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--enable-expert-parallel',",
    "]",
    "for _flag, _value in _h200_glm_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _h200_glm_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# NOTE: compilation/eager is intentionally NOT forced here. Inference runs",
    "# COMPILED by default (vLLM CompilationMode.VLLM_COMPILE + CUDA graphs); the",
    "# PoC forward runs eager on its own via gonka_poc poc_model_runner",
    "# (set_forward_context skip_compiled=True). For the EXCEPTIONAL eager-",
    "# inference (pure-mining) case the operator passes --enforce-eager; we",
    "# neither force nor strip it so that override is honored.",
    "# --- end Kaitaku H200-GLM-5.2 plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: H200-GLM-5.2 plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # mlnode 0.25.1 reads MLNODE_VLLM_MODULE natively, in a different textual
    # form -- any support for the variable counts as swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: H200-GLM-5.2 plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku H200-GLM-5.2 plugin hardcodes" in src
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
                "ERROR: H200-GLM-5.2 plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: H200-GLM-5.2 plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for H200-GLM-5.2 plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
