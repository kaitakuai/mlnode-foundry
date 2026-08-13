"""B300 DeepSeek-V4-Flash PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner in runner.py (same shape as b300-glm-5-2-plugin.py;
this is the FIRST DeepSeek-V4 leaf — a test image on the vLLM 0.25.1 line):

1. Insert a forced-args block after the constructor anchor
   in __init__. Forces the B300 DeepSeek-V4-Flash config:

     --tensor-parallel-size 1        (149 GiB FP8 weights fit on ONE B300 / 275 GiB
                                      with ~126 GiB headroom; an 8-GPU box auto-runs
                                      8 independent TP=1 engines. PR gonka-poc#14
                                      validated 1×B300 TP1 ~928 nonces/min @ bs=16)
     --gpu-memory-utilization 0.90
     --max-model-len 200000          (TEST cap, NOT governance. V4 native context is
                                      1,048,576; capped low for a first bring-up so KV
                                      profiling is cheap. The real max-model-len is a
                                      chain-governance parameter delivered via DAPI —
                                      raise per experiment B4, do not treat as final.)
     --max-num-batched-tokens 16384
     --kv-cache-dtype fp8            (MANDATORY: FlashMLA coerces to the fp8_ds_mla
                                      layout and asserts without it)
     --logprobs-mode processed_logprobs
     --worker-extension-cls gonka_poc.worker.PoCWorkerExtension   (PLUGIN: worker-
                                      side PoC via collective_rpc)
     (flags) --trust-remote-code

   NOT forced — and deliberately so for consensus safety:
     - NO --attention-backend: the default is deterministic FlashMLA-DSV4 on both
       sm_90 and sm_100 (_select_dsv4_attn_cls). Pinning FLASHINFER_MLA_SPARSE_DSV4
       would engage placeholder FP8 Q/KV scales and blow cross-hardware L2.
     - NO --enforce-eager / --compilation-config: vLLM auto-forces
       CompilationMode.NONE + breakable-cudagraph for deepseek_v4; the PoC forward
       is eager on its own via gonka-poc skip_compiled. NEVER set
       VLLM_USE_BREAKABLE_CUDAGRAPH=0 (untested upstream).
     - NO tool-call / reasoning parser, NO --enable-expert-parallel: those are
       chain-governance/serving choices not yet defined for V4 (#1408) — add when
       governance mandates them, not before.
     - NO DeepGEMM / VLLM_DISABLED_KERNELS env: V4 FP8 carries native ue8m0 scales
       (no GLM-style float32-requant linear crash) — let the MoE oracle auto-select.

2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint (mounts /api/v1/pow/* + gating) instead of stock api_server:
   `"-m", "vllm.entrypoints.openai.api_server"` →
   `"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`.

Surgical / fail-loud: errors if either anchor is missing. Idempotent.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b300-deepseek-v4-flash-plugin.py /tmp/b300-deepseek-v4-flash-plugin.py
    RUN python3 /tmp/b300-deepseek-v4-flash-plugin.py && rm /tmp/b300-deepseek-v4-flash-plugin.py
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
    "# --- Kaitaku B300-DeepSeek-V4-Flash plugin hardcodes (tools/runner-patches/b300-deepseek-v4-flash-plugin.py) ---",
    "_b300_dsv4_plugin_forced = [",
    "    ('--tensor-parallel-size', '1'),",
    "    ('--gpu-memory-utilization', '0.90'),",
    "    ('--max-model-len', '200000'),",
    "    ('--max-num-batched-tokens', '16384'),",
    "    ('--kv-cache-dtype', 'fp8'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b300_dsv4_plugin_flags = [",
    "    '--trust-remote-code',",
    "]",
    "for _flag, _value in _b300_dsv4_plugin_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b300_dsv4_plugin_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# NOTE: attention-backend / compilation / parsers are intentionally NOT forced.",
    "# V4 auto-forces CompilationMode.NONE; PoC forward is eager via gonka-poc",
    "# skip_compiled. Default attention is deterministic FlashMLA-DSV4 — pinning a",
    "# backend (esp. FlashInfer-DSV4 placeholder scales) would break cross-hw L2.",
    "# --- end Kaitaku B300-DeepSeek-V4-Flash plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: B300-DeepSeek-V4-Flash plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # mlnode 0.25.1 reads MLNODE_VLLM_MODULE natively, in a different textual
    # form -- any support for the variable counts as swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: B300-DeepSeek-V4-Flash plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored.\n"
        )
        return 1

    already_injected = "Kaitaku B300-DeepSeek-V4-Flash plugin hardcodes" in src
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
                "ERROR: B300-DeepSeek-V4-Flash plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: B300-DeepSeek-V4-Flash plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B300-DeepSeek-V4-Flash plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
