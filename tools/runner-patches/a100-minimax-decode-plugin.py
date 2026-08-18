"""A100 MiniMax-M2.7 DECODE-PoC plugin hardcodes for MLNode runner.py.

Per-card decode variant (campaign 2026-08-15/16, migration-025-kit/PROFILES.md).
Same two structural edits as a100-minimax-decode-plugin.py. Forces the decode
launch profile for 4xA100: TP=4, moe=marlin, mns 704,
compilation-config {"max_cudagraph_capture_size":608}
(capture 608 per the wall>512 rule).
gpu-memory-utilization 0.92: 0.92 (conservative; the capture-640 +22% chat lead is unverified).
Env pairing at the profile level: POC_DECODE_CAPTURE=1,
POC_DECODE_MAX_BATCH=600, POC_BATCH_SIZE_DEFAULT=600.
Additive/surgical; fails loud if an anchor is missing.
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8  # VLLMRunner.__init__ method body indent

# Edit 2: launch-module swap. runner.py already imports `os` (used for
# VLLM_PYTHON_PATH), so os.getenv is safe. The literal below is matched exactly
# and replaced in place (indentation preserved — only the substring changes).
MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = (
    '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'
)

INJECTION_LINES = [
    "",
    "# --- Kaitaku A100-MiniMax DECODE plugin hardcodes (tools/runner-patches/a100-minimax-decode-plugin.py) ---",
    "_a100_minimax_decode_forced = {",
    "    '--worker-extension-cls': 'gonka_poc.worker.PoCWorkerExtension',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--attention-backend': 'FLASHINFER',",
    "    '--tensor-parallel-size': '4',",
    "    '--max-model-len': '180000',",
    "    '--moe-backend': 'marlin',",
    "    '--max-num-seqs': '704',",
    '''    '--compilation-config': '{"max_cudagraph_capture_size":608}',''',
    "}",
    "for _flag, _value in _a100_minimax_decode_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# Governance-owned args: DAPI broadcasts them when the node runs in the",
    "# network, and a broadcast value must win. Added only when ABSENT, so a",
    "# standalone launch still gets a working default instead of a 400 on",
    "# tool calls and silent reasoning-in-content.",
    "_a100_minimax_decode_defaults = {",
    "    '--tool-call-parser': 'minimax_m2',",
    "    '--reasoning-parser': 'minimax_m2_append_think',",
    "    '--kv-cache-dtype': 'fp8',",
    "    '--gpu-memory-utilization': '0.92',",
    "}",
    "for _flag, _value in _a100_minimax_decode_defaults.items():",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.extend([_flag, _value])",
    "if '--enable-auto-tool-choice' not in self.additional_args:",
    "    self.additional_args.append('--enable-auto-tool-choice')",
    "# Campaign parity (Pasha 2026-08-17): the base enables prefix caching,",
    "# which costs a measured -12% chat on unique prompts (25.08 vs 28.18",
    "# req/s @536); every campaign number was taken with it OFF.",
    "if '--no-enable-prefix-caching' not in self.additional_args:",
    "    self.additional_args.append('--no-enable-prefix-caching')",
    "# NOTE: --enforce-eager must NOT be set. Unlike the prefill scheme (whose",
    "# golden reference is the 0.20 eager PoC forward), the decode branch has a",
    "# SINGLE execution path: compiled, with a private CUDA-graph capture of the",
    "# step. Eager is not an execution mode here (only a debug fallback behind",
    "# POC_DECODE_SKIP_COMPILED, off by default and outside the consensus",
    "# contract), so global eager would both break the graph capture and drop",
    "# CUDA graphs for inference.",
    "# --- end Kaitaku A100-MiniMax DECODE plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    # --- Preconditions (fail loud if upstream refactored either anchor) ---
    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: Kaitaku A100-MiniMax DECODE plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # The release-line runner.py (gonka-ai vllm-0.25.1-upgrade) ships its own
    # MLNODE_VLLM_MODULE swap in a different textual form -- any support for
    # the variable counts as already swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: Kaitaku A100-MiniMax DECODE plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored — re-verify "
            "the MLNODE_VLLM_MODULE swap.\n"
        )
        return 1

    already_injected = "Kaitaku B300-MiniMax plugin hardcodes" in src
    if already_injected and already_swapped:
        print("runner.py already patched — skipping")
        return 0

    out = src

    # --- Edit 1: inject the forced-args block after the additional_args marker ---
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
                "ERROR: Kaitaku A100-MiniMax DECODE plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    # --- Edit 2: swap the launched module to honour MLNODE_VLLM_MODULE ---
    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: Kaitaku A100-MiniMax DECODE plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for A100-MiniMax DECODE plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
