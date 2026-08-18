"""B300 MiniMax-M2.7 DECODE-PoC plugin hardcodes for MLNode runner.py.

Decode variant of b300-minimax-plugin.py. Same two structural edits (forced
args block + MLNODE_VLLM_MODULE launch swap); the forced set carries the
launch profile validated in the 2026-08-15/16 hardware campaign (see
migration-025-kit/PROFILES.md):

    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
        Worker-side decode entry: execute_poc_decode via public collective_rpc.
    --logprobs-mode processed_logprobs
        Inference-validation replay compares processed logprobs (unchanged).
    --attention-backend FLASHINFER
        Pinned: the 0.25 auto-selector ignores VLLM_ATTENTION_BACKEND and
        FLASH_ATTN loses to FLASHINFER on the PoC forms (campaign A/B).
    --moe-backend flashinfer_trtllm
        The 0.20-parity kernel for Blackwell; triton costs -34% PoC on B300,
        deep_gemm hits a chronic JIT illegal-instruction (tech-debt #3).
    --max-num-seqs 704 and --compilation-config {"max_cudagraph_capture_size":608}
        The "wall above 512" rule: without BOTH, the direct decode path fails
        above batch 512 (metadata buffers sized by max-num-seqs -> "provided
        out is the wrong size"; graph-builder buffers sized by capture size ->
        CUDA illegal memory access). Passed as one JSON argv token — argv goes
        straight to execve, no shell, so no brace-expansion hazard.
    --gpu-memory-utilization 0.95
        B300 profile decision 2026-08-15: 288 GB card, weights take 229 GB;
        0.92->0.95 lifts the KV wall 400->536 (+23% PoC) and releases the
        chat KV squeeze (chat 24.77 -> 30.00 req/s). Added-if-absent so a
        governance broadcast still wins.

Env pairing (set at the profile level, not here): POC_DECODE_CAPTURE=1,
POC_DECODE_MAX_BATCH=536 (chunk clamp at the KV wall: the lease is taken for
batch_size, and a client batch above the wall pays a 6-7 s lease timeout per
chunk — measured, this poisoned the first A2 run).

Additive/surgical — no upstream code is removed; fails loud if either anchor
is missing (= upstream refactored, re-verify the patch).
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
    "# --- Kaitaku B300-MiniMax DECODE plugin hardcodes (tools/runner-patches/b300-minimax-decode-plugin.py) ---",
    "_b300_minimax_decode_forced = {",
    "    '--worker-extension-cls': 'gonka_poc.worker.PoCWorkerExtension',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--attention-backend': 'FLASHINFER',",
    "    '--moe-backend': 'flashinfer_trtllm',",
    "    '--max-num-seqs': '704',",
    '''    '--compilation-config': '{"max_cudagraph_capture_size":608}',''',
    "}",
    "for _flag, _value in _b300_minimax_decode_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# Governance-owned args: DAPI broadcasts them when the node runs in the",
    "# network, and a broadcast value must win. Added only when ABSENT, so a",
    "# standalone launch still gets a working default instead of a 400 on",
    "# tool calls and silent reasoning-in-content.",
    "_b300_minimax_decode_defaults = {",
    "    '--tool-call-parser': 'minimax_m2',",
    "    '--reasoning-parser': 'minimax_m2_append_think',",
    "    '--kv-cache-dtype': 'fp8',",
    "    '--gpu-memory-utilization': '0.95',",
    "}",
    "for _flag, _value in _b300_minimax_decode_defaults.items():",
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
    "# --- end Kaitaku B300-MiniMax DECODE plugin hardcodes ---",
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
            f"ERROR: Kaitaku B300-MiniMax DECODE plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # The release-line runner.py (gonka-ai vllm-0.25.1-upgrade) ships its own
    # MLNODE_VLLM_MODULE swap in a different textual form -- any support for
    # the variable counts as already swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: Kaitaku B300-MiniMax DECODE plugin patch: launch-module line {MODULE_MARKER!r} "
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
                "ERROR: Kaitaku B300-MiniMax DECODE plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    # --- Edit 2: swap the launched module to honour MLNODE_VLLM_MODULE ---
    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: Kaitaku B300-MiniMax DECODE plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B300-MiniMax DECODE plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
