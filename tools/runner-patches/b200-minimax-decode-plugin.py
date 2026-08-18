"""B200 MiniMax-M2.7 DECODE-PoC plugin hardcodes for MLNode runner.py.

Decode variant of b200-minimax-m2-7-plugin.py, structurally identical to
b300-minimax-decode-plugin.py (same two edits: forced-args block +
MLNODE_VLLM_MODULE launch swap). Card-specific deltas vs the B300 patch:

    --tensor-parallel-size 2
        2 x B200 (180 GB HBM each) is the minimum that fits the 320 GB chain
        VRam requirement — same as the prod b200 profile. Forced (a wrong
        operator TP silently changes the consensus surface).
    --moe-backend flashinfer_trtllm / --attention-backend FLASHINFER
        Blackwell sm_100, same kernels as B300 (the 0.20-parity pair).

UNTUNED — this is a TEST image: no decode campaign point exists for B200.
The launch flags mirror the B300 campaign profile; the KV-wall batch
(POC_DECODE_MAX_BATCH / POC_BATCH_SIZE_DEFAULT at the profile level) is a
B300-derived placeholder, NOT a measured wall. First hardware task on this
image is the batch sweep (see HANDOVER-PASHA.md, B200 section). The
"wall above 512" pair (--max-num-seqs 704 + max_cudagraph_capture_size 608)
is carried so the sweep can probe above 512 at all.
fuse_allreduce_rms is disabled preemptively (H200 precedent — see the inline
comment in the forced block); unmeasured on B200, optional A/B after the sweep.

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
    "# --- Kaitaku B200-MiniMax DECODE plugin hardcodes (tools/runner-patches/b200-minimax-decode-plugin.py) ---",
    "_b200_minimax_decode_forced = {",
    "    '--worker-extension-cls': 'gonka_poc.worker.PoCWorkerExtension',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--attention-backend': 'FLASHINFER',",
    "    '--moe-backend': 'flashinfer_trtllm',",
    "    '--tensor-parallel-size': '2',",
    "    '--max-num-seqs': '704',",
    "    # fuse_allreduce_rms=false: preemptive H200 mirror, UNMEASURED on",
    "    # B200. The 0.25 fused-AR pass runs on any TP>1 and its workspace",
    "    # costs ~15k KV tokens (H200: wall 584->552, PoC 31.50->30.49);",
    "    # the mnnvl/FlashInfer-allreduce band regression (vLLM PR #47219,",
    "    # bug #44079) hit the NVLink+FLASHINFER combo — exactly what B200",
    "    # is. On H200 disabling was strictly better on both axes; H100/A100",
    "    # keep fusion ON only because they showed no symptom. Optional A/B",
    "    # after the batch sweep may revisit this.",
    '''    '--compilation-config': '{"max_cudagraph_capture_size":608,"pass_config":{"fuse_allreduce_rms":false}}',''',
    "}",
    "for _flag, _value in _b200_minimax_decode_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# Governance-owned args: DAPI broadcasts them when the node runs in the",
    "# network, and a broadcast value must win. Added only when ABSENT, so a",
    "# standalone launch still gets a working default instead of a 400 on",
    "# tool calls and silent reasoning-in-content.",
    "_b200_minimax_decode_defaults = {",
    "    '--tool-call-parser': 'minimax_m2',",
    "    '--reasoning-parser': 'minimax_m2_append_think',",
    "    '--kv-cache-dtype': 'fp8',",
    "    # UNTUNED on B200 — B300 campaign value carried as a default; the",
    "    # batch sweep on this test image decides the final gmu.",
    "    '--gpu-memory-utilization': '0.95',",
    "}",
    "for _flag, _value in _b200_minimax_decode_defaults.items():",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.extend([_flag, _value])",
    "if '--enable-auto-tool-choice' not in self.additional_args:",
    "    self.additional_args.append('--enable-auto-tool-choice')",
    "# Campaign parity (Pasha 2026-08-17): the base enables prefix caching,",
    "# which costs a measured -12% chat on unique prompts (25.08 vs 28.18",
    "# req/s @536 on B300); every campaign number was taken with it OFF.",
    "if '--no-enable-prefix-caching' not in self.additional_args:",
    "    self.additional_args.append('--no-enable-prefix-caching')",
    "# NOTE: --enforce-eager must NOT be set. Unlike the prefill scheme (whose",
    "# golden reference is the 0.20 eager PoC forward), the decode branch has a",
    "# SINGLE execution path: compiled, with a private CUDA-graph capture of the",
    "# step. Eager is not an execution mode here (only a debug fallback behind",
    "# POC_DECODE_SKIP_COMPILED, off by default and outside the consensus",
    "# contract), so global eager would both break the graph capture and drop",
    "# CUDA graphs for inference.",
    "# --- end Kaitaku B200-MiniMax DECODE plugin hardcodes ---",
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
            f"ERROR: Kaitaku B200-MiniMax DECODE plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # The release-line runner.py (gonka-ai vllm-0.25.1-upgrade) ships its own
    # MLNODE_VLLM_MODULE swap in a different textual form -- any support for
    # the variable counts as already swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: Kaitaku B200-MiniMax DECODE plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored — re-verify "
            "the MLNODE_VLLM_MODULE swap.\n"
        )
        return 1

    already_injected = "Kaitaku B200-MiniMax DECODE plugin hardcodes" in src
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
                "ERROR: Kaitaku B200-MiniMax DECODE plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    # --- Edit 2: swap the launched module to honour MLNODE_VLLM_MODULE ---
    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: Kaitaku B200-MiniMax DECODE plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for B200-MiniMax DECODE plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
