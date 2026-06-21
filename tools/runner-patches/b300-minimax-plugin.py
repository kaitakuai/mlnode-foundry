"""B300 MiniMax-M2.7 PLUGIN-base hardcodes for MLNode runner.py.

Inserts a small block right after `self.additional_args = additional_args or []`
in `VLLMRunner.__init__`. Additive — no upstream code is removed; fails loud if
the marker line is missing (= upstream refactored, re-verify the patch).

Why: this image builds on the vllm-poc PLUGIN base (residual vLLM + gonka-poc
package; see ADR-0013), NOT the fat-fork monolith. The PoC math is no longer in
the vLLM tree — it ships as an out-of-tree plugin attached via vLLM's official
extension points. Two args wire that plugin in, and two more pin the
correctness/throughput knobs this profile needs:

Forces (overwrite if present — operator / chain broadcast cannot drop them):
    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
        Worker-side PoC: registers execute_poc_forward as a worker method
        callable via the PUBLIC collective_rpc API. This is the plugin
        replacement for the monolith's AsyncLLM.poc_request monkey-patch
        (ADR-0013 Layer 1). Without it the composed entrypoint
        (MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router) has no worker
        method to drive and PoC forwards fail.
    --enforce-eager
        PoC forward MUST run eager for cross-validator bit-compatibility;
        compiled drift fails the L2 gate → epoch exclusion. (The fat-fork
        relied on vllm/poc/ skip_compiled=True; on the plugin base we pin
        eager at the engine level to be unambiguous.)
    --logprobs-mode processed_logprobs
        PoC v2 correctness — processed (post-sampler) logprobs are what the
        cross-node validation compares.
    --attention-backend FLASHINFER
        Pinned so the vLLM auto-selector heuristic can't silently regress this
        B300 profile across vLLM releases (Blackwell auto-selects FLASHINFER
        today, but we don't trust the heuristic across versions).

NOT forced here (intentionally) — chain governance / DAPI broadcast owns these:
    --max-model-len 180000, --kv-cache-dtype fp8, --tool-call-parser minimax_m2,
    --reasoning-parser minimax_m2_append_think, --enable-auto-tool-choice.
    The network-node DAPI broadcasts the v0.2.13 minimaxGovernanceModel args
    into mlnode runner.py's `additional_args` at runtime; the mlnode runner
    assembles `self.additional_args` identically regardless of which entrypoint
    module it launches, so those args reach the composed gonka-poc engine
    unchanged after the plugin flip. Re-injecting them here would (a) duplicate
    flags and (b) risk pinning a stale governance value if the chain bumps it.
    Verified: governance args flow to the engine after the flip (Q1 = YES).

Idempotent: re-running on an already-patched file is a no-op.

Usage inside a Dockerfile RUN step:
    COPY tools/runner-patches/b300-minimax-plugin.py /tmp/b300-minimax-plugin.py
    RUN python3 /tmp/b300-minimax-plugin.py && rm /tmp/b300-minimax-plugin.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.additional_args = additional_args or []"
INDENT = " " * 8  # VLLMRunner.__init__ method body indent

INJECTION_LINES = [
    "",
    "# --- Kaitaku B300-MiniMax plugin hardcodes (tools/runner-patches/b300-minimax-plugin.py) ---",
    "_b300_minimax_plugin_forced = {",
    "    '--worker-extension-cls': 'gonka_poc.worker.PoCWorkerExtension',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "    '--attention-backend': 'FLASHINFER',",
    "}",
    "for _flag, _value in _b300_minimax_plugin_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# --enforce-eager is a valueless flag; ensure it is present exactly once.",
    "if '--enforce-eager' not in self.additional_args:",
    "    self.additional_args.append('--enforce-eager')",
    "# --- end Kaitaku B300-MiniMax plugin hardcodes ---",
]


def main() -> int:
    injection = "".join(
        (INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES
    )

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: Kaitaku B300-MiniMax plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    if "Kaitaku B300-MiniMax plugin hardcodes" in src:
        print("runner.py already patched — skipping")
        return 0

    lines = src.splitlines(keepends=True)
    new_lines: list[str] = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and MARKER in line:
            new_lines.append(injection)
            inserted = True

    if not inserted:
        sys.stderr.write(
            "ERROR: Kaitaku B300-MiniMax plugin patch: marker present but insertion did not fire\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.writelines(new_lines)
    print("runner.py patched for B300-MiniMax plugin hardcodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
