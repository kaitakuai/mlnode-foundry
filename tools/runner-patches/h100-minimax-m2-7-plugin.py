"""H100 MiniMax-M2.7 plugin hardcodes — 0.25.1 line, params inherited from the 2026-05 vLLM 0.20 campaigns (NOT yet revalidated on 0.25.1).

B300 MiniMax-M2.7 PLUGIN-base hardcodes for MLNode runner.py.

Two edits to `VLLMRunner` in `runner.py`:

1. Insert a forced-args block right after `self.additional_args =
   additional_args or []` in `__init__` (worker-extension-cls + correctness/
   throughput knobs).
2. Swap the launched vLLM module so the subprocess runs the gonka-poc COMPOSED
   entrypoint instead of vLLM's stock api_server. The `"-m",
   "vllm.entrypoints.openai.api_server"` literal becomes `"-m",
   os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server")`, so
   the baked `MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router` env takes
   effect. This is the SERVER-side half of the plugin flip and is REQUIRED:
   the PoC HTTP routes (/api/v1/pow/*) and gating middleware are mounted only
   by `gonka_poc.entrypoint.api_router` (app.include_router) — stock
   api_server does NOT mount them even with the plugin auto-loaded
   (gonka_poc.plugin.register is a sentinel, not a route installer). Without
   this swap the subprocess serves stock vLLM with no /pow routes and PoC
   fails. (Equivalent to the upstream PR-A change to gonka mlnode; done here
   as a build-time patch because foundry pins stock mlnode f3b3893.)

Additive/surgical — no upstream code is removed; fails loud if either anchor is
missing (= upstream refactored, re-verify the patch).

Why the plugin base: this image builds on the vllm-poc PLUGIN base (residual
vLLM + gonka-poc package; see ADR-0013), NOT the fat-fork monolith. The PoC
math is no longer in the vLLM tree — it ships as an out-of-tree plugin attached
via vLLM's official extension points.

Forces (overwrite if present — operator / chain broadcast cannot drop them):
    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
        Worker-side PoC: registers execute_poc_forward as a worker method
        callable via the PUBLIC collective_rpc API. This is the plugin
        replacement for the monolith's AsyncLLM.poc_request monkey-patch
        (ADR-0013 Layer 1). Without it the composed entrypoint
        (MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router) has no worker
        method to drive and PoC forwards fail.
    NOT forced: --enforce-eager. The PoC forward already runs eager via
        gonka_poc.poc.poc_model_runner (set_forward_context(..., skip_compiled=True),
        same mechanism the fat-fork used), so bit-compat is guaranteed at the
        PoC-forward level. A global --enforce-eager would ALSO disable CUDA
        graphs for ordinary inference (throughput regression) for no PoC
        benefit — the fat-fork b300 image ran with NO --enforce-eager and
        still produced L2-valid nonces. So inference keeps CUDA graphs; only
        the PoC forward is eager.
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
    COPY tools/runner-patches/h100-minimax-m2-7-plugin.py /tmp/h100-minimax-m2-7-plugin.py
    RUN python3 /tmp/h100-minimax-m2-7-plugin.py && rm /tmp/h100-minimax-m2-7-plugin.py
"""

from __future__ import annotations

import sys

FILE = "/app/packages/api/src/api/inference/vllm/runner.py"
MARKER = "self.additional_args = additional_args or []"
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
    "# --- Kaitaku H100-MiniMax plugin hardcodes (tools/runner-patches/h100-minimax-m2-7-plugin.py) ---",
    "_h100_minimax_plugin_forced = {",
    "    '--tensor-parallel-size': '4',",
    "    '--gpu-memory-utilization': '0.92',",
    "    '--max-model-len': '131072',",
    "    '--moe-backend': 'triton',",
    "    '--attention-backend': 'FLASHINFER',",
    "    '--worker-extension-cls': 'gonka_poc.worker.PoCWorkerExtension',",
    "    '--logprobs-mode': 'processed_logprobs',",
    "}",
    "for _flag, _value in _h100_minimax_plugin_forced.items():",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "# NOTE: --enforce-eager is intentionally NOT forced — the PoC forward is",
    "# already eager via gonka_poc.poc.poc_model_runner (skip_compiled=True);",
    "# forcing global eager would needlessly drop CUDA graphs for inference.",
    "# --- end Kaitaku H100-MiniMax plugin hardcodes ---",
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
            f"ERROR: Kaitaku H100-MiniMax plugin patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    # The release-line runner.py ships its own MLNODE_VLLM_MODULE swap
    # (gonka-ai vllm-0.25.1-upgrade, hardened revision) in a different
    # textual form -- any MLNODE_VLLM_MODULE support counts as swapped.
    already_swapped = "MLNODE_VLLM_MODULE" in src
    if MODULE_MARKER not in src and not already_swapped:
        sys.stderr.write(
            f"ERROR: Kaitaku H100-MiniMax plugin patch: launch-module line {MODULE_MARKER!r} "
            f"not found in {FILE}. Upstream runner.py may have been refactored — re-verify "
            "the MLNODE_VLLM_MODULE swap.\n"
        )
        return 1

    already_injected = "Kaitaku H100-MiniMax plugin hardcodes" in src
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
                "ERROR: Kaitaku H100-MiniMax plugin patch: marker present but insertion did not fire\n"
            )
            return 1
        out = "".join(new_lines)

    # --- Edit 2: swap the launched module to honour MLNODE_VLLM_MODULE ---
    if not already_swapped:
        if MODULE_MARKER not in out:
            sys.stderr.write(
                "ERROR: Kaitaku H100-MiniMax plugin patch: launch-module marker vanished before swap\n"
            )
            return 1
        out = out.replace(MODULE_MARKER, MODULE_REPLACEMENT, 1)

    with open(FILE, "w") as f:
        f.write(out)
    print(
        "runner.py patched for H100-MiniMax plugin hardcodes + MLNODE_VLLM_MODULE launch swap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
