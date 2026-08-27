"""H100 GLM-5.3-Flash PLUGIN-base hardcodes for MLNode runner.py.

Two edits to VLLMRunner (same shape as every *-plugin.py patch): a forced-args
block after the constructor anchor, and the launch-module swap.

Forced config — 8x H100/80 GiB. The serving flags come from Crash_Bash_FL's
bring-up command (2026-08-27), which is the only configuration this checkpoint
has been started with so far:

    --tensor-parallel-size 8         (328 GiB of FP8 weights need the whole box)
    --no-enable-flashinfer-autotune  (his run; autotune is a startup cost with
                                      no measured benefit here yet)
    --tool-call-parser glm47
    --reasoning-parser glm45
    (flags) --trust-remote-code --enable-auto-tool-choice

Added on top, as on every PoC leaf:

    --logprobs-mode processed_logprobs
    --worker-extension-cls gonka_poc.worker.PoCWorkerExtension

NOT forced, deliberately:
    - NO --kv-cache-dtype: the Hopper arm of the bring-up runs the default
      dtype. The Blackwell leaf pins fp8; keeping the two apart is the point of
      having two images.
    - NO --max-model-len: native context is 1,048,576 and no governance value
      exists for this checkpoint yet. Leave it to DAPI rather than invent a cap.
    - NO --attention-backend, NO compilation pins: nothing has been measured on
      this model, and a wrong pin is a consensus hazard, not a slow path.
    - NO batch sizing: defaults until a benchmark says otherwise.

TEST image. GLM-5.3-Flash has no chain governance record, and its vLLM support
is an open upstream PR — see tools/model-registry.cue.
"""

import os
import sys

# Release-line Dockerfile installs mlnode under /app/packages/api/src; the
# pre-0.25.1 fat-fork used /app/src. Resolve at runtime so one patch works
# on both, and FAIL if neither exists (a silent skip ships an unconfigured
# image -- see the stage-4 fail-loud guard).
_CANDIDATES = (
    "/app/packages/api/src/api/inference/vllm/runner.py",
    "/app/src/api/inference/vllm/runner.py",
)
FILE = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])
MARKER = "self.processes: List[subprocess.Popen] = []"
INDENT = " " * 8

MODULE_MARKER = '"-m", "vllm.entrypoints.openai.api_server",'
MODULE_REPLACEMENT = '"-m", os.getenv("MLNODE_VLLM_MODULE", "vllm.entrypoints.openai.api_server"),'

INJECTION_LINES = [
    "",
    "# --- Kaitaku H100-GLM-5.3-Flash plugin hardcodes (tools/runner-patches/h100-glm-5-3-flash-plugin.py) ---",  # noqa: E501
    "_h100_glm53_forced = [",
    "    ('--tensor-parallel-size', '8'),",
    "    ('--tool-call-parser', 'glm47'),",
    "    ('--reasoning-parser', 'glm45'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_h100_glm53_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--no-enable-flashinfer-autotune',",
    "]",
    "for _flag, _value in _h100_glm53_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _h100_glm53_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# --- end Kaitaku H100-GLM-5.3-Flash plugin hardcodes ---",
]


def main() -> int:
    injection = "".join((INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES)

    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            "ERROR: h100-glm-5-3-flash patch: forced-args marker not found. "
            "Upstream runner.py may have been refactored - re-verify the patch.\n"
        )
        return 1
    if "_h100_glm53_forced" in src:
        sys.stderr.write("patch already applied; skipping\n")
        return 0

    idx = src.index(MARKER) + len(MARKER)
    src = src[:idx] + "\n" + injection + src[idx:]

    if MODULE_MARKER in src:
        src = src.replace(MODULE_MARKER, MODULE_REPLACEMENT)
    elif "MLNODE_VLLM_MODULE" not in src:
        sys.stderr.write(
            "ERROR: h100-glm-5-3-flash patch: launch-module marker not found and no "
            "MLNODE_VLLM_MODULE support present.\n"
        )
        return 1

    with open(FILE, "w") as f:
        f.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
