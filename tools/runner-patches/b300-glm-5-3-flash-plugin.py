"""B300 GLM-5.3-Flash PLUGIN-base hardcodes for MLNode runner.py.

Three flag classes, applied to self.additional_args at VLLMRunner.__init__:
  forced    set or override (TP, fp8 KV, block size, seq cap, parsers, plugin plumbing)
  defaults  set only when the operator did not pass the flag
  flags     boolean switches, appended if missing

Values are the ones the GLM-5.3-Flash measurements ran with
(kaitakuai/experiments/2026-09). 2x275 GB, TP=2 (328 GiB of FP8 weights fit in 550 GiB); inference validation on B300 passed the same block-size / max-num-seqs / autotune flags explicitly; 65536 batched tokens (batch x 1024 for batch 32 and above) per Crash_Bash_FL, 2026-09-10.

Model-side reasons for the common set:
  --kv-cache-dtype fp8      FlashInfer 0.6.18 SM90 sparse-MLA path; bf16 KV
                            yields 16-100 nonces then an illegal memory access
  --block-size 2304         divisible by both kpool and MLA paging
  --max-num-seqs 256        the hybrid architecture allocates one Mamba block
                            per sequence, 512 is the hard cap
  --no-enable-flashinfer-autotune  autotune hits an illegal memory access on Hopper (2/2)
  glm47 / glm45             tool-call and reasoning parsers for this checkpoint
No --max-model-len: native context is 1,048,576 and no governance value exists.
"""
import os
import sys

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
    "# --- Kaitaku B300-GLM-5.3-Flash plugin hardcodes (tools/runner-patches/b300-glm-5-3-flash-plugin.py) ---",  # noqa: E501
    "_b300_glm53_forced = [",
    "    ('--tensor-parallel-size', '2'),",
    "    ('--kv-cache-dtype', 'fp8'),",
    "    ('--block-size', '2304'),",
    "    ('--max-num-seqs', '256'),",
    "    ('--tool-call-parser', 'glm47'),",
    "    ('--reasoning-parser', 'glm45'),",
    "    ('--logprobs-mode', 'processed_logprobs'),",
    "    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),",
    "]",
    "_b300_glm53_defaults = [",
    "    ('--max-num-batched-tokens', '65536'),",
    "]",
    "_b300_glm53_flags = [",
    "    '--trust-remote-code',",
    "    '--enable-auto-tool-choice',",
    "    '--no-enable-flashinfer-autotune',",
    "]",
    "for _flag, _value in _b300_glm53_forced:",
    "    if _flag in self.additional_args:",
    "        self.additional_args[self.additional_args.index(_flag) + 1] = _value",
    "    else:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag, _value in _b300_glm53_defaults:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.extend([_flag, _value])",
    "for _flag in _b300_glm53_flags:",
    "    if _flag not in self.additional_args:",
    "        self.additional_args.append(_flag)",
    "# --- end Kaitaku B300-GLM-5.3-Flash plugin hardcodes ---",
]


def main() -> int:
    injection = "".join((INDENT + line + "\n") if line else "\n" for line in INJECTION_LINES)
    with open(FILE) as f:
        src = f.read()
    if MARKER not in src:
        sys.stderr.write(
            "ERROR: b300-glm-5-3-flash patch: forced-args marker not found. "
            "Upstream runner.py may have been refactored - re-verify the patch.\n"
        )
        return 1
    if "_b300_glm53_forced" in src:
        sys.stderr.write("patch already applied; skipping\n")
        return 0
    idx = src.index(MARKER) + len(MARKER)
    src = src[:idx] + "\n" + injection + src[idx:]
    if MODULE_MARKER in src:
        src = src.replace(MODULE_MARKER, MODULE_REPLACEMENT)
    elif "MLNODE_VLLM_MODULE" not in src:
        sys.stderr.write(
            "ERROR: b300-glm-5-3-flash patch: launch-module marker not found and no "
            "MLNODE_VLLM_MODULE support present.\n"
        )
        return 1
    with open(FILE, "w") as f:
        f.write(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
