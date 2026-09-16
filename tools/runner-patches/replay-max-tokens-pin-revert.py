"""Stage-4 form of gonka-ai/vllm#111: drop the replay max_tokens pin.

kaitakuai/vllm#21 (a34011049 in the release residual) pinned a replay's
max_tokens to the recorded length after SamplingParams had been validated on
the API side. The engine core re-runs __post_init__ when it deserialises the
request, so a client request carrying min_tokens above the recorded length
raised `min_tokens must be less than or equal to max_tokens` inside
process_input_sockets and killed the input thread: /health stayed green,
every request got 200 headers and no tokens, the next PoC /init/generate
hung on collective_rpc with 0 nonces. Two GLM-5.3-Flash nodes on 2026-09-16.

Vlad removed the pin outright (#111, after a first attempt to clamp
min_tokens in #110), so the replay block is the one 3.0.16 / 3.0.17 ship on
release/v0.25.1: the validator runs the recorded tokens under the request's
own limit. This removes the same block from an image whose residual still has
it — both the pin and, if present, the #110 clamp — and reports no-op once
the base is past #111. Anchored on the pin's own comment; fails loudly if it
moved.
"""
import os
import re
import sys

_CANDIDATES = (
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py",
    "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py",
)
FILE = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])

# From "# A replay must emit the recorded sequence" up to (not including) the
# eos_token_id line that follows the block in every version of the pin.
BLOCK = re.compile(
    r"                    # A replay must emit the recorded sequence and nothing past\n"
    r"(?:                    .*\n)*?"
    r"                        sampling_params\.max_tokens = replay_len\n"
    r"(?:                    # The params were validated before the pin; a min_tokens\n"
    r"(?:                    .*\n)*?"
    r"                        sampling_params\.min_tokens = replay_len\n)?"
    r"(?=                    eos_token_id = tokenizer\.eos_token_id\n)"
)


def main() -> int:
    with open(FILE) as f:
        src = f.read()
    if "replay_len" not in src:
        print("replay-max-tokens-pin-revert: no pin in this residual; no-op")
        return 0
    new, n = BLOCK.subn("", src, count=1)
    if n != 1 or "replay_len" in new:
        sys.stderr.write(
            f"ERROR: replay-max-tokens-pin-revert: could not remove the pin block cleanly "
            f"from {FILE} (matched {n}, replay_len still present: {'replay_len' in new}). "
            "The replay branch may have moved - re-verify.\n"
        )
        return 1
    with open(FILE, "w") as f:
        f.write(new)
    print("replay-max-tokens-pin-revert: replay max_tokens pin removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
