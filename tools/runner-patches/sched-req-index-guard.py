"""Skip requests the model runner produced no output for, instead of crashing.

`Scheduler.update_from_output` indexes `req_id_to_index[req_id]` directly. Under
async scheduling a request can sit in `num_scheduled_tokens` while the runner
emitted no row for it that step — aborted or preempted between schedule and
execution — and the lookup takes EngineCore down with a KeyError. The guard just
above only covers requests that are None or already finished, which is a
different case: here the request is alive and unfinished, it simply has no row.

Observed as a hard engine crash on the production DeepSeek-V4-Flash-0731 nodes;
the scheduler is model-agnostic, so every profile carries this.

S4 form of kaitakuai/vllm#19. That PR targets our residual tree, which the
release-line images no longer build from — they overlay gonka's published
mlnode, whose vLLM comes from gonka-ai/vllm release/v0.25.1. Delete this patch
once the same fix lands there and a base image carries it; the marker check
below turns it into a loud failure rather than a silent no-op when it does.
"""

import importlib.util
import sys
from pathlib import Path

MARKER = "            req_index = model_runner_output.req_id_to_index[req_id]\n"
REPLACEMENT = """            req_index = model_runner_output.req_id_to_index.get(req_id)
            if req_index is None:
                # Async-scheduling race: the request is in num_scheduled_tokens
                # but the model runner produced no output for it this step
                # (aborted or preempted between schedule and execution). Skip
                # it instead of crashing EngineCore with a KeyError.
                continue
"""
GUARD = "req_id_to_index.get(req_id)"


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.stderr.write("ERROR: sched-req-index-guard: vllm is not installed\n")
        return 1
    path = Path(spec.submodule_search_locations[0]) / "v1/core/sched/scheduler.py"
    src = path.read_text()

    if GUARD in src:
        print("sched-req-index-guard: already guarded; no-op")
        return 0
    if src.count(MARKER) != 1:
        sys.stderr.write(
            f"ERROR: sched-req-index-guard: expected exactly one unguarded lookup in "
            f"{path}, found {src.count(MARKER)}. vLLM's scheduler may have been "
            "refactored — re-verify against kaitakuai/vllm#19.\n"
        )
        return 1

    path.write_text(src.replace(MARKER, REPLACEMENT, 1))
    print(f"sched-req-index-guard: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
