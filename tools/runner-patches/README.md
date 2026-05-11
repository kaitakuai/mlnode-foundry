# `tools/runner-patches/` — Python patchers for mlnode's `runner.py`

Each `*.py` script mutates `runner.py` (in the built image) to inject GPU+model-specific vLLM flags that can't be cleanly expressed as ENV vars. Profile references patcher by basename (without `.py` extension):

```cue
runner_patch: "b300-kimi"
```

The Stage 3 build resolves to `tools/runner-patches/<name>.py`, COPYs it into the image at `/tmp/runner-patch.py`, runs it, and removes it.

## Inventory

| Patch | What it does |
|-------|--------------|
| `b300.py` | B300 Qwen baseline: forces TP=1, gpu_memory_utilization=0.95, max_model_len, logprobs_mode=processed |
| `b300-kimi.py` | B300 Kimi-K2.6 INT4: forces TP=4, max_num_batched_tokens=131072, compilation mode=0, cudagraph=NONE |
| `cold-start-tolerance.py` | Patches WAIT_FOR_SERVER_TIMEOUT + watcher grace window for slow cold starts |

## Style guidelines

- Each patcher is idempotent (safe to re-run on already-patched runner.py)
- Use `marker-based` safety check (e.g., `if "# already patched" in source: skip`)
- Mutations should be additive (insert hardcoded flag dicts) rather than rewriting
- Document forced vs default flags clearly in the patcher's docstring

Source: migrated from legacy `kaitakuai/mlnode/tools/fragments/hw-patches/runner-py-patches/`.
