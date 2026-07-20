# `tools/runner-patches/` — Python patchers for mlnode's `runner.py`

Each `*.py` script mutates `runner.py` (in the built image) to inject GPU+model-specific vLLM flags that can't be cleanly expressed as ENV vars. Profile references patcher by basename (without `.py` extension):

```cue
runner_patch: "b300-kimi"
```

The Stage 4 build resolves to `tools/runner-patches/<name>.py`, COPYs it into the image at `/tmp/runner-patch.py`, runs it, and removes it.

## Inventory

| Patch | What it does |
|-------|--------------|
| `b300.py` | B300 Qwen baseline: forces TP=1, gpu_memory_utilization=0.95, max_model_len, logprobs_mode=processed |
| `b300-kimi.py` | B300 Kimi-K2.6 INT4: forces TP=4, max_num_batched_tokens=131072, compilation mode=0, cudagraph=NONE |
| `b300-minimax-plugin.py` | B300 MiniMax-M2.7 on the vllm-poc PLUGIN base: swaps the launch module to `MLNODE_VLLM_MODULE` and forces `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`, `--logprobs-mode processed_logprobs`, `--attention-backend FLASHINFER`. Does NOT force `--enforce-eager` (PoC eager via gonka-poc `skip_compiled`; inference keeps CUDA graphs). Governance args (max-model-len/kv-cache-dtype/parsers) NOT re-injected — they flow from the DAPI broadcast. |
| `b300-kimi-k2-6-plugin.py` | B300 Kimi-K2.6 INT4 on the vllm-poc PLUGIN base: swaps the launch module to `MLNODE_VLLM_MODULE` and forces the B300 Kimi tune (TP=4, gpu-memory-utilization=0.85, max-num-batched-tokens=131072, max-num-seqs=128, `--compilation-config {"mode":0,"cudagraph_mode":"NONE"}` eager, CUTLASS_MLA, kimi_k2 parsers, mm-encoder-tp-mode=data, expert-parallel, processed_logprobs) plus `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`. REMOVES `--enforce-eager` (conflicts with `--compilation-config`; eager is set via mode=0). max-model-len NOT capped (256K fits on 4×B300). |
| `b200-kimi-k2-6-plugin.py` | B200 Kimi-K2.6 INT4 on the vllm-poc PLUGIN base: same shape as b300-kimi but the validated B200 envelope (experiment kimi_k26_int4_4xb200) — TP=4, gpu-memory-utilization=0.93, max-num-batched-tokens=32768, max-num-seqs=32, **max-model-len=120000** (capped; 256K OOMs cudagraph capture on B200's 178 GiB/GPU), **`--compilation-config {"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}` CUDA graphs** (NOT eager — at B200 batch=32 cudagraph works and the experiment showed it does not change PoC throughput), CUTLASS_MLA, kimi_k2 parsers, mm-encoder-tp-mode=data, expert-parallel, processed_logprobs, `--worker-extension-cls`. REMOVES `--enforce-eager` (conflicts with `--compilation-config`). PoC-forward eager via gonka-poc skip_compiled. |
| `b200-glm-5-2-plugin.py` | B200 (×8) GLM-5.2 FP8 on the vllm-poc PLUGIN base: swaps the launch module to `MLNODE_VLLM_MODULE` and forces the GLM-5.2 tune (TP=8, gmu=0.85, max-model-len=400000, max-num-batched-tokens=16384, max-num-seqs=16, kv-cache-dtype=fp8_e4m3, glm47/glm45 parsers, processed_logprobs, trust-remote-code, enable-auto-tool-choice) plus `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`. Does NOT force `--compilation-config`/`--enforce-eager` (inference COMPILED by default; PoC eager via gonka-poc `skip_compiled`; operator passes `--enforce-eager` for the pure-mining exception). Does NOT pin `--attention-backend` (GLM-5.2 is DSA, not MLA). DeepGEMM split (MoE-on + linear→Cutlass via `VLLM_DISABLED_KERNELS`) is set via env in the `b200-glm-5-2` profile leaf, not here. |
| `b300-glm-5-2-plugin.py` | B300 (×4, 2 engines/box) GLM-5.2 FP8 on the PLUGIN base: same shape as b200-glm but the B300 tune — TP=4, gmu=0.92, max-model-len=400000, max-num-batched-tokens=16384, max-num-seqs=64, kv-cache-dtype=fp8_e4m3, glm47/glm45. DeepGEMM split via the `b300-glm-5-2` leaf env. Compilation/eager NOT forced (same policy as b200-glm). |
| `b300-deepseek-v4-flash-plugin.py` | B300 (×1, 8 engines/box) DeepSeek-V4-Flash FP8 on the vLLM 0.25.1 PLUGIN base (first V4 leaf, TEST bring-up): swaps the launch module to `MLNODE_VLLM_MODULE` and forces TP=1, gmu=0.90, max-model-len=200000 (TEST cap — real cap is governance-level), max-num-batched-tokens=16384, **kv-cache-dtype=fp8 (MANDATORY — FlashMLA fp8_ds_mla assert)**, processed_logprobs, plus flag trust-remote-code, and `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`. Deliberately does NOT force `--attention-backend` (default FlashMLA-DSV4 is deterministic; FlashInfer-DSV4 placeholder scales = cross-hw L2 hazard), `--enforce-eager`/`--compilation-config` (V4 auto-forces NONE; PoC eager via skip_compiled), tool/reasoning parsers or expert-parallel (governance-level, undefined for V4), or any DeepGEMM env (native ue8m0). |
| `h200-glm-5-2-plugin.py` | H200 (×8, Hopper sm_90) GLM-5.2 FP8 on the PLUGIN base: TP=8, gmu=0.90, max-model-len=400000, max-num-batched-tokens=16384, max-num-seqs=128, **kv-cache-dtype=fp8 (FLASHMLA_SPARSE alias for fp8_ds_mla compressed KV — NOT the rejected fp8_e4m3; Pasha-verified 2026-06-26)**, **tool-call-parser glm45 (NOT glm47)**, reasoning glm45, **`--moe-backend triton`** (DeepGEMM off on Hopper), processed_logprobs, plus flags trust-remote-code, enable-auto-tool-choice, **`--enable-expert-parallel`** (H200-unique). No `--attention-backend` pin (FLASHMLA_SPARSE auto). DeepGEMM-off env in the `h200-glm-5-2` leaf. Compilation/eager NOT forced. |
| `cold-start-tolerance.py` | Patches WAIT_FOR_SERVER_TIMEOUT + watcher grace window for slow cold starts |

## Style guidelines

- Each patcher is idempotent (safe to re-run on already-patched runner.py)
- Use `marker-based` safety check (e.g., `if "# already patched" in source: skip`)
- Mutations should be additive (insert hardcoded flag dicts) rather than rewriting
- Document forced vs default flags clearly in the patcher's docstring

Source: migrated from legacy `kaitakuai/mlnode/tools/fragments/hw-patches/runner-py-patches/`.
