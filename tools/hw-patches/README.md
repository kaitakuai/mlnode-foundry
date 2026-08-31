# `tools/hw-patches/` — hardware-specific Dockerfile fragments

Each `*.dockerfile` is a fragment applied to the Stage 4 image build via the profile's `hw_patches:` list. Fragments are **idempotent** (re-applying produces same result).

Profile references patches by basename (without `.dockerfile` extension):

```cue
hw_patches: [
    "triton-ptxas-from-system-cuda",
    "flashinfer-jit-uninstall",
    "libcuda-compat-580-driver",
    "nvidia-headers-symlinks",
    "cold-start-tolerance",
]
```

The Stage 4 build resolves each name to `tools/hw-patches/<name>.dockerfile`, inlines via Docker build context, and applies in order.

## Inventory

| Patch | What it does | Applies to |
|-------|--------------|------------|
| `triton-ptxas-from-system-cuda` | Overwrites Triton's bundled `ptxas` (lacks newer SM targets) with system CUDA's | sm_103a (B300), sm_120 (RTX PRO 6000), any newer Blackwell |
| `flashinfer-jit-uninstall` | Removes pre-compiled FlashInfer JIT cache (ships sm_120 only) — forces JIT compile-on-first-launch | sm_103a, any non-sm_120 Blackwell |
| `libcuda-compat-580-driver` | Replaces CUDA compat stub libcuda.so with symlink to real driver — fixes broken GPU detection with NVIDIA driver 580+ | Any B300/Blackwell host with 580+ driver |
| `nvidia-headers-symlinks` | Symlinks all CUDA dev headers from nvidia-* pip packages into `/usr/local/cuda/include` for FlashInfer JIT | Any FlashInfer JIT consumer |
| `cold-start-tolerance` | Patches mlnode runner timeout + watcher grace period for slow cold-starts (B300 Kimi-K2.6 INT4 can take 10-20 min) | Any large model / slow cold start |
| `content-type-injector` | Adds the mlnode proxy's Content-Type middleware — S4 form of `patches/0001`, for images that overlay a published mlnode instead of building from our Stage 3 | Any `upstream-overlay` profile |
| `libnvrtc-symlink` | Links `libnvrtc.so` into `/usr/local/lib` so `-lnvrtc` resolves; the CUDA toolkit dir is not on the linker search path and the first JIT link dies without it. Merged upstream as gonka#1560 — a no-op from rc4 on | Any image whose engine JIT-compiles (all of them, first surfaced on Hopper) |
| `sched-req-index-guard` | Skips requests the model runner produced no output for instead of raising `KeyError` and killing EngineCore — S4 form of kaitakuai/vllm#19, needed because release-line images no longer build from our residual tree | Any `upstream-overlay` profile; scheduler is model-agnostic |
| `dsv4-nvfp4-draft-moe` | Keeps DSpark draft (`mtp.*`) experts off the NVFP4 kernel path by consulting the checkpoint's `quantized_layers` map — S4 form of kaitakuai/vllm#20. Without it the NVFP4 draft emits garbage and acceptance collapses to ~1.2 tok/chunk (2.7-5.6x decode loss). | `b300-deepseek-v4-flash-0731` ONLY — NVFP4 is the technically-primary variant on B300 alone |
| `flashinfer-0-6-18-nightly` | Replaces FlashInfer with the 0.6.18 nightly of 2026-08-19 (all three distributions, jit-cache built for cu130). There is no stable 0.6.18 — PyPI and the flashinfer.ai index both stop at 0.6.17, which is what the base ships and what vLLM pins — so the pin is a nightly wheel URL. Must come BEFORE `flashinfer-jit-uninstall` where both are used: the bump installs all three distributions together, and the uninstall then drops the jit-cache so a non-sm_120 GPU JIT-compiles its own. Reversed, the bump puts the prebuilt cache straight back. | GLM-5.3-Flash bring-up, on request |
| `glm53-indexer-init` | Initializes the kpool top-k receiver (`torch.full(-1)`, two sites) and bounds the pool-expand kernel — the two fixes from Crash_Bash_FL's Hopper investigation. Without them the sparse indexer gathers from uninitialized memory: IMAs, or silent zero vectors with nothing in the log. | GLM-5.3-Flash profiles |
| `poc-householder-compile` | Wraps `vllm/poc/gpu_random.py::apply_householder` with `@torch.compile(dynamic=False, fullgraph=True)`. PR #36 measured +10-12% PoC throughput on Qwen3-235B-FP8. Opt-in per profile because legacy production Kimi image deliberately omits it. | Qwen3 + MiniMax-M2.7 profiles; NOT Kimi-K2.6-INT4 (untested, legacy evidence suggests no gain) |

## Adding a new patch

1. Create `tools/hw-patches/<name>.dockerfile`
2. Ensure it's idempotent (re-runnable without error)
3. Reference from one or more profiles via `hw_patches: ["<name>", ...]`
4. Document in this README

## Style guidelines

- Single-purpose: one fragment = one concern
- Idempotent: must work on first apply AND re-apply
- No `ARG`/`ENV` declarations spanning multiple fragments (use the profile's `env:` block for ENV vars)
- Keep RUN steps under 100 lines each for cache granularity

Source: migrated from legacy `kaitakuai/mlnode/tools/fragments/hw-patches/_shared/`.
