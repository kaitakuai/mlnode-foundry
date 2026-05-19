# `tools/hw-patches/` — hardware-specific Dockerfile fragments

Each `*.dockerfile` is a fragment applied to the Stage 3 image build via the profile's `hw_patches:` list. Fragments are **idempotent** (re-applying produces same result).

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

The Stage 3 build resolves each name to `tools/hw-patches/<name>.dockerfile`, inlines via Docker build context, and applies in order.

## Inventory

| Patch | What it does | Applies to |
|-------|--------------|------------|
| `triton-ptxas-from-system-cuda` | Overwrites Triton's bundled `ptxas` (lacks newer SM targets) with system CUDA's | sm_103a (B300), sm_120 (RTX PRO 6000), any newer Blackwell |
| `flashinfer-jit-uninstall` | Removes pre-compiled FlashInfer JIT cache (ships sm_120 only) — forces JIT compile-on-first-launch | sm_103a, any non-sm_120 Blackwell |
| `libcuda-compat-580-driver` | Replaces CUDA compat stub libcuda.so with symlink to real driver — fixes broken GPU detection with NVIDIA driver 580+ | Any B300/Blackwell host with 580+ driver |
| `nvidia-headers-symlinks` | Symlinks all CUDA dev headers from nvidia-* pip packages into `/usr/local/cuda/include` for FlashInfer JIT | Any FlashInfer JIT consumer |
| `cold-start-tolerance` | Patches mlnode runner timeout + watcher grace period for slow cold-starts (B300 Kimi-K2.6 INT4 can take 10-20 min) | Any large model / slow cold start |
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
