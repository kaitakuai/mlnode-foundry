# Wrap vllm/poc/gpu_random.py::apply_householder with
#   @torch.compile(dynamic=False, fullgraph=True)
# Source: product-science/vllm PR #36 (OPEN, target v0.15.1).
#
# WHY OPT-IN (not in Stage 2 base):
#   - Measured +10.2% on 8×B200 + Qwen3-235B-FP8, +12.5% on 8×H100 + Qwen3-235B-FP8
#     (PR #36 A/B with bitwise-equiv verification).
#   - NOT measured on Kimi-K2.6-INT4. Legacy production Kimi image
#     (ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k5-kimi-1)
#     explicitly omits this decorator — strong signal that the gain doesn't
#     replicate (or regresses) for INT4 weight models.
#   - Profiles opt in via their hw_patches list.
#
# Idempotent: detects existing decorator and skips. Anchor is exact;
# if upstream moves the def, the script exits 1 with a clear error.
RUN python3 <<'EOF'
import importlib.util, sys
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    sys.exit("ERROR: vllm not importable in Stage 3 base; mlnode-base missing vLLM?")
path = Path(spec.origin).parent / "poc" / "gpu_random.py"
if not path.exists():
    sys.exit(f"ERROR: {path} missing — vLLM layout changed?")

lines = path.read_text().splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.startswith("def apply_householder("):
        if i > 0 and lines[i - 1].startswith("@torch.compile"):
            print(f"→ {path}: skip (already patched)")
            sys.exit(0)
        lines.insert(i, "@torch.compile(dynamic=False, fullgraph=True)\n")
        path.write_text("".join(lines))
        print(f"→ {path}: patched")
        sys.exit(0)
sys.exit(f"ERROR: `def apply_householder(` not found in {path}; "
         f"vLLM upstream moved? Verify against tools/stage2.lock.cue.")
EOF
