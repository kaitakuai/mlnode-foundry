# Initialize the kpool top-k receiver and bound the pool-expand kernel — the
# two fixes from Crash_Bash_FL's 2026-08-29 Hopper investigation, applied at
# image level so nobody re-applies patch_v7.py by hand.
#
# The top-k kernels only fill the first min(k, valid) elements, so a
# torch.empty receiver leaves garbage block ids in rows with fewer valid
# pools; attention then gathers from uninitialized memory (IMAs, or worse,
# silent zero vectors with nothing in the log). Two call sites, not one.
# The Triton pool-expand kernel additionally kept out-of-range ids that a
# full receiver would have masked: bound them too.
#
# Verified on 4xH200: with these two edits plus --no-enable-flashinfer-autotune
# the fp8 path mined 101k+ nonces in one run, every checked vector non-zero.
RUN python3 - <<'PYEOF'
import re, pathlib, sys

SITE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")

f = SITE / "model_executor/layers/sparse_attn_indexer_kpool.py"
s = f.read_text()
if "pool_topk = torch.full(" in s:
    print("glm53-indexer-init: already patched; no-op")
else:
    n = s.count("pool_topk = torch.empty(")
    if n != 2:
        sys.exit(f"glm53-indexer-init: expected 2 torch.empty sites, found {n} - re-verify")
    s = s.replace("pool_topk = torch.empty(\n", "pool_topk = torch.full(\n")
    s = re.sub(
        r"(pool_topk = torch\.full\(\s*\n\s*\(num_rows, select_k\),)(\s*dtype=torch\.int32)",
        r"\1 -1,\2", s)
    f.write_text(s)
    print(f"glm53-indexer-init: patched {n} top-k receivers")

f2 = SITE / "models/glm5next/nvidia/ops/kpool_compress.py"
s2 = f2.read_text()
old = "hist_out = tl.where(pid >= 0, hist_val, -1)"
new = "hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)"
if new in s2:
    print("glm53-indexer-init: kpool bound already present; no-op")
elif old not in s2:
    sys.exit("glm53-indexer-init: kpool anchor not found - re-verify")
else:
    f2.write_text(s2.replace(old, new))
    print("glm53-indexer-init: kpool expand bounded")
PYEOF
