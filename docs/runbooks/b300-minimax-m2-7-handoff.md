# Handoff — MiniMax-M2.7 + B300 (vLLM 0.20 fat-fork) image

> **Status:** colleague-handoff draft. The image is **already built, pushed,
> and cosign-signed** on GHCR — nothing needs to be rebuilt to hand this over.
> Built by `build-stage3.yml` CI run `26410844148` on 2026-05-25.

## TL;DR

Pull this exact digest (multi-arch index — Docker resolves the right platform child):

```
ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.20.0-k1
  @ sha256:fa3b66471f85237ef99c1bc5f8d729ca53b76ae014d0b298210ec62575b807e1
```

- **What:** fat-fork PoC (PoC baked into the vLLM fork in-tree), **TP=1, single B300**.
- **Throughput:** **1792 PoC nonces/min @ batch=32** — highest per-GPU of any hardware we've measured.
- **No plugin wiring:** no `MLNODE_VLLM_MODULE`, no `--worker-extension-cls`, no `pip install gonka-poc`.
- **Starts out-of-the-box on Blackwell** — no MoE env override needed (unlike the A100 sibling).

> ⚠️ **Digest gotcha.** The 2026-05-23 validation report bakes the digest
> `sha256:45e14a91…`. That is the **superseded initial build** (now untagged).
> The tag was re-pointed to `sha256:fa3b664…` by the 2026-05-25 rebuild
> (PRs #57/#58). **Pin `fa3b664…`, not `45e14a91…`.**

---

## 1. What this image is

Three-stage foundry chain, all on GHCR:

| Stage | Image | Notes |
|-------|-------|-------|
| S1 (vLLM PoC fork) | `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` @ `sha256:7955b846…` | CUDA 13.0; built 2026-05-19 from `kaitakuai/vllm@mb/feat/port-pocv2-vllm-0.20` (HEAD `ccbe7cd8d`). PoC math (`vllm/poc/`) lives **in-tree** in this fork. |
| S2 (mlnode-base) | `ghcr.io/kaitakuai/mlnode-base:0.2.13-vllm0.20.0-k1` | mlnode from `gonka-ai/gonka@f3b3893` (0.2.13) + `0001-content-type-middleware.patch`. Bakes the required env. |
| S3 (this image) | `ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.20.0-k1` @ `sha256:fa3b664…` | b300 + minimax profile applied; cosign-signed; `build_smoke=pass`. |

This is the **fat-fork** lineage (ADR-0013 "Option A: fork + monkey-patch",
ADR-0014 "residual fork = permanent infra"). It is **not** the in-progress
0.23 `gonka-poc` plugin — that path (`--worker-extension-cls`,
`pip install gonka-poc`) would *replace* this fork but has **not** here.

OCI labels on the image (verifiable with `docker inspect`):
`gonka.kaitaku.gpu=b300`, `model=minimax`, `model_revision=m2-7`,
`gonka.vllm.version=0.20.0`, `gonka.mlnode.version=0.2.13`,
`gonka.kaitaku.stage=3`, `profile_hash=2ee9baf1…`.

---

## 2. Launch mechanism (fat-fork, not plugin)

- **ENTRYPOINT** = `/app/entrypoint.sh` (creates `appuser`, symlinks
  `libcuda.so`, activates `/app/packages/api/.venv`, then `exec "$@"`).
- **CMD** = none → the **mlnode api server command is supplied by the
  caller** (DAPI/fleet supplies it automatically; for a standalone box you
  pass it yourself — see §6).
- The mlnode api server (`api.app:app`) mounts `pow_router` + `pow_v2_router`
  under `/api/v1` **in-process**. vLLM is launched in-process by
  `InferenceManager` on `POST /api/v1/inference/up`; the fork's
  `AsyncLLM.poc_request` makes PoC calls reach the compute path **without any
  plugin extension class**.

**Net:** no `MLNODE_VLLM_MODULE`, no `--worker-extension-cls`, no pip plugin.
PoC ships inside the S1 fork.

---

## 3. Baked env (do **not** re-pass, do **not** unset)

Confirmed directly from the image config:

| Env | Value | Why |
|-----|-------|-----|
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | **Required** — the fork serializes the synthetic `inputs_embeds` tensor over the vLLM RPC boundary; 0.20 refuses it otherwise. |
| `VLLM_USE_V1` | `1` | PoC-v2 path is V1-engine only. |
| `VLLM_RUNNER_TIMEOUT` | `3600` | Tolerates the ~4 min ptxas cold-JIT so the watcher doesn't kill a still-warming engine. |
| `WATCHER_GRACE_FIRST_HEALTHY` | `1` | Suppresses 3-strike crash-kill until first `healthy=True` (MiniMax FP8-KV + 180k cold start is slow). |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `1` | **Enabled — correct for Blackwell** (native FP8). *(A100 is the exception where we override this to `0`; do NOT do that here.)* |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | FlashInfer MoE FP4 enabled. |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | — |

**Not needed for MiniMax:**
- `HF_HUB_OFFLINE` — Kimi-K2.6-only workaround (buggy `tokenization_kimi_fast.py`).
  MiniMax uses a normal tokenizer. Set `=1` **only** if you pre-stage weights
  and want to forbid hub calls.
- No MoE backend override — B300 (sm_100) auto-selects `FLASHINFER_TRTLLM`.
- `HF_TOKEN` — only if the HF cache is empty and the pull 401s
  (MiniMax-M2.7 is Apache-2.0, usually unauthenticated).

---

## 4. Serve config (chain-mandated — supplied by the runner, **not** baked CLI args)

These are the resolved `runtime_defaults` from the profile; DAPI/the mlnode
runner supplies them to vLLM at `/inference/up` time (they are **not** baked
as CLI args into the image):

| Arg | Value |
|-----|-------|
| `tensor_parallel_size` | `1` (single B300) |
| `max_model_len` | `180000` — chain-mandated. ⚠️ KV-pool fit validated (221,616 tokens), but generation quality at full 180k context was **not** specifically swept (`tuning_note: warning`). Experiments ran at 131072. |
| `kv_cache_dtype` | `fp8` |
| `gpu_memory_utilization` | `0.92` |
| `max_num_seqs` | `128` |
| `logprobs_mode` | `processed_logprobs` |
| `enable_auto_tool_choice` | `true` |
| `tool_call_parser` | `minimax_m2` |
| `reasoning_parser` | `minimax_m2_append_think` |
| `trust_remote_code` | `true` |
| `attention_backend` | `FLASHINFER` (pinned; auto-selects `FLASHINFER_TRTLLM` MoE on Blackwell) |
| model | `MiniMaxAI/MiniMax-M2.7`, `hf_revision d494266a4affc0d2995ba1fa35c8481cbd84294b` |

---

## 5. Cold-start expectations — "wait it out, not a hang"

- Weights load ~29 s.
- **~4 min one-time ptxas JIT** on first kernel. **Do not kill it.**
- Total to first-healthy ~567 s.
- KV pool lands at **221,616 tokens** (1.23× concurrency @ 180k).
- GPU ~91 % full (~249 / 275 GiB). MiniMax-M2.7 FP8 = ~230 GB weights.

---

## 6. How to run

### Primary path — attach to the fleet (recommended)

This image is designed to be driven by the gonka network node / DAPI
lifecycle, which supplies the server command and drives `/inference/up` with
the chain-governed args from §4. **On a fleet-attached node, let DAPI own the
lifecycle — do not manually `POST /inference/up`.**

### Standalone path — isolated GPU test box (manual bring-up)

Only for a box that is **not** fleet-attached. Pin by digest:

```bash
IMG=ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7@sha256:fa3b66471f85237ef99c1bc5f8d729ca53b76ae014d0b298210ec62575b807e1

docker run -d --name mlnode-b300-minimax \
  --gpus '"device=0"' \
  --ipc=host \
  -p 8080:8080 \
  -v /data/hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  "$IMG" \
  <mlnode-api-server-start-command>
# ENTRYPOINT=/app/entrypoint.sh wraps the command (CMD is empty by design).
# The api server start command + listen port are owned by the mlnode api
# package — confirm them against your mlnode 0.2.13 rev (or copy how DAPI
# starts the container on the fleet). Port 8080 below is illustrative.
# VLLM_ALLOW_INSECURE_SERIALIZATION / VLLM_USE_V1 / RUNNER_TIMEOUT /
# WATCHER_GRACE / MoE-FP8 are baked — do not re-pass or unset them.
```

Then load the model (this launches vLLM; the fat-fork carries PoC):

```bash
curl -s -X POST localhost:8080/api/v1/inference/up \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.7",
    "hf_revision": "d494266a4affc0d2995ba1fa35c8481cbd84294b",
    "args": {
      "tensor_parallel_size": 1,
      "max_model_len": 180000,
      "kv_cache_dtype": "fp8",
      "gpu_memory_utilization": 0.92,
      "max_num_seqs": 128,
      "logprobs_mode": "processed_logprobs",
      "enable_auto_tool_choice": true,
      "tool_call_parser": "minimax_m2",
      "reasoning_parser": "minimax_m2_append_think",
      "trust_remote_code": true,
      "attention_backend": "FLASHINFER"
    }
  }'
```

> The exact `/inference/up` request schema is owned by mlnode
> (`api.inference.routes`); if a field name differs in this rev, drive
> bring-up via DAPI or check the route's pydantic model. The body above is
> illustrative.

---

## 7. Acceptance gates (run in order)

| # | Gate | Pass condition |
|---|------|----------------|
| 0 | **Image identity** | `docker inspect` shows digest `sha256:fa3b664…`; labels `gonka.kaitaku.gpu=b300`, `model=minimax`, `gonka.vllm.version=0.20.0`. Confirms fat-fork base. |
| 1 | **Container starts** | `docker ps` healthy; logs show `entrypoint.sh` activated the api venv; `docker exec … env` shows `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. |
| 2 | **`/health` 200** | `curl -fsS localhost:8080/health` → 200 **before** model load. PoC routes registered (POST `/api/v1/…/pow/…` returns 405/422, **not** 404). |
| 3 | **Model loads** | `POST /inference/up`; poll until `healthy=True`. Logs: weights ~29 s, MoE = `FLASHINFER_TRTLLM Fp8` (auto), ptxas JIT ~4 min (one-time — don't kill), KV ≈ 221,616 tokens, GPU ~249/275 GiB. **No** `seq_lens_cpu is not None` assert, **no** `per_token_group_quant` crash. |
| 4 | **Single PoC nonce + self-validate** | One PoC-v2 generate (batch=1) returns a nonce with logprobs. Self-consistency: re-run same seed → ≈ identical (L2 ≈ 0). **Cross-hardware gate = MiniMax chain gate `mean L2 < 0.75`, `≤10%` mismatch** (measured **0.266 / 0.9 % → PASS**). ⚠️ Do **not** use an L2<0.2 bar — TP=1-vs-B200-TP=2 reduction order gives ~0.27 by design; it is **not** bit-identical and passes the *chain* gate, not 0.2. |
| 5 | **Chat inference** | Short prompt → coherent output, `finish_reason=stop`, 0 failures. |
| 6 | **Throughput sanity** | PoC sweep @ batch=32 → **~1792 nonces/min** (≥~1700 ok; batch=8→~1392, 16→~1728). **HARD GUARD: never batch=64 — the PoC engine hangs (OOM-stuck, not a crash; never recovers).** Inference (60 req / 20 concurrent): TTFT ~1.28 s, ~746 output tok/s, 0 failures. |

---

## 8. Baseline numbers to compare against

Source: [`kaitakuai/experiments/2026-05/minimax_m27_1xb300_sxm6`](https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md)
(1×B300 SXM6, 275 GiB HBM, driver 580.126.09, sm_100, 2026-05-23).

- **PoC throughput:** batch 8→1392, 16→1728, **32→1792 (best)**, 64→**HANG**. PoC weight `1792 × 0.3024 = 542`/GPU.
- **Per-GPU ranking:** B300 1792 > B200 1312 > H200 864 > H100 576 > A100 224.
- **Quality / L2:** mean L2 **0.2655**, 9/1000 mismatch (0.9 %) @ thr 0.75 → **PASS** (p=1.0) under the MiniMax chain gate. Not bit-identical to B200 (TP=1 vs TP=2) — by design.
- **Inference (compressa-perf, 60 req / 20 conc):** TTFT **1.28 s** (fastest of all hw), latency 7.92 s, TPOT 26.6 ms/tok, total 2104 tok/s, **output 746 tok/s**, RPS 2.51, **0/60 failed**.
- **Fit:** 230 GB weights on 275 GiB at TP=1, 1.23× concurrency @ 180k, KV 221,616 tokens, ~91 % GPU.

---

## 9. Gotchas / flags

- **Digest:** hand over `fa3b664…` (current, signed). `45e14a91…` in the report is the superseded 2026-05-23 build (now untagged).
- **L2 gate:** chain gate `0.75 / ≤10%`, **not** 0.2. B300 TP=1 ≈ 0.27 vs B200, passes by design.
- **batch=64 hangs** the PoC engine — hard guard.
- **`max_model_len=180000`** is chain-mandated but quality at full 180k context not swept (warning); KV fit is validated.
- **832 ≠ 1792:** 832 nonces/min was 1×B300 **Qwen3-235B**; this image is **MiniMax-M2.7** at **1792**. Different model — don't conflate. Topology here is **TP=1 single GPU**, not PP.
- **ptxas ~4 min JIT** on first kernel is normal, not a hang.

---

## 10. Promotion state (be honest with the colleague)

The foundry state file
(`state/mlnode-b300-minimax-m2-7-0.2.13-vllm0.20.0-k1.json`) records this
image as **`status=draft`**: it is **built, pushed, cosign-signed, and
`build_smoke=pass`**, but it has **not** been run through the `promote.yml`
workflow. The B300 hardware validation (L2 0.266, throughput 1792) is
documented in the profile `tuning_notes` and the experiments report, but the
state is still `draft` pending an explicit promote. "Published artifact
exists" = yes; "promoted/blessed for fleet" = not yet.

---

## 11. References

- Experiments report: `kaitakuai/experiments/2026-05/minimax_m27_1xb300_sxm6/README.md`
- Profile: `mlnode-foundry/profiles/b300-minimax-m2-7.cue` (+ `bases/b300.cue`, `bases/minimax_m2_7.cue`)
- Version pins: `mlnode-foundry/tools/stage2.lock.cue`
- Build workflow: `mlnode-foundry/.github/workflows/build-stage3.yml` (run `26410844148`, 2026-05-25)
- Architecture: ADR-0013 (PoC integration), ADR-0014 (residual fork = permanent infra)
