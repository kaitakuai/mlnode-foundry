# B300 + MiniMax-M2.7 (vllm-poc plugin base) — colleague GPU acceptance handoff

> This runbook is the acceptance gate for the **plugin-base** B300 MiniMax-M2.7
> image (`b300-minimax-m2-7` profile, migrated in place from the fat-fork). It migrates the PoC engine from
> the fat-fork monolith (in-tree `vllm/poc/`) to the **vllm-poc** base
> (residual vLLM 0.23 + out-of-tree `gonka-poc` package; ADR-0013). Because the
> 0.20.0 → 0.23.0 base bump and the plugin entrypoint/worker-extension wiring
> have **never been run on a real B300**, every throughput/quality number in
> the profile is INHERITED from the fat-fork 1×B300 validation (2026-05-23) and
> is unverified here. The colleague with B300 access MUST walk all 8 gates
> below before this image is treated as production-validated.

## Image under test

```
ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.23.0-k1
```

- This is the **Stage 4** (colleague) image rendered from the
  `b300-minimax-m2-7` profile. The exact digest is produced by foundry
  CI when PR-C merges; the operator hand-off message MUST carry the digest, not
  just the tag (the tag is mutable until the build lands).
- Rollback baseline (do NOT delete): the fat-fork image
  `ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.20.0-k1`. Tags do
  not collide — the vllm version axis (`0.20.0` vs `0.23.0`) distinguishes them.

## Prerequisites (must be true before any gate runs)

1. **Real S2 digest is locked.** `tools/stage3.lock.cue` `stage2.digest` is a
   placeholder (`sha256:` + 64 zeros). The image cannot actually build until
   the real `ghcr.io/kaitakuai/vllm-poc@sha256:...` digest replaces it. See the
   USER-REQUIRED list — gates 1-8 are unreachable until the build chain
   completes with a real digest.
2. **gonka-poc package contract holds** (assumed from ADR-0013, NOT verified
   against a published wheel here):
   - composed entrypoint module: `gonka_poc.entrypoint.api_router`
   - worker extension class: `gonka_poc.worker.PoCWorkerExtension`
   - console script: `gonka-vllm-serve = gonka_poc.entrypoint.api_router:main`
   - plugin entry point: `vllm.general_plugins -> gonka_poc.plugin:register`
   If any import path moved, the runner-patch marker check fails loud but it
   CANNOT validate the `gonka_poc` symbol names — confirm them at build time.
3. **One B300 SXM6** (275 GiB HBM) with a working NVIDIA driver. MiniMax-M2.7
   FP8 (~230 GB) at `--max-model-len 180000` fills the GPU to ~91%.

## Mandatory environment

These are baked into the image (S2 base + the plugin profile) but verify they
are present in the running container — a missing one silently breaks PoC:

| Env var | Value | Why |
| --- | --- | --- |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | `collective_rpc` msgpack channel carries PoC artifacts between API and worker. Without it PoC forwards fail at serialization. |
| `MLNODE_VLLM_MODULE` | `gonka_poc.entrypoint.api_router` | Server-side plugin flip: the mlnode runner launches the gonka-poc composed entrypoint (stock `build_app` + PoC router + gating middleware) instead of `vllm.entrypoints.openai.api_server`. |
| `VLLM_USE_V1` | `1` | V1 engine is required for the worker-extension / `collective_rpc` plugin path. Inherited from the B300 base; do NOT drop it. |
| `VLLM_RUNNER_TIMEOUT` | `3600` | MiniMax-M2.7 FP8 + 180k context cold start is slow; short timeouts kill the runner mid-load. |
| `WATCHER_GRACE_FIRST_HEALTHY` | `1` | Same slow-cold-start tolerance for the health watcher. |

The runner-patch (`tools/runner-patches/b300-minimax-plugin.py`) additionally
**forces** these engine args (operator / chain broadcast cannot drop them):
`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`,
`--logprobs-mode processed_logprobs`, `--attention-backend FLASHINFER`. It does
**not** force `--enforce-eager`: PoC eager is handled inside gonka-poc
(`poc_model_runner` `skip_compiled=True`), so inference keeps CUDA graphs.

Chain-governance MiniMax args (`--max-model-len 180000`, `--kv-cache-dtype fp8`,
`--tool-call-parser minimax_m2`, `--reasoning-parser minimax_m2_append_think`,
`--enable-auto-tool-choice`) are NOT in the image — they flow from the
network-node DAPI broadcast into mlnode `runner.py` `self.additional_args` at
runtime. Confirmed (Q1=YES): the runner assembles `additional_args` identically
regardless of which entrypoint module it launches, so they reach the composed
gonka-poc engine unchanged after the flip.

## How to launch

### Option A — production path (DAPI-managed, preferred)

On a fleet-attached node the DAPI owns the inference lifecycle. Do **not**
manually `up`/`inference` a fleet-attached node (DAPI owns it). For acceptance
testing use a node that is NOT yet fleet-attached, or coordinate with the
network-node owner to point a DAPI at this image and let it issue
`inference-up`. The DAPI broadcasts the governance args listed above.

### Option B — standalone container (acceptance testing only)

Run the mlnode container directly so the gates can be exercised without the
fleet. The mlnode runner inside the container reads `MLNODE_VLLM_MODULE` and
launches the composed entrypoint; you supply the governance args as the
runner's `additional_args` (mimicking the DAPI broadcast):

```bash
docker run --rm --gpus all \
  --name b300-minimax-plugin-accept \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router \
  -e VLLM_USE_V1=1 \
  -e VLLM_RUNNER_TIMEOUT=3600 \
  -e WATCHER_GRACE_FIRST_HEALTHY=1 \
  -p 8080:8080 \
  ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.23.0-k1
```

Then drive the mlnode API to start inference with the governance args (the
DAPI normally sends this). Exact API call depends on the mlnode build under
test; the governance arg set that MUST land in `additional_args` is:

```
--max-model-len 180000 --kv-cache-dtype fp8 \
--tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think \
--enable-auto-tool-choice
```

(The runner-patch appends the four forced plugin args on top of these.)

---

## Acceptance gates (run in order; STOP on first hard fail)

### Gate 1 — image pulls + container smoke

- [ ] `docker pull ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.23.0-k1@sha256:<digest>` succeeds.
- [ ] Plugin import graph + console script resolve **without a GPU**:
  ```bash
  docker run --rm --entrypoint bash <image> -c \
    'gonka-vllm-serve --help >/dev/null && python -c "import vllm, gonka_poc; print(vllm.__version__)"'
  ```
  Expected: prints `0.23.0...` (the residual `+gonka.sampler1` wheel), exit 0,
  no `ModuleNotFoundError`, no plugin registration traceback.
- [ ] `python -c "import vllm; print(vllm.__version__)"` shows the **patched**
  residual wheel (e.g. `0.23.0+gonka.sampler1`), NOT a vanilla `0.23.0`. A
  vanilla version means `--no-deps` was defeated and the sampler stack
  reverted — hard fail, rebuild S2.

### Gate 2 — launch-swap took effect

- [ ] After launch, the backend serving process is
  `python -m gonka_poc.entrypoint.api_router ...`, NOT
  `python -m vllm.entrypoints.openai.api_server ...`:
  ```bash
  docker exec b300-minimax-plugin-accept ps -ef | grep -E 'gonka_poc|api_server'
  ```
  Expected: a `gonka_poc.entrypoint.api_router` process; **no**
  `vllm.entrypoints.openai.api_server`. If you see `api_server`,
  `MLNODE_VLLM_MODULE` did not reach the runner (PR-A regression or env not
  propagated) — hard fail.
- [ ] The forced engine args are present in the launched command line:
  `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`,
  `--logprobs-mode processed_logprobs`, `--attention-backend FLASHINFER`.
  (`--enforce-eager` is intentionally ABSENT — see the eager note below.)

### Gate 3 — PoC router mounted

- [ ] The PoC status endpoint is served by the composed entrypoint:
  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/api/v1/pow/status
  ```
  Expected: `200` with body status `IDLE` (engine up, no PoC running yet). A
  `404` means the router was not mounted — the stock `api_server` is running,
  not the composed entrypoint — hard fail (re-check Gate 2). Adjust the host/
  port to whatever the mlnode exposes the engine on.

### Gate 4 — PoC forward = first real `collective_rpc` on hardware

- [ ] Trigger a PoC init + generate (via the DAPI broadcast, or the mlnode PoC
  API). Expected: returns artifacts, HTTP 200.
- [ ] **No** `AttributeError` on `execute_poc_forward` in the worker logs (that
  means `--worker-extension-cls` did not register the worker method — Gate 2
  arg missing or wrong class path).
- [ ] **No** 500 from the PoC route, **no** msgpack/serialization error (that
  means `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is missing).
- [ ] This is the FIRST time the `gonka_poc.worker.PoCWorkerExtension`
  `collective_rpc` path runs on a B300 — watch logs closely; any traceback
  here is a real plugin-wiring bug, not a config typo.

### Gate 5 — cross-validate vs 2×B200 baseline (MiniMax chain gate)

- [ ] Run the PoC nonce comparison against the established 2×B200 baseline
  under the MiniMax chain gate thresholds:
  - mean L2 ≤ **0.75**
  - mismatch fraction ≤ **0.10**
  - verdict **PASS**
- [ ] Record the measured `mean L2` and `mismatch`. The fat-fork base scored
  mean L2 0.266 on this hardware; a 0.23-plugin score materially worse than
  that (even if still under 0.75) is a yellow flag worth escalating before
  fleet rollout, because eager bit-compat should be preserved across the base
  bump.
- [ ] If mean L2 > 0.75 or mismatch > 0.10: hard fail. Most likely cause is the
  PoC forward not running eager (gonka-poc's `skip_compiled` path broken →
  compiled drift) or a logprobs-mode mismatch — re-check Gate 2 args.

### Gate 6 — inference serves

- [ ] A standard chat completion against the served model returns a coherent
  response:
  ```bash
  curl -s http://localhost:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"<served-model-id>","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}'
  ```
  Expected: HTTP 200 with a non-empty `choices[0].message.content`. Tool-call /
  reasoning parsers (`minimax_m2`, `minimax_m2_append_think`) load without
  error.

### Gate 7 — gating returns 503 while PoC active

- [ ] While a PoC run is active, inference requests are gated:
  ```bash
  # during an active PoC forward
  curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/v1/chat/completions -d '{...}'
  ```
  Expected: `503` (gating middleware rejects inference while PoC holds the
  engine). After PoC completes, the same request returns `200` again
  (back to Gate 6 behavior). A `200` during active PoC means the gating
  middleware from the composed entrypoint is not engaged — hard fail.

### Gate 8 — SIGTERM clean teardown

- [ ] `docker stop` (SIGTERM) the container; the engine + worker shut down
  cleanly within the stop grace period — no orphaned `gonka_poc` /`vllm`
  processes, no GPU memory left allocated (`nvidia-smi` shows the GPU freed),
  no Python traceback on shutdown.
- [ ] Restart once and confirm it comes back up healthy (no stale lock / shared
  memory segment left behind by the previous run).

---

## Known gotchas

- **Placeholder S2 digest blocks everything.** Until
  `tools/stage3.lock.cue` `stage2.digest` is a real
  `ghcr.io/kaitakuai/vllm-poc@sha256:...`, the image will not build and no gate
  can run. There is an inline `TODO(user)` at that line.
- **Shared lock repoint side effect.** Bumping `stage3.lock.cue` `stage3.tag`
  to `0.2.13-vllm0.23.0-k1` repoints the BASE_IMAGE for ALL kaitakuai-base
  profiles (the lock is a single shared file). The OLD fat-fork b300 profile
  and the 5 others now resolve to `mlnode-base:0.2.13-vllm0.23.0-k1`. The OLD
  profile is retained only as a rollback baseline; its own
  `identity.version.vllm` stays `0.20.0`.
- **CUDA 13.0 (vLLM's recommended default).** The residual S1 bases on
  `vllm/vllm-openai:v0.23.0` (bare tag → **CUDA 13.0.2**), not the pinned
  `-cu129` (12.9). This matches the 5 fat-fork 0.20 profiles (also CUDA 13.0)
  and the previously validated 0.20 B300 image, so the shared `stage2.cuda`
  field is accurate for every profile. **This image's actual CUDA is 13.0.2.**
- **householder-compile is intentionally GONE on this base.** The fat-fork's
  `poc-householder-compile` hw-patch edited the monolith's
  `vllm/poc/gpu_random.py`, which does not exist on the plugin base (PoC math
  lives in the gonka-poc package). Any equivalent compile wrap is now a
  gonka-poc-internal concern, not a foundry hw-patch. Its perf delta is
  UNMEASURED on the plugin base — do not expect the fat-fork +10-12% on this
  image until gonka-poc reintroduces an equivalent and it is re-benchmarked.
- **PoC must run eager — but via `skip_compiled`, not global `--enforce-eager`.**
  The PoC forward is eager via gonka-poc's `poc_model_runner`
  (`skip_compiled=True`), the same mechanism the fat-fork used — so the
  runner-patch deliberately does **not** force global `--enforce-eager` (that
  would also drop CUDA graphs for ordinary inference, a throughput regression
  for no PoC benefit; the fat-fork b300 image ran with no `--enforce-eager` and
  still produced L2-valid nonces). Compiled drift in the PoC forward still fails
  the cross-validator L2 gate → epoch exclusion; if Gate 5 regresses, the
  suspect is a broken `skip_compiled` path in gonka-poc, not a missing
  `--enforce-eager`.
- **Do not benchmark on a fleet-attached node by hand.** The DAPI owns the
  inference lifecycle; manual `up`/`inference` on an attached node conflicts
  with it. Use a detached node for acceptance, or coordinate with the
  network-node owner.
- **`batch_size=64` hangs the PoC engine** on every MiniMax profile (OOM-stuck,
  engine never recovers) — same failure mode across A100/H100/H200/B200/B300.
  Keep operator-supplied PoC batch overrides at or below 32 on B300.
- **Inherited numbers, not measured.** 1792 nonces/min @ batch=32 and mean L2
  0.266 are from the fat-fork 0.20.0 validation. They are placeholders in the
  profile carrying a `warning` tuning_note until Gate 5 + a fresh throughput
  sweep are recorded in a new `kaitakuai/experiments` report (see
  `add-validation-report.md`).

## After acceptance passes

- Write the hardware-validation report in `kaitakuai/experiments` and link it
  per `docs/runbooks/add-validation-report.md` (prepend the `validation-report`
  tuning_note, patch `registry-view/<package>-<tag>.json`). Until then this
  image shows the amber **in progress** chip on the public dashboard.
- Record the measured 8-GPU-normalized `nonces` figure and the cross-validation
  `mean L2` / `mismatch` in that report.
