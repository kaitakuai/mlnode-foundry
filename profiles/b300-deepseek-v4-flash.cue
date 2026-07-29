// Profile: B300 Blackwell Ultra (×1) + DeepSeek-V4-Flash FP8 — vllm-poc PLUGIN,
// vLLM 0.25.1, UPSTREAM-OVERLAY mode.
//
// FIRST DeepSeek-V4 leaf and FIRST overlay-mode profile. It builds FROM an
// explicit mlnode-base digest (base{} below) instead of tools/stage3.lock.cue,
// so the whole 0.23 fleet keeps its shared stage3 lock untouched while this
// leaf rides a one-off mlnode-base-0.25.1 (built from the 0.25.1 PoC chain:
// S1 residual poc-sampler-residual-v0.25 -> S2 vllm-poc:0.25.1 -> this base).
// vLLM 0.25.1 has FIRST-CLASS DeepSeek-V4 support.
//
// Deployment: 149 GiB FP8 weights fit on ONE B300 (275 GiB) at TP=1 with ~126
// GiB headroom; an 8-GPU box auto-runs 8 independent TP=1 engines. gonka-poc#14
// validated 1×B300 TP1 ~928 nonces/min @ bs=16. Config is a TEST bring-up
// (max-model-len capped, no governance mandate yet — #1408 is still discussion).
//
// Consensus-safety choices baked here and in the runner-patch:
//   - --kv-cache-dtype fp8 (FlashMLA fp8_ds_mla; assert otherwise).
//   - NO --attention-backend pin (default FlashMLA-DSV4 is deterministic;
//     FlashInfer-DSV4 has placeholder FP8 scales — cross-hw L2 hazard).
//   - NO forced compilation (V4 auto-forces NONE; PoC eager via skip_compiled).
//   - NO backend/DeepGEMM env (native ue8m0 scales; MoE oracle auto-selects).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b300_deepseek_v4_flash: #OverlayProfile & bases.B300 & bases.DEEPSEEK_V4_FLASH & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "deepseek"
			model_revision: "v4-flash"
			// quant axis omitted — DeepSeek-V4-Flash ships only as this FP8
			// (MXFP4 experts + FP8-block linears) checkpoint today.
		}
		version: {
			// Overlay identity: the "upstream" here is OUR 0.25.1 mlnode-base, not a
			// product-science release. rev=7 = the first stack built entirely from
			// upstream sources: gonka-ai/vllm release/v0.25.1 (port merged as #78) +
			// gonka-ai/gonka-vllm-plugins v0.1.1 (abort-by-internal-id fix, 0.23
			// dropped) + mlnode 0.2.14 with the node-side metrics exporter.
			// Tag renders as ...:0.2.14-vllm0.25.1-overlay-k7.
			upstream: "0.2.14-vllm0.25.1"
			rev:      7
		}
	}
	mode: "upstream-overlay"
	base: {
		image: "ghcr.io/kaitakuai/mlnode-base"
		// mlnode-base:0.2.14-vllm0.25.1-k3, built via build-stage3 workflow_dispatch
		// (run 30476136950) from S2 vllm-poc:0.25.1 @sha256:5453a9b0... -- the first
		// S2 whose both inputs are the upstream sources: gonka-ai/vllm
		// release/v0.25.1 (#78 merged) + gonka-ai/gonka-vllm-plugins v0.1.1
		// (469aaf2ae) -- + mlnode 0.2.14 with the node-side metrics exporter
		// (#1501, kaitakuai/gonka@26f7db5) + patches/0001+0002+0003. Monitoring-enabled
		// base for hardware testing; tools/stage3.lock.cue on main NOT changed (the
		// fleet flip to 0.25.1 is a separate migration).
		digest:           "sha256:86ecf8ac3565433c2a3b2158ddeec652bc1591fdfdc197f447cf6d1a5ef1e268"
		upstream_version: "0.2.14-vllm0.25.1"
	}
	// hw_patches inherited from bases.B300 (triton-ptxas, flashinfer-jit-uninstall,
	// libcuda-compat-580-driver, nvidia-headers-symlinks, cold-start-tolerance) —
	// the slow 149 GiB FP8 load + MLA/sparse warmup needs cold-start-tolerance.
	runner_patch: "b300-deepseek-v4-flash-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// VLLM_RUNNER_TIMEOUT / WATCHER_GRACE_FIRST_HEALTHY come from the
		// DEEPSEEK_V4_FLASH + B300 bases; VLLM_USE_V1 from B300.
	}
	runtime_defaults: {
		// Forced via the runner-patch; reproduced here for the dashboard.
		tensor_parallel_size:   1
		gpu_memory_utilization: 0.90
		max_model_len:          200000
		max_num_batched_tokens: 16384
		logprobs_mode:          "processed_logprobs"
		trust_remote_code:      true
		// kv_cache_dtype fp8 + compilation_mode 0 / cudagraph NONE come from the
		// DEEPSEEK_V4_FLASH base. NO attention_backend, NO expert_parallel,
		// NO tool/reasoning parser (governance-level, undefined for V4 yet).
	}
	description: "B300 Blackwell Ultra SXM6 (×1, 8 engines/box) + DeepSeek-V4-Flash FP8 (TP=1, FlashMLA fp8 KV, eager PoC) — vllm-poc 0.25.1 PLUGIN, overlay-mode TEST image"
	notes: """
		FIRST DeepSeek-V4-Flash image + first overlay-mode leaf, on the vLLM 0.25.1
		PoC chain. TEST bring-up for the experiment programme (.work/deepseek-v4-flash/
		EXPERIMENTS.md) — NOT production, no chain governance model yet (#1408).
		k4 = FULL stack for hardware validation: 9 sampler-residual rows +
		gonka-poc#14 (V4 per-group PoC) + gonka-poc#18 (borrowed-lease
		validation, probe-gated — V4 is fp8-KV/scratch-free so the feature
		self-enables; experiment A7 covers it) + mlnode patches/0003
		(heartbeat liveness + damped proxy probe, gonka-ai/gonka#1421).

		Config: TP=1 (149 GiB FP8 on one B300; box = 8 TP=1 engines), gmu 0.90,
		max-model-len 200000 (TEST cap; V4 native is 1,048,576 — real cap is a
		governance param via DAPI, raise per experiment B4), max-num-batched-tokens
		16384, kv-cache-dtype fp8 (FlashMLA fp8_ds_mla — mandatory), processed_logprobs.

		Overlay-mode: builds FROM base{} mlnode-base-0.2.13-vllm0.25.1 digest, NOT
		tools/stage3.lock.cue — the 0.23 fleet's shared lock is untouched. The base
		is the one-off 0.25.1 mlnode-base (S1 poc-sampler-residual-v0.25 -> S2
		vllm-poc:0.25.1 -> mlnode-base). k1 was PoC-inoperative on V4 (gonka-poc
		predated PR#14 — per-group KV layouts missing); k2 rides gonka-poc 3a119d8
		with the per-group/positions/pseudo-ids fixes — first PoC-capable V4 rev. registry-view will label line=mlnode-overlay
		/ vllm_base_version=null (known overlay cost).

		Backend/compilation left to V4 defaults on purpose (see header + runner-patch):
		deterministic FlashMLA-DSV4, auto NONE + breakable-cudagraph, native ue8m0
		MoE — no pins, no DeepGEMM env. PoC forward eager via gonka-poc skip_compiled.

		NOT yet GPU-validated on 0.25.1 — run experiments A2/A6 (backend/cache_dtype
		dump, PoC sweep, 1000-nonce L2 vs the gonka-poc#14 B300/H100 references) before
		trusting. sm_80 (A100) + sm_120 (RTX PRO 6000) cannot run V4 (sparse backends
		need sm major in [9,10]).
		"""
	tuning_notes: [
		{
			knob:     "tensor_parallel_size=1 (8-GPU box -> 8 engines)"
			source:   "https://github.com/kaitakuai/gonka-poc/pull/14"
			reason:   "149 GiB FP8 fits on one B300 (275 GiB, ~126 GiB headroom). gonka-poc#14 validated 1×B300 TP1 ~928 nonces/min @ bs=16, seq 1024; a box runs 8 independent TP=1 engines. TP1 vs packing math is experiment A2/B-scope."
			added_at: "2026-07-20"
		},
		{
			knob:     "kv_cache_dtype=fp8 (FlashMLA fp8_ds_mla — mandatory)"
			source:   "vLLM 0.25.1 DeepseekV4 FlashMLA path"
			reason:   "The FlashMLA V4 path coerces fp8 to the fp8_ds_mla layout and asserts without it. NOT optional. fp8 changes attention math vs bf16 — all V4 references are already fp8 (no bf16 baseline exists; the 0.23 no-kv-fp8 path hard-fails)."
			added_at: "2026-07-20"
		},
		{
			knob:     "max_model_len=200000 (TEST cap)"
			source:   "test bring-up (no governance)"
			reason:   "Cut far below V4's native 1,048,576 for a cheap first KV profile. This is NOT the production cap — max-model-len is a chain-governance parameter delivered via DAPI (#1408 undefined). Prompts >200k truncate. Raise per experiment B4."
			severity: "warning"
			added_at: "2026-07-20"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/gonka-poc/pull/14"
			reason:   "gonka-poc#14 validated V4 PoC on 0.23 (B300 TP1/TP2 bit-identical, H100 TP4/TP8, cross-L2≈0.20). This image is the 0.25.1 rebuild — NOT yet re-validated on 0.25.1 kernels. Run experiments A2/A5/A6 (start clean, backend=FlashMLA-DSV4 + cache_dtype=fp8_ds_mla dump, PoC sweep, enforced_tokens replay, 1000-nonce L2 vs #14 refs) before production."
			severity: "warning"
			added_at: "2026-07-20"
		},
	]
}
