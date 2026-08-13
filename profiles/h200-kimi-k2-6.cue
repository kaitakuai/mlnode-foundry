// Profile: H200 Hopper (×8) + Kimi-K2.6 INT4 — release-matrix leaf.
//
// Composition: #OverlayProfile & H200 & KIMI_INT4 with TP=8, overlaid on
// cortima's published release mlnode (vLLM 0.25.1 residual + gonka-poc plugin;
// ADR-0013).
//
// UNLIKE the b200/b300 Kimi leaves, none of the serving flags here are ours:
// they are copied from gonka's own H200 Kimi profile
// (deploy/join/node-config-kimik26-H200.json), which is the only Kimi tuning
// that exists for Hopper. Two Blackwell-only choices are consequently absent:
//   - CUTLASS_MLA -> FLASHMLA (CUTLASS_MLA is sm_100+).
//   - KIMI_INT4_FLASHINFER_MOE is NOT unified in — FlashInfer's INT4 MoE
//     kernels are trtllm-gen, i.e. Blackwell-only. MoE backend is left to
//     vLLM's auto-selection, as in gonka's profile.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

h200_kimi_k2_6: #OverlayProfile & bases.H200 & bases.KIMI_INT4 & {
	identity: {
		axes: {
			gpu:            "h200"
			model:          "kimi"
			model_revision: "k2-6"
			// quant axis omitted — Kimi-K2.6 ships only as INT4 (compressed-tensors
			// W4A16). Mirrors b200/b300-kimi-k2-6.cue.
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// rev=3 — drop the dead VLLM_USE_V1 (removed from vLLM).
			upstream: "3.0.16"
			rev:      3
		}
	}
	mode: "upstream-overlay"
	base: {
		// Cortima's PUBLISHED release image — see b300-kimi-k2-6.cue on the switch.
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:79550026c5c567f2bdc3ae181a3cce1586e00957271fdd1554da124ff6f50b19"
		upstream_version: "3.0.16"
	}
	hw_patches: list.Concat([bases.H200.hw_patches, bases.GONKA_BASE_PATCHES])
	runner_patch: "h200-kimi-k2-6-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint
		// (build_app + PoC router + gating) instead of stock api_server (ADR-0013).
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// Kimi-K2.6's newer revision ships a buggy tokenization_kimi_fast.py;
		// HF_HUB_OFFLINE forces the cached (pre-staged) tokenizer and avoids the
		// hub pull that trips it. Requires the weights/tokenizer pre-staged in the
		// HF cache — pre-stage before a standalone run.
		HF_HUB_OFFLINE: "1"
		// ~530 GiB INT4 load spread over 8 GPUs + MLA warmup.
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		// Forced via the runner-patch — values reproduced here for dashboard display.
		tensor_parallel_size:    8
		gpu_memory_utilization:  0.90
		max_model_len:           240000
		attention_backend:       "FLASHMLA"
		tool_call_parser:        "kimi_k2"
		reasoning_parser:        "kimi_k2"
		mm_encoder_tp_mode:      "data"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		enable_expert_parallel:  true
		// Compilation left to vLLM: the KIMI_INT4 eager default exists for a
		// Blackwell batch=128 cudagraph hang that has never been reproduced on
		// Hopper, and gonka's H200 profile does not pin it either.
		compilation_mode: -1
		cudagraph_mode:   "AUTO"
	}
	description: "H200 Hopper (×8) + Kimi-K2.6 INT4 (TP=8, FLASHMLA, 240k ctx) — vllm-poc 0.25.1 PLUGIN, release-matrix CANDIDATE image"
	notes: """
		First Hopper Kimi-K2.6 leaf. Serving flags are gonka's H200 Kimi profile
		verbatim (deploy/join/node-config-kimik26-H200.json) plus the two PoC
		additions every leaf carries (--logprobs-mode processed_logprobs,
		--worker-extension-cls gonka_poc.worker.PoCWorkerExtension).

		NOT BENCHMARKED BY US. There is no kaitakuai experiment for Kimi on
		Hopper — the b200 (~2240 nonces/min @ batch=32) and b300 (5120 nonces/min
		@ batch=64) numbers do not transfer: different attention backend
		(FLASHMLA vs CUTLASS_MLA), different MoE kernels (no FlashInfer INT4 on
		sm_90), different TP (8 vs 4). Treat throughput as unknown and PoC
		cross-hardware L2 as unverified until an H200 run exists.

		Batch sizing is deliberately unset. Both Blackwell leaves cap
		max-num-batched-tokens (131072 on B300, 32768 on B200) because their
		measured envelopes demanded it; picking a number here without a
		measurement would just be a guess with a false provenance.
		"""
	tuning_notes: [
		{
			knob:     "tensor_parallel_size=8"
			source:   "gonka-ai/gonka deploy/join/node-config-kimik26-H200.json"
			reason:   "1.06T-param Kimi-K2.6 INT4 (~530 GiB) does not fit Hopper's 141 GiB/GPU below TP=8. Gonka's own H200 profile; not independently measured."
			added_at: "2026-08-07"
		},
		{
			knob:     "attention_backend=FLASHMLA"
			source:   "gonka-ai/gonka deploy/join/node-config-kimik26-H200.json"
			reason:   "CUTLASS_MLA (pinned on the b200/b300 leaves) is sm_100+. FLASHMLA is the Hopper MLA path."
			added_at: "2026-08-07"
		},
		{
			knob:     "max_model_len=240000"
			source:   "gonka-ai/gonka deploy/join/node-config-kimik26-H200.json"
			reason:   "Below Kimi's native 262144. Gonka caps it on both their Kimi profiles; b300 leaves it uncapped on 275 GiB/GPU."
			added_at: "2026-08-07"
		},
		{
			knob:     "VLLM_USE_FLASHINFER_MOE_INT4 unset"
			source:   "bases/kimi_int4.cue (KIMI_INT4_FLASHINFER_MOE)"
			reason:   "FlashInfer's INT4 MoE kernels are trtllm-gen (Blackwell). Leaving the backend to vLLM's auto-selection matches gonka's H200 profile."
			added_at: "2026-08-07"
		},
		{
			knob:     "validation-report"
			source:   "none — no H200 Kimi experiment exists"
			reason:   "CANDIDATE image. Throughput, PoC nonce rate and cross-hardware L2 are all unmeasured on this (GPU, model) pair. Benchmark on 8xH200 before treating as production."
			severity: "warning"
			added_at: "2026-08-07"
		},
	]
}
