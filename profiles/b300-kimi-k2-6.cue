// Profile: B300 Blackwell Ultra (×4) + Kimi-K2.6 INT4 — release-matrix leaf.
//
// Composition: #OverlayProfile & B300 & KIMI_INT4 with TP=4, overlaid on
// cortima's published release mlnode (vLLM 0.25.1 residual + gonka-poc plugin;
// ADR-0013), NOT the fat-fork monolith. Serving flags come from the fat-fork
// B300 tune (tools/runner-patches/b300-kimi.py, measured 5120 nonces/min @
// batch=64 on 8×B300), not the B200 tune: eager (compilation mode=0 / cudagraph
// NONE) at batch=128, because cudagraph hangs at batch=128 on Kimi MLA and eager
// also saves ~5% over cudagraph FULL on this model. B300's 275 GiB/GPU has room
// for that envelope; B200's 178 GiB forced batch=32 and a 120000 context cap.
//
// Plugin differences vs the fat-fork b300-kimi.py:
//   - env adds MLNODE_VLLM_MODULE → the runner launches the composed gonka-poc
//     entrypoint; runner_patch forces --worker-extension-cls (worker-side PoC).
//   - no --enforce-eager: it conflicts with the forced --compilation-config, and
//     eager is already set via mode=0; PoC-forward eager (bit-compat) is handled
//     inside gonka-poc (poc_model_runner skip_compiled=True).
//
// Throughput/quality numbers are INHERITED from the fat-fork B300 Kimi tune and
// are NOT yet re-validated on the 0.25.1 plugin base — see the tuning_notes.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b300_kimi_k2_6: #OverlayProfile & bases.B300 & bases.KIMI_INT4 & bases.KIMI_INT4_FLASHINFER_MOE & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "kimi"
			model_revision: "k2-6"
			// quant axis intentionally omitted — Kimi-K2.6 ships only as INT4
			// today (compressed-tensors W4A16). Adding `quant: "int4"` would
			// suffix the tag with `-q.int4` without distinguishing anything.
			// Mirrors b200-kimi-k2-6.cue.
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
		// Cortima's PUBLISHED release image — their Stage-3 equivalent. Taking it
		// as our base drops a whole build stage and makes drifting from their
		// mlnode source impossible; everything we still add lives in hw_patches +
		// runner_patch below. See schema.cue on the switch.
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:1b9b7ce55feecab837f1d7ce974fc5f377ae0a04a4fb403eeeb50130e7728ee1"
		upstream_version: "3.0.16"
	}
	// The fat-fork's poc-householder-compile is NOT referenced — that edited the
	// monolith's vllm/poc/ tree, which does not exist on the plugin base.
	hw_patches: list.Concat([bases.B300.hw_patches, bases.GONKA_BASE_PATCHES])
	runner_patch: "b300-kimi-k2-6-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint
		// (build_app + PoC router + gating) instead of stock api_server (ADR-0013).
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// Kimi-K2.6's newer revision ships a buggy tokenization_kimi_fast.py;
		// HF_HUB_OFFLINE forces the cached (pre-staged) tokenizer and avoids the
		// hub pull that trips it. Requires the weights/tokenizer pre-staged in the
		// HF cache (true on fleet nodes; pre-stage before a standalone run).
		HF_HUB_OFFLINE: "1"
		// Long Kimi-K2.6 INT4 load (~530 GiB on TP=4) + MLA warmup; long runner
		// timeout + first-healthy grace (inherited from B300 / KIMI_INT4 bases,
		// reproduced for clarity).
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	// Eager path (compilation_mode=0 / cudagraph_mode=NONE) is the KIMI_INT4 base
	// default — kept (NOT overridden to the B200 rev=2 mode=3 compiled tune),
	// because cudagraph hangs at batch=128 on Kimi MLA. Surfaced for the dashboard.
	runtime_defaults: {
		// Forced via the runner-patch — values reproduced here for dashboard display.
		tensor_parallel_size:    4
		gpu_memory_utilization:  0.85
		max_num_batched_tokens:  131072
		max_num_seqs:            128
		attention_backend:       "CUTLASS_MLA"
		tool_call_parser:        "kimi_k2"
		reasoning_parser:        "kimi_k2"
		mm_encoder_tp_mode:      "data"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		enable_expert_parallel:  true
		// max_model_len intentionally NOT capped: Kimi's native 262144 context
		// fits on 4×B300 (~3-4× concurrency); leave it to the chain broadcast.
	}
	description: "B300 Blackwell Ultra SXM6 (×4) + Kimi-K2.6 INT4 (TP=4, eager) — vllm-poc PLUGIN base (gonka-poc entrypoint + worker extension)"
	notes: """
		First B300 Kimi-K2.6 profile on the 0.25.1 release base. Config taken from the
		fat-fork B300 tune (tools/runner-patches/b300-kimi.py): TP=4, eager
		(compilation mode=0 / cudagraph NONE), batch=128 (max-num-batched-tokens
		131072), gpu-memory-utilization 0.85, CUTLASS_MLA, expert-parallel,
		kimi_k2 tool/reasoning parsers, INT4 FlashInfer MoE (latency backend, from
		the KIMI_INT4 base). Native 256K context left uncapped (fits on 4×B300).

		Plugin flip vs the fat-fork:
		  - MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router (composed server).
		  - runner-patch forces --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
		    (worker-side PoC via collective_rpc), plus the B300 tune above.
		  - --enforce-eager removed (conflicts with --compilation-config; mode=0 is
		    already eager). PoC-forward eager comes from gonka-poc skip_compiled.
		  - HF_HUB_OFFLINE=1 baked (Kimi tokenizer workaround) — pre-stage weights.

		Throughput/quality INHERITED from the fat-fork B300 Kimi run (5120
		nonces/min @ batch=64 on 8×B300 = TP=4 ×2 instances; INT4 nonces
		L2-cross-validate under the Kimi chain gate). NOT yet re-validated on the
		0.25.1 base — re-benchmark on B300 before treating as production.

		RISK to verify on GPU: vLLM 0.23's lockable MoE WorkspaceManager broke
		modular-kernel PoC forwards (DEEPGEMM/TRITON) — workspace sized on
		inference shapes + locked before the PoC forward runs. If Kimi's INT4
		FlashInfer MoE routes through the modular kernel, the first PoC generate
		hits `AssertionError: Workspace is locked`. Whether the 0.25.1 line still
		does this is UNVERIFIED — no Kimi PoC run on it yet. The DeepSeek and
		MiniMax leaves on the same base do not exercise this path.
		"""
	tuning_notes: [
		{
			knob:     "tensor_parallel_size=4"
			source:   "tools/runner-patches/b300-kimi.py (B300 Kimi tune, 8×B300 = TP=4 ×2)"
			reason:   "1.06T-param Kimi-K2.6 INT4 (~530 GiB) only splits onto B300 at TP=4. Validated on the fat-fork B300 image."
			added_at: "2026-06-22"
		},
		{
			knob:     "gpu_memory_utilization=0.85"
			source:   "tools/runner-patches/b300-kimi.py"
			reason:   "Leaves HBM headroom for the Kimi vision encoder on B300. Below the 0.92-0.95 used elsewhere; intentional for this model — info, not a regression."
			added_at: "2026-06-22"
		},
		{
			knob:     "max_num_batched_tokens=131072"
			source:   "tools/runner-patches/b300-kimi.py"
			reason:   "One prefill chunk for batch=128 on B300's 275 GiB/GPU (B200 rev=2 had to drop to 32768 under cudagraph FULL on 178 GiB). Above the upstream 8192 chunked-prefill default — perf opt, no baseline violation; info."
			added_at: "2026-06-22"
		},
		{
			knob:     "compilation_mode=0"
			source:   "tools/runner-patches/b300-kimi.py"
			reason:   "Eager (cudagraph NONE). Cudagraph hangs at batch=128 on Kimi MLA, and eager also saves ~5% throughput vs cudagraph FULL on this model. KIMI_INT4 base default; B200 rev=2 overrode to mode=3 for an experimental compiled tune — B300 stays eager."
			added_at: "2026-06-22"
		},
		{
			knob:     "attention_backend=CUTLASS_MLA"
			source:   "tools/runner-patches/b300-kimi.py + b200-kimi-k2-6.cue"
			reason:   "Required for Kimi-K2.6 MLA on TP=4. Pinned (also forced via the runner-patch) so the auto-selector can't regress it."
			added_at: "2026-06-22"
		},
		{
			knob:     "MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router"
			source:   "docs/adr/0013-poc-integration-architecture.md"
			reason:   "Server-side plugin flip — launches the composed gonka-poc entrypoint (PoC router + gating) instead of stock api_server. Required on the plugin base. No throughput trade-off; info."
			added_at: "2026-06-22"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/kimi_k26_int4_4xb200_q-int4-k2/README.md"
			reason:   "B300 Kimi throughput/quality INHERITED from the fat-fork tune (b300-kimi.py: 5120 nonces/min @ batch=64 on 8×B300; INT4 nonces L2-valid under the Kimi chain gate). NOT yet re-validated on the 0.25.1 release base — re-benchmark on B300, and verify the MoE WorkspaceManager-lock risk (see notes) before production."
			severity: "warning"
			added_at: "2026-06-22"
		},
	]
}
