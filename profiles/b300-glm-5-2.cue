// Profile: B300 Blackwell Ultra (×4) + GLM-5.2 FP8 — vllm-poc PLUGIN base.
//
// Composition: #BaseProfile & B300 & GLM_5_2 with TP=4, built on the vllm-poc
// PLUGIN base (residual vLLM 0.23 + gonka-poc; ADR-0013). Source of truth:
// experiments/2026-06/glm-5.2-deepgemm-4xb300. B300's 275 GiB HBM/GPU fits
// GLM-5.2 FP8 (753B, ~704 GB) on just 4×B300 at TP=4 (176 GB weights/card), so an
// 8-GPU box auto-runs TWO independent TP=4 engines → 1792 nonces/min/box vs 1078
// for a single TP=8 engine (less all-reduce overhead per engine). The mlnode
// runner auto-detects the spare-GPU box and launches 2 instances; the profile only
// encodes tensor_parallel_size=4 (mirrors b300-minimax-m2-7.cue).
//
// DeepGEMM split (B200/B300 sm_100): MoE experts on DeepGEMM + the block-FP8 LINEAR
// kernel routed to Cutlass via VLLM_DISABLED_KERNELS (GLM fused_qkv_a_proj N=2624,
// N%128==64, crashes the DeepGEMM linear kernel at memory profiling). Same env as
// b200-glm-5-2; lives in THIS leaf (NOT the GLM_5_2 base, so H200 can run
// DeepGEMM-off). Compilation NOT forced (compiled inference + eager PoC via
// gonka-poc skip_compiled; --enforce-eager is the operator pure-mining exception).
//
// max-model-len 400000 is the operator directive (Pasha) for all GLM GPUs; B300
// trivially fits it (the experiment proved the full 1,048,576 context at TP=4 /
// gmu0.92, KV pool 1,124,928 tokens). Throughput NOT yet re-validated on the
// foundry-built leaf — see tuning_notes.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b300_glm_5_2: #BaseProfile & bases.B300 & bases.GLM_5_2 & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "glm"
			model_revision: "5-2"
			// quant axis omitted — GLM-5.2 ships only as FP8 (zai-org/GLM-5.2-FP8).
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.23.0"
			rev:    1
		}
	}
	mode: "kaitakuai-base"
	// hw_patches inherited from bases.B300 (triton-ptxas, flashinfer-jit-uninstall,
	// libcuda-compat-580-driver, nvidia-headers-symlinks, cold-start-tolerance) —
	// the B300 default already includes cold-start-tolerance for the slow 753B+DSA
	// init, so no override needed (unlike b200 whose base omits it).
	runner_patch: "b300-glm-5-2-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// DeepGEMM split (B300 sm_100) — GPU-specific, same as b200-glm-5-2:
		//   - MoE experts ON DeepGEMM; PoC workspace lock handled by gonka-poc df73e1c.
		//   - LINEAR block-FP8 routed to Cutlass via VLLM_DISABLED_KERNELS (GLM
		//     fused_qkv_a_proj N=2624, N%128==64, crashes the DeepGEMM linear kernel).
		// E8M0 + the disabled-kernels route are image-inherited (the deepgemm-4xb300
		// experiment reused the baked b200-glm image) — NOT independently re-proven on
		// B300 driver 595.71.05; GPU-confirm DeepGEMM-on start. NONCE NOTE: Cutlass
		// linear is not bit-identical to a pure-DeepGEMM validator — fine in benchmark
		// phase; re-confirm L2 before production. FALLBACK: drop VLLM_DISABLED_KERNELS
		// + set VLLM_USE_DEEP_GEMM_E8M0=0; last resort triton-only (both flags 0).
		VLLM_USE_DEEP_GEMM:          "1"
		VLLM_MOE_USE_DEEP_GEMM:      "1"
		VLLM_USE_DEEP_GEMM_E8M0:     "1"
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
		VLLM_DISABLED_KERNELS:       "DeepGemmFp8BlockScaledMMKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
		// VLLM_RUNNER_TIMEOUT / WATCHER_GRACE_FIRST_HEALTHY come from the B300 +
		// GLM_5_2 bases (identical concretes — CUE unifies cleanly).
	}
	// Compiled inference (vLLM default cudagraph) + eager PoC via gonka-poc
	// skip_compiled — NOT forced by the runner-patch (operator passes --enforce-eager
	// only for the exceptional pure-mining eager case). Display from the GLM_5_2 base.
	runtime_defaults: {
		// From experiments/2026-06/glm-5.2-deepgemm-4xb300; forced via the
		// runner-patch, reproduced here for dashboard display.
		tensor_parallel_size:    4
		gpu_memory_utilization:  0.92
		max_model_len:           400000
		max_num_batched_tokens:  16384
		max_num_seqs:            64
		kv_cache_dtype:          "fp8_e4m3"
		tool_call_parser:        "glm47"
		reasoning_parser:        "glm45"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		// NO attention_backend: GLM-5.2 is DSA, not MLA — let vLLM auto-select.
		// NO enable_expert_parallel: B300 experiment runs TP=4 without EP.
	}
	description: "B300 Blackwell Ultra SXM6 (×4, 2 engines/box) + GLM-5.2 FP8 (TP=4, eager PoC / compiled inference, DeepGEMM MoE + Cutlass linear) — vllm-poc PLUGIN base"
	notes: """
		GLM-5.2 FP8 on B300 at TP=4, from experiments/2026-06/glm-5.2-deepgemm-4xb300.
		Config: TP=4, gpu-memory-utilization 0.92, max-model-len 400000
		(operator-forced; B300 fits the full 1048576 at TP=4 so this is comfortable),
		max-num-batched-tokens 16384 (DeepGEMM survives memory profiling only at small
		mnbt — batch>16 → cudaErrorInvalidValue), max-num-seqs 64, kv-cache-dtype
		fp8_e4m3 (B300 sm_100 accepts e4m3, no fp8_ds_mla needed unlike H200), glm47
		tool-call + glm45 reasoning parsers.

		Deployment: 753B FP8 fits on 4×B300 (176 GB/card of 275), so an 8-GPU box
		auto-runs 2 independent TP=4 engines → 1792 nonces/min/box (2×896) vs 1078 for
		a single TP=8 engine (+66%). The mlnode runner auto-detects this and launches
		2 instances; the profile encodes only TP=4 (same as b300-minimax). To force a
		single engine for a clean per-engine number: CUDA_VISIBLE_DEVICES=0,1,2,3.

		DeepGEMM split + compilation/eager policy: identical to b200-glm-5-2 (MoE on
		DeepGEMM, linear→Cutlass via VLLM_DISABLED_KERNELS; compiled inference + eager
		PoC via skip_compiled, --enforce-eager is the pure-mining exception).

		Throughput (experiment, single TP=4 engine): PoC 896 nonces/min @ batch 16
		(box 1792), 864/min even at full 1M context (no context↔PoC tradeoff on B300).
		NOTE: the experiment's 896/1792 was measured under a FORCED
		--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'; the
		shipped image uses the vLLM compiled default (cudagraph mode may differ) — to
		reproduce exactly the operator adds that flag. NOT yet re-benchmarked on the
		foundry-built leaf; re-confirm DeepGEMM-on start on driver 595.71.05 + L2
		cross-validate before production.
		"""
	tuning_notes: [
		{
			knob:     "tensor_parallel_size=4 (8-GPU box → 2 engines)"
			source:   "https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-deepgemm-4xb300"
			reason:   "753B FP8 (~704 GB) fits on 4×B300 (176 GB/card of 275). Box runs 2 TP=4 engines = 1792 nonces/min, +66% over a single TP=8 (1078) from less all-reduce overhead. Runner auto-detects 2 instances; profile encodes TP=4 only."
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "max_model_len=400000 (operator-forced)"
			source:   "Pasha directive + glm-5.2-deepgemm-4xb300"
			reason:   "400000 on all GLM GPUs (operator). B300 trivially fits — the experiment proved the full 1,048,576 context at TP=4/gmu0.92 (KV pool 1,124,928 tokens) with PoC still at 864/min, so 400000 has ample headroom."
			added_at: "2026-06-26"
		},
		{
			knob:     "max_num_batched_tokens=16384"
			source:   "glm-5.2-deepgemm-4xb300"
			reason:   "DeepGEMM survives vLLM memory profiling only at small mnbt; PoC batch 32/64 → 0 nonces (cudaErrorInvalidValue), capped at batch 16. Same constraint as B200."
			added_at: "2026-06-26"
		},
		{
			knob:     "MoE backend=DeepGEMM (linear→Cutlass via VLLM_DISABLED_KERNELS)"
			source:   "glm-5.2-deepgemm-4xb300 (reused baked b200-glm image)"
			reason:   "MoE on DeepGEMM (the throughput lever); block-FP8 LINEAR routed to Cutlass to dodge the GLM fused_qkv_a_proj crash. Image-inherited (NOT independently re-proven on B300 driver 595.71.05) — GPU-confirm DeepGEMM-on start. MoE workspace lock handled by gonka-poc df73e1c."
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "compiled inference + eager PoC (compilation NOT forced)"
			source:   "GLM_5_2 base + gonka-poc skip_compiled"
			reason:   "Inference COMPILED by default; PoC eager via skip_compiled. The experiment's 896/1792 was under FORCED --compilation-config mode:3 FULL_AND_PIECEWISE; the shipped default cudagraph mode may differ — operator adds that flag to reproduce, or --enforce-eager for pure-mining."
			added_at: "2026-06-26"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-deepgemm-4xb300"
			reason:   "PoC 896/engine (1792/box) @ batch 16, 864/min at full 1M context; honest FP8 L2-cross-validates vs AWQ-INT4 fraud. NOT yet re-benchmarked on the vllm-poc 0.23 PLUGIN leaf — re-confirm DeepGEMM start on B300 driver 595.71.05 + L2 before production."
			severity: "warning"
			added_at: "2026-06-26"
		},
	]
}
