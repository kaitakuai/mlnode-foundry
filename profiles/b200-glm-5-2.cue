// Profile: B200 Blackwell (×8) + GLM-5.2 FP8 — vllm-poc PLUGIN base.
//
// Composition: #BaseProfile & B200 & GLM_5_2 with TP=8, built on the vllm-poc
// PLUGIN base (residual vLLM 0.23 + gonka-poc package; ADR-0013). New model on
// the foundry — first GLM profile. Config is the operator-supplied GLM-5.2
// recommendation (zai-org/GLM-5.2-FP8) plus the PoC additions (eager PoC +
// compiled inference, plugin worker wiring, DeepGEMM MoE + Cutlass linear).
//
// GLM-5.2 = GlmMoeDsaForCausalLM: DeepSeek-style MLA + DSA sparse attention
// (index_topk=2048), 753B total / ~40B active, 256 experts × top-8, FP8 block
// e4m3 [128,128]. DeepGEMM is SPLIT on our 0.23 image: MoE experts run on
// DeepGEMM (throughput lever) while the block-FP8 LINEAR kernel is routed to
// Cutlass — GLM's fused_qkv_a_proj (N=2624, N%128==64 partial block) crashes the
// DeepGEMM linear kernel at memory profiling. Both via the GLM_5_2 base env
// (VLLM_MOE_USE_DEEP_GEMM=1 + VLLM_DISABLED_KERNELS=...). The MoE PoC-forward
// workspace lock is covered by gonka-poc df73e1c (unlock/relock).
//
// Plugin flip vs a stock serve:
//   - env adds MLNODE_VLLM_MODULE → the runner launches the composed gonka-poc
//     entrypoint; runner_patch forces --worker-extension-cls (worker-side PoC).
//   - compilation NOT forced: inference runs COMPILED (vLLM default + CUDA
//     graphs) for decode throughput; the PoC forward runs eager on its own via
//     gonka-poc poc_model_runner (skip_compiled=True). Forcing global eager would
//     drop inference CUDA graphs (~5× decode loss on GLM: 157 vs 817 tok/s);
//     operator passes --enforce-eager only for the exceptional eager-inference
//     case (neither forced nor stripped here).
//   - NOT pinned: --attention-backend. GLM-5.2 is DSA (not MLA) — do NOT force
//     CUTLASS_MLA (a Kimi/DeepSeek-MLA backend); let vLLM auto-select.
//
// Throughput numbers are from the standalone GLM experiment, NOT yet re-validated
// on the 0.23 plugin base — see tuning_notes.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b200_glm_5_2: #BaseProfile & bases.B200 & bases.GLM_5_2 & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "glm"
			model_revision: "5-2"
			// quant axis intentionally omitted — GLM-5.2 ships only as the FP8
			// block-wise checkpoint (zai-org/GLM-5.2-FP8). Adding `quant: "fp8"`
			// would suffix the tag with `-q.fp8` without distinguishing anything.
			// Mirrors b300-kimi-k2-6.cue / b200-kimi-k2-6.cue.
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.23.0"
			rev:    1
		}
	}
	mode: "kaitakuai-base"
	// B200 GPU-env hw-patches + cold-start-tolerance: 753B FP8 on 8×B200 + DSA
	// warmup is a slow first init (the B200 default list omits cold-start; added
	// here, same as the Kimi-INT4 leaves note in bases/b200.cue).
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
	]
	runner_patch: "b200-glm-5-2-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint
		// (build_app + PoC router + gating) instead of stock api_server (ADR-0013).
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// DeepGEMM split (B200 sm_100) — GPU-specific, lives in THIS leaf (moved out
		// of the GLM_5_2 base so H200 can run DeepGEMM-off):
		//   - MoE experts ON DeepGEMM (AND-gated on USE_DEEP_GEMM + MOE_USE_DEEP_GEMM;
		//     PoC workspace lock handled by gonka-poc df73e1c).
		//   - LINEAR block-FP8 kernel routed to Cutlass via VLLM_DISABLED_KERNELS:
		//     GLM's fused_qkv_a_proj (N=2624, N%128==64) + E8M0 trips
		//     cudaErrorInvalidValue in the DeepGEMM linear kernel at memory profiling;
		//     Cutlass pads to 16-align and starts clean, MoE keeps DeepGEMM+E8M0.
		// E8M0 is image-inherited (baked in mlnode-b200-glm-5-2; the deepgemm-8xb200
		// experiment started cleanly WITH it) — not independently experiment-proven.
		// NONCE NOTE: the Cutlass linear is NOT bit-identical to a pure-DeepGEMM
		// validator — fine in benchmark phase (GLM not governance-activated); re-confirm
		// L2 before production. FALLBACK if it won't start: drop VLLM_DISABLED_KERNELS +
		// set VLLM_USE_DEEP_GEMM_E8M0=0; last resort triton-only (both flags 0).
		VLLM_USE_DEEP_GEMM:          "1"
		VLLM_MOE_USE_DEEP_GEMM:      "1"
		VLLM_USE_DEEP_GEMM_E8M0:     "1"
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
		VLLM_DISABLED_KERNELS:       "DeepGemmFp8BlockScaledMMKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
		// VLLM_RUNNER_TIMEOUT / WATCHER_GRACE_FIRST_HEALTHY come from the GLM_5_2 base.
	}
	// Compiled inference (compilation_mode=3 / vLLM default cudagraph) from the
	// GLM_5_2 base — the PoC forward is eager on its own via gonka-poc
	// skip_compiled. Surfaced for the dashboard; compilation is NOT forced by the
	// runner-patch (operator passes --enforce-eager only for the exceptional
	// eager-inference case).
	runtime_defaults: {
		// Operator-supplied GLM-5.2 recommendation; forced via the runner-patch,
		// reproduced here for dashboard display.
		tensor_parallel_size:    8
		gpu_memory_utilization:  0.85
		max_model_len:           400000
		max_num_batched_tokens:  16384
		max_num_seqs:            16
		kv_cache_dtype:          "fp8_e4m3"
		tool_call_parser:        "glm47"
		reasoning_parser:        "glm45"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		// NO attention_backend: GLM-5.2 is DSA, not MLA — let vLLM auto-select.
		// NO enable_expert_parallel: the operator recommendation omits it.
	}
	description: "B200 Blackwell SXM6 (×8) + GLM-5.2 FP8 (TP=8, eager PoC / compiled inference, DeepGEMM MoE + Cutlass linear) — vllm-poc PLUGIN base (gonka-poc entrypoint + worker extension)"
	notes: """
		First GLM-5.2 profile on the foundry (0.23 plugin base). Config is the
		operator-supplied GLM-5.2 recommendation: TP=8, gpu-memory-utilization 0.85,
		max-model-len 400000 (capped below the native 1048576 to fit 8×B200 KV),
		max-num-batched-tokens 16384, max-num-seqs 16, kv-cache-dtype fp8_e4m3,
		glm47 tool-call parser + glm45 reasoning parser, --trust-remote-code,
		--enable-auto-tool-choice.

		PoC additions:
		  - MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router (composed server).
		  - runner-patch forces --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
		    (worker-side PoC via collective_rpc).
		  - compilation NOT forced: inference runs COMPILED by default (vLLM
		    VLLM_COMPILE + CUDA graphs, GLM 817 vs 157 tok/s eager) while the PoC
		    forward runs eager on its own via gonka-poc skip_compiled (+~25% nonces).
		    Global eager is NOT forced — it would drop inference CUDA graphs (~5×
		    decode loss). Operator passes --enforce-eager only for the exceptional
		    eager-inference case (the patch neither forces nor strips it).
		  - DeepGEMM strategy is SPLIT (GLM_5_2 base): MoE experts ON DeepGEMM
		    (VLLM_USE_DEEP_GEMM=1 + VLLM_MOE_USE_DEEP_GEMM=1 + E8M0=1, the throughput
		    lever) while the block-FP8 LINEAR kernel is routed to Cutlass via
		    VLLM_DISABLED_KERNELS=DeepGemmFp8BlockScaledMMKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel.
		    GLM's MLA fused_qkv_a_proj (N=2624, N%128==64 partial block) + E8M0
		    requant trips cudaErrorInvalidValue in the DeepGEMM linear kernel at
		    memory profiling (the "image didn't start" bug); Cutlass pads to
		    16-alignment and starts cleanly. MoE-DeepGEMM's PoC-forward workspace
		    lock is handled by gonka-poc df73e1c (unlock/relock), same as MiniMax.

		NONCE CAVEAT: the Cutlass linear changes fused_qkv_a_proj FP8 rounding vs a
		pure-DeepGEMM validator (NOT bit-identical). Fine for the benchmark phase
		(GLM not governance-activated, no live L2 gate); re-confirm L2-equivalence
		against the validator kernel before production mining.

		Throughput reference from the standalone GLM experiment
		(kaitakuai/experiments/2026-06/glm-5.2-fp8-8xb200, triton-only run): eager
		PoC 1016 nonces/min @ batch 64 vs cudagraph mode-3 768 (−24%); inference
		mode-3 817 tok/s (+6.8× decode). DeepGEMM-MoE is expected to lift PoC above
		that triton baseline (MiniMax/Kimi saw +49% from DeepGEMM-MoE) but is NOT yet
		measured on GLM — GPU-confirm DeepGEMM-on start + throughput vs Gleb's branch
		on the vllm-poc 0.23 PLUGIN base before treating as production.
		"""
	tuning_notes: [
		{
			knob:     "tensor_parallel_size=8"
			source:   "operator GLM-5.2 recommendation (zai-org/GLM-5.2-FP8)"
			reason:   "753B FP8 (~full 8×B200 HBM) requires TP=8 to fit the weights + KV; single-instance per 8-GPU box. Validated in the standalone GLM experiment."
			added_at: "2026-06-24"
		},
		{
			knob:     "max_model_len=400000 (operator-forced; +14% over the 350000 GPU-proven fit)"
			source:   "operator GLM-5.2 recommendation"
			reason:   "Operator directive (Pasha): 400000 on all GLM GPUs. The deepgemm-8xb200 experiment proved 350000 at fp8_e4m3/gmu0.85; 400000 is +14% and NOT GPU-validated — re-confirm it does not OOM at memory profiling on 8×B200 before production. Cut from native 1048576 (full window won't hold at useful concurrency)."
			severity: "warning"
			added_at: "2026-06-24"
		},
		{
			knob:     "max_num_batched_tokens=16384"
			source:   "operator GLM-5.2 recommendation"
			reason:   "Chunked-prefill budget (above the upstream 8192 default) paired with max_num_seqs=16. Perf opt, no baseline violation; info."
			added_at: "2026-06-24"
		},
		{
			knob:     "max_num_seqs=16"
			source:   "operator GLM-5.2 recommendation"
			reason:   "Concurrency cap for the large 753B model at max_model_len=400000. Couples to max_num_batched_tokens=16384. Info, not a regression vs upstream."
			added_at: "2026-06-24"
		},
		{
			knob:     "kv_cache_dtype=fp8_e4m3"
			source:   "operator GLM-5.2 recommendation"
			reason:   "FP8 KV halves the cache footprint, enabling max_model_len=400000 at max_num_seqs=16 on 8×B200. Accepted natively on sm_100. Standard for the FP8 Blackwell profiles."
			added_at: "2026-06-24"
		},
		{
			knob:     "compiled inference + eager PoC (compilation NOT forced)"
			source:   "GLM_5_2 base + gonka-poc skip_compiled + b300-minimax-plugin policy"
			reason:   "Inference runs COMPILED by default (vLLM VLLM_COMPILE + CUDA graphs): GLM decode 817 vs 157 tok/s eager (+6.8× TPOT). The PoC forward is eager on its own via gonka-poc poc_model_runner (set_forward_context skip_compiled=True): eager PoC is +~25% (1016 vs 768 nonces/min) AND bit-compat. The runner-patch does NOT force --compilation-config / --enforce-eager (forcing global eager would drop inference CUDA graphs). Operator passes --enforce-eager only for the EXCEPTIONAL eager-inference case."
			added_at: "2026-06-24"
		},
		{
			knob:     "MoE backend=DeepGEMM (linear→Cutlass via VLLM_DISABLED_KERNELS)"
			source:   "GLM_5_2 base env + fused_moe/oracle/fp8.py:360 + kernels/linear (v0.23.0)"
			reason:   "DeepGEMM's two flags are split: MoE experts ON DeepGEMM (the throughput lever, AND-gated on VLLM_USE_DEEP_GEMM=1 + VLLM_MOE_USE_DEEP_GEMM=1 + E8M0=1) while the block-FP8 LINEAR path is disabled (DeepGemmFp8BlockScaledMMKernel + FlashInferFp8DeepGEMMDynamicBlockScaledKernel) so fused_qkv_a_proj (N=2624, N%128==64) falls to CutlassFp8BlockScaledMMKernel instead of crashing the DeepGEMM linear kernel at memory profiling. MoE workspace lock handled by gonka-poc df73e1c. The earlier 'triton only' read was wrong — DeepGEMM-MoE runs once the linear is routed off it. GPU-confirm start + throughput."
			severity: "warning"
			added_at: "2026-06-24"
		},
		{
			knob:     "no attention_backend pin (DSA, not MLA)"
			source:   "GlmMoeDsaForCausalLM arch + operator recommendation"
			reason:   "GLM-5.2 uses DSA sparse attention, not MLA — do NOT pin CUTLASS_MLA (a Kimi/DeepSeek-MLA backend). The runner-patch deliberately omits --attention-backend; vLLM auto-selects the DSA path."
			added_at: "2026-06-24"
		},
		{
			knob:     "MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router"
			source:   "docs/adr/0013-poc-integration-architecture.md"
			reason:   "Server-side plugin flip — launches the composed gonka-poc entrypoint (PoC router + gating) instead of stock api_server. Required on the plugin base. No throughput trade-off; info."
			added_at: "2026-06-24"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-06/glm-5.2-fp8-8xb200/README.md"
			reason:   "GLM-5.2 throughput from the standalone experiment (eager PoC 1016 nonces/min @ batch 64; triton-only MoE). NOT yet re-validated on the vllm-poc 0.23 PLUGIN base — re-benchmark on 8×B200 and verify nonces L2-cross-validate under the GLM chain gate before production."
			severity: "warning"
			added_at: "2026-06-24"
		},
	]
}
