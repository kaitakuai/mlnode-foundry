// Profile: B200 Blackwell (×8) + GLM-5.2 FP8 — vllm-poc PLUGIN base.
//
// Composition: #BaseProfile & B200 & GLM_5_2 with TP=8, built on the vllm-poc
// PLUGIN base (residual vLLM 0.23 + gonka-poc package; ADR-0013). New model on
// the foundry — first GLM profile. Config is the operator-supplied GLM-5.2
// recommendation (zai-org/GLM-5.2-FP8) plus the PoC additions (eager mining,
// plugin worker wiring, triton-only MoE).
//
// GLM-5.2 = GlmMoeDsaForCausalLM: DeepSeek-style MLA + DSA sparse attention
// (index_topk=2048), 753B total / ~40B active, 256 experts × top-8, FP8 block
// e4m3 [128,128]. On our 0.23 image DeepGEMM crashes (cudaErrorInvalidValue in
// the BlockScaledMM LINEAR kernel — a DSA-arch bug, NOT the MoE workspace lock)
// and FlashInfer-CUTLASS MoE hangs → triton MoE is the only backend that
// composes (forced off via the GLM_5_2 base env). Because DeepGEMM is disabled,
// the MoE WorkspaceManager-lock path is not exercised here — no workspace risk.
//
// Plugin flip vs a stock serve:
//   - env adds MLNODE_VLLM_MODULE → the runner launches the composed gonka-poc
//     entrypoint; runner_patch forces --worker-extension-cls (worker-side PoC).
//   - eager via --compilation-config (mode=0 / cudagraph NONE) for PoC mining
//     (+~25% nonces vs cudagraph mode 3 on GLM); NO --enforce-eager (it conflicts
//     with --compilation-config; PoC-forward eager bit-compat is handled inside
//     gonka-poc poc_model_runner skip_compiled=True).
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
		// DeepGEMM/FlashInfer-MoE off (triton-only) + long runner timeout +
		// first-healthy grace come from the GLM_5_2 base.
	}
	// Eager path (compilation_mode=0 / cudagraph_mode=NONE) is the GLM_5_2 base
	// default — kept for PoC mining (eager is ~+25% over cudagraph mode 3 on GLM).
	// Surfaced here for the dashboard; forced via the runner-patch.
	runtime_defaults: {
		// Operator-supplied GLM-5.2 recommendation; forced via the runner-patch,
		// reproduced here for dashboard display.
		tensor_parallel_size:    8
		gpu_memory_utilization:  0.85
		max_model_len:           350000
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
	description: "B200 Blackwell SXM6 (×8) + GLM-5.2 FP8 (TP=8, eager, DeepGEMM MoE + Cutlass linear) — vllm-poc PLUGIN base (gonka-poc entrypoint + worker extension)"
	notes: """
		First GLM-5.2 profile on the foundry (0.23 plugin base). Config is the
		operator-supplied GLM-5.2 recommendation: TP=8, gpu-memory-utilization 0.85,
		max-model-len 350000 (capped below the native 1048576 to fit 8×B200 KV),
		max-num-batched-tokens 16384, max-num-seqs 16, kv-cache-dtype fp8_e4m3,
		glm47 tool-call parser + glm45 reasoning parser, --trust-remote-code,
		--enable-auto-tool-choice.

		PoC additions:
		  - MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router (composed server).
		  - runner-patch forces --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
		    (worker-side PoC via collective_rpc).
		  - eager via --compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'
		    (+~25% PoC nonces vs cudagraph mode 3 on GLM). --enforce-eager removed
		    (conflicts with --compilation-config). PoC-forward eager (bit-compat)
		    comes from gonka-poc skip_compiled.
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
			knob:     "max_model_len=350000"
			source:   "operator GLM-5.2 recommendation"
			reason:   "Cut from the native 1048576 context. KV at fp8_e4m3 for 753B on 8×B200 cannot hold the full window at useful concurrency; 350000 keeps headroom. Operators needing longer contexts must add GPUs or drop concurrency."
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
			reason:   "Concurrency cap for the large 753B model at max_model_len=350000. Couples to max_num_batched_tokens=16384. Info, not a regression vs upstream."
			added_at: "2026-06-24"
		},
		{
			knob:     "kv_cache_dtype=fp8_e4m3"
			source:   "operator GLM-5.2 recommendation"
			reason:   "FP8 KV halves the cache footprint, enabling max_model_len=350000 at max_num_seqs=16 on 8×B200. Accepted natively on sm_100. Standard for the FP8 Blackwell profiles."
			added_at: "2026-06-24"
		},
		{
			knob:     "compilation_mode=0 (eager, cudagraph NONE)"
			source:   "GLM_5_2 base default (eager PoC; GLM benchmark in the validation-report note)"
			reason:   "Eager for PoC mining: 1016 nonces/min @ batch 64 vs 768 under cudagraph mode 3 (+~25%). Forced via --compilation-config (NOT --enforce-eager — that conflicts with it). A future inference-serving leaf can flip to mode 3 (+6.8× decode tok/s)."
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
