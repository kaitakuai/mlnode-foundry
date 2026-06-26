// Profile base: GLM-5.2 FP8 common tunings.
// Imported by all glm+5-2 profiles.
package bases

GLM_5_2: {
	env: {
		// COMMON GLM-5.2 env only. The MoE/linear BACKEND env (DeepGEMM on/off,
		// VLLM_DISABLED_KERNELS, E8M0, FlashInfer-MoE, KV dtype) is GPU-SPECIFIC and
		// lives in each leaf profile — Blackwell (b200/b300, sm_100) runs DeepGEMM
		// (MoE-DeepGEMM + linear→Cutlass split); Hopper (h200, sm_90) runs DeepGEMM
		// OFF + triton MoE (FLASHMLA_SPARSE rejects fp8_e4m3, so it also flips KV to
		// bf16/auto). Keeping those keys out of this shared base is what lets H200
		// run a different backend than B200/B300 — see the per-profile env blocks.
		//
		// 753B FP8 load + DSA warmup is slow on every card — long runner timeout +
		// first-healthy grace are the only universally-safe GLM env, kept here.
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		// Inference runs COMPILED by default (vLLM CompilationMode.VLLM_COMPILE +
		// CUDA graphs) for decode throughput — on GLM that is 817 vs 157 tok/s
		// (+6.8× TPOT) over eager. The PoC forward runs EAGER on its own via
		// gonka-poc poc_model_runner (set_forward_context skip_compiled=True), so
		// one image gives eager PoC (bit-compat, ~+25% nonces) + compiled inference.
		// NOT forced by the runner-patch (matches b300-minimax) — these are the
		// effective vLLM defaults, shown for the dashboard. For the EXCEPTIONAL
		// case where inference must also be eager, the operator passes
		// --enforce-eager at launch (the patch neither forces nor strips it).
		compilation_mode: *3 | int           // VLLM_COMPILE (compiled inference)
		cudagraph_mode:   *"PIECEWISE" | string // vLLM compiled default; not forced
	}
}
